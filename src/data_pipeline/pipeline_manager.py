"""
Main pipeline orchestrator for managing interpolation data preparation
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np

from .manifest_manager import ManifestManager
from .data_processor import DataProcessor
from .experiment_configs import ExperimentConfig, PRESET_EXPERIMENTS
from .utils import (
    Task, DataSample, create_directory_structure, 
    load_patient_splits, save_patient_splits, get_patient_from_path
)


logger = logging.getLogger(__name__)


class PipelineManager:
    """Main orchestrator for the interpolation data pipeline"""
    
    def __init__(self, base_dir: Path, processed_data_dir: Path, 
                 manifest_file: str = 'manifest.csv'):
        """
        Initialize pipeline manager
        
        Args:
            base_dir: Base directory for interpolated data output
            processed_data_dir: Directory containing processed 4D-Lung-Cycles data
            manifest_file: Name of manifest file
        """
        self.base_dir = Path(base_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Create directory structure
        create_directory_structure(self.base_dir)
        
        # Initialize components
        self.manifest_manager = ManifestManager(self.base_dir / manifest_file)
        self.data_processor = DataProcessor(self.processed_data_dir, self.base_dir)
        
        # Load or initialize patient splits
        self.splits_file = self.base_dir / 'splits' / 'patient_splits.json'
        self.patient_splits = self._load_or_create_splits()
        
        # Cache for processed data info
        self._processed_data_cache = None
    
    def _load_or_create_splits(self) -> Dict[str, List[str]]:
        """Load existing splits or create new ones"""
        if self.splits_file.exists():
            return load_patient_splits(self.splits_file)
        else:
            # Create default splits from available patients
            patients = self._get_available_patients()
            if not patients:
                logger.warning("No patients found in processed data directory")
                return {'train': [], 'val': [], 'test': []}
            
            # Default split: 70% train, 15% val, 15% test
            n_patients = len(patients)
            n_train = int(0.7 * n_patients)
            n_val = int(0.15 * n_patients)
            
            # Shuffle for randomness (with fixed seed for reproducibility)
            np.random.seed(42)
            shuffled = np.random.permutation(patients).tolist()
            
            splits = {
                'train': shuffled[:n_train],
                'val': shuffled[n_train:n_train + n_val],
                'test': shuffled[n_train + n_val:]
            }
            
            save_patient_splits(splits, self.splits_file)
            logger.info(f"Created patient splits: train={len(splits['train'])}, "
                       f"val={len(splits['val'])}, test={len(splits['test'])}")
            
            return splits
    
    def _get_available_patients(self) -> List[str]:
        """Get list of available patients from processed data"""
        if not self.processed_data_dir.exists():
            return []
        
        # List directories in processed data dir
        patients = [d.name for d in self.processed_data_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
        return sorted(patients)
    
    def _scan_processed_data(self) -> pd.DataFrame:
        """Scan processed data directory and create inventory"""
        if self._processed_data_cache is not None:
            return self._processed_data_cache
        
        data_records = []
        
        for patient_dir in self.processed_data_dir.iterdir():
            if not patient_dir.is_dir():
                continue
                
            patient_id = patient_dir.name
            
            # Each subdirectory is a breathing cycle (series)
            for cycle_dir in patient_dir.iterdir():
                if not cycle_dir.is_dir():
                    continue
                
                series_id = cycle_dir.name
                
                # Count phase files to determine number of slices
                phase_files = sorted(cycle_dir.glob('phase_*.npy'))
                if not phase_files:
                    continue
                
                # Get dimensions from first phase
                first_phase = np.load(phase_files[0], mmap_mode='r')
                num_slices = first_phase.shape[0]
                
                data_records.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'num_slices': num_slices,
                    'num_phases': len(phase_files),
                    'shape': first_phase.shape
                })
        
        self._processed_data_cache = pd.DataFrame(data_records)
        return self._processed_data_cache
    
    def get_missing_tasks(self, filter_config: Optional[Dict] = None) -> List[Task]:
        """
        Get list of tasks that need to be processed
        
        Args:
            filter_config: Optional filters
                - patients: List of patient IDs
                - experiments: List of experiment names
                - phase_ranges: List of phase ranges (e.g., ['0-50', '0-100'])
                - max_slices_per_patient: Limit slices per patient
                - priority: Only get experiments with this priority level
        """
        filter_config = filter_config or {}
        
        # Scan available data
        processed_data = self._scan_processed_data()
        
        if processed_data.empty:
            logger.warning("No processed data found")
            return []
        
        # Apply patient filter
        if 'patients' in filter_config:
            processed_data = processed_data[
                processed_data['patient_id'].isin(filter_config['patients'])
            ]
        
        # Get experiments to check
        if 'experiments' in filter_config:
            experiments = {k: v for k, v in PRESET_EXPERIMENTS.items() 
                         if k in filter_config['experiments']}
        else:
            experiments = PRESET_EXPERIMENTS
        
        # Filter by priority
        if 'priority' in filter_config:
            experiments = {k: v for k, v in experiments.items() 
                         if v.priority == filter_config['priority']}
        
        # Get phase ranges
        phase_ranges = filter_config.get('phase_ranges', ['0-50', '0-100'])
        
        tasks = []
        
        # Generate tasks for each combination
        for _, row in processed_data.iterrows():
            patient_id = row['patient_id']
            series_id = row['series_id']
            num_slices = row['num_slices']
            
            for exp_name, exp_config in experiments.items():
                # Get slice indices based on experiment config
                slice_indices = exp_config.get_slice_indices(num_slices)
                
                # Apply max slices filter
                if 'max_slices_per_patient' in filter_config:
                    slice_indices = slice_indices[:filter_config['max_slices_per_patient']]
                
                for slice_num in slice_indices:
                    for phase_range in phase_ranges:
                        # Check if experiment supports this phase range
                        if not exp_config.supports_phase_range(phase_range):
                            continue
                        
                        task = Task(
                            patient_id=patient_id,
                            experiment_id=exp_name,
                            series_id=series_id,
                            slice_num=slice_num,
                            phase_range=phase_range,
                            cycle_id=series_id  # For 4D-Lung-Cycles, series_id is cycle_id
                        )
                        
                        # Check if already exists in manifest
                        output_path = task.get_output_path(self.base_dir)
                        if not output_path.exists():
                            tasks.append(task)
        
        logger.info(f"Found {len(tasks)} missing tasks")
        return tasks
    
    def process_tasks(self, tasks: List[Task], max_workers: int = 4,
                     show_progress: bool = True) -> Dict[str, int]:
        """
        Process a list of tasks
        
        Returns:
            Dictionary with counts of 'success', 'failed', 'skipped'
        """
        if not tasks:
            logger.warning("No tasks to process")
            return {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Group tasks by experiment
        tasks_by_experiment = {}
        for task in tasks:
            if task.experiment_id not in tasks_by_experiment:
                tasks_by_experiment[task.experiment_id] = []
            tasks_by_experiment[task.experiment_id].append(task)
        
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Process each experiment's tasks
        for exp_id, exp_tasks in tasks_by_experiment.items():
            if exp_id not in PRESET_EXPERIMENTS:
                logger.error(f"Unknown experiment: {exp_id}")
                results['failed'] += len(exp_tasks)
                continue
            
            experiment = PRESET_EXPERIMENTS[exp_id]
            
            # Mark tasks as in progress
            self.manifest_manager.mark_in_progress(exp_tasks)
            
            # Process in batches
            logger.info(f"Processing {len(exp_tasks)} tasks for experiment: {exp_id}")
            
            if show_progress:
                pbar = tqdm(total=len(exp_tasks), desc=f"Processing {exp_id}")
            
            # Process tasks
            batch_results = self.data_processor.process_batch(
                exp_tasks, experiment, max_workers=max_workers
            )
            
            # Update manifest and counts
            for task in batch_results['success']:
                output_path = task.get_output_path(self.base_dir)
                
                # Calculate file size
                try:
                    size_mb = sum(
                        (output_path / f).stat().st_size / 1024 / 1024
                        for f in ['input_frames.npy', 'target_frames.npy']
                        if (output_path / f).exists()
                    )
                except:
                    size_mb = 0
                
                # Get metadata for HFR info
                metadata_path = output_path / 'metadata.json'
                num_frames = None
                is_hfr = False
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        num_frames = metadata.get('num_frames')
                        is_hfr = metadata.get('is_hfr', False)
                
                self.manifest_manager.update_status(
                    task, 'completed', output_path, size_mb
                )
                results['success'] += 1
                
                if show_progress:
                    pbar.update(1)
            
            for task in batch_results['failed']:
                self.manifest_manager.update_status(task, 'failed')
                results['failed'] += 1
                
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        logger.info(f"Processing complete: {results}")
        return results
    
    def get_available_data(self, split: str = 'train', 
                          experiment_filter: Optional[List[str]] = None) -> List[DataSample]:
        """
        Get list of available data samples for training/evaluation
        
        Args:
            split: 'train', 'val', or 'test'
            experiment_filter: Optional list of experiment IDs to include
        """
        # Get completed samples from manifest
        completed = self.manifest_manager.get_completed_samples(split, self.patient_splits)
        
        if experiment_filter:
            completed = completed[completed['experiment_id'].isin(experiment_filter)]
        
        samples = []
        
        for _, row in completed.iterrows():
            output_path = Path(row['output_path'])
            
            sample = DataSample(
                patient_id=row['patient_id'],
                experiment_id=row['experiment_id'],
                series_id=row['series_id'],
                slice_num=row['slice_num'],
                phase_range=row['phase_range'],
                split=split,
                input_path=output_path / 'input_frames.npy',
                target_path=output_path / 'target_frames.npy',
                metadata={
                    'cycle_id': row.get('cycle_id'),
                    'data_type': '2d'  # For future 3D extension
                }
            )
            samples.append(sample)
        
        logger.info(f"Found {len(samples)} samples for split '{split}'")
        return samples
    
    def get_status_summary(self) -> Dict:
        """Get comprehensive status summary"""
        summary = {
            'manifest_status': self.manifest_manager.get_status_summary(),
            'storage_usage': self.manifest_manager.get_storage_usage(),
            'patient_splits': {k: len(v) for k, v in self.patient_splits.items()},
            'experiments': {}
        }
        
        # Add per-experiment progress
        for exp_id in PRESET_EXPERIMENTS:
            summary['experiments'][exp_id] = self.manifest_manager.get_experiment_progress(exp_id)
        
        return summary
    
    def export_training_config(self, output_path: Path, 
                             experiments: List[str],
                             phase_ranges: List[str] = ['0-50']) -> None:
        """Export configuration for training scripts"""
        config = {
            'base_dir': str(self.base_dir),
            'experiments': experiments,
            'phase_ranges': phase_ranges,
            'splits': self.patient_splits,
            'data_samples': {}
        }
        
        # Add sample counts for each split
        for split in ['train', 'val', 'test']:
            samples = self.get_available_data(split, experiments)
            config['data_samples'][split] = len(samples)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Exported training config to {output_path}")
    
    def cleanup_failed_tasks(self) -> int:
        """Reset failed tasks to allow reprocessing"""
        failed = self.manifest_manager.manifest[
            self.manifest_manager.manifest['status'] == 'failed'
        ]
        
        # Remove failed entries
        self.manifest_manager.manifest = self.manifest_manager.manifest[
            self.manifest_manager.manifest['status'] != 'failed'
        ]
        self.manifest_manager.save_manifest()
        
        return len(failed)