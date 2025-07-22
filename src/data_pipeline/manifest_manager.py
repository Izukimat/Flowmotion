"""
Manifest management for tracking processed data inventory
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import threading
import json
from .utils import Task


class ManifestManager:
    """Manages the inventory of processed interpolation data"""
    
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        
        # Define manifest schema
        self.columns = [
            'patient_id', 'experiment_id', 'series_id', 'slice_num', 
            'phase_range', 'status', 'output_path', 'created_at', 
            'file_size_mb', 'cycle_id', 'num_frames', 'is_hfr'
        ]
        
        # Initialize or load manifest
        if self.manifest_path.exists():
            self.manifest = self.load_manifest()
        else:
            self.manifest = pd.DataFrame(columns=self.columns)
            self.save_manifest()
    
    def load_manifest(self) -> pd.DataFrame:
        """Load manifest from CSV file"""
        with self._lock:
            df = pd.read_csv(self.manifest_path)
            # Ensure all columns exist
            for col in self.columns:
                if col not in df.columns:
                    df[col] = None
            return df
    
    def save_manifest(self) -> None:
        """Save manifest to CSV file"""
        with self._lock:
            self.manifest.to_csv(self.manifest_path, index=False)
    
    def update_status(self, task: Task, status: str, output_path: Optional[Path] = None,
                     file_size_mb: Optional[float] = None) -> None:
        """Update the status of a task in the manifest"""
        with self._lock:
            # Create a unique identifier for the task
            mask = (
                (self.manifest['patient_id'] == task.patient_id) &
                (self.manifest['experiment_id'] == task.experiment_id) &
                (self.manifest['series_id'] == task.series_id) &
                (self.manifest['slice_num'] == task.slice_num) &
                (self.manifest['phase_range'] == task.phase_range)
            )
            
            if mask.any():
                # Update existing entry
                idx = self.manifest[mask].index[0]
                self.manifest.loc[idx, 'status'] = status
                if output_path:
                    self.manifest.loc[idx, 'output_path'] = str(output_path)
                if file_size_mb is not None:
                    self.manifest.loc[idx, 'file_size_mb'] = file_size_mb
                self.manifest.loc[idx, 'created_at'] = datetime.now().isoformat()
            else:
                # Add new entry
                new_row = {
                    'patient_id': task.patient_id,
                    'experiment_id': task.experiment_id,
                    'series_id': task.series_id,
                    'slice_num': task.slice_num,
                    'phase_range': task.phase_range,
                    'status': status,
                    'output_path': str(output_path) if output_path else None,
                    'created_at': datetime.now().isoformat(),
                    'file_size_mb': file_size_mb,
                    'cycle_id': task.cycle_id
                }
                self.manifest = pd.concat([self.manifest, pd.DataFrame([new_row])], ignore_index=True)
            
            self.save_manifest()
    
    def get_missing(self, patients: Optional[List[str]] = None,
                   experiments: Optional[List[str]] = None,
                   phase_ranges: Optional[List[str]] = None,
                   series_ids: Optional[List[str]] = None) -> List[Dict]:
        """Get list of missing combinations based on filters"""
        # Get all expected combinations from processed data catalog
        expected = self._get_expected_combinations(patients, experiments, phase_ranges, series_ids)
        
        # Get existing combinations
        existing = self._get_existing_combinations()
        
        # Find missing
        missing = []
        for exp in expected:
            key = (exp['patient_id'], exp['experiment_id'], exp['series_id'], 
                  exp['slice_num'], exp['phase_range'])
            if key not in existing:
                missing.append(exp)
        
        return missing
    
    def _get_existing_combinations(self) -> Set[Tuple]:
        """Get set of existing combinations from manifest"""
        completed = self.manifest[self.manifest['status'] == 'completed']
        return set(
            tuple(row) for row in completed[
                ['patient_id', 'experiment_id', 'series_id', 'slice_num', 'phase_range']
            ].values
        )
    
    def _get_expected_combinations(self, patients: Optional[List[str]],
                                 experiments: Optional[List[str]],
                                 phase_ranges: Optional[List[str]],
                                 series_ids: Optional[List[str]]) -> List[Dict]:
        """Generate expected combinations based on filters"""
        # This would normally scan the processed data directory
        # For now, return empty list - to be implemented with actual data scanning
        # This is where you'd integrate with the processed data manifest
        return []
    
    def mark_in_progress(self, tasks: List[Task]) -> None:
        """Mark multiple tasks as in progress"""
        for task in tasks:
            self.update_status(task, 'in_progress')
    
    def get_status_summary(self) -> Dict[str, int]:
        """Get summary of task statuses"""
        if self.manifest.empty:
            return {'completed': 0, 'in_progress': 0, 'failed': 0, 'missing': 0}
        
        status_counts = self.manifest['status'].value_counts().to_dict()
        # Add zero counts for missing statuses
        for status in ['completed', 'in_progress', 'failed']:
            if status not in status_counts:
                status_counts[status] = 0
        
        return status_counts
    
    def get_completed_samples(self, split: Optional[str] = None,
                            patient_splits: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """Get completed samples, optionally filtered by split"""
        completed = self.manifest[self.manifest['status'] == 'completed'].copy()
        
        if split and patient_splits:
            # Filter by patient split
            split_patients = patient_splits.get(split, [])
            completed = completed[completed['patient_id'].isin(split_patients)]
        
        return completed
    
    def get_experiment_progress(self, experiment_id: str) -> Dict:
        """Get detailed progress for a specific experiment"""
        exp_data = self.manifest[self.manifest['experiment_id'] == experiment_id]
        
        total = len(exp_data)
        if total == 0:
            return {'total': 0, 'completed': 0, 'progress': 0.0}
        
        completed = len(exp_data[exp_data['status'] == 'completed'])
        in_progress = len(exp_data[exp_data['status'] == 'in_progress'])
        failed = len(exp_data[exp_data['status'] == 'failed'])
        
        return {
            'total': total,
            'completed': completed,
            'in_progress': in_progress,
            'failed': failed,
            'progress': completed / total * 100
        }
    
    def cleanup_orphaned_entries(self, base_dir: Path) -> int:
        """Remove manifest entries where output files don't exist"""
        removed = 0
        to_remove = []
        
        for idx, row in self.manifest.iterrows():
            if row['status'] == 'completed' and row['output_path']:
                output_path = Path(row['output_path'])
                if not output_path.exists():
                    to_remove.append(idx)
                    removed += 1
        
        if to_remove:
            self.manifest = self.manifest.drop(to_remove)
            self.save_manifest()
        
        return removed
    
    def export_summary_report(self, output_path: Path) -> None:
        """Export a summary report of the current inventory"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_entries': len(self.manifest),
            'status_summary': self.get_status_summary(),
            'experiments': {},
            'patients': {}
        }
        
        # Per-experiment summary
        for exp_id in self.manifest['experiment_id'].unique():
            report['experiments'][exp_id] = self.get_experiment_progress(exp_id)
        
        # Per-patient summary
        for patient_id in self.manifest['patient_id'].unique():
            patient_data = self.manifest[self.manifest['patient_id'] == patient_id]
            report['patients'][patient_id] = {
                'total_tasks': len(patient_data),
                'completed': len(patient_data[patient_data['status'] == 'completed'])
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def get_storage_usage(self) -> Dict[str, float]:
        """Calculate storage usage statistics"""
        completed = self.manifest[self.manifest['status'] == 'completed']
        
        if 'file_size_mb' not in completed.columns or completed.empty:
            return {'total_mb': 0, 'total_gb': 0, 'avg_size_mb': 0}
        
        total_mb = completed['file_size_mb'].sum()
        avg_mb = completed['file_size_mb'].mean()
        
        return {
            'total_mb': total_mb,
            'total_gb': total_mb / 1024,
            'avg_size_mb': avg_mb,
            'num_files': len(completed)
        }