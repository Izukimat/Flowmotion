#!/usr/bin/env python
"""
Create train/val/test patient splits and metadata CSV for NumPy array lung CT data
Updated to work with preprocessed data structure at /home/ragenius_admin/azureblob/4D-Lung-Interpolated/data/
"""

import argparse
import sys
from pathlib import Path
import logging
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import re


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def discover_data_structure(data_root: Path) -> List[Dict]:
    """
    Discover all available samples in the NumPy data directory structure
    
    Returns:
        List of sample dictionaries with metadata
    """
    logging.info(f"Scanning data directory: {data_root}")
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    
    samples = []
    patients_found = set()
    experiments_found = set()
    
    # Walk through directory structure: patient/experiment/series/slice/phase/
    for patient_dir in data_root.iterdir():
        if not patient_dir.is_dir():
            continue
            
        patient_id = patient_dir.name
        patients_found.add(patient_id)
        
        # Look for experiment directories
        for experiment_dir in patient_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
                
            experiment_id = experiment_dir.name
            experiments_found.add(experiment_id)
            
            # Look for series directories
            for series_dir in experiment_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                    
                series_id = series_dir.name
                
                # Look for slice directories
                for slice_dir in series_dir.iterdir():
                    if not slice_dir.is_dir():
                        continue
                    
                    # Extract slice number from directory name (slice_XXXX)
                    slice_match = re.match(r'slice_(\d+)', slice_dir.name)
                    if not slice_match:
                        continue
                    
                    slice_num = int(slice_match.group(1))
                    
                    # Look for phase directories
                    for phase_dir in slice_dir.iterdir():
                        if not phase_dir.is_dir():
                            continue
                        
                        # Extract phase range from directory name (phase_X-Y)
                        phase_match = re.match(r'phase_(.+)', phase_dir.name)
                        if not phase_match:
                            continue
                        
                        phase_range = phase_match.group(1)
                        
                        # Check if required files exist
                        input_path = phase_dir / 'input_frames.npy'
                        target_path = phase_dir / 'target_frames.npy'
                        metadata_path = phase_dir / 'metadata.json'
                        
                        if input_path.exists() and target_path.exists():
                            # Load metadata if available
                            sample_metadata = {}
                            if metadata_path.exists():
                                try:
                                    with open(metadata_path, 'r') as f:
                                        sample_metadata = json.load(f)
                                except Exception as e:
                                    logging.warning(f"Error loading metadata from {metadata_path}: {e}")
                            
                            # Verify file sizes and shapes
                            try:
                                input_frames = np.load(input_path, mmap_mode='r')
                                target_frames = np.load(target_path, mmap_mode='r')
                                
                                sample = {
                                    'patient_id': patient_id,
                                    'experiment_id': experiment_id,
                                    'series_id': series_id,
                                    'slice_num': slice_num,
                                    'phase_range': phase_range,
                                    'input_path': str(input_path),
                                    'target_path': str(target_path),
                                    'metadata_path': str(metadata_path) if metadata_path.exists() else '',
                                    'input_shape': list(input_frames.shape),
                                    'target_shape': list(target_frames.shape),
                                    'input_dtype': str(input_frames.dtype),
                                    'target_dtype': str(target_frames.dtype),
                                    'sample_metadata': sample_metadata
                                }
                                
                                samples.append(sample)
                                
                            except Exception as e:
                                logging.warning(f"Error loading NumPy files from {phase_dir}: {e}")
                        else:
                            logging.debug(f"Missing required files in {phase_dir}")
    
    logging.info(f"Discovery complete:")
    logging.info(f"  - Found {len(patients_found)} patients: {sorted(patients_found)}")
    logging.info(f"  - Found {len(experiments_found)} experiments: {sorted(experiments_found)}")
    logging.info(f"  - Found {len(samples)} total samples")
    
    return samples


def analyze_data_distribution(samples: List[Dict]) -> Dict:
    """Analyze the distribution of discovered data"""
    analysis = {
        'total_samples': len(samples),
        'patients': set(),
        'experiments': defaultdict(int),
        'phase_ranges': defaultdict(int),
        'slice_distribution': defaultdict(list),
        'shape_distribution': defaultdict(int),
        'dtype_distribution': defaultdict(int)
    }
    
    for sample in samples:
        analysis['patients'].add(sample['patient_id'])
        analysis['experiments'][sample['experiment_id']] += 1
        analysis['phase_ranges'][sample['phase_range']] += 1
        analysis['slice_distribution'][sample['patient_id']].append(sample['slice_num'])
        
        input_shape = tuple(sample['input_shape'])
        target_shape = tuple(sample['target_shape'])
        analysis['shape_distribution'][f"input_{input_shape}"] += 1
        analysis['shape_distribution'][f"target_{target_shape}"] += 1
        
        analysis['dtype_distribution'][f"input_{sample['input_dtype']}"] += 1
        analysis['dtype_distribution'][f"target_{sample['target_dtype']}"] += 1
    
    # Convert sets to counts
    analysis['num_patients'] = len(analysis['patients'])
    analysis['patients'] = sorted(list(analysis['patients']))
    
    # Calculate slice statistics per patient
    slice_stats = {}
    for patient, slices in analysis['slice_distribution'].items():
        slice_stats[patient] = {
            'min_slice': min(slices),
            'max_slice': max(slices),
            'num_slices': len(set(slices)),
            'total_samples': len(slices)
        }
    analysis['slice_stats'] = slice_stats
    
    return analysis


def create_patient_splits(
    patients: List[str], 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create patient-level train/val/test splits
    
    Args:
        patients: List of patient IDs
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys and patient lists
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle patients
    patients_shuffled = np.random.permutation(patients).tolist()
    
    # Calculate split sizes
    n_total = len(patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Remainder goes to test
    
    # Create splits
    splits = {
        'train': patients_shuffled[:n_train],
        'val': patients_shuffled[n_train:n_train + n_val],
        'test': patients_shuffled[n_train + n_val:n_train + n_val + n_test]
    }
    
    logging.info(f"Created patient splits:")
    logging.info(f"  - Train: {len(splits['train'])} patients ({len(splits['train'])/n_total:.1%})")
    logging.info(f"  - Val: {len(splits['val'])} patients ({len(splits['val'])/n_total:.1%})")
    logging.info(f"  - Test: {len(splits['test'])} patients ({len(splits['test'])/n_total:.1%})")
    
    return splits


def assign_splits_to_samples(samples: List[Dict], patient_splits: Dict[str, List[str]]) -> List[Dict]:
    """Assign train/val/test split to each sample based on patient"""
    # Create patient to split mapping
    patient_to_split = {}
    for split_name, patient_list in patient_splits.items():
        for patient in patient_list:
            patient_to_split[patient] = split_name
    
    # Assign splits to samples
    samples_with_splits = []
    split_counts = defaultdict(int)
    
    for sample in samples:
        patient_id = sample['patient_id']
        if patient_id in patient_to_split:
            sample_copy = sample.copy()
            sample_copy['split'] = patient_to_split[patient_id]
            samples_with_splits.append(sample_copy)
            split_counts[patient_to_split[patient_id]] += 1
        else:
            logging.warning(f"Patient {patient_id} not found in splits")
    
    logging.info(f"Sample distribution:")
    for split_name, count in split_counts.items():
        logging.info(f"  - {split_name}: {count} samples")
    
    return samples_with_splits


def create_metadata_csv(samples: List[Dict], output_path: Path) -> pd.DataFrame:
    """
    Create metadata CSV with required columns for training pipeline
    
    Required columns: patient_id, split, experiment_id, series_id, slice_num, phase_range, input_path, target_path
    """
    # Create DataFrame with required columns
    df_data = []
    
    for sample in samples:
        row = {
            'patient_id': sample['patient_id'],
            'split': sample['split'],
            'experiment_id': sample['experiment_id'],
            'series_id': sample['series_id'],
            'slice_num': sample['slice_num'],
            'phase_range': sample['phase_range'],
            'input_path': sample['input_path'],
            'target_path': sample['target_path']
        }
        
        # Add optional metadata columns
        if sample.get('metadata_path'):
            row['metadata_path'] = sample['metadata_path']
        
        # Add shape and dtype information for reference
        row['input_shape'] = str(sample['input_shape'])
        row['target_shape'] = str(sample['target_shape'])
        row['input_dtype'] = sample['input_dtype']
        row['target_dtype'] = sample['target_dtype']
        
        df_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Sort by patient, experiment, series, slice, phase for consistent ordering
    df = df.sort_values(['patient_id', 'experiment_id', 'series_id', 'slice_num', 'phase_range'])
    df = df.reset_index(drop=True)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logging.info(f"Saved metadata CSV to: {output_path}")
    logging.info(f"CSV contains {len(df)} samples with columns: {list(df.columns)}")
    
    return df


def save_analysis_report(analysis: Dict, patient_splits: Dict, output_dir: Path):
    """Save detailed analysis report"""
    report_path = output_dir / 'data_analysis_report.json'
    
    # Convert sets and defaultdicts to regular dicts for JSON serialization
    serializable_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, (set, list, tuple)):
            serializable_analysis[key] = list(value)
        elif isinstance(value, defaultdict):
            serializable_analysis[key] = dict(value)
        else:
            serializable_analysis[key] = value
    
    report = {
        'analysis': serializable_analysis,
        'patient_splits': patient_splits,
        'summary': {
            'total_patients': len(analysis['patients']),
            'total_samples': analysis['total_samples'],
            'experiments': list(analysis['experiments'].keys()),
            'phase_ranges': list(analysis['phase_ranges'].keys())
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"Saved analysis report to: {report_path}")


def main():
    """Main function to create splits and metadata CSV"""
    parser = argparse.ArgumentParser(
        description='Create train/val/test splits and metadata CSV for lung CT NumPy data'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='/home/ragenius_admin/azureblob/4D-Lung-Interpolated/data/',
        help='Root directory containing patient data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./splits',
        help='Output directory for splits and metadata'
    )
    parser.add_argument(
        '--csv-filename',
        type=str,
        default='metadata.csv',
        help='Name of output CSV file'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of patients for training'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Fraction of patients for validation'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Fraction of patients for testing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits'
    )
    parser.add_argument(
        '--phase-filter',
        type=str,
        default=None,
        help='Optional phase range filter (e.g., "0-100" to only include full cycle data)'
    )
    parser.add_argument(
        '--experiment-filter',
        type=str,
        nargs='+',
        default=None,
        help='Optional experiment filter (e.g., "hfr_optical_flow_8fps")'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    if not data_root.exists():
        logging.error(f"Data root directory not found: {data_root}")
        sys.exit(1)
    
    try:
        # Discover data structure
        logging.info("=== Discovering Data Structure ===")
        samples = discover_data_structure(data_root)
        
        if not samples:
            logging.error("No valid samples found in data directory")
            sys.exit(1)
        
        # Apply filters if specified
        if args.phase_filter:
            logging.info(f"Filtering by phase range: {args.phase_filter}")
            samples = [s for s in samples if s['phase_range'] == args.phase_filter]
            logging.info(f"After phase filtering: {len(samples)} samples")
        
        if args.experiment_filter:
            logging.info(f"Filtering by experiments: {args.experiment_filter}")
            samples = [s for s in samples if s['experiment_id'] in args.experiment_filter]
            logging.info(f"After experiment filtering: {len(samples)} samples")
        
        if not samples:
            logging.error("No samples remaining after filtering")
            sys.exit(1)
        
        # Analyze data distribution
        logging.info("=== Analyzing Data Distribution ===")
        analysis = analyze_data_distribution(samples)
        
        # Print analysis summary
        logging.info(f"Data Analysis Summary:")
        logging.info(f"  - Total samples: {analysis['total_samples']}")
        logging.info(f"  - Unique patients: {analysis['num_patients']}")
        logging.info(f"  - Experiments: {dict(analysis['experiments'])}")
        logging.info(f"  - Phase ranges: {dict(analysis['phase_ranges'])}")
        logging.info(f"  - Input shapes: {dict(analysis['shape_distribution'])}")
        
        # Create patient splits
        logging.info("=== Creating Patient Splits ===")
        patient_splits = create_patient_splits(
            analysis['patients'],
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
        
        # Assign splits to samples
        logging.info("=== Assigning Splits to Samples ===")
        samples_with_splits = assign_splits_to_samples(samples, patient_splits)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata CSV
        logging.info("=== Creating Metadata CSV ===")
        csv_path = output_dir / args.csv_filename
        metadata_df = create_metadata_csv(samples_with_splits, csv_path)
        
        # Save patient splits as JSON
        splits_path = output_dir / 'patient_splits.json'
        with open(splits_path, 'w') as f:
            json.dump(patient_splits, f, indent=2)
        logging.info(f"Saved patient splits to: {splits_path}")
        
        # Save analysis report
        save_analysis_report(analysis, patient_splits, output_dir)
        
        # Final summary
        logging.info("=== Summary ===")
        logging.info(f"Successfully created splits for {len(samples_with_splits)} samples")
        logging.info(f"Output files:")
        logging.info(f"  - Metadata CSV: {csv_path}")
        logging.info(f"  - Patient splits: {splits_path}")
        logging.info(f"  - Analysis report: {output_dir / 'data_analysis_report.json'}")
        
        # Display split statistics
        split_stats = metadata_df.groupby('split').agg({
            'patient_id': 'nunique',
            'experiment_id': lambda x: len(set(x)),
            'slice_num': 'count'
        }).rename(columns={
            'patient_id': 'num_patients',
            'experiment_id': 'num_experiments', 
            'slice_num': 'num_samples'
        })
        
        logging.info("\nFinal split statistics:")
        logging.info(f"\n{split_stats}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error creating splits: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())