#!/usr/bin/env python
"""
Initialize the interpolation pipeline and scan available data
"""

import argparse
import sys
from pathlib import Path
import logging
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline import PipelineManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Initialize interpolation pipeline and scan available data'
    )
    
    # Required arguments
    parser.add_argument(
        '--base-dir', 
        type=Path, 
        required=True,
        help='Base directory for interpolated data output'
    )
    parser.add_argument(
        '--processed-dir', 
        type=Path, 
        required=True,
        help='Directory containing processed 4D-Lung-Cycles data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--scan-only', 
        action='store_true',
        help='Only scan and report available data without creating structure'
    )
    parser.add_argument(
        '--force-splits', 
        action='store_true',
        help='Force recreation of patient splits even if they exist'
    )
    parser.add_argument(
        '--train-ratio', 
        type=float, 
        default=0.7,
        help='Ratio of patients for training set (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio', 
        type=float, 
        default=0.15,
        help='Ratio of patients for validation set (default: 0.15)'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate directories
    if not args.processed_dir.exists():
        logger.error(f"Processed data directory not found: {args.processed_dir}")
        return 1
    
    # Create base directory if needed
    if not args.scan_only:
        args.base_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline manager...")
        pipeline = PipelineManager(args.base_dir, args.processed_dir)
        
        # Force recreate splits if requested
        if args.force_splits and not args.scan_only:
            logger.info("Recreating patient splits...")
            from src.data_pipeline.utils import save_patient_splits
            import numpy as np
            
            patients = pipeline._get_available_patients()
            n_patients = len(patients)
            n_train = int(args.train_ratio * n_patients)
            n_val = int(args.val_ratio * n_patients)
            
            np.random.seed(42)
            shuffled = np.random.permutation(patients).tolist()
            
            splits = {
                'train': shuffled[:n_train],
                'val': shuffled[n_train:n_train + n_val],
                'test': shuffled[n_train + n_val:]
            }
            
            save_patient_splits(splits, pipeline.splits_file)
            pipeline.patient_splits = splits
            logger.info(f"Created new splits: train={len(splits['train'])}, "
                       f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        # Scan available data
        logger.info("Scanning processed data directory...")
        processed_data = pipeline._scan_processed_data()
        
        if processed_data.empty:
            logger.warning("No processed data found!")
            return 1
        
        # Report findings
        print("\n" + "="*60)
        print("PROCESSED DATA SUMMARY")
        print("="*60)
        print(f"Total patients: {processed_data['patient_id'].nunique()}")
        print(f"Total breathing cycles: {len(processed_data)}")
        print(f"Average cycles per patient: {len(processed_data) / processed_data['patient_id'].nunique():.1f}")
        
        # Show sample of data
        print("\nSample data structure:")
        print(processed_data.head())
        
        # Show patient splits
        print("\n" + "="*60)
        print("PATIENT SPLITS")
        print("="*60)
        for split, patients in pipeline.patient_splits.items():
            print(f"{split}: {len(patients)} patients")
            if args.verbose:
                print(f"  {', '.join(patients[:5])}{'...' if len(patients) > 5 else ''}")
        
        if not args.scan_only:
            # Create experiment configs directory and save presets
            exp_dir = args.base_dir / 'experiment_configs'
            exp_dir.mkdir(exist_ok=True)
            
            from src.data_pipeline import PRESET_EXPERIMENTS
            for exp_name, exp_config in PRESET_EXPERIMENTS.items():
                exp_path = exp_dir / f"{exp_name}.json"
                if not exp_path.exists():
                    exp_config.save(exp_path)
                    logger.info(f"Saved experiment config: {exp_name}")
            
            # Estimate storage requirements
            print("\n" + "="*60)
            print("STORAGE ESTIMATES")
            print("="*60)
            
            # Calculate for different scenarios focusing on HFR
            scenarios = [
                ("Test run (1 patient, 3 HFR methods, 3 slices)", 1, 3, 3, 82),
                ("Small HFR (3 patients, 3 HFR methods, 10 slices)", 3, 3, 10, 82),
                ("Medium HFR (10 patients, 3 HFR methods, 30 slices)", 10, 3, 30, 82),
                ("Full HFR (20 patients, 3 HFR methods, 100 slices)", 20, 3, 100, 82)
            ]
            
            print("HFR experiments generate ~82 frames per task (10 original + 72 interpolated)")
            print()
            
            for desc, n_pat, n_exp, n_slice, n_frames in scenarios:
                # Rough estimate: 2MB per frame pair for 512x512 images
                mb_per_task = n_frames * 0.5  # ~0.5MB per frame in compressed .npy
                total_tasks = n_pat * n_exp * n_slice
                total_gb = (total_tasks * mb_per_task) / 1024
                print(f"{desc:.<55} ~{total_gb:.1f} GB")
            
            print("\n" + "="*60)
            print("AVAILABLE EXPERIMENTS")
            print("="*60)
            
            # Group experiments by type
            hfr_exps = {k: v for k, v in PRESET_EXPERIMENTS.items() if 'hfr_' in k}
            test_exps = {k: v for k, v in PRESET_EXPERIMENTS.items() if 'test_' in k}
            
            print("\nHigh Frame Rate (HFR) Training Data Generation:")
            for exp_name, exp_config in sorted(hfr_exps.items()):
                fps = exp_config.params.get('frames_per_interval', 0)
                print(f"  {exp_name:.<30} {exp_config.method}, {fps} fps between phases")
            
            print("\nTest/Evaluation Experiments (for trained models):")
            for exp_name, exp_config in sorted(test_exps.items()):
                phases = f"{exp_config.input_phases[0]}-{exp_config.input_phases[-1]}"
                print(f"  {exp_name:.<30} {exp_config.method}, phases {phases}")
            
            print("\n" + "="*60)
            print(f"Pipeline initialized successfully at: {args.base_dir}")
            print("="*60)
            
            # Save initialization summary
            summary = {
                'base_dir': str(args.base_dir),
                'processed_dir': str(args.processed_dir),
                'num_patients': processed_data['patient_id'].nunique(),
                'num_cycles': len(processed_data),
                'patient_splits': {k: len(v) for k, v in pipeline.patient_splits.items()},
                'available_experiments': list(PRESET_EXPERIMENTS.keys())
            }
            
            summary_path = args.base_dir / 'pipeline_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nNext steps:")
            print(f"1. Test with small batch: ./test_interpolation.sh")
            print(f"2. Run full processing: ./run_all_interpolation.sh")
            print(f"3. Check HFR status: python check_hfr_status.py --base-dir {args.base_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())