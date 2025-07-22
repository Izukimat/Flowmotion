#!/usr/bin/env python
"""
Create or update train/val/test patient splits
"""

import argparse
import sys
from pathlib import Path
import logging
import json
import numpy as np
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline import PipelineManager
from src.data_pipeline.utils import save_patient_splits, load_patient_splits


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_splits(splits: Dict[str, List[str]]) -> bool:
    """Validate split configuration"""
    # Check required keys
    required_keys = {'train', 'val', 'test'}
    if not all(key in splits for key in required_keys):
        logging.error(f"Splits must contain keys: {required_keys}")
        return False
    
    # Check for duplicates
    all_patients = []
    for split_patients in splits.values():
        all_patients.extend(split_patients)
    
    if len(all_patients) != len(set(all_patients)):
        logging.error("Duplicate patients found across splits")
        return False
    
    # Check for empty splits
    if any(len(patients) == 0 for patients in splits.values()):
        logging.warning("One or more splits are empty")
    
    return True


def create_stratified_splits(patients: List[str], train_ratio: float, 
                           val_ratio: float, seed: int = 42) -> Dict[str, List[str]]:
    """Create stratified splits ensuring good distribution"""
    np.random.seed(seed)
    
    # Shuffle patients
    shuffled = np.random.permutation(patients).tolist()
    
    # Calculate split sizes
    n_total = len(patients)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    # Create splits
    splits = {
        'train': shuffled[:n_train],
        'val': shuffled[n_train:n_train + n_val],
        'test': shuffled[n_train + n_val:]
    }
    
    return splits


def create_fixed_test_splits(patients: List[str], test_patients: List[str],
                           train_ratio: float = 0.8, seed: int = 42) -> Dict[str, List[str]]:
    """Create splits with fixed test set"""
    # Remove test patients from available pool
    remaining = [p for p in patients if p not in test_patients]
    
    if not remaining:
        raise ValueError("No patients left for train/val after fixing test set")
    
    np.random.seed(seed)
    shuffled = np.random.permutation(remaining).tolist()
    
    # Split remaining into train/val
    n_train = int(train_ratio * len(remaining))
    
    splits = {
        'train': shuffled[:n_train],
        'val': shuffled[n_train:],
        'test': test_patients
    }
    
    return splits


def main():
    parser = argparse.ArgumentParser(
        description='Create or update patient train/val/test splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default 70/15/15 splits
  python create_splits.py --base-dir /data/interpolated --processed-dir /data/processed
  
  # Create 60/20/20 splits with specific seed
  python create_splits.py --base-dir /data/interpolated --processed-dir /data/processed --train-ratio 0.6 --val-ratio 0.2 --seed 123
  
  # Create splits with fixed test patients
  python create_splits.py --base-dir /data/interpolated --processed-dir /data/processed --test-patients Patient_1,Patient_2,Patient_3
  
  # Load splits from JSON file
  python create_splits.py --base-dir /data/interpolated --from-file custom_splits.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--base-dir', 
        type=Path, 
        required=True,
        help='Base directory for interpolated data'
    )
    
    # Data source (required for automatic split creation)
    parser.add_argument(
        '--processed-dir', 
        type=Path,
        help='Directory containing processed data (required for automatic splits)'
    )
    
    # Split creation options
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
        '--test-patients',
        type=str,
        help='Comma-separated list of patient IDs to fix as test set'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Alternative: load from file
    parser.add_argument(
        '--from-file',
        type=Path,
        help='Load splits from JSON file instead of creating new ones'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=Path,
        help='Save splits to custom location (in addition to default)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without saving'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing splits without confirmation'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Determine splits to use
        if args.from_file:
            # Load from file
            logger.info(f"Loading splits from {args.from_file}")
            splits = load_patient_splits(args.from_file)
            
        else:
            # Create new splits
            if not args.processed_dir:
                logger.error("--processed-dir is required when creating new splits")
                return 1
            
            if not args.processed_dir.exists():
                logger.error(f"Processed data directory not found: {args.processed_dir}")
                return 1
            
            # Get available patients
            logger.info("Scanning for available patients...")
            pipeline = PipelineManager(args.base_dir, args.processed_dir)
            patients = pipeline._get_available_patients()
            
            if not patients:
                logger.error("No patients found in processed data directory")
                return 1
            
            logger.info(f"Found {len(patients)} patients")
            
            # Create splits based on options
            if args.test_patients:
                # Fixed test set
                test_list = [p.strip() for p in args.test_patients.split(',')]
                invalid_test = [p for p in test_list if p not in patients]
                if invalid_test:
                    logger.error(f"Invalid test patients: {invalid_test}")
                    return 1
                
                logger.info(f"Creating splits with fixed test set: {test_list}")
                splits = create_fixed_test_splits(
                    patients, test_list, 
                    train_ratio=0.8,  # Use 80/20 for remaining train/val
                    seed=args.seed
                )
            else:
                # Standard ratio-based splits
                logger.info(f"Creating {args.train_ratio:.0%}/{args.val_ratio:.0%}/"
                           f"{1-args.train_ratio-args.val_ratio:.0%} splits")
                splits = create_stratified_splits(
                    patients, args.train_ratio, args.val_ratio, args.seed
                )
        
        # Validate splits
        if not validate_splits(splits):
            logger.error("Invalid splits configuration")
            return 1
        
        # Display splits
        print("\n" + "="*60)
        print("PATIENT SPLITS")
        print("="*60)
        
        total_patients = sum(len(patients) for patients in splits.values())
        print(f"Total patients: {total_patients}\n")
        
        for split_name, split_patients in splits.items():
            percentage = len(split_patients) / total_patients * 100 if total_patients > 0 else 0
            print(f"{split_name.upper()} ({percentage:.1f}%): {len(split_patients)} patients")
            
            if args.verbose or len(split_patients) <= 10:
                # Show all if verbose or small number
                for patient in sorted(split_patients):
                    print(f"  - {patient}")
            else:
                # Show sample
                sorted_patients = sorted(split_patients)
                for patient in sorted_patients[:5]:
                    print(f"  - {patient}")
                print(f"  ... and {len(split_patients) - 5} more")
        
        if args.dry_run:
            print("\n[DRY RUN] No files were saved")
            return 0
        
        # Check for existing splits
        default_splits_file = args.base_dir / 'splits' / 'patient_splits.json'
        if default_splits_file.exists() and not args.force:
            print(f"\nWARNING: Splits already exist at {default_splits_file}")
            response = input("Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted")
                return 0
        
        # Save splits
        save_patient_splits(splits, default_splits_file)
        logger.info(f"Saved splits to {default_splits_file}")
        
        # Save to additional location if specified
        if args.output:
            save_patient_splits(splits, args.output)
            logger.info(f"Also saved splits to {args.output}")
        
        # Generate train/val/test file lists (useful for other tools)
        splits_dir = args.base_dir / 'splits'
        for split_name, split_patients in splits.items():
            list_file = splits_dir / f'{split_name}_patients.txt'
            with open(list_file, 'w') as f:
                for patient in sorted(split_patients):
                    f.write(f"{patient}\n")
            logger.info(f"Created patient list: {list_file}")
        
        print("\n" + "="*60)
        print("Splits created successfully!")
        print(f"Location: {default_splits_file}")
        print("\nNext steps:")
        print(f"1. Process data: python process_batch.py --base-dir {args.base_dir} --processed-dir {args.processed_dir}")
        print(f"2. Check available training data: python check_status.py --base-dir {args.base_dir} --split train")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating splits: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())