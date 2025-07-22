#!/usr/bin/env python
"""
Check status specifically for high frame rate (HFR) experiments
"""

import argparse
import sys
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline import PipelineManager, PRESET_EXPERIMENTS


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
        description='Check status of HFR interpolation experiments'
    )
    
    # Required arguments
    parser.add_argument(
        '--base-dir', 
        type=Path, 
        required=True,
        help='Base directory for interpolated data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--experiment',
        type=str,
        help='Show details for specific HFR experiment'
    )
    parser.add_argument(
        '--sample-video',
        action='store_true',
        help='Show path to a sample video if available'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed statistics about frame counts'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate base directory
    if not args.base_dir.exists():
        logger.error(f"Base directory not found: {args.base_dir}")
        return 1
    
    try:
        # Initialize pipeline
        pipeline = PipelineManager(args.base_dir, args.base_dir)
        
        # Get HFR experiments
        hfr_experiments = {k: v for k, v in PRESET_EXPERIMENTS.items() 
                          if 'hfr_' in k}
        
        if not hfr_experiments:
            print("No HFR experiments found in configuration")
            return 1
        
        # Print header
        print("\n" + "="*70)
        print("HIGH FRAME RATE (HFR) EXPERIMENT STATUS")
        print(f"Base directory: {args.base_dir}")
        print("="*70)
        
        # Get manifest data
        manifest = pipeline.manifest_manager.manifest
        
        # Filter for HFR experiments
        hfr_data = manifest[manifest['experiment_id'].str.contains('hfr_')]
        
        if hfr_data.empty:
            print("\nNo HFR experiments have been processed yet.")
            print("\nAvailable HFR experiments:")
            for exp_id, exp_config in hfr_experiments.items():
                fps = exp_config.params.get('frames_per_interval', 0)
                method = exp_config.method
                print(f"  - {exp_id}: {method} interpolation, {fps} frames between phases")
            return 0
        
        # Overall HFR statistics
        print("\nOVERALL HFR STATISTICS")
        print("-"*50)
        
        total_hfr = len(hfr_data)
        completed_hfr = len(hfr_data[hfr_data['status'] == 'completed'])
        
        print(f"Total HFR tasks: {total_hfr}")
        print(f"Completed: {completed_hfr} ({completed_hfr/total_hfr*100:.1f}%)")
        
        # Per-experiment breakdown
        print("\nPER-EXPERIMENT STATUS")
        print("-"*50)
        
        for exp_id in sorted(hfr_data['experiment_id'].unique()):
            exp_data = hfr_data[hfr_data['experiment_id'] == exp_id]
            completed = len(exp_data[exp_data['status'] == 'completed'])
            total = len(exp_data)
            
            if exp_id in hfr_experiments:
                config = hfr_experiments[exp_id]
                fps = config.params.get('frames_per_interval', 0)
                expected_frames = fps * 9 + 10  # 9 intervals + 10 original
                
                print(f"\n{exp_id}:")
                print(f"  Method: {config.method}")
                print(f"  Frames per interval: {fps}")
                print(f"  Expected total frames: {expected_frames}")
                print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")
            else:
                print(f"\n{exp_id}: {completed}/{total} tasks")
        
        # Detailed statistics if requested
        if args.stats:
            print("\n\nDETAILED FRAME STATISTICS")
            print("-"*50)
            
            # Sample some completed tasks to check frame counts
            completed_tasks = hfr_data[hfr_data['status'] == 'completed'].head(10)
            
            frame_stats = []
            for _, row in completed_tasks.iterrows():
                output_path = Path(row['output_path'])
                metadata_path = output_path / 'metadata.json'
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    frame_stats.append({
                        'experiment': row['experiment_id'],
                        'patient': row['patient_id'],
                        'slice': row['slice_num'],
                        'input_frames': metadata.get('input_shape', [0])[0],
                        'output_frames': metadata.get('target_shape', [0])[0],
                        'expected_frames': metadata.get('num_frames', 0)
                    })
            
            if frame_stats:
                df = pd.DataFrame(frame_stats)
                print("\nSample of processed data:")
                print(df.to_string(index=False))
        
        # Show sample video path if requested
        if args.sample_video:
            print("\n\nSAMPLE VIDEO LOCATIONS")
            print("-"*50)
            
            # Find tasks with videos
            video_found = False
            for _, row in completed_tasks.head(5).iterrows():
                output_path = Path(row['output_path'])
                video_path = output_path / 'interpolated_sequence.mp4'
                
                if video_path.exists():
                    print(f"\nVideo available:")
                    print(f"  Patient: {row['patient_id']}")
                    print(f"  Experiment: {row['experiment_id']}")
                    print(f"  Path: {video_path}")
                    video_found = True
                    break
            
            if not video_found:
                print("No video files found. Videos are optional and may not have been generated.")
        
        # Show specific experiment details
        if args.experiment:
            if args.experiment not in hfr_experiments:
                print(f"\n\nError: '{args.experiment}' is not an HFR experiment")
                print("Available HFR experiments:", list(hfr_experiments.keys()))
                return 1
            
            print(f"\n\nDETAILED VIEW: {args.experiment}")
            print("-"*50)
            
            exp_data = hfr_data[hfr_data['experiment_id'] == args.experiment]
            if exp_data.empty:
                print("No data processed for this experiment yet")
            else:
                # Group by patient
                patient_stats = exp_data.groupby('patient_id').agg({
                    'status': 'count',
                    'slice_num': 'nunique'
                }).rename(columns={'status': 'total_tasks', 'slice_num': 'unique_slices'})
                
                print("\nPer-patient breakdown:")
                print(patient_stats.to_string())
                
                # Storage usage
                completed_exp = exp_data[exp_data['status'] == 'completed']
                if 'file_size_mb' in completed_exp.columns:
                    total_size = completed_exp['file_size_mb'].sum()
                    avg_size = completed_exp['file_size_mb'].mean()
                    print(f"\nStorage usage:")
                    print(f"  Total: {total_size/1024:.2f} GB")
                    print(f"  Average per task: {avg_size:.2f} MB")
        
        # Training readiness
        print("\n\nTRAINING DATA READINESS")
        print("-"*50)
        
        # Check if we have enough data for training
        for split in ['train', 'val', 'test']:
            # Get available HFR samples
            samples = pipeline.get_available_data(split)
            hfr_samples = [s for s in samples if 'hfr_' in s.experiment_id]
            
            if hfr_samples:
                exp_counts = {}
                for sample in hfr_samples:
                    if sample.experiment_id not in exp_counts:
                        exp_counts[sample.experiment_id] = 0
                    exp_counts[sample.experiment_id] += 1
                
                print(f"\n{split.upper()} set:")
                for exp_id, count in sorted(exp_counts.items()):
                    print(f"  {exp_id}: {count} samples")
        
        print("\n" + "="*70)
        print("Use 'python process_batch.py' to generate more HFR data")
        print("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error checking HFR status: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())