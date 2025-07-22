#!/usr/bin/env python
"""
Check the status of the interpolation pipeline
"""

import argparse
import sys
from pathlib import Path
import logging
import json
import pandas as pd
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


def format_size(size_mb: float) -> str:
    """Format size in MB/GB"""
    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb/1024:.2f} GB"


def print_progress_bar(progress: float, width: int = 40) -> str:
    """Create a text progress bar"""
    filled = int(width * progress / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {progress:.1f}%"


def main():
    parser = argparse.ArgumentParser(
        description='Check status of interpolation pipeline'
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
        '--processed-dir', 
        type=Path,
        help='Directory containing processed data (for detailed analysis)'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='Show detailed status for specific experiment'
    )
    parser.add_argument(
        '--patient',
        type=str,
        help='Show detailed status for specific patient'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        help='Show available data for specific split'
    )
    parser.add_argument(
        '--export-report',
        type=Path,
        help='Export detailed report to JSON file'
    )
    parser.add_argument(
        '--show-missing',
        action='store_true',
        help='Show sample of missing tasks'
    )
    parser.add_argument(
        '--cleanup-failed',
        action='store_true',
        help='Remove failed tasks from manifest to allow retry'
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
        # Initialize pipeline (with dummy processed_dir if not provided)
        processed_dir = args.processed_dir or args.base_dir
        pipeline = PipelineManager(args.base_dir, processed_dir)
        
        # Get overall status
        status = pipeline.get_status_summary()
        
        # Print header
        print("\n" + "="*70)
        print(f"INTERPOLATION PIPELINE STATUS")
        print(f"Base directory: {args.base_dir}")
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Overall progress
        manifest_status = status['manifest_status']
        total_tasks = sum(manifest_status.values())
        
        print("\nOVERALL PROGRESS")
        print("-"*50)
        print(f"Total tasks: {total_tasks:,}")
        print(f"Completed: {manifest_status.get('completed', 0):,}")
        print(f"In progress: {manifest_status.get('in_progress', 0):,}")
        print(f"Failed: {manifest_status.get('failed', 0):,}")
        
        if total_tasks > 0:
            completion_rate = manifest_status.get('completed', 0) / total_tasks * 100
            print(f"\nCompletion: {print_progress_bar(completion_rate)}")
        
        # Storage usage
        storage = status['storage_usage']
        print("\nSTORAGE USAGE")
        print("-"*50)
        print(f"Total size: {format_size(storage['total_mb'])}")
        print(f"Number of files: {storage['num_files']:,}")
        if storage['num_files'] > 0:
            print(f"Average per task: {format_size(storage['avg_size_mb'])}")
        
        # Patient splits
        print("\nPATIENT SPLITS")
        print("-"*50)
        for split, count in status['patient_splits'].items():
            print(f"{split.capitalize():.<20} {count} patients")
        
        # Experiment progress
        print("\nEXPERIMENT PROGRESS")
        print("-"*50)
        
        exp_data = []
        for exp_id, exp_status in sorted(status['experiments'].items()):
            if exp_status['total'] > 0:
                exp_data.append({
                    'Experiment': exp_id,
                    'Method': PRESET_EXPERIMENTS.get(exp_id, {}).method if exp_id in PRESET_EXPERIMENTS else 'unknown',
                    'Total': exp_status['total'],
                    'Completed': exp_status['completed'],
                    'Progress': f"{exp_status['progress']:.1f}%"
                })
        
        if exp_data:
            df = pd.DataFrame(exp_data)
            print(df.to_string(index=False))
        else:
            print("No experiments processed yet")
        
        # Detailed experiment view
        if args.experiment:
            if args.experiment not in status['experiments']:
                print(f"\nExperiment '{args.experiment}' not found")
            else:
                exp_detail = status['experiments'][args.experiment]
                print(f"\n\nDETAILED VIEW: {args.experiment}")
                print("-"*50)
                print(f"Total tasks: {exp_detail['total']}")
                print(f"Completed: {exp_detail['completed']}")
                print(f"In progress: {exp_detail.get('in_progress', 0)}")
                print(f"Failed: {exp_detail.get('failed', 0)}")
                print(print_progress_bar(exp_detail['progress']))
        
        # Show available data for specific split
        if args.split:
            print(f"\n\nAVAILABLE DATA FOR {args.split.upper()}")
            print("-"*50)
            
            samples = pipeline.get_available_data(args.split)
            if samples:
                # Group by experiment
                exp_counts = {}
                for sample in samples:
                    if sample.experiment_id not in exp_counts:
                        exp_counts[sample.experiment_id] = 0
                    exp_counts[sample.experiment_id] += 1
                
                print(f"Total samples: {len(samples)}")
                print("\nSamples by experiment:")
                for exp_id, count in sorted(exp_counts.items()):
                    print(f"  {exp_id}: {count}")
                
                # Show sample entries
                print("\nSample entries:")
                for sample in samples[:3]:
                    print(f"  {sample.patient_id} / {sample.experiment_id} / "
                          f"slice_{sample.slice_num} / phase_{sample.phase_range}")
                if len(samples) > 3:
                    print(f"  ... and {len(samples) - 3} more")
            else:
                print(f"No samples available for {args.split} split")
        
        # Show missing tasks
        if args.show_missing and args.processed_dir:
            print("\n\nSAMPLE OF MISSING TASKS")
            print("-"*50)
            
            # Get a small sample of missing tasks
            missing_tasks = pipeline.get_missing_tasks({'priority': 1})[:10]
            
            if missing_tasks:
                print(f"Showing {len(missing_tasks)} of many missing tasks:")
                for task in missing_tasks:
                    print(f"  {task.patient_id} / {task.experiment_id} / "
                          f"slice_{task.slice_num} / phase_{task.phase_range}")
            else:
                print("No missing high-priority tasks found")
        
        # Cleanup failed tasks
        if args.cleanup_failed:
            num_cleaned = pipeline.cleanup_failed_tasks()
            print(f"\n\nCleaned up {num_cleaned} failed tasks")
        
        # Export detailed report
        if args.export_report:
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'base_dir': str(args.base_dir),
                'status': status,
                'available_experiments': list(PRESET_EXPERIMENTS.keys())
            }
            
            # Add sample data info
            if args.processed_dir:
                for split in ['train', 'val', 'test']:
                    samples = pipeline.get_available_data(split)
                    report_data[f'{split}_samples'] = len(samples)
            
            with open(args.export_report, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"\n\nDetailed report exported to: {args.export_report}")
        
        # Footer with next steps
        print("\n" + "="*70)
        if manifest_status.get('completed', 0) == 0:
            print("No tasks completed yet. Run process_batch.py to start processing.")
        elif manifest_status.get('completed', 0) < total_tasks:
            print(f"Processing in progress. {total_tasks - manifest_status.get('completed', 0)} tasks remaining.")
        else:
            print("All tasks completed!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())