#!/usr/bin/env python
"""
Process a batch of interpolation tasks
"""

import argparse
import sys
from pathlib import Path
import logging
import json
from typing import List

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


def parse_patient_list(patient_str: str) -> List[str]:
    """Parse patient list from string (comma-separated or range)"""
    patients = []
    
    for part in patient_str.split(','):
        part = part.strip()
        if '-' in part and not part.startswith('Patient'):
            # Range like "1-5"
            start, end = part.split('-')
            for i in range(int(start), int(end) + 1):
                patients.append(f"Patient_{i}")
        else:
            # Direct patient ID
            patients.append(part)
    
    return patients


def main():
    parser = argparse.ArgumentParser(
        description='Process interpolation tasks in batch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process high priority tasks for first 3 patients
  python process_batch.py --base-dir /data/interpolated --processed-dir /data/processed --patients Patient_1,Patient_2,Patient_3 --priority 1
  
  # Process specific experiments with limited slices
  python process_batch.py --base-dir /data/interpolated --processed-dir /data/processed --experiments linear_0_50,flow_match_0_50 --max-slices 5
  
  # Process all missing tasks for linear interpolation
  python process_batch.py --base-dir /data/interpolated --processed-dir /data/processed --experiments linear_0_50 --phase-ranges 0-50
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--base-dir', 
        type=Path, 
        required=True,
        help='Base directory for interpolated data'
    )
    parser.add_argument(
        '--processed-dir', 
        type=Path, 
        required=True,
        help='Directory containing processed 4D-Lung-Cycles data'
    )
    
    # Task selection arguments
    parser.add_argument(
        '--patients',
        type=str,
        help='Comma-separated list of patient IDs or ranges (e.g., "Patient_1,Patient_2" or "1-5")'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        help='Comma-separated list of experiment IDs to process'
    )
    parser.add_argument(
        '--phase-ranges',
        type=str,
        default='0-50',
        help='Comma-separated list of phase ranges (default: 0-50)'
    )
    parser.add_argument(
        '--priority',
        type=int,
        choices=[1, 2, 3],
        help='Only process experiments with this priority level'
    )
    parser.add_argument(
        '--max-slices',
        type=int,
        help='Maximum slices per patient to process'
    )
    parser.add_argument(
        '--max-tasks',
        type=int,
        help='Maximum total tasks to process'
    )
    
    # Processing options
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    parser.add_argument(
        '--list-experiments',
        action='store_true',
        help='List available experiments and exit'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Handle list experiments
    if args.list_experiments:
        print("\nAvailable experiments:")
        print("="*80)
        for exp_id, exp_config in PRESET_EXPERIMENTS.items():
            print(f"\n{exp_id} (priority={exp_config.priority}):")
            print(f"  Method: {exp_config.method}")
            print(f"  Description: {exp_config.description}")
            print(f"  Input phases: {exp_config.input_phases}")
            print(f"  Target phases: {exp_config.target_phases}")
            print(f"  Slice selection: {exp_config.slice_selection}")
        return 0
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
        
    # Validate directories
    if not args.base_dir.exists():
        logger.error(f"Base directory not found: {args.base_dir}")
        return 1
    
    if not args.processed_dir.exists():
        logger.error(f"Processed data directory not found: {args.processed_dir}")
        return 1
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline manager...")
        pipeline = PipelineManager(args.base_dir, args.processed_dir)
        
        # Build filter config
        filter_config = {}
        
        if args.patients:
            filter_config['patients'] = parse_patient_list(args.patients)
            logger.info(f"Filtering to patients: {filter_config['patients']}")
        
        if args.experiments:
            filter_config['experiments'] = [e.strip() for e in args.experiments.split(',')]
            # Validate experiments
            invalid = [e for e in filter_config['experiments'] if e not in PRESET_EXPERIMENTS]
            if invalid:
                logger.error(f"Invalid experiments: {invalid}")
                logger.info(f"Available experiments: {list(PRESET_EXPERIMENTS.keys())}")
                return 1
            logger.info(f"Filtering to experiments: {filter_config['experiments']}")
        
        if args.phase_ranges:
            filter_config['phase_ranges'] = [r.strip() for r in args.phase_ranges.split(',')]
            logger.info(f"Phase ranges: {filter_config['phase_ranges']}")
        
        if args.priority:
            filter_config['priority'] = args.priority
            logger.info(f"Filtering to priority: {args.priority}")
        
        if args.max_slices:
            filter_config['max_slices_per_patient'] = args.max_slices
            logger.info(f"Max slices per patient: {args.max_slices}")
        
        # Get missing tasks
        logger.info("Scanning for missing tasks...")
        tasks = pipeline.get_missing_tasks(filter_config)
        
        if not tasks:
            logger.info("No missing tasks found with the given filters")
            return 0
        
        # Apply max tasks limit
        if args.max_tasks and len(tasks) > args.max_tasks:
            tasks = tasks[:args.max_tasks]
            logger.info(f"Limited to {args.max_tasks} tasks")
        
        # Report what will be processed
        print("\n" + "="*60)
        print("TASKS TO PROCESS")
        print("="*60)
        print(f"Total tasks: {len(tasks)}")
        
        # Group by experiment for summary
        tasks_by_exp = {}
        for task in tasks:
            if task.experiment_id not in tasks_by_exp:
                tasks_by_exp[task.experiment_id] = 0
            tasks_by_exp[task.experiment_id] += 1
        
        print("\nTasks by experiment:")
        for exp_id, count in sorted(tasks_by_exp.items()):
            print(f"  {exp_id}: {count} tasks")
        
        # Show sample tasks
        print("\nSample tasks:")
        for task in tasks[:5]:
            print(f"  {task.patient_id} / {task.experiment_id} / slice_{task.slice_num} / phase_{task.phase_range}")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks) - 5} more")
        
        if args.dry_run:
            print("\n[DRY RUN] No tasks were processed")
            return 0
        
        # Confirm before processing
        print(f"\nThis will process {len(tasks)} tasks using {args.workers} workers")
        
        # Process tasks
        print("\nProcessing tasks...")
        results = pipeline.process_tasks(tasks, max_workers=args.workers, show_progress=True)
        
        # Report results
        print("\n" + "="*60)
        print("PROCESSING RESULTS")
        print("="*60)
        print(f"Success: {results['success']} tasks")
        print(f"Failed: {results['failed']} tasks")
        print(f"Skipped: {results['skipped']} tasks")
        
        # Get updated status
        status = pipeline.get_status_summary()
        print("\nStorage usage:")
        storage = status['storage_usage']
        print(f"  Total: {storage['total_gb']:.2f} GB")
        print(f"  Files: {storage['num_files']}")
        print(f"  Average size: {storage['avg_size_mb']:.2f} MB per task")
        
        # Save processing log
        log_data = {
            'timestamp': str(Path.cwd()),
            'filter_config': filter_config,
            'tasks_processed': len(tasks),
            'results': results,
            'workers': args.workers
        }
        
        log_dir = args.base_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"process_log_{Path.cwd().name}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Processing log saved to: {log_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())