import os
import re
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# Configuration
PROCESSED_DIR = Path("/mnt/tcia_data/processed/4D-Lung")
CYCLES_DIR = Path("/mnt/tcia_data/processed/4D-Lung-Cycles")
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

def extract_series_identifier(series_desc):
    """Extract series identifier from series description"""
    # Look for patterns like S111, S125, S101 in the series description
    # Example: "CT_P4^P100^S111^I0,_Gated,_0.0%"
    match = re.search(r'S(\d+)', series_desc)
    if match:
        return f"S{match.group(1)}"
    
    # Alternative pattern matching if needed
    # Could also try extracting the full series UID parts
    match = re.search(r'P4\^P100\^([^,]+)', series_desc)
    if match:
        return match.group(1)
    
    return None

def analyze_breathing_cycles(manifest_df):
    """Analyze the breathing cycles in the manifest"""
    print("=== Analyzing Breathing Cycles ===")
    
    # Extract series identifiers
    manifest_df['series_id'] = manifest_df['series_desc'].apply(extract_series_identifier)
    
    # Group by patient and series
    cycles_by_patient = defaultdict(lambda: defaultdict(list))
    
    for _, row in manifest_df.iterrows():
        patient_id = row['patient_id']
        series_id = row['series_id']
        breathing_phase = row['breathing_phase_percent']
        
        if series_id:
            cycles_by_patient[patient_id][series_id].append({
                'phase': breathing_phase,
                'file_path': row['file_path'],
                'phase_index': row['phase_index']
            })
    
    # Analyze cycle completeness
    cycle_analysis = []
    
    for patient_id, series_dict in cycles_by_patient.items():
        print(f"\nPatient {patient_id}:")
        
        for series_id, phases in series_dict.items():
            phase_list = sorted([p['phase'] for p in phases])
            phase_count = len(phase_list)
            phase_range = max(phase_list) - min(phase_list) if phase_list else 0
            
            # Check if it's a complete breathing cycle
            expected_phases = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
            is_complete = set(phase_list) == set(expected_phases)
            
            cycle_info = {
                'patient_id': patient_id,
                'series_id': series_id,
                'phase_count': phase_count,
                'phases': phase_list,
                'phase_range': phase_range,
                'is_complete': is_complete,
                'phases_data': phases
            }
            
            cycle_analysis.append(cycle_info)
            
            status = "✓ Complete" if is_complete else f"✗ Incomplete ({phase_count} phases)"
            print(f"  {series_id}: {status} - Phases: {phase_list}")
    
    return cycle_analysis

def create_breathing_cycles(cycle_analysis):
    """Create organized breathing cycle structure"""
    print(f"\n=== Creating Breathing Cycle Structure ===")
    
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    
    complete_cycles = [c for c in cycle_analysis if c['is_complete']]
    incomplete_cycles = [c for c in cycle_analysis if not c['is_complete']]
    
    print(f"Complete breathing cycles: {len(complete_cycles)}")
    print(f"Incomplete cycles (will be skipped): {len(incomplete_cycles)}")
    
    cycle_manifest = []
    
    for cycle_idx, cycle in enumerate(complete_cycles):
        patient_id = cycle['patient_id']
        series_id = cycle['series_id']
        
        # Create cycle directory
        cycle_name = f"{patient_id}_{series_id}"
        cycle_dir = CYCLES_DIR / patient_id / cycle_name
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing cycle {cycle_idx+1}/{len(complete_cycles)}: {cycle_name}")
        
        # Sort phases by breathing percentage
        sorted_phases = sorted(cycle['phases_data'], key=lambda x: x['phase'])
        
        # Copy and rename files to temporal sequence
        cycle_files = {}
        for temporal_idx, phase_data in enumerate(sorted_phases):
            # New filename: phase_00.npy, phase_01.npy, etc.
            new_filename = f"phase_{temporal_idx:02d}.npy"
            new_path = cycle_dir / new_filename
            
            # Copy the existing file
            shutil.copy2(phase_data['file_path'], new_path)
            
            cycle_files[phase_data['phase']] = {
                'temporal_index': temporal_idx,
                'breathing_phase': phase_data['phase'],
                'file_path': str(new_path)
            }
        
        # Add to cycle manifest
        cycle_manifest.append({
            'cycle_id': cycle_name,
            'patient_id': patient_id,
            'series_id': series_id,
            'cycle_dir': str(cycle_dir),
            'num_phases': len(sorted_phases),
            'phase_range': cycle['phase_range'],
            'breathing_phases': [p['phase'] for p in sorted_phases],
            'file_paths': [cycle_files[p['phase']]['file_path'] for p in sorted_phases]
        })
    
    return cycle_manifest, complete_cycles

def save_cycle_manifest(cycle_manifest):
    """Save the breathing cycle manifest"""
    
    # Save as CSV
    cycle_df = pd.DataFrame(cycle_manifest)
    csv_path = CYCLES_DIR / "breathing_cycles_manifest.csv"
    cycle_df.to_csv(csv_path, index=False)
    
    # Save detailed JSON with file mappings
    detailed_manifest = []
    for cycle in cycle_manifest:
        cycle_detail = cycle.copy()
        
        # Load one volume to get shape info
        sample_volume = np.load(cycle['file_paths'][0])
        cycle_detail.update({
            'volume_shape': list(sample_volume.shape),
            'dtype': str(sample_volume.dtype),
            'total_frames': len(cycle['file_paths']),
            'temporal_resolution': '10% breathing phase'
        })
        
        detailed_manifest.append(cycle_detail)
    
    json_path = CYCLES_DIR / "breathing_cycles_detailed.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_manifest, f, indent=2)
    
    print(f"\nManifests saved:")
    print(f"- CSV: {csv_path}")
    print(f"- Detailed JSON: {json_path}")
    
    return cycle_df

def create_train_test_splits(cycle_df, train_ratio=0.7, val_ratio=0.15):
    """Create patient-level train/test splits for breathing cycles"""
    
    patients = cycle_df['patient_id'].unique()
    np.random.seed(42)  # Reproducible splits
    np.random.shuffle(patients)
    
    n_train = int(len(patients) * train_ratio)
    n_val = int(len(patients) * val_ratio)
    
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train+n_val]
    test_patients = patients[n_train+n_val:]
    
    # Create splits
    splits = {
        'train': cycle_df[cycle_df['patient_id'].isin(train_patients)],
        'val': cycle_df[cycle_df['patient_id'].isin(val_patients)],
        'test': cycle_df[cycle_df['patient_id'].isin(test_patients)]
    }
    
    # Save split files
    splits_dir = CYCLES_DIR / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    for split_name, split_df in splits.items():
        split_path = splits_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        
        # Also save just the cycle IDs for easy loading
        cycle_ids_path = splits_dir / f"{split_name}_cycle_ids.txt"
        with open(cycle_ids_path, 'w') as f:
            for cycle_id in split_df['cycle_id']:
                f.write(f"{cycle_id}\n")
    
    print(f"\n=== Train/Test Splits ===")
    print(f"Train: {len(splits['train'])} cycles from {len(train_patients)} patients")
    print(f"Val: {len(splits['val'])} cycles from {len(val_patients)} patients")
    print(f"Test: {len(splits['test'])} cycles from {len(test_patients)} patients")
    print(f"Splits saved to: {splits_dir}")
    
    return splits

def main():
    """Main function to regroup breathing cycles"""
    
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return
    
    print("Loading existing manifest...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded {len(manifest_df)} processed phases")
    
    # Analyze breathing cycles
    cycle_analysis = analyze_breathing_cycles(manifest_df)
    
    # Create breathing cycle structure
    cycle_manifest, complete_cycles = create_breathing_cycles(cycle_analysis)
    
    # Save manifests
    cycle_df = save_cycle_manifest(cycle_manifest)
    
    # Create train/test splits
    splits = create_train_test_splits(cycle_df)
    
    print(f"\n=== Summary ===")
    print(f"Original phases: {len(manifest_df)}")
    print(f"Complete breathing cycles: {len(complete_cycles)}")
    print(f"Patients with complete cycles: {cycle_df['patient_id'].nunique()}")
    print(f"Average cycles per patient: {len(complete_cycles) / cycle_df['patient_id'].nunique():.1f}")
    print(f"")
    print(f"Breathing cycles organized in: {CYCLES_DIR}")
    print(f"Ready for flow matching training!")
    
    # Sample cycle info
    if len(complete_cycles) > 0:
        sample_cycle = complete_cycles[0]
        print(f"\nSample cycle structure:")
        print(f"Patient: {sample_cycle['patient_id']}")
        print(f"Series: {sample_cycle['series_id']}")
        print(f"Phases: {sample_cycle['phases']}")

if __name__ == "__main__":
    main()