import os
import re
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("/mnt/tcia_data/raw/4D-Lung")
PROCESSED_DIR = Path("/mnt/tcia_data/processed/4D-Lung")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def extract_breathing_phase(series_desc):
    """Extract breathing phase percentage from series description"""
    # Look for patterns like "50.0%", "70%", "0.0%", etc.
    match = re.search(r'(\d+\.?\d*)%', series_desc)
    if match:
        return float(match.group(1))
    return None

def is_respiratory_gated_ct(series_desc):
    """Check if this is a respiratory-gated CT series for breathing motion"""
    series_desc_lower = series_desc.lower()
    
    # Must have all these keywords
    required_keywords = ['gated', 'ct']
    has_required = all(keyword in series_desc_lower for keyword in required_keywords)
    
    # Must have breathing phase percentage
    has_percentage = '%' in series_desc_lower
    
    # Exclude non-breathing series
    exclude_keywords = ['planning', 'scout', 'localizer', 'dose', 'struct', 'rtss']
    has_excluded = any(keyword in series_desc_lower for keyword in exclude_keywords)
    
    return has_required and has_percentage and not has_excluded

def get_gated_series_info(series_dir):
    """Extract info from a gated CT series"""
    series_name = series_dir.name
    
    # Check if this is respiratory gated
    if not is_respiratory_gated_ct(series_name):
        return None
    
    # Extract breathing phase
    breathing_phase = extract_breathing_phase(series_name)
    if breathing_phase is None:
        return None
    
    # Only include standard breathing phases (0-100%)
    if not (0 <= breathing_phase <= 100):
        return None
    
    return {
        'series_name': series_name,
        'series_dir': series_dir,
        'breathing_phase': breathing_phase
    }

manifest_rows = []

print("=== Processing 4D-Lung for Respiratory-Gated CT Only ===")

# Process each patient
for patient_dir in tqdm(list(RAW_DIR.iterdir()), desc="Processing patients"):
    if not patient_dir.is_dir():
        continue
    
    patient_id = patient_dir.name
    print(f"\nProcessing patient: {patient_id}")
    
    # Process each study (breathing cycle session)
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue
        
        study_name = study_dir.name
        print(f"  Study: {study_name}")
        
        # Find all respiratory-gated series in this study
        gated_series = []
        all_series_count = 0
        
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            
            all_series_count += 1
            series_info = get_gated_series_info(series_dir)
            
            if series_info is not None:
                gated_series.append(series_info)
                print(f"    ✓ Gated CT: {series_info['breathing_phase']}%")
            else:
                print(f"    ✗ Skipped: {series_dir.name[:50]}...")
        
        print(f"    Found {len(gated_series)}/{all_series_count} respiratory-gated series")
        
        # Need at least 5 breathing phases for a meaningful cycle
        if len(gated_series) < 5:
            print(f"    WARNING: Only {len(gated_series)} gated phases, skipping study")
            continue
        
        # Sort by breathing phase
        gated_series.sort(key=lambda x: x['breathing_phase'])
        breathing_phases = [s['breathing_phase'] for s in gated_series]
        print(f"    Breathing phases: {breathing_phases}")
        
        # Check if we have a reasonable breathing cycle (should span significant range)
        phase_range = max(breathing_phases) - min(breathing_phases)
        if phase_range < 50:  # Should span at least 50% of breathing cycle
            print(f"    WARNING: Phase range only {phase_range}%, skipping study")
            continue
        
        # Process each gated series
        study_output_dir = PROCESSED_DIR / patient_id / study_name
        study_output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_phases = []
        
        for phase_idx, series_info in enumerate(gated_series):
            try:
                series_dir = series_info['series_dir']
                
                # Gather DICOM files
                dicom_files = sorted(series_dir.glob("*.dcm"))
                if not dicom_files:
                    dicom_files = sorted(series_dir.rglob("*.dcm"))
                if not dicom_files:
                    print(f"      No DICOM files in: {series_dir.name}")
                    continue
                
                # Read and stack slices
                slices = [pydicom.dcmread(f) for f in dicom_files]
                
                # Sort by z-position
                slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
                
                # Check slice uniformity
                shapes = [s.pixel_array.shape for s in slices]
                if len(set(shapes)) > 1:
                    print(f"      WARNING: Non-uniform slice shapes: {set(shapes)}")
                    continue
                
                # Create 3D volume
                volume = np.stack([s.pixel_array for s in slices], axis=0)
                
                # Output filename: phase_00.npy, phase_01.npy, etc.
                output_filename = f"phase_{phase_idx:02d}.npy"
                output_path = study_output_dir / output_filename
                
                # Save volume
                np.save(output_path, volume)
                
                # Add to manifest
                manifest_rows.append({
                    "patient_id": patient_id,
                    "study_name": study_name,
                    "phase_index": phase_idx,
                    "breathing_phase_percent": series_info['breathing_phase'],
                    "series_desc": series_info['series_name'],
                    "series_dir": str(series_info['series_dir']),
                    "file_path": str(output_path),
                    "num_slices": len(slices),
                    "height": volume.shape[1],
                    "width": volume.shape[2],
                    "dtype": str(volume.dtype),
                    "breathing_cycle_position": phase_idx,
                    "cycle_length": len(gated_series),
                    "phase_range": phase_range
                })
                
                successful_phases.append(series_info['breathing_phase'])
                print(f"      ✓ Processed phase {series_info['breathing_phase']}%: {volume.shape}")
                
            except Exception as e:
                print(f"      ✗ Error processing {series_info['series_name']}: {e}")
                continue
        
        print(f"    Successfully processed {len(successful_phases)} phases: {successful_phases}")

# Create manifest
manifest_df = pd.DataFrame(manifest_rows)

if len(manifest_df) > 0:
    # Sort by patient, study, and breathing phase
    manifest_df = manifest_df.sort_values(['patient_id', 'study_name', 'breathing_phase_percent']).reset_index(drop=True)
    
    # Save manifest
    manifest_path = PROCESSED_DIR / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total processed breathing phases: {len(manifest_df)}")
    print(f"Patients: {manifest_df['patient_id'].nunique()}")
    print(f"Breathing cycles: {len(manifest_df.groupby(['patient_id', 'study_name']))}")
    print(f"Average phases per cycle: {manifest_df['cycle_length'].mean():.1f}")
    print(f"Breathing phase range: {manifest_df['breathing_phase_percent'].min()}% - {manifest_df['breathing_phase_percent'].max()}%")
    print(f"Manifest saved to: {manifest_path}")
    
    # Quality analysis
    print(f"\n=== Quality Analysis ===")
    
    # Phase distribution
    phase_dist = manifest_df['breathing_phase_percent'].value_counts().sort_index()
    print("Breathing phase distribution:")
    for phase, count in phase_dist.head(10).items():
        print(f"  {phase:4.1f}%: {count:2d} series")
    
    # Cycle completeness
    cycle_stats = manifest_df.groupby(['patient_id', 'study_name']).agg({
        'breathing_phase_percent': ['min', 'max', 'count'],
        'phase_range': 'first'
    }).round(1)
    cycle_stats.columns = ['min_phase', 'max_phase', 'phase_count', 'phase_range']
    
    print(f"\nBreathing cycle statistics:")
    print(f"Average phases per cycle: {cycle_stats['phase_count'].mean():.1f}")
    print(f"Average phase range: {cycle_stats['phase_range'].mean():.1f}%")
    print(f"Cycles with 8+ phases: {(cycle_stats['phase_count'] >= 8).sum()}/{len(cycle_stats)}")
    print(f"Cycles with 70%+ range: {(cycle_stats['phase_range'] >= 70).sum()}/{len(cycle_stats)}")

else:
    print("No respiratory-gated CT series found!")
    
print(f"\nProcessed data saved to: {PROCESSED_DIR}")