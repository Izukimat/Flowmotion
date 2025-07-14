#!/usr/bin/env python3
"""
Smart 4D-Lung Download Script
Downloads 4D-Lung dataset with proper error handling and progress tracking
"""

import os
import zipfile
import time
from pathlib import Path
from tqdm import tqdm
from tciaclient.core import TCIAClient
import json

# Configuration
COLLECTION = "4D-Lung"
DATA_DIR = Path("/mnt/tcia_data/raw") / COLLECTION
LOG_FILE = Path("download_log.json")
MAX_PATIENTS = 20  # Start with 3 patients for testing

# Initialize
DATA_DIR.mkdir(parents=True, exist_ok=True)
client = TCIAClient()

def load_progress():
    """Load download progress from log file"""
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {"completed_series": set(), "failed_series": set(), "patients_done": []}

def save_progress(progress):
    """Save download progress to log file"""
    # Convert sets to lists for JSON serialization
    progress_copy = progress.copy()
    progress_copy["completed_series"] = list(progress["completed_series"])
    progress_copy["failed_series"] = list(progress["failed_series"])
    
    with open(LOG_FILE, 'w') as f:
        json.dump(progress_copy, f, indent=2)

def download_series(series_uid, output_dir, max_retries=3):
    """Download a single series with retry logic"""
    for attempt in range(max_retries):
        try:
            zip_file = output_dir / "series.zip"
            
            # Download
            client.get_image(
                seriesInstanceUid=series_uid, 
                downloadPath=output_dir, 
                zipFileName="series.zip"
            )
            
            # Extract if zip exists
            if zip_file.exists():
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(output_dir)
                os.remove(zip_file)
                return True, "Success"
            else:
                return False, "No zip file created"
                
        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"Failed after {max_retries} attempts: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return False, "Max retries exceeded"

def main():
    print(f"=== Smart 4D-Lung Download ===")
    print(f"Target: {MAX_PATIENTS} patients")
    print(f"Output: {DATA_DIR}")
    
    # Load progress
    progress = load_progress()
    completed_series = set(progress["completed_series"])
    failed_series = set(progress["failed_series"])
    patients_done = progress["patients_done"]
    
    # Get patients
    try:
        patients = client.get_patient(collection=COLLECTION)
        print(f"Found {len(patients)} patients in collection")
    except Exception as e:
        print(f"Error getting patients: {e}")
        return
    
    # Limit to target number
    patients = patients[:MAX_PATIENTS]
    
    for patient_idx, patient in enumerate(patients):
        patient_id = patient["PatientID"]
        
        if patient_id in patients_done:
            print(f"Patient {patient_id} already completed, skipping...")
            continue
            
        print(f"\n--- Patient {patient_idx+1}/{len(patients)}: {patient_id} ---")
        
        try:
            # Get studies
            studies = client.get_patient_study(collection=COLLECTION, patientId=patient_id)
            print(f"Found {len(studies)} studies")
            
            patient_series_count = 0
            patient_success_count = 0
            
            for study_idx, study in enumerate(studies):
                study_uid = study["StudyInstanceUID"]
                study_desc = study.get("StudyDescription", "Unknown").replace(" ", "_")
                
                print(f"  Study {study_idx+1}/{len(studies)}: {study_desc}")
                
                # Get all series for this study
                series_list = client.get_series(collection=COLLECTION, studyInstanceUid=study_uid)
                print(f"    Found {len(series_list)} series")
                
                for series_idx, series in enumerate(series_list):
                    series_uid = series["SeriesInstanceUID"]
                    series_desc = series.get("SeriesDescription", "Unknown").replace(" ", "_")
                    modality = series.get("Modality", "Unknown")
                    
                    # Skip if already processed
                    if series_uid in completed_series:
                        print(f"      Series {series_idx+1}: Already completed")
                        patient_success_count += 1
                        continue
                    
                    if series_uid in failed_series:
                        print(f"      Series {series_idx+1}: Previously failed, skipping")
                        continue
                    
                    # Create output directory
                    out_dir = DATA_DIR / patient_id / study_desc / f"{modality}_{series_desc}_{series_uid}"
                    
                    # Skip if files already exist
                    if out_dir.exists() and any(out_dir.glob("*.dcm")):
                        print(f"      Series {series_idx+1}: Files exist, marking as completed")
                        completed_series.add(series_uid)
                        patient_success_count += 1
                        continue
                    
                    out_dir.mkdir(parents=True, exist_ok=True)
                    patient_series_count += 1
                    
                    print(f"      Series {series_idx+1}: Downloading {modality} {series_desc}")
                    
                    # Download with retry
                    success, message = download_series(series_uid, out_dir)
                    
                    if success:
                        completed_series.add(series_uid)
                        patient_success_count += 1
                        print(f"        ✓ {message}")
                    else:
                        failed_series.add(series_uid)
                        print(f"        ✗ {message}")
                    
                    # Save progress periodically
                    progress["completed_series"] = completed_series
                    progress["failed_series"] = failed_series
                    save_progress(progress)
                    
                    # Rate limiting
                    time.sleep(1)
            
            # Mark patient as done
            patients_done.append(patient_id)
            progress["patients_done"] = patients_done
            save_progress(progress)
            
            print(f"  Patient {patient_id} completed: {patient_success_count}/{patient_series_count + patient_success_count} series successful")
            
        except Exception as e:
            print(f"  Error processing patient {patient_id}: {e}")
            continue
    
    # Final summary
    print(f"\n=== Download Summary ===")
    print(f"Completed series: {len(completed_series)}")
    print(f"Failed series: {len(failed_series)}")
    print(f"Completed patients: {len(patients_done)}")
    
    # Check downloaded data
    total_dicoms = 0
    total_size = 0
    
    for dcm_file in DATA_DIR.rglob("*.dcm"):
        total_dicoms += 1
        total_size += dcm_file.stat().st_size
    
    print(f"Total DICOM files: {total_dicoms}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Data location: {DATA_DIR}")

if __name__ == "__main__":
    main()