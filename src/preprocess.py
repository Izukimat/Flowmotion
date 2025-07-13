import os
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("/mnt/tcia_data/raw/RIDER Lung CT")
PROCESSED_DIR = Path("/mnt/tcia_data/processed/RIDER Lung CT")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

manifest_rows = []

for patient_dir in tqdm(list(RAW_DIR.iterdir()), desc="Patients"):
    if not patient_dir.is_dir():
        continue
    # Sort series folders for reproducible phase labeling
    series_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
    num_phases = len(series_dirs)
    if num_phases == 0:
        continue

    for idx, series_dir in enumerate(series_dirs):
        phase_id = idx  # 0, 1, 2, ...
        phase_percent = int(round(100 * idx / (num_phases-1))) if num_phases > 1 else 0

        # Gather all DICOM files (support nested structure)
        dicom_files = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            dicom_files = sorted(series_dir.rglob("*.dcm"))
        if not dicom_files:
            continue

        # Read and stack all slices by z-location
        slices = [pydicom.dcmread(f) for f in dicom_files]
        # Sort slices by position in z-axis (ImagePositionPatient[2])
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        shapes = [s.pixel_array.shape for s in slices]
        if len(set(shapes)) > 1:
            print(f"WARNING: Skipping {series_dir} (non-uniform slice shapes: {set(shapes)})")
            continue  # Skip this series
        volume = np.stack([s.pixel_array for s in slices], axis=0)

        # Output path: /mnt/tcia_data/processed/RIDER Lung CT/{patient_id}/phase_{phase_id}.npy
        out_dir = PROCESSED_DIR / patient_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"phase_{phase_id}.npy"
        out_path = out_dir / out_filename
        np.save(out_path, volume)

        # Save manifest row (no phase info in DICOM, so all is by index)
        manifest_rows.append({
            "patient_id": patient_dir.name,
            "phase_id": phase_id,
            "phase_percent": phase_percent,
            "series_dir": str(series_dir),
            "file_path": str(out_path),
            "num_slices": len(slices),
            "height": volume.shape[1],
            "width": volume.shape[2],
            "dtype": str(volume.dtype)
        })

# Save manifest CSV
manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(PROCESSED_DIR / "manifest.csv", index=False)

print(f"Processing complete. Manifest saved to {PROCESSED_DIR}/manifest.csv")
