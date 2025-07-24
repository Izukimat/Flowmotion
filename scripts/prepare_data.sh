#!/bin/bash
# prepare_data.sh - Data preparation script

DATA_ROOT="/path/to/raw_data"
OUTPUT_DIR="/path/to/processed_data"

# Create directory structure
mkdir -p $OUTPUT_DIR/{train,val,test}

# Process each patient
for patient_dir in $DATA_ROOT/*/; do
    patient_id=$(basename $patient_dir)
    echo "Processing $patient_id..."
    
    # Your preprocessing code here
    # e.g., convert DICOM to NIfTI, extract breathing phases, etc.
done

# Create metadata CSV
python create_metadata.py \
    --data-dir $OUTPUT_DIR \
    --output-csv $OUTPUT_DIR/metadata.csv \
    --train-split 0.8 \
    --val-split 0.1 \
    --test-split 0.1

echo "Data preparation complete!"