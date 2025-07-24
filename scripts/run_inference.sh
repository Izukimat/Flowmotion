#!/bin/bash
# run_inference.sh - Inference script

# Set checkpoint path
CHECKPOINT="/path/to/best_model.pt"
OUTPUT_DIR="inference_results/$(date +%Y%m%d_%H%M%S)"

# Single pair inference
python inference_lungct_flf2v.py \
    --checkpoint $CHECKPOINT \
    --output-dir $OUTPUT_DIR \
    --first-frame /path/to/inhale.nii.gz \
    --last-frame /path/to/exhale.nii.gz \
    --num-frames 40 \
    --guidance-scale 1.0 \
    --num-inference-steps 50 \
    --save-formats nifti video figure \
    --video-fps 10 \
    --verbose

# Batch inference from CSV
python inference_lungct_flf2v.py \
    --checkpoint $CHECKPOINT \
    --output-dir $OUTPUT_DIR \
    --input-csv /path/to/test_pairs.csv \
    --num-frames 40 \
    --batch-size 4 \
    --save-formats nifti video \
    --device cuda