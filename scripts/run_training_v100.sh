#!/bin/bash
# run_training_v100.sh - Training script for single V100

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory with timestamp
OUTPUT_DIR="outputs/flf2v_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Copy config for reference
cp config_v100.yaml $OUTPUT_DIR/

# Run training
python train_lungct_flf2v.py \
    --config config_v100.yaml \
    --data-dir /path/to/lung_ct_data \
    --csv-file /path/to/metadata.csv \
    --output-dir $OUTPUT_DIR \
    --wandb-project lungct-flf2v \
    --wandb-run "v100_$(date +%Y%m%d_%H%M%S)" \
    --num-workers 4 \
    --prefetch-factor 2 \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed! Results saved to $OUTPUT_DIR"
