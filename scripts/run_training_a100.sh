#!/bin/bash
# run_training_a100.sh - Training script for A100 spot instances

# For Azure spot instances
OUTPUT_DIR="outputs/flf2v_a100_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Copy config
cp config_a100.yaml $OUTPUT_DIR/

# Run on 4 A100s
torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    train_lungct_flf2v.py \
    --config config_a100.yaml \
    --data-dir /path/to/lung_ct_data \
    --csv-file /path/to/metadata.csv \
    --output-dir $OUTPUT_DIR \
    --wandb-project lungct-flf2v \
    --wandb-run "a100_final_$(date +%Y%m%d_%H%M%S)" \
    --num-workers 8 \
    --distributed \
    --resume /path/to/v100_checkpoint.pt \
    2>&1 | tee $OUTPUT_DIR/training.log
