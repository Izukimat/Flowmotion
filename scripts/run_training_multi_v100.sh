#!/bin/bash
# run_training_multi_v100.sh - Distributed training on 4 V100s

# Set distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Create output directory
OUTPUT_DIR="outputs/flf2v_distributed_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Copy config
cp config_v100.yaml $OUTPUT_DIR/

# Run distributed training
torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_lungct_flf2v.py \
    --config config_v100.yaml \
    --data-dir /path/to/lung_ct_data \
    --csv-file /path/to/metadata.csv \
    --output-dir $OUTPUT_DIR \
    --wandb-project lungct-flf2v \
    --wandb-run "v100_4gpu_$(date +%Y%m%d_%H%M%S)" \
    --num-workers 4 \
    --distributed \
    2>&1 | tee $OUTPUT_DIR/training.log
