#!/bin/bash
# run_training_v100.sh - Updated training script for single V100 with NumPy data
# Works with preprocessed NumPy array data structure

# for OMM error
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# Create output directory with timestamp
OUTPUT_DIR="$HOME/azureblob/output/flf2v_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Copy config for reference
cp configs/config_v100.yaml $OUTPUT_DIR/

echo "Starting training with NumPy array data..."
echo "Data root: $DATA_ROOT"
echo "Metadata CSV: $METADATA_CSV"
echo "Output directory: $OUTPUT_DIR"

# Data paths - Updated for NumPy array structure
DATA_ROOT="/home/ragenius_admin/azureblob/4D-Lung-Interpolated/data/"
METADATA_CSV="/home/ragenius_admin/flowmotion/splits/metadata.csv"

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data directory not found: $DATA_ROOT"
    echo "Please ensure the data is available at the specified location"
    exit 1
fi

# Check if metadata CSV exists
if [ ! -f "$METADATA_CSV" ]; then
    echo "ERROR: Metadata CSV not found: $METADATA_CSV"
    echo "Please run scripts/create_splits.py first to generate the metadata CSV:"
    echo "  python scripts/create_splits.py --data-root $DATA_ROOT --output-dir /home/ragenius_admin/azureblob/4D-Lung-Interpolated/splits/"
    exit 1
fi

echo "✓ Data directory found: $DATA_ROOT"
echo "✓ Metadata CSV found: $METADATA_CSV"
echo "✓ Output directory: $OUTPUT_DIR"

# Run training with updated arguments
python scripts/train_flf2v.py \
    --config configs/config_v100.yaml \
    --csv-file $METADATA_CSV \
    --data-root $DATA_ROOT \
    --output-dir $OUTPUT_DIR \
    --wandb-project lungct-flf2v-numpy \
    --wandb-run "v100_numpy_$(date +%Y%m%d_%H%M%S)" \
    --num-workers 4 \
    --prefetch-factor 2 \
    2>&1 | tee $OUTPUT_DIR/training.log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully! Results saved to $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    echo "  - Training log: $OUTPUT_DIR/training.log"
    echo "  - Config: $OUTPUT_DIR/config_v100.yaml"
    echo "  - Checkpoints: $OUTPUT_DIR/checkpoint_*.pt"
    echo "  - Final model: $OUTPUT_DIR/final_model.pt"
else
    echo "Training failed! Check the log for details: $OUTPUT_DIR/training.log"
    exit 1
fi