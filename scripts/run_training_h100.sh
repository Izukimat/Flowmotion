#!/usr/bin/env bash
# run_training_h100.sh  –  Launch FLF2V training on a single H100 80 GB
# Uses 2-D MedVAE (medvae_8x1_2d) and the large DiT in configs/config_h100_2d.yaml

set -e  # exit on first error
set -o pipefail          # ← propagate errors through pipes

# ───────── Environment ──────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0                       # pick the H100
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoids frag

# ───────── Data paths (edit if different) ───────────────────────────
DATA_DIR="/home/ubuntu/azureblob/4D-Lung-Interpolated/data"
METADATA_CSV="/home/ubuntu/flowmotion/splits/metadata.csv"

# ───────── Output dir with timestamp ────────────────────────────────
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$HOME/azureblob/output/flf2v_h100_$STAMP"
mkdir -p "$OUTPUT_DIR"

# Copy config for record
cp configs/config_h100_2d.yaml "$OUTPUT_DIR/"

echo "Launching training on H100..."
echo "  Data dir     : $DATA_DIR"
echo "  Metadata CSV : $METADATA_CSV"
echo "  Output dir   : $OUTPUT_DIR"

# Sanity checks
[ -d "$DATA_DIR"     ] || { echo "Data dir not found"; exit 1; }
[ -f "$METADATA_CSV" ] || { echo "Metadata CSV not found"; exit 1; }

# ───────── Train ────────────────────────────────────────────────────
python scripts/train_flf2v.py \
    --config configs/config_h100_2d.yaml \
    --csv-file $METADATA_CSV \
    --output-dir $OUTPUT_DIR \
    --wandb-project lungct-flf2v-numpy \
    --wandb-run "h100_numpy_$(date +%Y%m%d_%H%M%S)" \
    --num-workers 24 \
    --prefetch-factor 4 \
    2>&1 | tee $OUTPUT_DIR/training.log


# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully! Results saved to $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    echo "  - Training log: $OUTPUT_DIR/training.log"
    echo "  - Config: $OUTPUT_DIR/config_h100_2d.yaml"
    echo "  - Checkpoints: $OUTPUT_DIR/checkpoint_*.pt"
    echo "  - Final model: $OUTPUT_DIR/final_model.pt"
else
    echo "Training failed! Check the log for details: $OUTPUT_DIR/training.log"
    exit 1
fi