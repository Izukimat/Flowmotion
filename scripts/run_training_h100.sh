#!/usr/bin/env bash
# run_training_h100.sh  –  Launch FLF2V training on a single H100 80 GB
# Uses 2-D MedVAE (medvae_8x1_2d) and the large DiT in configs/config_h100_2d.yaml

set -e  # exit on first error

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
    --config        configs/config_h100_2d.yaml \
    --data-dir      "$DATA_DIR" \
    --csv-file      "$METADATA_CSV" \
    --output-dir    "$OUTPUT_DIR" \
    --wandb-project lungct-flf2v \
    --wandb-run     "h100_${STAMP}" \
    --num-workers   8 \
    --prefetch-factor 4 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "✓ Training completed. Artifacts in $OUTPUT_DIR"
else
  echo "✗ Training failed — see $OUTPUT_DIR/training.log"
fi
