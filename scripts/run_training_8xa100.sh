#!/usr/bin/env bash
# run_training_8xa100.sh – Launch FLF2V training on 8×A100 80 GB (single node)

set -euo pipefail

# ───────── Environment ──────────────────────────────────────────────
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # same as before

# If you *only* want the 8 GPUs on this box, do nothing.
# Otherwise uncomment and list them explicitly, e.g.:
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ───────── Data paths (edit if different) ───────────────────────────
DATA_DIR="/home/ubuntu/azureblob/4D-Lung-Interpolated/data"
METADATA_CSV="/home/ubuntu/flowmotion/splits/metadata.csv"

# ───────── Output dir with timestamp ────────────────────────────────
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$HOME/azureblob/output/flf2v_8xa100_$STAMP"
mkdir -p "$OUTPUT_DIR"
cp configs/config_a100.yaml "$OUTPUT_DIR/"      # keep a record

echo "Launching distributed training on 8×A100..."
echo "  Data dir     : $DATA_DIR"
echo "  Metadata CSV : $METADATA_CSV"
echo "  Output dir   : $OUTPUT_DIR"

# ───────── torchrun ─────────────────────────────────────────────────
torchrun \
  --nnodes 1 --nproc_per_node 8 \
  --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  scripts/train_flf2v.py \
    --config configs/config_h100.yaml \
    --csv-file "$METADATA_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --distributed \                        # <── tell the script to use DDP
    --wandb-project lungct-flf2v-a100 \
    --wandb-run "8xa100_${STAMP}" \
    --num-workers 24 \                     # per *process*; tune as needed
    --prefetch-factor 4 | tee "$OUTPUT_DIR/training.log"

STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "✅ Training finished, results in $OUTPUT_DIR"
else
  echo "❌ Training failed, see $OUTPUT_DIR/training.log"
fi
