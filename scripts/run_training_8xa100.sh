#!/usr/bin/env bash
# run_training_8xa100.sh – Launch FLF2V training on 8×A100 80 GB (single node)

set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="/home/ubuntu/azureblob/4D-Lung-Interpolated/data"
METADATA_CSV="/home/ubuntu/flowmotion/splits/metadata.csv"

STAMP=$(date +%Y%m%d_%H%M%S)
#OUTPUT_DIR="$HOME/azureblob/output/flf2v_8xa100_$STAMP"
OUTPUT_DIR="$HOME/azureblob/output/flf2v_8xa100_20250808_153946/"
mkdir -p "$OUTPUT_DIR"
cp configs/config_a100.yaml "$OUTPUT_DIR/"

echo "Launching distributed training on 8×A100..."
echo "  Data dir     : $DATA_DIR"
echo "  Metadata CSV : $METADATA_CSV"
echo "  Output dir   : $OUTPUT_DIR"

# ---- Resume logic ----
RESUME_ARG=""
if [[ -f "$OUTPUT_DIR/best_model.pt" ]]; then
  echo "⏩ Resuming from checkpoint: $OUTPUT_DIR/best_model.pt"
  RESUME_ARG="--resume $OUTPUT_DIR/best_model.pt"
fi

# ---- Run torchrun ----
torchrun \
  --nnodes 1 --nproc_per_node 8 \
  --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  scripts/train_flf2v.py \
    --config configs/config_h100_2d.yaml \
    --csv-file "$METADATA_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --distributed \
    --wandb-project lungct-flf2v-a100 \
    --wandb-run "8xa100_${STAMP}" \
    --num-workers 24 \
    --prefetch-factor 4 \
    $RESUME_ARG | tee "$OUTPUT_DIR/training.log"

STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "✅ Training finished, results in $OUTPUT_DIR"
else
  echo "❌ Training failed, see $OUTPUT_DIR/training.log"
fi
