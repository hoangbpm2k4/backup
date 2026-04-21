#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vietnam/voice_to_text"
BASE_DIR="${1:-$ROOT/data/data_label_auto_yt_1900h}"
ZIP_MODEL_DIR="${ZIP_MODEL_DIR:-$ROOT/models/zipformer_hynt_30m_onnx}"
GIPFORMER_REPO="${GIPFORMER_REPO:-$ROOT/gipformer}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT/.hf-cache-local}"

BATCH_SIZE_CUDA="${BATCH_SIZE_CUDA:-6}"
BATCH_SIZE_CPU="${BATCH_SIZE_CPU:-4}"
SALVAGE_BATCH_SIZE="${SALVAGE_BATCH_SIZE:-256}"
LOAD_WORKERS="${LOAD_WORKERS:-32}"
ZIP_THREADS="${ZIP_THREADS:-2}"
GIP_THREADS="${GIP_THREADS:-8}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
MAX_ITEMS="${MAX_ITEMS:-}"

MANIFEST_DIR="$BASE_DIR/manifests"
mkdir -p "$MANIFEST_DIR"
PH1_LOG="$MANIFEST_DIR/voting_phase1_exact_lowvram.log"
PH2_LOG="$MANIFEST_DIR/voting_phase2_timestamp_only_lowvram.log"

source /home/vietnam/miniconda3/etc/profile.d/conda.sh
conda activate voice_to_text

# Required for sherpa_onnx CUDA provider to load cuDNN/cuBLAS from the conda env.
export LD_LIBRARY_PATH="/home/vietnam/.conda/envs/voice_to_text/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/vietnam/.conda/envs/voice_to_text/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date '+%F %T')] Phase 1 (exact) start | low VRAM target"
python "$ROOT/voting_label.py" \
  --base-dir "$BASE_DIR" \
  --zip-model-dir "$ZIP_MODEL_DIR" \
  --gipformer-repo "$GIPFORMER_REPO" \
  --hf-cache-dir "$HF_CACHE_DIR" \
  --provider auto \
  --zip-provider cuda \
  --gip-provider cpu \
  --steps exact \
  --batch-size-cuda "$BATCH_SIZE_CUDA" \
  --batch-size-cpu "$BATCH_SIZE_CPU" \
  --load-workers "$LOAD_WORKERS" \
  --zip-num-threads "$ZIP_THREADS" \
  --gip-num-threads "$GIP_THREADS" \
  --resume \
  --save-every "$SAVE_EVERY" \
  ${MAX_ITEMS:+--max-items "$MAX_ITEMS"} \
  2>&1 | tee "$PH1_LOG"
echo "[$(date '+%F %T')] Phase 1 (exact) done"

echo "[$(date '+%F %T')] Phase 2 (timestamp-only salvage+combine) start"
python "$ROOT/voting_label.py" \
  --base-dir "$BASE_DIR" \
  --zip-model-dir "$ZIP_MODEL_DIR" \
  --gipformer-repo "$GIPFORMER_REPO" \
  --hf-cache-dir "$HF_CACHE_DIR" \
  --provider cpu \
  --steps salvage,combine \
  --batch-size-cpu "$BATCH_SIZE_CPU" \
  --salvage-batch-size "$SALVAGE_BATCH_SIZE" \
  --load-workers "$LOAD_WORKERS" \
  --salvage-reuse-phase1 \
  --no-salvage-allow-model-infer \
  --save-every "$SAVE_EVERY" \
  2>&1 | tee "$PH2_LOG"
echo "[$(date '+%F %T')] Phase 2 (timestamp-only salvage+combine) done"

echo "Logs:"
echo "  - $PH1_LOG"
echo "  - $PH2_LOG"
