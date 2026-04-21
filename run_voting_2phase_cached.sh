#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vietnam/voice_to_text"
BASE_DIR="${1:-$ROOT/data/data_label_auto_yt_run_full}"
ZIP_MODEL_DIR="${ZIP_MODEL_DIR:-$ROOT/hf_zipformer30m_streaming_benchmark/models}"
GIPFORMER_REPO="${GIPFORMER_REPO:-$ROOT/gipformer}"

PROVIDER="${PROVIDER:-auto}"
LOAD_WORKERS="${LOAD_WORKERS:-128}"
BATCH_SIZE_CUDA="${BATCH_SIZE_CUDA:-96}"
BATCH_SIZE_CPU="${BATCH_SIZE_CPU:-24}"
SALVAGE_BATCH_SIZE="${SALVAGE_BATCH_SIZE:-512}"

MANIFEST_DIR="$BASE_DIR/manifests"
mkdir -p "$MANIFEST_DIR"
PH1_LOG="$MANIFEST_DIR/voting_phase1_exact.log"
PH2_LOG="$MANIFEST_DIR/voting_phase2_timestamp_only.log"

echo "[$(date '+%F %T')] Phase 1 (exact) started"
conda run -n voice_to_text python "$ROOT/voting_label.py" \
  --base-dir "$BASE_DIR" \
  --zip-model-dir "$ZIP_MODEL_DIR" \
  --gipformer-repo "$GIPFORMER_REPO" \
  --ort-probe \
  --provider "$PROVIDER" \
  --steps exact \
  --batch-size-cuda "$BATCH_SIZE_CUDA" \
  --batch-size-cpu "$BATCH_SIZE_CPU" \
  --load-workers "$LOAD_WORKERS" \
  --resume \
  --save-every 1000 \
  2>&1 | tee "$PH1_LOG"
echo "[$(date '+%F %T')] Phase 1 (exact) finished"

echo "[$(date '+%F %T')] Phase 2 (timestamp-only salvage+combine) started"
conda run -n voice_to_text python "$ROOT/voting_label.py" \
  --base-dir "$BASE_DIR" \
  --zip-model-dir "$ZIP_MODEL_DIR" \
  --gipformer-repo "$GIPFORMER_REPO" \
  --ort-probe \
  --provider "$PROVIDER" \
  --steps salvage,combine \
  --batch-size-cpu "$BATCH_SIZE_CPU" \
  --salvage-batch-size "$SALVAGE_BATCH_SIZE" \
  --load-workers "$LOAD_WORKERS" \
  --salvage-reuse-phase1 \
  --no-salvage-allow-model-infer \
  --save-every 1000 \
  2>&1 | tee "$PH2_LOG"
echo "[$(date '+%F %T')] Phase 2 (timestamp-only salvage+combine) finished"

echo "Logs:"
echo "  - $PH1_LOG"
echo "  - $PH2_LOG"
