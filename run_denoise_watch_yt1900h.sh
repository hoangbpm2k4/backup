#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vietnam/voice_to_text"
BASE_DIR="${BASE_DIR:-$ROOT/data/data_label_auto_yt_1900h}"
NOTEBOOK="${NOTEBOOK:-$ROOT/data_preprocess.ipynb}"
LOG="${LOG:-$BASE_DIR/logs/denoise_watch.log}"
SLEEP_SEC="${SLEEP_SEC:-30}"

mkdir -p "$BASE_DIR/logs"

export DATA_LABEL_BASE="$BASE_DIR"
# Denoise CPU tuning
export DENOISE_TORCH_THREADS="${DENOISE_TORCH_THREADS:-64}"
export DENOISE_INTEROP_THREADS="${DENOISE_INTEROP_THREADS:-8}"
export DENOISE_BATCH_SIZE="${DENOISE_BATCH_SIZE:-32}"
export DENOISE_PREFETCH_FILES_PER_ROUND="${DENOISE_PREFETCH_FILES_PER_ROUND:-800}"
export DENOISE_MAX_FILES_PER_ROUND="${DENOISE_MAX_FILES_PER_ROUND:-8000}"
export DENOISE_MAX_RUN_MINUTES_PER_ROUND="${DENOISE_MAX_RUN_MINUTES_PER_ROUND:-20}"
export DENOISE_SAVE_EVERY="${DENOISE_SAVE_EVERY:-200}"
export DENOISE_SLEEP_BETWEEN_ROUNDS_SEC="${DENOISE_SLEEP_BETWEEN_ROUNDS_SEC:-1}"

while true; do
  echo "[$(date '+%F %T')] Denoise cycle start" | tee -a "$LOG"
  conda run --no-capture-output -n voice_to_text \
    python "$ROOT/run_data_preprocess_notebook.py" \
      --notebook "$NOTEBOOK" \
      --cell-order 0 \
    2>&1 | tee -a "$LOG" || true
  echo "[$(date '+%F %T')] Denoise cycle end" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
