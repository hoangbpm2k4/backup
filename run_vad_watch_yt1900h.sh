#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vietnam/voice_to_text"
BASE_DIR="${BASE_DIR:-$ROOT/data/data_label_auto_yt_1900h}"
RAW_WAV_DIR="${RAW_WAV_DIR:-$ROOT/data/youtube_wav_16k_1900h}"
NOTEBOOK="${NOTEBOOK:-$ROOT/data_preprocess.ipynb}"
LOG="${LOG:-$BASE_DIR/logs/vad_watch.log}"
SLEEP_SEC="${SLEEP_SEC:-30}"

mkdir -p "$BASE_DIR/logs"

export DATA_LABEL_BASE="$BASE_DIR"
export RAW_WAV_DIR="$RAW_WAV_DIR"
export VAD_NUM_WORKERS="${VAD_NUM_WORKERS:-96}"
export VAD_EXECUTOR="${VAD_EXECUTOR:-thread}"
export VAD_MAX_IN_FLIGHT="${VAD_MAX_IN_FLIGHT:-256}"
export VAD_SORT_LONG_FIRST="${VAD_SORT_LONG_FIRST:-0}"
export VAD_CHECKPOINT_EVERY_FILES="${VAD_CHECKPOINT_EVERY_FILES:-1}"

while true; do
  echo "[$(date '+%F %T')] VAD cycle start" | tee -a "$LOG"
  conda run --no-capture-output -n voice_to_text \
    python "$ROOT/run_data_preprocess_notebook.py" \
      --notebook "$NOTEBOOK" \
      --cell-order 1 \
    2>&1 | tee -a "$LOG" || true
  echo "[$(date '+%F %T')] VAD cycle end" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
