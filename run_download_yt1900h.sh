#!/usr/bin/env bash
set -euo pipefail
cd /home/vietnam/voice_to_text
LOG="/home/vietnam/voice_to_text/data/youtube_1900h_download.log"
OUTDIR="/home/vietnam/voice_to_text/data/youtube_wav_16k_1900h"
ARCHIVE="/home/vietnam/voice_to_text/data/youtube_wav_16k_1900h.archive.txt"
YT_WORKERS="${YT_WORKERS:-8}"
YT_RETRIES="${YT_RETRIES:-25}"
mkdir -p /home/vietnam/voice_to_text/data

echo "[$(date '+%F %T')] start download" | tee -a "$LOG"
conda run --no-capture-output -n voice_to_text python /home/vietnam/voice_to_text/download_youtube_wav16k_from_csv.py \
  --csv /home/vietnam/voice_to_text/link_ytb_1900h.csv \
  --url-column URL \
  --cookies /home/vietnam/voice_to_text/youtube_auth_cookies.txt \
  --output-dir "$OUTDIR" \
  --archive-file "$ARCHIVE" \
  --workers "$YT_WORKERS" \
  --retries "$YT_RETRIES" \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] done download" | tee -a "$LOG"
