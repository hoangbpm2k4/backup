#!/usr/bin/env bash
set -euo pipefail

cd /home/vietnam/voice_to_text/mms_msg
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voice_to_text

OUT_DIR="/home/vietnam/voice_to_text/data/overlap_zipformer_prep/full_overlap_yt_vlsp"
LOG_FILE="${OUT_DIR}/run.log"
mkdir -p "${OUT_DIR}"

python scripts/generate_multidataset_overlap.py \
  --manifest /home/vietnam/voice_to_text/data/overlap_zipformer_prep/input_manifest_full.jsonl \
  --output-dir "${OUT_DIR}" \
  --workers 96 \
  --batch-size 2048 \
  --progress-every 1000 \
  --verify-audio-exists \
  --no-write-cutset \
  --no-delete-used-source \
  --min-disk-gb 60 \
  2>&1 | tee -a "${LOG_FILE}"
