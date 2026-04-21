#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="/home/vietnam/.conda/envs/voice_to_text/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/vietnam/.conda/envs/voice_to_text/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
LOG="/home/vietnam/voice_to_text/data/data_label_auto_yt_run_full/voting_full_gpu.log"
cd /home/vietnam/voice_to_text
/home/vietnam/.conda/envs/voice_to_text/bin/python -u /home/vietnam/voice_to_text/voting_label.py \
  --base-dir /home/vietnam/voice_to_text/data/data_label_auto_yt_run_full \
  --zip-model-dir /home/vietnam/voice_to_text/models/zipformer_hynt_30m_onnx \
  --gipformer-repo /home/vietnam/voice_to_text/gipformer \
  --hf-cache-dir /home/vietnam/voice_to_text/.hf-cache-local \
  --provider auto \
  --zip-provider cuda \
  --gip-provider cpu \
  --steps exact,salvage,combine \
  --batch-size-cuda 96 \
  --batch-size-cpu 16 \
  --salvage-batch-size 384 \
  --load-workers 128 \
  --gip-num-threads 128 \
  --zip-num-threads 2 \
  --save-every 1000 \
  --resume \
  2>&1 | tee -a "$LOG"
