#!/bin/bash

set -euo pipefail

# ==============================================================================
# PHASE 1: SSL PRETRAINING (HuBERT/WAV2VEC2)
# ==============================================================================
#
# Usage:
#   bash zipformer/train_phase_1.sh smoke
#   bash zipformer/train_phase_1.sh full
#
# Useful overrides:
#   SKIP_DOWNLOAD=1 SSL_SOURCE_AUDIO_DIR=/path/to/audio bash zipformer/train_phase_1.sh smoke
#   SSL_LINKS_TXT=/path/to/urls.txt bash zipformer/train_phase_1.sh full
#   FORCE_PREP=1 bash zipformer/train_phase_1.sh smoke
#   SSL_FEATURE_TYPE=spectral bash zipformer/train_phase_1.sh smoke
#   SSL_WAV2VEC2_MODEL=nguyenvulebinh/wav2vec2-base-vietnamese-250h SSL_WAV2VEC2_LAYER=6 bash zipformer/train_phase_1.sh smoke
# ==============================================================================

MODE="${1:-smoke}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREP_SCRIPT="$ROOT_DIR/SSL_prepare_data/prepare_ssl_data.py"
LINKS_TXT="${SSL_LINKS_TXT:-$ROOT_DIR/SSL_prepare_data/all_audio_watch_urls.txt}"
DATA_ROOT="${SSL_DATA_ROOT:-$ROOT_DIR/data/ssl_youtube/$MODE}"
MANIFEST_DIR="$DATA_ROOT/manifests"
METADATA_PATH="$MANIFEST_DIR/ssl_data_metadata.json"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "/root/venv/bin/python" ]]; then
  PYTHON_BIN="/root/venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python interpreter not found. Set PYTHON=/path/to/python"
  exit 1
fi

PYTHON_SITE_PACKAGES="$(
"$PYTHON_BIN" - <<'PY'
import site
for path in site.getsitepackages():
    if "site-packages" in path:
        print(path)
        break
PY
)"

if [[ -n "${PYTHON_SITE_PACKAGES}" ]]; then
  EXTRA_LD_PATHS=()
  for rel in \
    torch/lib \
    k2/lib64 \
    nvidia/cublas/lib \
    nvidia/cuda_cupti/lib \
    nvidia/cuda_nvrtc/lib \
    nvidia/cuda_runtime/lib \
    nvidia/cudnn/lib \
    nvidia/cufft/lib \
    nvidia/curand/lib \
    nvidia/cusolver/lib \
    nvidia/cusparse/lib \
    nvidia/nccl/lib \
    nvidia/nvtx/lib
  do
    if [[ -d "${PYTHON_SITE_PACKAGES}/${rel}" ]]; then
      EXTRA_LD_PATHS+=("${PYTHON_SITE_PACKAGES}/${rel}")
    fi
  done

  if (( ${#EXTRA_LD_PATHS[@]} > 0 )); then
    EXTRA_LD_PATHS_JOINED="$(IFS=:; echo "${EXTRA_LD_PATHS[*]}")"
    export LD_LIBRARY_PATH="${EXTRA_LD_PATHS_JOINED}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
MASTER_PORT="${MASTER_PORT:-12354}"
USE_FP16="${USE_FP16:-1}"
EXTRACTOR_MODE="${EXTRACTOR_MODE:-default}"
EXP_DIR="${EXP_DIR:-zipformer/exp_phase1_ssl_$MODE}"
mkdir -p "$EXP_DIR"

case "$MODE" in
  smoke)
    NUM_EPOCHS="${NUM_EPOCHS:-2}"
    MAX_DURATION="${MAX_DURATION:-30}"
    ACCUM_GRAD="${ACCUM_GRAD:-1}"
    DROP_LAST="${DROP_LAST:-False}"
    SSL_NUM_JOBS="${SSL_NUM_JOBS:-4}"
    SSL_NORMALIZE_WORKERS="${SSL_NORMALIZE_WORKERS:-4}"
    ;;
  full)
    NUM_EPOCHS="${NUM_EPOCHS:-100}"
    MAX_DURATION="${MAX_DURATION:-87.5}"
    ACCUM_GRAD="${ACCUM_GRAD:-4}"
    DROP_LAST="${DROP_LAST:-True}"
    SSL_NUM_JOBS="${SSL_NUM_JOBS:-32}"
    SSL_NORMALIZE_WORKERS="${SSL_NORMALIZE_WORKERS:-16}"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Expected one of: smoke, full"
    exit 1
    ;;
esac

PREP_ARGS=(
  --mode "$MODE"
  --links-txt "$LINKS_TXT"
  --output-root "$DATA_ROOT"
)

if [[ -n "${SSL_SOURCE_AUDIO_DIR:-}" ]]; then
  PREP_ARGS+=(--source-audio-dir "$SSL_SOURCE_AUDIO_DIR")
fi

if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  PREP_ARGS+=(--skip-download true)
fi

if [[ "${FORCE_PREP:-0}" == "1" ]]; then
  PREP_ARGS+=(--force true)
fi

if [[ -n "${SSL_COOKIES:-}" ]]; then
  PREP_ARGS+=(--cookies "$SSL_COOKIES")
fi

if [[ -n "${SSL_MAX_URLS:-}" ]]; then
  PREP_ARGS+=(--max-urls "$SSL_MAX_URLS")
fi

if [[ -n "${SSL_DOWNLOAD_WORKERS:-}" ]]; then
  PREP_ARGS+=(--download-workers "$SSL_DOWNLOAD_WORKERS")
fi

if [[ -n "${SSL_NUM_JOBS:-}" ]]; then
  PREP_ARGS+=(--num-jobs "$SSL_NUM_JOBS")
fi

if [[ -n "${SSL_NORMALIZE_WORKERS:-}" ]]; then
  PREP_ARGS+=(--normalize-workers "$SSL_NORMALIZE_WORKERS")
fi

if [[ -n "${SSL_NUM_CLUSTERS:-}" ]]; then
  PREP_ARGS+=(--num-clusters "$SSL_NUM_CLUSTERS")
fi

if [[ -n "${SSL_SEGMENT_DURATION:-}" ]]; then
  PREP_ARGS+=(--segment-duration "$SSL_SEGMENT_DURATION")
fi

if [[ -n "${SSL_SEGMENT_HOP:-}" ]]; then
  PREP_ARGS+=(--segment-hop "$SSL_SEGMENT_HOP")
fi

# Feature extraction: wav2vec2 (default) or spectral
SSL_FEATURE_TYPE="${SSL_FEATURE_TYPE:-wav2vec2}"
PREP_ARGS+=(--feature-type "$SSL_FEATURE_TYPE")

if [[ "$SSL_FEATURE_TYPE" == "wav2vec2" ]]; then
  if [[ -n "${SSL_WAV2VEC2_MODEL:-}" ]]; then
    PREP_ARGS+=(--wav2vec2-model "$SSL_WAV2VEC2_MODEL")
  fi
  if [[ -n "${SSL_WAV2VEC2_LAYER:-}" ]]; then
    PREP_ARGS+=(--wav2vec2-layer "$SSL_WAV2VEC2_LAYER")
  fi
fi

echo "Preparing SSL data in mode: $MODE"
"$PYTHON_BIN" "$PREP_SCRIPT" "${PREP_ARGS[@]}"

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "Missing metadata after preparation: $METADATA_PATH"
  exit 1
fi

NUM_CLASSES="$("$PYTHON_BIN" - "$METADATA_PATH" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["num_classes"])
PY
)"

TRAIN_MANIFEST="$MANIFEST_DIR/ssl_cuts_train.jsonl.gz"
VALID_MANIFEST="$MANIFEST_DIR/ssl_cuts_valid.jsonl.gz"

echo "Starting Phase 1 SSL Pretraining..."
echo "Mode: $MODE"
echo "Experiment Dir: $EXP_DIR"
echo "GPUs: $WORLD_SIZE"
echo "Manifest Dir: $MANIFEST_DIR"
echo "Train Manifest: $TRAIN_MANIFEST"
echo "Valid Manifest: $VALID_MANIFEST"
echo "Num Classes: $NUM_CLASSES"
echo "Extractor Mode: $EXTRACTOR_MODE"

"$PYTHON_BIN" "$ROOT_DIR/zipformer/pretrain.py" \
  --world-size "$WORLD_SIZE" \
  --master-port "$MASTER_PORT" \
  --num-epochs "$NUM_EPOCHS" \
  --start-epoch 1 \
  --use-fp16 "$USE_FP16" \
  --extractor-mode "$EXTRACTOR_MODE" \
  --exp-dir "$EXP_DIR" \
  --full-libri 0 \
  --manifest-dir "$MANIFEST_DIR" \
  --train-manifest "$TRAIN_MANIFEST" \
  --valid-manifest "$VALID_MANIFEST" \
  --num-classes "$NUM_CLASSES" \
  --max-duration "$MAX_DURATION" \
  --drop-last "$DROP_LAST" \
  --accum-grad "$ACCUM_GRAD" \
  --tensorboard 1 \
  --save-every-n 5000 \
  --keep-last-k 10 \
  --average-period 200

# Note:
#   EXTRACTOR_MODE=default uses GroupNorm in the frontend and matches the
#   original HuBERT-style setup.
#   If you want to reduce norm shift before later streaming fine-tuning,
#   you can run:
#     EXTRACTOR_MODE=layer_norm bash zipformer/train_phase_1.sh smoke
