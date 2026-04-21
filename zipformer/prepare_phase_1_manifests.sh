#!/bin/bash

set -euo pipefail

MODE="${1:-full}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREP_SCRIPT="$ROOT_DIR/SSL_prepare_data/prepare_ssl_data.py"
LINKS_TXT="${SSL_LINKS_TXT:-$ROOT_DIR/SSL_prepare_data/all_audio_watch_urls.txt}"
DATA_ROOT="${SSL_DATA_ROOT:-$ROOT_DIR/data/ssl_youtube/$MODE}"

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

# Avoid leaking user-site packages into this environment.
export PYTHONNOUSERSITE=1

# Ensure Hugging Face cache points to a writable path.
DEFAULT_HF_CACHE="${HOME}/.cache/huggingface"
HF_CACHE_BASE="${SSL_HF_CACHE_DIR:-${HF_HOME:-$DEFAULT_HF_CACHE}}"
if ! mkdir -p "$HF_CACHE_BASE" >/dev/null 2>&1 || [[ ! -w "$HF_CACHE_BASE" ]]; then
  HF_CACHE_BASE="$ROOT_DIR/.hf-cache"
  mkdir -p "$HF_CACHE_BASE"
fi
export HF_HOME="$HF_CACHE_BASE"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

case "$MODE" in
  smoke)
    : "${SSL_NUM_JOBS:=4}"
    : "${SSL_NORMALIZE_WORKERS:=4}"
    ;;
  full)
    : "${SSL_NUM_JOBS:=32}"
    : "${SSL_NORMALIZE_WORKERS:=16}"
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
  --skip-download true
  --num-jobs "$SSL_NUM_JOBS"
  --normalize-workers "$SSL_NORMALIZE_WORKERS"
)

if [[ "${FORCE_PREP:-0}" == "1" ]]; then
  PREP_ARGS+=(--force true)
fi

if [[ -n "${SSL_SOURCE_AUDIO_DIR:-}" ]]; then
  PREP_ARGS+=(--source-audio-dir "$SSL_SOURCE_AUDIO_DIR")
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
  if [[ -n "${SSL_WAV2VEC2_BATCH_SIZE:-}" ]]; then
    PREP_ARGS+=(--wav2vec2-batch-size "$SSL_WAV2VEC2_BATCH_SIZE")
  fi
fi

echo "Preparing Phase 1 manifests in mode: $MODE"
"$PYTHON_BIN" "$PREP_SCRIPT" "${PREP_ARGS[@]}"
