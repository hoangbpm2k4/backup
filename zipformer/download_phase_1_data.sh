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

case "$MODE" in
  smoke)
    : "${SSL_DOWNLOAD_WORKERS:=4}"
    ;;
  full)
    : "${SSL_DOWNLOAD_WORKERS:=8}"
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
  --download-workers "$SSL_DOWNLOAD_WORKERS"
)

if [[ -n "${SSL_COOKIES:-}" ]]; then
  PREP_ARGS+=(--cookies "$SSL_COOKIES")
fi

if [[ -n "${SSL_MAX_URLS:-}" ]]; then
  PREP_ARGS+=(--max-urls "$SSL_MAX_URLS")
fi

echo "Downloading Phase 1 SSL data in mode: $MODE"
"$PYTHON_BIN" "$PREP_SCRIPT" "${PREP_ARGS[@]}"
