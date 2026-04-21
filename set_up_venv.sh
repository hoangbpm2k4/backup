#!/bin/bash
set -euo pipefail

MINICONDA_DIR="$HOME/miniconda3"
CONDA="$MINICONDA_DIR/bin/conda"

# Install Miniconda if not already present
if [ ! -f "$CONDA" ]; then
  echo ">>> Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
  rm /tmp/miniconda.sh
fi

# Init conda for this script
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Accept Anaconda Terms of Service
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create env if not exists
if ! "$CONDA" env list | grep -q "^voice_to_text "; then
  echo ">>> Creating conda env voice_to_text (Python 3.10)..."
  "$CONDA" create -y -n voice_to_text python=3.10
fi

conda activate voice_to_text

pip install --upgrade pip

pip install --no-cache-dir \
  torch==2.2.0+cu121 \
  torchaudio==2.2.0+cu121 \
  torchvision==0.17.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir \
  numpy==1.26.4 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  lhotse==1.32.2 \
  lilcom==1.8.2 \
  einops==0.8.2 \
  yt-dlp==2026.2.21 \
  SoundFile==0.13.1 \
  tqdm==4.67.3 \
  kaldialign==0.9.3 \
  sentencepiece==0.2.1 \
  tensorboard==2.20.0

pip install --no-cache-dir \
  k2==1.24.4.dev20240223+cuda12.1.torch2.2.0 \
  -f https://k2-fsa.github.io/k2/cuda.html

echo ""
echo ">>> Done! Activate env with: conda activate voice_to_text"