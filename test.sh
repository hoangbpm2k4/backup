#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate voice_to_text

CKPT="${CHECKPOINT:-zipformer_kd/exp_kd_v2/best-valid-loss.pt}"
if [[ ! -f "$CKPT" ]]; then
  CKPT="zipformer_kd/exp_hybrid/epoch-2.pt"
fi

DEFAULT_AUDIO="data/augmentation_samples/single_clean/09_test_00002083_single-2082.wav"
AUDIO="$DEFAULT_AUDIO"

# If first arg is a positional path (not an option), treat it as audio.
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  AUDIO="$1"
  shift
fi

EXTRA_ARGS=("$@")

echo "Using checkpoint: $CKPT"
echo "Using audio: $AUDIO"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra args: ${EXTRA_ARGS[*]}"
fi

python3 test_asr_quick.py \
  --checkpoint "$CKPT" \
  --audio "$AUDIO" \
  --use-model \
  --decode-chunk-size 32 \
  --left-context-frames 128 \
  --blank-penalty 0.0 \
  "${EXTRA_ARGS[@]}"
