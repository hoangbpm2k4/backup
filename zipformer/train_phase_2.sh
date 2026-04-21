#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# PHASE 2: SUPERVISED MULTI-SPEAKER RNN-T TRAINING
# ==============================================================================
#
# This script launches Phase 2: Supervised fine-tuning using the pretrained
# HuBERT encoder and the MultiSpeakerRnntModel.
#
# Prerequisites:
# 1. Complete Phase 1 SSL Pretraining (or download a pretrained Hubert model).
# 2. Prepare your supervised audio + transcripts + timestamps using Lhotse into manifest files.
#    The `cut.supervisions` must contain words/sentences with start/duration
#    and speaker IDs (e.g., sup.speaker = 0, 1, 2).
#
# Usage:
#   chmod +x train_phase_2.sh
#   ./train_phase_2.sh
# ==============================================================================

# 1. Hardware config
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORLD_SIZE="${WORLD_SIZE:-$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}"
MASTER_PORT="${MASTER_PORT:-12355}" # Change if port is in use

# 2. Experiment directory
EXP_DIR="${EXP_DIR:-zipformer/exp_phase2_supervised}"
mkdir -p "$EXP_DIR"

# 3. Path to the model from Phase 1
PRETRAINED_CKPT="${PRETRAINED_CKPT:-zipformer/epoch-20.pt}"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "Error: PRETRAINED_CKPT not found: $PRETRAINED_CKPT"
  exit 1
fi

# 4. Training config
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_DURATION="${MAX_DURATION:-1200}"  # With gradient checkpointing: ~80-85GB VRAM
ACCUM_GRAD="${ACCUM_GRAD:-2}"         # Gradient accumulation
USE_FP16="${USE_FP16:-1}"             # Mixed precision
USE_BF16="${USE_BF16:-1}"             # BFloat16 (wider range, no grad scaler needed)
BASE_LR="${BASE_LR:-0.001}"           # Learning rate (use 0.0002 for unfreeze)

# 5. Multi-Speaker config
NUM_SPEAKERS="${NUM_SPEAKERS:-3}"       # Max speakers
CHUNK_SIZE_FRAMES="${CHUNK_SIZE_FRAMES:-32}"  # Fallback when dynamic chunk sampling is disabled
CAUSAL="${CAUSAL:-1}"

# Diarization: pyannote pretrained
USE_PYANNOTE_DIAR="${USE_PYANNOTE_DIAR:-0}"            # Disable pyannote for now (teacher forcing)
PYANNOTE_WEIGHT_DIR="${PYANNOTE_WEIGHT_DIR:-weight}"  # Path to fine-tuned pyannote segmentation weights
FREEZE_PYANNOTE="${FREEZE_PYANNOTE:-1}"               # Freeze pyannote weights (0 trainable params)
PYANNOTE_PHASE="${PYANNOTE_PHASE:-1}"                 # 1=oracle activity, 2=pyannote-pred activity

# Encoder freezing: freeze HuBERT encoder to let decoder/joiner learn first
FREEZE_ENCODER="${FREEZE_ENCODER:-0}"   # Set to 1 for Stage A (freeze encoder)

# Data augmentation: comma-separated types (clean,noise,rir,rir_noise)
AUGMENT_TYPES="${AUGMENT_TYPES:-clean}"

# SSA-style training: one target speaker per sample
SSA_SINGLE_TARGET="${SSA_SINGLE_TARGET:-1}"
SSA_TARGET_POLICY="${SSA_TARGET_POLICY:-round_robin}"  # random | round_robin

# Multi-chunk causal training (Zipformer will sample from these values)
# All chunk sizes must be divisible by max downsampling factor (8)
# Paired configs: (8,16,32) ↔ (64,64,128)
ZIPFORMER_CHUNK_SIZE="${ZIPFORMER_CHUNK_SIZE:-8,16,32}"
LEFT_CONTEXT_FRAMES="${LEFT_CONTEXT_FRAMES:-64,64,128}"
# Streaming decode config
DECODE_CHUNK_LEN="${DECODE_CHUNK_LEN:-16}"   # Chunk size for streaming decoding (encoder frames)
SHORT_CHUNK_SIZE="${SHORT_CHUNK_SIZE:-50}"   # Max dynamic chunk for training (sampled from 1..this)

csv_contains() {
  local needle="$1"
  local csv="$2"
  [[ ",$csv," == *",$needle,"* ]]
}

for cs in ${ZIPFORMER_CHUNK_SIZE//,/ }; do
  if (( cs <= 0 )); then
    echo "Error: ZIPFORMER_CHUNK_SIZE must contain positive values, got: $ZIPFORMER_CHUNK_SIZE"
    exit 1
  fi
done

for lc in ${LEFT_CONTEXT_FRAMES//,/ }; do
  if (( lc < 0 )); then
    echo "Error: LEFT_CONTEXT_FRAMES must be non-negative, got: $LEFT_CONTEXT_FRAMES"
    exit 1
  fi
  if (( lc % 8 != 0 )); then
    echo "Error: LEFT_CONTEXT_FRAMES must be divisible by 8 (default max downsampling factor), got: $lc"
    exit 1
  fi
done

if ! csv_contains "$DECODE_CHUNK_LEN" "$ZIPFORMER_CHUNK_SIZE"; then
  echo "Warning: DECODE_CHUNK_LEN=$DECODE_CHUNK_LEN is not in ZIPFORMER_CHUNK_SIZE=$ZIPFORMER_CHUNK_SIZE"
  echo "         This may increase train/inference mismatch."
fi

echo "Starting Phase 2 Supervised Multi-Speaker RNN-T Training..."
echo "Experiment Dir: $EXP_DIR"
echo "GPUs: $WORLD_SIZE"
echo "Streaming config: chunk_size={$ZIPFORMER_CHUNK_SIZE}, left_context={$LEFT_CONTEXT_FRAMES}, decode_chunk=$DECODE_CHUNK_LEN"

# 6. Launch training
python zipformer/finetune.py \
  --world-size "$WORLD_SIZE" \
  --master-port "$MASTER_PORT" \
  --num-epochs "$NUM_EPOCHS" \
  --start-epoch "${START_EPOCH:-1}" \
  --start-batch "${START_BATCH:-0}" \
  --use-fp16 "$USE_FP16" \
  --use-bf16 "$USE_BF16" \
  --exp-dir "$EXP_DIR" \
  --pretrained-dir "$PRETRAINED_CKPT" \
  --bpe-model "zipformer/tokens.txt" \
  --manifest-dir "data/lhotse_cutsets" \
  --max-duration "$MAX_DURATION" \
  --accum-grad "$ACCUM_GRAD" \
  --base-lr "$BASE_LR" \
  --num-speakers "$NUM_SPEAKERS" \
  --chunk-size-frames "$CHUNK_SIZE_FRAMES" \
  --use-pyannote-diar "$USE_PYANNOTE_DIAR" \
  --pyannote-weight-dir "$PYANNOTE_WEIGHT_DIR" \
  --freeze-pyannote "$FREEZE_PYANNOTE" \
  --pyannote-phase "$PYANNOTE_PHASE" \
  --enable-pre-asr-stream-routing 1 \
  --causal "$CAUSAL" \
  --chunk-size "$ZIPFORMER_CHUNK_SIZE" \
  --left-context-frames "$LEFT_CONTEXT_FRAMES" \
  --decode-chunk-len "$DECODE_CHUNK_LEN" \
  --short-chunk-size "$SHORT_CHUNK_SIZE" \
  --ssa-single-target "$SSA_SINGLE_TARGET" \
  --ssa-target-policy "$SSA_TARGET_POLICY" \
  --augment-types "$AUGMENT_TYPES" \
  --freeze-encoder "$FREEZE_ENCODER" \
  --encoder-lr-scale "${ENCODER_LR_SCALE:-0.1}" \
  --ctc-only-steps "${CTC_ONLY_STEPS:-2000}" \
  --activity-teacher-forcing 1.0 \
  --activity-teacher-forcing-warmup-steps 50000 \
  --tensorboard 1 \
  --save-every-n 500 \
  --keep-last-k 10 \
  --average-period 200 \
  --use-transducer True \
  --use-ctc True \
  --am-scale "${AM_SCALE:-0.1}" \
  --ctc-loss-scale "${CTC_LOSS_SCALE:-0.3}" \
  --delay-penalty "${DELAY_PENALTY:-0.01}" \
  --simple-loss-scale "${SIMPLE_LOSS_SCALE:-0.5}" \
  --prune-range "${PRUNE_RANGE:-5}" \
  --spk-kernel-dropout 0.5 \
  --bg-threshold 0.5 \
  --num-workers 64 \
  --num-buckets 50
