#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# CTC-only fine-tuning (no RNNT pruning, no KD / teacher).
#
# Run from repo root (voice_to_text/):
#   # Step 0 (one-time): build Lhotse CutSets from new manifest
#   conda activate voice_to_text
#   python mms_msg/scripts/build_lhotse_cutset_augmented.py \
#     --overlap-dir data/overlap_zipformer_prep/full_overlap_yt_vlsp_trainonly_merged \
#     --output-dir  data/lhotse_cutsets_new_train \
#     --splits train --skip-augment --workers 32
#
#   # Step 1: CTC training
#   PRETRAINED_CKPT=zipformer/exp_phase2_supervised/best-valid-loss.pt \
#   bash zipformer_kd/train_ctc.sh
# ==============================================================================

# 1. Hardware
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORLD_SIZE="${WORLD_SIZE:-$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}"
MASTER_PORT="${MASTER_PORT:-12357}"

# 2. Directories
EXP_DIR="${EXP_DIR:-zipformer_kd/exp_ctc}"
mkdir -p "$EXP_DIR"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
if [[ -n "$PRETRAINED_CKPT" && ! -f "$PRETRAINED_CKPT" ]]; then
  echo "Error: PRETRAINED_CKPT not found: $PRETRAINED_CKPT"
  exit 1
fi

# 3. Data
# Val/test cuts come from --manifest-dir (existing lhotse_cutsets).
# Train cuts come from --train-manifest-dir (new converted cutsets).
MANIFEST_DIR="${MANIFEST_DIR:-data/lhotse_cutsets}"
TRAIN_MANIFEST_DIR="${TRAIN_MANIFEST_DIR:-data/lhotse_cutsets_new_train}"

# 4. Training config
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MAX_DURATION="${MAX_DURATION:-700}"
ACCUM_GRAD="${ACCUM_GRAD:-2}"
USE_FP16="${USE_FP16:-1}"
USE_BF16="${USE_BF16:-1}"
BASE_LR="${BASE_LR:-0.0005}"
LR_BATCHES="${LR_BATCHES:-5000}"
LR_EPOCHS="${LR_EPOCHS:-5}"
WARMUP_BATCHES="${WARMUP_BATCHES:-2000}"  # Eden LR warmup: 0.5×→1× over N batches
NUM_WORKERS="${NUM_WORKERS:-128}"
NUM_BUCKETS="${NUM_BUCKETS:-80}"
ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-0.1}"
DROPOUT_INPUT="${DROPOUT_INPUT:-0.0}"
DROPOUT_FEATURES="${DROPOUT_FEATURES:-0.0}"
SPK_KERNEL_DROPOUT="${SPK_KERNEL_DROPOUT:-0.5}"

# 5. Multi-Speaker config
NUM_SPEAKERS="${NUM_SPEAKERS:-3}"
CHUNK_SIZE_FRAMES="${CHUNK_SIZE_FRAMES:-32}"
CAUSAL="${CAUSAL:-1}"
USE_PYANNOTE_DIAR="${USE_PYANNOTE_DIAR:-0}"
PYANNOTE_WEIGHT_DIR="${PYANNOTE_WEIGHT_DIR:-weight}"
FREEZE_PYANNOTE="${FREEZE_PYANNOTE:-1}"
PYANNOTE_PHASE="${PYANNOTE_PHASE:-1}"
FREEZE_ENCODER="${FREEZE_ENCODER:-0}"
AUGMENT_TYPES="${AUGMENT_TYPES:-clean}"
SSA_SINGLE_TARGET="${SSA_SINGLE_TARGET:-1}"
SSA_TARGET_POLICY="${SSA_TARGET_POLICY:-round_robin}"
ZIPFORMER_CHUNK_SIZE="${ZIPFORMER_CHUNK_SIZE:-8,16,32}"
LEFT_CONTEXT_FRAMES="${LEFT_CONTEXT_FRAMES:-64,64,128}"
DECODE_CHUNK_LEN="${DECODE_CHUNK_LEN:-16}"
SHORT_CHUNK_SIZE="${SHORT_CHUNK_SIZE:-50}"

echo "Starting CTC-only training..."
echo "Experiment Dir    : $EXP_DIR"
echo "Train manifest dir: $TRAIN_MANIFEST_DIR"
echo "Val manifest dir  : $MANIFEST_DIR"
echo "Pretrained ckpt   : ${PRETRAINED_CKPT:-none}"
echo "GPUs              : $WORLD_SIZE"
echo "Num workers       : $NUM_WORKERS"
echo "Num buckets       : $NUM_BUCKETS"
echo "LR                : base=$BASE_LR warmup=$WARMUP_BATCHES enc_scale=$ENCODER_LR_SCALE"
echo "Dropout           : input=$DROPOUT_INPUT features=$DROPOUT_FEATURES spk_kernel=$SPK_KERNEL_DROPOUT"

export PYTHONPATH="$(pwd)/zipformer_kd:$(pwd)/zipformer:${PYTHONPATH:-}"

PRETRAINED_ARGS=()
if [[ -n "$PRETRAINED_CKPT" ]]; then
  PRETRAINED_ARGS=(--pretrained-dir "$PRETRAINED_CKPT")
fi

# 6. Resume behavior
# If the requested checkpoint is missing, fall back to the latest available
# checkpoint in EXP_DIR instead of crashing.
AUTO_RESUME="${AUTO_RESUME:-1}"
START_EPOCH="${START_EPOCH:-1}"
START_BATCH="${START_BATCH:-0}"

latest_epoch_in_exp() {
  local latest
  latest="$(find "$EXP_DIR" -maxdepth 1 -type f -name 'epoch-*.pt' -printf '%f\n' \
    | sed -E 's/^epoch-([0-9]+)\.pt$/\1/' \
    | sort -n \
    | tail -1)"
  echo "$latest"
}

latest_batch_in_exp() {
  local latest
  latest="$(find "$EXP_DIR" -maxdepth 1 -type f -name 'checkpoint-*.pt' -printf '%f\n' \
    | sed -E 's/^checkpoint-([0-9]+)\.pt$/\1/' \
    | sort -n \
    | tail -1)"
  echo "$latest"
}

if [[ "$START_BATCH" -gt 0 ]]; then
  REQ_BATCH_CKPT="$EXP_DIR/checkpoint-$START_BATCH.pt"
  if [[ ! -f "$REQ_BATCH_CKPT" ]]; then
    if [[ "$AUTO_RESUME" == "1" ]]; then
      LATEST_BATCH="$(latest_batch_in_exp)"
      if [[ -n "${LATEST_BATCH:-}" ]]; then
        echo "Warning: requested $REQ_BATCH_CKPT not found; resume from checkpoint-$LATEST_BATCH.pt"
        START_BATCH="$LATEST_BATCH"
      else
        echo "Warning: requested $REQ_BATCH_CKPT not found; no checkpoint-* found. Falling back to epoch resume logic."
        START_BATCH=0
      fi
    else
      echo "Error: $REQ_BATCH_CKPT not found (set AUTO_RESUME=1 to auto-fallback)."
      exit 1
    fi
  fi
fi

if [[ "$START_BATCH" -eq 0 && "$START_EPOCH" -gt 1 ]]; then
  REQ_EPOCH_CKPT="$EXP_DIR/epoch-$((START_EPOCH-1)).pt"
  if [[ ! -f "$REQ_EPOCH_CKPT" ]]; then
    if [[ "$AUTO_RESUME" == "1" ]]; then
      LATEST_EPOCH="$(latest_epoch_in_exp)"
      if [[ -n "${LATEST_EPOCH:-}" ]]; then
        START_EPOCH="$((LATEST_EPOCH + 1))"
        echo "Warning: requested $REQ_EPOCH_CKPT not found; resume from epoch-$LATEST_EPOCH.pt (start-epoch=$START_EPOCH)"
      else
        START_EPOCH=1
        echo "Warning: no epoch checkpoint found in $EXP_DIR; starting from epoch=1"
      fi
    else
      echo "Error: $REQ_EPOCH_CKPT not found (set AUTO_RESUME=1 to auto-fallback)."
      exit 1
    fi
  fi
fi

echo "Resume config     : start_epoch=$START_EPOCH start_batch=$START_BATCH auto_resume=$AUTO_RESUME"

python zipformer_kd/finetune.py \
  --world-size "$WORLD_SIZE" \
  --master-port "$MASTER_PORT" \
  --num-epochs "$NUM_EPOCHS" \
  --start-epoch "$START_EPOCH" \
  --start-batch "$START_BATCH" \
  --use-fp16 "$USE_FP16" \
  --use-bf16 "$USE_BF16" \
  --exp-dir "$EXP_DIR" \
  "${PRETRAINED_ARGS[@]}" \
  --bpe-model "zipformer/tokens.txt" \
  --manifest-dir "$MANIFEST_DIR" \
  --train-manifest-dir "$TRAIN_MANIFEST_DIR" \
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
  --encoder-lr-scale "$ENCODER_LR_SCALE" \
  --lr-batches "$LR_BATCHES" \
  --lr-epochs "$LR_EPOCHS" \
  --warmup-batches "$WARMUP_BATCHES" \
  --ctc-only-steps "${CTC_ONLY_STEPS:-0}" \
  --activity-teacher-forcing 1.0 \
  --activity-teacher-forcing-warmup-steps 50000 \
  --tensorboard 1 \
  --save-every-n 500 \
  --keep-last-k 10 \
  --average-period 200 \
  --use-transducer False \
  --use-ctc True \
  --ctc-loss-scale "${CTC_LOSS_SCALE:-1.0}" \
  --am-scale 0.0 \
  --simple-loss-scale 0.0 \
  --prune-range 0 \
  --delay-penalty 0.0 \
  --spk-kernel-dropout "$SPK_KERNEL_DROPOUT" \
  --dropout-input "$DROPOUT_INPUT" \
  --dropout-features "$DROPOUT_FEATURES" \
  --bg-threshold 0.5 \
  --num-workers "$NUM_WORKERS" \
  --num-buckets "$NUM_BUCKETS" \
  --kd-teacher-path "" \
  --kd-objective "ctc" \
  --kd-loss-scale 0.0 \
  --joiner-kd-scale 0.0 \
  --kd-fitnet-scale 0.0 \
  --kd-mgd-scale 0.0
