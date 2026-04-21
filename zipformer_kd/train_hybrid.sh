#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Joint CTC + RNNT fine-tuning (no KD / teacher).
# Uses icefall-standard loss weights: simple=0.5, pruned=1.0, ctc=0.2
#
# Run from repo root (voice_to_text/):
#   conda activate voice_to_text
#   PRETRAINED_CKPT=zipformer/exp_phase2_supervised/best-valid-loss.pt \
#   bash zipformer_kd/train_hybrid.sh
# ==============================================================================

# 1. Hardware
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORLD_SIZE="${WORLD_SIZE:-$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}"
MASTER_PORT="${MASTER_PORT:-12358}"

# 2. Directories
EXP_DIR="${EXP_DIR:-zipformer_kd/exp_hybrid}"
mkdir -p "$EXP_DIR"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
if [[ -n "$PRETRAINED_CKPT" && ! -f "$PRETRAINED_CKPT" ]]; then
  echo "Error: PRETRAINED_CKPT not found: $PRETRAINED_CKPT"
  exit 1
fi

# 3. Data
MANIFEST_DIR="${MANIFEST_DIR:-data/lhotse_cutsets}"
TRAIN_MANIFEST_DIR="${TRAIN_MANIFEST_DIR:-data/lhotse_cutsets_new_train}"

# 4. Training config
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MAX_DURATION="${MAX_DURATION:-700}"
ACCUM_GRAD="${ACCUM_GRAD:-2}"
USE_FP16="${USE_FP16:-1}"
USE_BF16="${USE_BF16:-1}"
BASE_LR="${BASE_LR:-0.0005}"           # 1/10 of original pretraining LR
LR_BATCHES="${LR_BATCHES:-5000}"
LR_EPOCHS="${LR_EPOCHS:-5}"
WARMUP_BATCHES="${WARMUP_BATCHES:-2000}"
NUM_WORKERS="${NUM_WORKERS:-128}"
NUM_BUCKETS="${NUM_BUCKETS:-80}"
ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-0.1}"
DROPOUT_INPUT="${DROPOUT_INPUT:-0.0}"
DROPOUT_FEATURES="${DROPOUT_FEATURES:-0.0}"
SPK_KERNEL_DROPOUT="${SPK_KERNEL_DROPOUT:-0.1}"

# 5. Joint loss weights (icefall-standard for Zipformer)
CTC_LOSS_SCALE="${CTC_LOSS_SCALE:-0.2}"
SIMPLE_LOSS_SCALE="${SIMPLE_LOSS_SCALE:-0.5}"
AM_SCALE="${AM_SCALE:-0.0}"
PRUNE_RANGE="${PRUNE_RANGE:-5}"
DELAY_PENALTY="${DELAY_PENALTY:-0.0}"

# 6. Multi-Speaker config
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

echo "Starting joint CTC+RNNT training (no KD)..."
echo "Experiment Dir    : $EXP_DIR"
echo "Train manifest dir: $TRAIN_MANIFEST_DIR"
echo "Val manifest dir  : $MANIFEST_DIR"
echo "Pretrained ckpt   : ${PRETRAINED_CKPT:-none}"
echo "Loss weights      : simple=$SIMPLE_LOSS_SCALE pruned=1.0 ctc=$CTC_LOSS_SCALE"
echo "LR                : base=$BASE_LR warmup=$WARMUP_BATCHES enc_scale=$ENCODER_LR_SCALE"
echo "Dropout           : input=$DROPOUT_INPUT features=$DROPOUT_FEATURES spk_kernel=$SPK_KERNEL_DROPOUT"
echo "GPUs              : $WORLD_SIZE"

export PYTHONPATH="$(pwd)/zipformer_kd:$(pwd)/zipformer:${PYTHONPATH:-}"

PRETRAINED_ARGS=()
if [[ -n "$PRETRAINED_CKPT" ]]; then
  PRETRAINED_ARGS=(--pretrained-dir "$PRETRAINED_CKPT")
fi

python zipformer_kd/finetune.py \
  --world-size "$WORLD_SIZE" \
  --master-port "$MASTER_PORT" \
  --num-epochs "$NUM_EPOCHS" \
  --start-epoch "${START_EPOCH:-1}" \
  --start-batch "${START_BATCH:-0}" \
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
  --use-transducer True \
  --use-ctc True \
  --ctc-loss-scale "$CTC_LOSS_SCALE" \
  --simple-loss-scale "$SIMPLE_LOSS_SCALE" \
  --am-scale "$AM_SCALE" \
  --prune-range "$PRUNE_RANGE" \
  --delay-penalty "$DELAY_PENALTY" \
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
