#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# PHASE 2-KD: KNOWLEDGE DISTILLATION from Teacher JIT (30M Zipformer) into
#             the Multi-Speaker Student model.
#
# Run from the repo root (voice_to_text/).
#
# Typical usage:
#   # Step 0 (one-time): transfer Teacher weights into Student init checkpoint
#   python zipformer_kd/transfer_weights.py \
#       --teacher-jit  hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt \
#       --student-ckpt zipformer/exp_phase2_supervised/best-valid-loss.pt \
#       --out-ckpt     zipformer_kd/exp_kd/init_from_teacher.pt
#
#   # Step 1: KD fine-tuning
#   KD_TEACHER_PATH=hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt \
#   PRETRAINED_CKPT=zipformer_kd/exp_kd/init_from_teacher.pt \
#   ./zipformer_kd/train_kd.sh
# ==============================================================================

# 1. Hardware config
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORLD_SIZE="${WORLD_SIZE:-$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}"
MASTER_PORT="${MASTER_PORT:-12356}"   # different port from normal training

# 2. Experiment directory
EXP_DIR="${EXP_DIR:-zipformer_kd/exp_kd}"
mkdir -p "$EXP_DIR"

# 3. Student init checkpoint (output of transfer_weights.py, or any existing ckpt)
PRETRAINED_CKPT="${PRETRAINED_CKPT:-zipformer_kd/student_teacher_init.pt}"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "Error: PRETRAINED_CKPT not found: $PRETRAINED_CKPT"
  exit 1
fi

# 4. Teacher JIT path (required for KD loss)
KD_TEACHER_PATH="${KD_TEACHER_PATH:-hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt}"
if [[ ! -f "$KD_TEACHER_PATH" ]]; then
  echo "Error: KD_TEACHER_PATH not found: $KD_TEACHER_PATH"
  exit 1
fi

# 5. Training config
NUM_EPOCHS="${NUM_EPOCHS:-1}"
# KD adds ~50-60% VRAM overhead vs normal training (TeacherPy + extra student fwd).
# At BF16 + grad checkpointing: 350 ≈ 28-32GB.  Lower to 280 if OOM.
MAX_DURATION="${MAX_DURATION:-600}"
ACCUM_GRAD="${ACCUM_GRAD:-2}"
USE_FP16="${USE_FP16:-1}"
USE_BF16="${USE_BF16:-1}"
BASE_LR="${BASE_LR:-0.001}"
LR_BATCHES="${LR_BATCHES:-5000}"   # Eden decay: halves LR after ~5k batches (default 100k = too slow)
LR_EPOCHS="${LR_EPOCHS:-5}"        # Eden decay: halves LR after ~5 epochs (default 100 = too slow)

# 6. KD-specific config
KD_LOSS_SCALE="${KD_LOSS_SCALE:-0.05}"     # LM-only RNNT KD (AM branch disabled: teacher AM is degenerate)
                                            # raw LM-KD ~1-5 × scale 0.05 → ~0.05-0.25 added to RNNT loss
KD_TEMPERATURE="${KD_TEMPERATURE:-4.0}"    # temperature for softening distributions
DECODER_CE_SCALE="${DECODER_CE_SCALE:-0.0}" # auxiliary CE on decoder-end logits
JOINER_CE_SCALE="${JOINER_CE_SCALE:-0.0}"   # auxiliary CE on joiner-end diagonal logits
JOINER_KD_SCALE="${JOINER_KD_SCALE:-0.3}"  # KD KL on joiner diagonal — USE THIS for acoustic KD
                                            # teacher joiner: 80% blank, entropy=0.12, correct decode
                                            # student joiner: blank < THÌ at t=0 → wrong decode (needs fix)
                                            # raw joiner KL×T² ≈ 5-10; scale 0.3 → ~1.5-3.0 added to loss
KD_LOG_RAW_NONBLANK="${KD_LOG_RAW_NONBLANK:-1}" # log raw + non-blank KD diagnostics
KD_LOG_ON_VALID="${KD_LOG_ON_VALID:-1}"     # compute/log KD diagnostics on validation
KD_OBJECTIVE="${KD_OBJECTIVE:-rnnt}"       # ctc | rnnt | both
KD_TEACHER_TOKENS="${KD_TEACHER_TOKENS:-}" # optional explicit teacher tokens file
KD_FITNET_SCALE="${KD_FITNET_SCALE:-0.0}"   # disabled: Zipformer features have tiny magnitude (~0)
KD_MGD_SCALE="${KD_MGD_SCALE:-0.0}"        # disabled: same reason as FitNet
KD_MGD_MASK_RATIO="${KD_MGD_MASK_RATIO:-0.75}"
KD_PHASE_A_STEPS="${KD_PHASE_A_STEPS:-0}"   # 0 = no Phase A
KD_PHASE_B_STEPS="${KD_PHASE_B_STEPS:-0}"   # 0 = DKD full scale from step 0 (no ramp)

# 7. Multi-Speaker config (same as train_phase_2.sh)
NUM_SPEAKERS="${NUM_SPEAKERS:-3}"
CHUNK_SIZE_FRAMES="${CHUNK_SIZE_FRAMES:-32}"
CAUSAL="${CAUSAL:-1}"
USE_PYANNOTE_DIAR="${USE_PYANNOTE_DIAR:-0}"
PYANNOTE_WEIGHT_DIR="${PYANNOTE_WEIGHT_DIR:-weight}"
FREEZE_PYANNOTE="${FREEZE_PYANNOTE:-1}"
PYANNOTE_PHASE="${PYANNOTE_PHASE:-1}"
FREEZE_ENCODER="${FREEZE_ENCODER:-0}"
AUGMENT_TYPES="${AUGMENT_TYPES:-noise}"
SSA_SINGLE_TARGET="${SSA_SINGLE_TARGET:-1}"
SSA_TARGET_POLICY="${SSA_TARGET_POLICY:-round_robin}"
ZIPFORMER_CHUNK_SIZE="${ZIPFORMER_CHUNK_SIZE:-8,16,32}"
LEFT_CONTEXT_FRAMES="${LEFT_CONTEXT_FRAMES:-64,64,128}"
DECODE_CHUNK_LEN="${DECODE_CHUNK_LEN:-16}"
SHORT_CHUNK_SIZE="${SHORT_CHUNK_SIZE:-50}"

echo "Starting Phase 2-KD: Knowledge Distillation Training..."
echo "Experiment Dir : $EXP_DIR"
echo "Teacher JIT    : $KD_TEACHER_PATH"
echo "Teacher tokens : ${KD_TEACHER_TOKENS:-auto-detect}"
echo "Student init   : $PRETRAINED_CKPT"
echo "KD objective   : $KD_OBJECTIVE"
echo "KD scale/temp  : ${KD_LOSS_SCALE} / ${KD_TEMPERATURE}"
echo "Decoder CE scale: ${DECODER_CE_SCALE}"
echo "Joiner CE scale : ${JOINER_CE_SCALE}"
echo "Joiner KD scale : ${JOINER_KD_SCALE}"
echo "KD raw/nonblank log: ${KD_LOG_RAW_NONBLANK}"
echo "KD log on valid : ${KD_LOG_ON_VALID}"
echo "GPUs           : $WORLD_SIZE"

# PYTHONPATH: let zipformer_kd/ override zipformer/ for modified modules
# (model.py, multi_speaker_model.py, teacher.py, kd_loss.py), while unchanged
# modules (zipformer.py, speaker_modules.py, scaling.py, etc.) come from zipformer/.
export PYTHONPATH="$(pwd)/zipformer_kd:$(pwd)/zipformer:${PYTHONPATH:-}"

python zipformer_kd/finetune.py \
  --world-size "$WORLD_SIZE" \
  --master-port "$MASTER_PORT" \
  --num-epochs "$NUM_EPOCHS" \
  --start-epoch "${START_EPOCH:-1}" \
  --start-batch "${START_BATCH:-0}" \
  --use-fp16 "$USE_FP16" \
  --use-bf16 "$USE_BF16" \
  --exp-dir "$EXP_DIR" \
  --pretrained-dir "$PRETRAINED_CKPT" \
  --bpe-model "zipformer_kd/teacher_tokens.txt" \
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
  --lr-batches "$LR_BATCHES" \
  --lr-epochs "$LR_EPOCHS" \
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
  --num-buckets 50 \
  --kd-teacher-path "$KD_TEACHER_PATH" \
  --kd-teacher-tokens "$KD_TEACHER_TOKENS" \
  --kd-objective "$KD_OBJECTIVE" \
  --kd-loss-scale "$KD_LOSS_SCALE" \
  --kd-temperature "$KD_TEMPERATURE" \
  --decoder-ce-scale "$DECODER_CE_SCALE" \
  --joiner-ce-scale "$JOINER_CE_SCALE" \
  --joiner-kd-scale "$JOINER_KD_SCALE" \
  --kd-log-raw-nonblank "$KD_LOG_RAW_NONBLANK" \
  --kd-log-on-valid "$KD_LOG_ON_VALID" \
  --kd-fitnet-scale "$KD_FITNET_SCALE" \
  --kd-mgd-scale "$KD_MGD_SCALE" \
  --kd-mgd-mask-ratio "$KD_MGD_MASK_RATIO" \
  --kd-phase-a-steps "$KD_PHASE_A_STEPS" \
  --kd-phase-b-steps "$KD_PHASE_B_STEPS"
