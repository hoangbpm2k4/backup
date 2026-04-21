"""One-time weight transfer from Teacher JIT (30M) to Student checkpoint.

Copies all Teacher weights whose keys and shapes exactly match the Student
into the Student checkpoint, then saves a new checkpoint for KD warm-start.

Transfer map (teacher key → student key, condition):
  decoder.*            → asr_model.decoder.*          (all 2 params match)
  simple_lm_proj.*     → asr_model.simple_lm_proj.*   (all 2 params match)
  joiner.*             → asr_model.joiner.*            (5/6 match; encoder_proj.weight skipped: 256 vs 512)
  simple_am_proj.bias  → asr_model.simple_am_proj.bias (shape [2000] matches)
  encoder.encoder.*    → asr_model.encoder.encoder.*  (337/611 match; partial copy)

Skipped (shape mismatch):
  joiner.encoder_proj.weight      [512,256] vs [512,512]
  simple_am_proj.weight           [2000,256] vs [2000,512]
  encoder.encoder.* (274 params)  different Zipformer stack sizes/dims

Usage:
    python zipformer_kd/transfer_weights.py \\
        --teacher-jit  hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt \\
        --student-ckpt zipformer/exp_phase2_supervised/best-valid-loss.pt \\
        --out-ckpt     zipformer_kd/exp_kd/init_from_teacher.pt

    # Copy decoder only (keep encoder/joiner/projections from student)
    python zipformer_kd/transfer_weights.py \\
        --teacher-jit  hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt \\
        --student-ckpt zipformer/exp_phase2_supervised/best-valid-loss.pt \\
        --out-ckpt     zipformer_kd/exp_kd/init_decoder_only.pt \\
        --copy-groups decoder
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# Teacher prefix → Student prefix per copy group
PREFIX_MAP_BY_GROUP = {
    "decoder": ("decoder.", "asr_model.decoder."),
    "simple_lm_proj": ("simple_lm_proj.", "asr_model.simple_lm_proj."),
    "joiner": ("joiner.", "asr_model.joiner."),
    "simple_am_proj": ("simple_am_proj.", "asr_model.simple_am_proj."),
    "encoder": ("encoder.", "asr_model.encoder."),
}

DEFAULT_COPY_GROUPS = (
    "decoder",
    "simple_lm_proj",
    "joiner",
    "simple_am_proj",
    "encoder",
)


def transfer_weights(
    teacher_jit_path: str,
    student_ckpt_path: str,
    out_ckpt_path: str,
    copy_groups: list[str],
) -> None:
    logger.info(f"Loading Teacher JIT from {teacher_jit_path}")
    teacher = torch.jit.load(teacher_jit_path, map_location="cpu")
    teacher_state = dict(teacher.named_parameters())
    logger.info(f"Teacher has {len(teacher_state)} parameter tensors")

    logger.info(f"Loading Student checkpoint from {student_ckpt_path}")
    ckpt = torch.load(student_ckpt_path, map_location="cpu", weights_only=False)
    student_state: dict = ckpt["model"]
    logger.info(f"Student has {len(student_state)} parameter tensors")

    new_state = copy.deepcopy(student_state)

    if not copy_groups:
        raise ValueError("copy_groups is empty; choose at least one transfer group")

    unknown = [g for g in copy_groups if g not in PREFIX_MAP_BY_GROUP]
    if unknown:
        raise ValueError(
            f"Unknown copy group(s): {unknown}. "
            f"Available: {list(PREFIX_MAP_BY_GROUP.keys())}"
        )

    prefix_map = [PREFIX_MAP_BY_GROUP[g] for g in copy_groups]
    logger.info("Active transfer groups: %s", copy_groups)

    copied = 0
    skipped_shape = 0
    skipped_nomatch = 0

    for t_key, t_val in teacher_state.items():
        # Find student key from prefix map
        s_key = None
        for t_pfx, s_pfx in prefix_map:
            if t_key.startswith(t_pfx):
                s_key = s_pfx + t_key[len(t_pfx):]
                break

        if s_key is None:
            skipped_nomatch += 1
            continue

        if s_key not in new_state:
            logger.debug(f"  SKIP (not in student): {t_key} → {s_key}")
            skipped_nomatch += 1
            continue

        s_val = new_state[s_key]
        if t_val.shape != s_val.shape:
            logger.info(
                f"  SKIP shape mismatch: {t_key} {list(t_val.shape)} "
                f"→ {s_key} {list(s_val.shape)}"
            )
            skipped_shape += 1
            continue

        new_state[s_key] = t_val.clone()
        logger.debug(f"  COPY {t_key} → {s_key}  {list(t_val.shape)}")
        copied += 1

    logger.info(
        f"Transfer complete: {copied} copied, "
        f"{skipped_shape} skipped (shape), "
        f"{skipped_nomatch} skipped (no match)"
    )

    out_path = Path(out_ckpt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_ckpt = copy.deepcopy(ckpt)
    new_ckpt["model"] = new_state
    # Reset optimizer and scheduler so training starts fresh from this init
    for key in ("optimizer", "scheduler", "grad_scaler", "sampler"):
        new_ckpt.pop(key, None)
    # Keep model metadata but reset epoch/batch counters so finetune.py can
    # pick the right starting point via --start-batch or --start-epoch.
    new_ckpt.pop("epoch", None)
    new_ckpt.pop("batch_idx_train", None)

    torch.save(new_ckpt, str(out_path))
    logger.info(f"Saved initialised checkpoint to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-jit",
        required=True,
        help="Path to Teacher JIT .pt file",
    )
    parser.add_argument(
        "--student-ckpt",
        required=True,
        help="Path to Student checkpoint .pt file (warm-start base)",
    )
    parser.add_argument(
        "--out-ckpt",
        required=True,
        help="Output path for the initialised checkpoint",
    )
    parser.add_argument(
        "--copy-groups",
        type=str,
        default=",".join(DEFAULT_COPY_GROUPS),
        help=(
            "Comma-separated transfer groups. Available: "
            f"{','.join(PREFIX_MAP_BY_GROUP.keys())}. "
            "Example: --copy-groups decoder"
        ),
    )
    args = parser.parse_args()

    copy_groups = [x.strip() for x in args.copy_groups.split(",") if x.strip()]

    transfer_weights(
        teacher_jit_path=args.teacher_jit,
        student_ckpt_path=args.student_ckpt,
        out_ckpt_path=args.out_ckpt,
        copy_groups=copy_groups,
    )


if __name__ == "__main__":
    main()
