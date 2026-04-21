"""
Copy teacher vocab-dependent weights into a student checkpoint.

Because teacher and student now share the same vocab (teacher_tokens.txt),
token-id semantics are identical → we can copy weight rows directly.

Layers copied (shape-matched, same vocab):
  decoder.embedding.weight      (2000, 512) → asr_model.decoder.embedding.weight
  joiner.output_linear.weight   (2000, 512) → asr_model.joiner.output_linear.weight
  joiner.output_linear.bias     (2000,)     → asr_model.joiner.output_linear.bias
  simple_lm_proj.weight         (2000, 512) → asr_model.simple_lm_proj.weight
  simple_lm_proj.bias           (2000,)     → asr_model.simple_lm_proj.bias
  joiner.decoder_proj.weight    (512, 512)  → asr_model.joiner.decoder_proj.weight  (if shapes match)
  joiner.decoder_proj.bias      (512,)      → asr_model.joiner.decoder_proj.bias

NOT copied (shape mismatch or degenerate):
  simple_am_proj  — teacher (2000,256) vs student (2000,512); also degenerate bias[<unk>]=80
  joiner.encoder_proj — teacher encoder dim=256 vs student dim=512
  ctc_output — no equivalent in teacher

Usage:
    python zipformer_kd/init_from_teacher.py \
        --teacher-jit hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt \
        --student-ckpt zipformer/exp_phase2_supervised/best-valid-loss.pt \
        --out zipformer_kd/student_teacher_init.pt
"""
import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")

# teacher key → student key (must have same shape)
COPY_MAP = {
    "decoder.embedding.weight":     "asr_model.decoder.embedding.weight",
    "joiner.output_linear.weight":  "asr_model.joiner.output_linear.weight",
    "joiner.output_linear.bias":    "asr_model.joiner.output_linear.bias",
    "simple_lm_proj.weight":        "asr_model.simple_lm_proj.weight",
    "simple_lm_proj.bias":          "asr_model.simple_lm_proj.bias",
    "joiner.decoder_proj.weight":   "asr_model.joiner.decoder_proj.weight",
    "joiner.decoder_proj.bias":     "asr_model.joiner.decoder_proj.bias",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-jit", required=True)
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.info(f"Loading teacher JIT: {args.teacher_jit}")
    teacher = torch.jit.load(args.teacher_jit, map_location="cpu")
    teacher_state = dict(teacher.named_parameters())

    logging.info(f"Loading student checkpoint: {args.student_ckpt}")
    ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    student_state = ckpt["model"]

    copied, skipped = 0, 0
    for t_key, s_key in COPY_MAP.items():
        if t_key not in teacher_state:
            logging.warning(f"  SKIP (not in teacher): {t_key}")
            skipped += 1
            continue
        if s_key not in student_state:
            logging.warning(f"  SKIP (not in student): {s_key}")
            skipped += 1
            continue

        t_val = teacher_state[t_key].data
        s_val = student_state[s_key]

        if t_val.shape != s_val.shape:
            logging.warning(
                f"  SKIP (shape mismatch): {t_key} {tuple(t_val.shape)} "
                f"vs {s_key} {tuple(s_val.shape)}"
            )
            skipped += 1
            continue

        student_state[s_key] = t_val.clone()
        logging.info(f"  Copied: {t_key} {tuple(t_val.shape)} → {s_key}")
        copied += 1

    logging.info(f"\nCopied {copied} layers, skipped {skipped}")

    # Reinitialize vocab-dependent heads that have wrong vocab ordering from Phase 2.
    # simple_am_proj: teacher's is (2000,256) vs student (2000,512) — shape mismatch,
    #   can't copy. Phase 2 weights have old vocab ordering → wrong semantics.
    #   Reinit to small random so training starts from neutral, not wrong.
    # ctc_output.1: no teacher equivalent. Phase 2 weights have old vocab ordering.
    #   Reinit to small random.
    REINIT_KEYS = [
        "asr_model.simple_am_proj.weight",
        "asr_model.simple_am_proj.bias",
        "asr_model.ctc_output.1.weight",
        "asr_model.ctc_output.1.bias",
    ]
    reinit_count = 0
    for key in REINIT_KEYS:
        if key in student_state:
            t = student_state[key]
            student_state[key] = torch.nn.init.trunc_normal_(
                torch.empty_like(t), std=0.02
            )
            logging.info(f"  Reinit (old vocab ordering): {key} {tuple(t.shape)}")
            reinit_count += 1
    logging.info(f"Reinitialized {reinit_count} keys with old vocab ordering")

    ckpt["model"] = student_state
    torch.save(ckpt, args.out)
    logging.info(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
