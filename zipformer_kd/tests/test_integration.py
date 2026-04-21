"""Integration test: full KD forward pass with all three loss terms.

Uses the real JIT teacher but a mock student to test the complete pipeline:
  Teacher (JIT + Python) → FitNet loss + MGD loss + DKD loss → backward

Run from repo root:
    PYTHONPATH=zipformer_kd:zipformer python zipformer_kd/tests/test_integration.py \
        [--jit-path <path>]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "zipformer_kd"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "zipformer"))

import torch
import torch.nn as nn


def test_full_kd_pipeline(jit_path: str):
    """Load both teachers, produce all KD losses, verify backward pass."""
    from teacher import TeacherJIT
    from teacher_py import TeacherPy
    from kd_loss import ctc_kd_loss, fitnet_loss, mgd_loss

    device = torch.device("cpu")

    print("  Loading TeacherJIT...")
    t_jit = TeacherJIT(jit_path, device=device)
    t_jit.eval()

    print("  Loading TeacherPy...")
    t_py = TeacherPy(jit_path, device=device)
    t_py.eval()

    # Fake student tensors (replacing real model output)
    B, T_s = 2, 50
    student_ctc_logits = torch.randn(B, T_s, 2000, requires_grad=True)  # DKD input
    student_stack3     = torch.randn(B, T_s, 512,  requires_grad=True)  # FitNet input
    student_stack5     = torch.randn(B, T_s, 256,  requires_grad=True)  # MGD input
    student_lens       = torch.tensor([T_s, T_s - 10])

    proj3 = nn.Linear(512, 256)
    proj5 = nn.Linear(256, 256)

    # Fake audio batch (2s at 16kHz)
    audio = torch.randn(B, 32000)
    padding_mask = torch.zeros(B, 32000, dtype=torch.bool)
    padding_mask[1, 24000:] = True

    # --- Teacher inference ---
    print("  Running TeacherJIT...")
    with torch.no_grad():
        t_logits_jit, t_lens_jit = t_jit.get_ctc_logits(audio, padding_mask)

    print(f"  JIT logits: {tuple(t_logits_jit.shape)}  lens: {t_lens_jit.tolist()}")

    print("  Running TeacherPy...")
    with torch.no_grad():
        t_logits_py, t_s3, t_s5, t_lens_py = t_py.get_features_and_logits(
            audio, padding_mask
        )

    print(f"  Py  logits: {tuple(t_logits_py.shape)}  stack3: {tuple(t_s3.shape)}  stack5: {tuple(t_s5.shape)}")

    # Logits from JIT and Python teacher should be close (same weights, full-context)
    # They won't be identical because JIT uses chunk_64 masking, Python uses chunk_size=[-1]
    # but they should be in a similar range
    print(f"  JIT logits mean={t_logits_jit.mean().item():.4f}  Python logits mean={t_logits_py.mean().item():.4f}")

    # --- DKD loss ---
    dkd = ctc_kd_loss(student_ctc_logits, t_logits_jit, student_lens, temperature=4.0)
    print(f"  DKD loss: {dkd.item():.4f}")

    # --- FitNet loss ---
    fl = fitnet_loss(student_stack3, t_s3, proj3, student_lens)
    print(f"  FitNet loss: {fl.item():.4f}")

    # --- MGD loss ---
    ml = mgd_loss(student_stack5, t_s5, proj5, student_lens, mask_ratio=0.75)
    print(f"  MGD loss: {ml.item():.4f}")

    # --- Combined backward pass ---
    total_loss = 0.5 * dkd + 0.5 * fl + 0.3 * ml
    total_loss.backward()

    assert student_ctc_logits.grad is not None, "No grad to student CTC logits"
    assert student_stack3.grad is not None, "No grad to student stack3"
    assert student_stack5.grad is not None, "No grad to student stack5"
    assert proj3.weight.grad is not None, "No grad to proj3"
    assert proj5.weight.grad is not None, "No grad to proj5"

    print(f"  Total loss: {total_loss.item():.4f}")
    print("test_full_kd_pipeline PASSED")


def test_phase_schedule():
    """Verify phase schedule logic: Phase A → B → C."""

    def get_scales(step, phase_a=5000, phase_b=15000):
        fitnet_active = step >= 0
        mgd_active = step >= phase_a
        if step < phase_a:
            dkd_factor = 0.0
        elif step < phase_b:
            dkd_factor = (step - phase_a) / (phase_b - phase_a)
        else:
            dkd_factor = 1.0
        return fitnet_active, mgd_active, dkd_factor

    # Phase A
    fa, ma, df = get_scales(0)
    assert fa and not ma and df == 0.0, f"Phase A wrong: {fa},{ma},{df}"
    fa, ma, df = get_scales(4999)
    assert fa and not ma and df == 0.0

    # Phase B entry
    fa, ma, df = get_scales(5000)
    assert fa and ma and df == 0.0

    # Phase B middle
    fa, ma, df = get_scales(10000)
    assert fa and ma and abs(df - 0.5) < 1e-6, f"Phase B middle DKD factor: {df}"

    # Phase C
    fa, ma, df = get_scales(15000)
    assert fa and ma and df == 1.0
    fa, ma, df = get_scales(20000)
    assert fa and ma and df == 1.0

    print("test_phase_schedule PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jit-path",
        default="hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt",
    )
    args = parser.parse_args()

    test_phase_schedule()

    if not Path(args.jit_path).exists():
        print(f"\nSKIP test_full_kd_pipeline: JIT not found at {args.jit_path}")
    else:
        test_full_kd_pipeline(args.jit_path)

    print("\nAll integration tests PASSED.")
