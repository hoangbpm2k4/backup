"""Tests for teacher_py.py — TeacherPy weight loading and forward pass.

Requires the JIT model to be present at:
    hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt

Run from repo root:
    PYTHONPATH=zipformer_kd:zipformer python zipformer_kd/tests/test_teacher_py.py \
        [--jit-path <path_to_jit.pt>]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "zipformer_kd"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "zipformer"))

import torch


def test_weight_loading(jit_path: str):
    from teacher_py import TeacherPy
    device = torch.device("cpu")
    teacher = TeacherPy(jit_path, device=device)
    teacher.eval()

    n_params = sum(p.numel() for p in teacher.zipformer.parameters())
    print(f"  Zipformer2 params: {n_params/1e6:.3f}M")
    assert 20e6 < n_params < 23e6, f"Unexpected param count: {n_params}"

    # All params should be frozen
    for name, p in teacher.zipformer.named_parameters():
        assert not p.requires_grad, f"Param {name} should be frozen"

    print("test_weight_loading PASSED")
    return teacher


def test_forward_shapes(jit_path: str):
    from teacher_py import TeacherPy
    device = torch.device("cpu")
    teacher = TeacherPy(jit_path, device=device)
    teacher.eval()

    B, T_samples = 2, 32000  # 2 sec at 16kHz
    audio = torch.randn(B, T_samples)
    padding_mask = torch.zeros(B, T_samples, dtype=torch.bool)
    padding_mask[1, 24000:] = True  # item 1 has 1.5 sec valid

    logits, stack3, stack5, out_lens = teacher.get_features_and_logits(audio, padding_mask)

    print(f"  logits:  {tuple(logits.shape)}")
    print(f"  stack3:  {tuple(stack3.shape)}")
    print(f"  stack5:  {tuple(stack5.shape)}")
    print(f"  out_lens: {out_lens.tolist()}")

    T_t = logits.shape[1]
    assert logits.shape[0] == B
    assert logits.shape[2] == 2000, f"Expected vocab=2000, got {logits.shape[2]}"

    # stack3 and stack5 should have the same T as logits output
    assert stack3.shape[0] == B
    assert stack3.shape[1] == T_t, f"stack3 T={stack3.shape[1]} != logits T={T_t}"
    assert stack3.shape[2] == 256, f"Expected stack3 dim=256, got {stack3.shape[2]}"

    assert stack5.shape[0] == B
    assert stack5.shape[1] == T_t, f"stack5 T={stack5.shape[1]} != logits T={T_t}"
    assert stack5.shape[2] == 256, f"Expected stack5 dim=256, got {stack5.shape[2]}"

    # Shorter input should give shorter output
    assert out_lens[0] > out_lens[1], f"Expected item0 longer, got {out_lens.tolist()}"

    # No gradients (frozen teacher)
    assert logits.grad_fn is None
    assert stack3.grad_fn is None

    print("test_forward_shapes PASSED")


def test_hooks_fire_independently(jit_path: str):
    """Each call should update hook outputs independently (no stale state)."""
    from teacher_py import TeacherPy
    device = torch.device("cpu")
    teacher = TeacherPy(jit_path, device=device)
    teacher.eval()

    B, T = 1, 16000
    audio1 = torch.randn(B, T)
    audio2 = torch.randn(B, T) * 10.0  # very different input
    pmask = torch.zeros(B, T, dtype=torch.bool)

    _, s3_1, s5_1, _ = teacher.get_features_and_logits(audio1, pmask)
    _, s3_2, s5_2, _ = teacher.get_features_and_logits(audio2, pmask)

    # Different inputs should give different features
    assert not torch.allclose(s3_1, s3_2, atol=1e-3), "stack3 unchanged between calls"
    assert not torch.allclose(s5_1, s5_2, atol=1e-3), "stack5 unchanged between calls"

    print("test_hooks_fire_independently PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jit-path",
        default="hf_zipformer30m_streaming_benchmark/models/jit_script_chunk_64_left_128.pt",
    )
    args = parser.parse_args()

    jit_path = args.jit_path
    if not Path(jit_path).exists():
        print(f"SKIP: JIT model not found at {jit_path}")
        sys.exit(0)

    print(f"Using JIT model: {jit_path}\n")
    test_weight_loading(jit_path)
    test_forward_shapes(jit_path)
    test_hooks_fire_independently(jit_path)
    print("\nAll teacher_py tests PASSED.")
