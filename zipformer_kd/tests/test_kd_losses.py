"""Tests for kd_loss.py — ctc_kd_loss, fitnet_loss, mgd_loss.

Run from repo root:
    PYTHONPATH=zipformer_kd:zipformer python zipformer_kd/tests/test_kd_losses.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # zipformer_kd/

import torch
import torch.nn as nn

from kd_loss import align_teacher_to_student, ctc_kd_loss, fitnet_loss, mgd_loss


def test_align_time():
    x = torch.randn(2, 10, 256)
    out = align_teacher_to_student(x, 20)
    assert out.shape == (2, 20, 256), out.shape
    # No-op when sizes match
    out2 = align_teacher_to_student(x, 10)
    assert out2 is x
    print("test_align_time PASSED")


def test_ctc_kd_loss():
    B, T_s, V = 2, 50, 2000
    T_t = 25  # teacher at half student rate
    student_logits = torch.randn(B, T_s, V)
    teacher_logits = torch.randn(B, T_t, V)
    lens = torch.tensor([50, 40])

    loss = ctc_kd_loss(student_logits, teacher_logits, lens, temperature=4.0)
    assert loss.ndim == 0, "expected scalar"
    assert loss.item() > 0, "loss should be positive"
    assert not torch.isnan(loss), "NaN loss"

    # Gradient check
    student_logits.requires_grad_(True)
    loss = ctc_kd_loss(student_logits, teacher_logits, lens)
    loss.backward()
    assert student_logits.grad is not None
    print("test_ctc_kd_loss PASSED")


def test_fitnet_loss_shapes():
    B, T_s, D_s, D_t = 2, 50, 512, 256
    T_t = 25
    proj = nn.Linear(D_s, D_t)

    s_feat = torch.randn(B, T_s, D_s, requires_grad=True)
    t_feat = torch.randn(B, T_t, D_t)
    lens = torch.tensor([50, 40])

    loss = fitnet_loss(s_feat, t_feat, proj, lens)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()
    assert s_feat.grad is not None, "no grad through student features"
    assert proj.weight.grad is not None, "no grad through proj head"
    print(f"test_fitnet_loss_shapes PASSED  loss={loss.item():.4f}")


def test_fitnet_loss_same_time():
    """Ensure no interpolation when teacher T == student T."""
    B, T, D = 2, 50, 256
    proj = nn.Linear(D, D)
    s = torch.randn(B, T, D, requires_grad=True)
    t = torch.randn(B, T, D)
    lens = torch.full((B,), T, dtype=torch.long)
    loss = fitnet_loss(s, t, proj, lens)
    loss.backward()
    print("test_fitnet_loss_same_time PASSED")


def test_mgd_loss_shapes():
    B, T_s, D_s, D_t = 2, 50, 256, 256
    T_t = 25
    proj = nn.Linear(D_s, D_t)

    s_feat = torch.randn(B, T_s, D_s, requires_grad=True)
    t_feat = torch.randn(B, T_t, D_t)
    lens = torch.tensor([50, 40])

    loss = mgd_loss(s_feat, t_feat, proj, lens, mask_ratio=0.75)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()
    assert s_feat.grad is not None
    print(f"test_mgd_loss_shapes PASSED  loss={loss.item():.4f}")


def test_mgd_mask_ratio_zero():
    """mask_ratio=0 → no frames selected → loss should be 0 (division by clamp(1))."""
    B, T, D = 2, 30, 128
    proj = nn.Linear(D, D)
    s = torch.randn(B, T, D, requires_grad=True)
    t = torch.randn(B, T, D)
    lens = torch.full((B,), T, dtype=torch.long)
    # With mask_ratio=0, rand < 0 is never True → combined mask = all False
    # → loss_per_frame.sum() = 0, total = clamp(0,1) = 1 → loss = 0
    loss = mgd_loss(s, t, proj, lens, mask_ratio=0.0)
    assert loss.item() == 0.0, f"expected 0 loss with mask_ratio=0, got {loss.item()}"
    print("test_mgd_mask_ratio_zero PASSED")


def test_padding_masking():
    """Padded frames should not contribute to loss."""
    B, T, V = 2, 20, 100
    logits = torch.randn(B, T, V)
    # Only first 10 frames valid for item 0, 15 for item 1
    lens = torch.tensor([10, 15])

    # Set padded region to extreme values — should not affect loss
    logits_extreme = logits.clone()
    logits_extreme[0, 10:] = 1e6

    l1 = ctc_kd_loss(logits, logits, lens, temperature=4.0)
    l2 = ctc_kd_loss(logits_extreme, logits, lens, temperature=4.0)
    assert abs(l1.item() - l2.item()) < 1e-3, (
        f"Padded frames affected loss: {l1.item()} vs {l2.item()}"
    )
    print("test_padding_masking PASSED")


if __name__ == "__main__":
    test_align_time()
    test_ctc_kd_loss()
    test_fitnet_loss_shapes()
    test_fitnet_loss_same_time()
    test_mgd_loss_shapes()
    test_mgd_mask_ratio_zero()
    test_padding_masking()
    print("\nAll kd_loss tests PASSED.")
