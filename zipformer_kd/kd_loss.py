"""Knowledge Distillation loss functions.

Three distillation objectives:
  1. ctc_kd_loss  - DKD-inspired logit KL divergence (CTC frame distributions)
  2. fitnet_loss  - FitNet L2 feature matching (after linear projection)
  3. mgd_loss     - Masked Generative Distillation (random frame masking + L2)

Frame-rate alignment:
    Teacher outputs at ~25fps (mel 100fps -> encoder_embed 2x -> zipformer 2x).
    Student outputs at ~50fps (HuBERT 50fps -> Zipformer symmetric output).
    We upsample Teacher tensors to match Student T via F.interpolate.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _align_time(
    x: torch.Tensor,  # [B, T_src, D]
    target_T: int,
) -> torch.Tensor:
    """Linearly interpolate x along the time axis to target_T frames."""
    if x.shape[1] == target_T:
        return x
    x = x.permute(0, 2, 1)                                           # [B, D, T_src]
    x = F.interpolate(x, size=target_T, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)                                         # [B, target_T, D]


def align_teacher_to_student(
    teacher_logits: torch.Tensor,   # [B, T_t, V]
    student_T: int,
) -> torch.Tensor:
    """Upsample Teacher logits along time to match Student frame count."""
    return _align_time(teacher_logits, student_T)


def _drop_blank_class(
    logits: torch.Tensor,
    blank_id: Optional[int],
) -> torch.Tensor:
    """Drop blank class from logits for non-blank KD diagnostics."""
    if blank_id is None:
        return logits

    v = logits.size(-1)
    if blank_id < 0 or blank_id >= v:
        return logits

    return torch.cat([logits[..., :blank_id], logits[..., blank_id + 1 :]], dim=-1)


def _masked_kl(
    student_logits: torch.Tensor,  # [B, T, V]
    teacher_logits: torch.Tensor,  # [B, T, V]
    valid_lens: torch.Tensor,      # [B]
    temperature: float,
    ignore_blank_id: Optional[int] = None,
) -> torch.Tensor:
    """KL(student || teacher) over valid frames, averaged by valid frame count."""
    student_logits = _drop_blank_class(student_logits, ignore_blank_id)
    teacher_logits = _drop_blank_class(teacher_logits, ignore_blank_id)

    B, T, V = student_logits.shape

    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_prob.reshape(-1, V),
        teacher_prob.reshape(-1, V),
        reduction="none",
    ).sum(-1).reshape(B, T)

    mask = (
        torch.arange(T, device=valid_lens.device).unsqueeze(0)
        < valid_lens.unsqueeze(1)
    )
    kl = kl * mask.float()

    total = valid_lens.sum().clamp(min=1)
    return (kl.sum() / total) * (temperature ** 2)


def ctc_kd_loss(
    student_logits: torch.Tensor,    # [B, T_s, V]  raw (before log_softmax)
    teacher_logits: torch.Tensor,    # [B, T_t, V]  raw (before softmax)
    student_lens: torch.Tensor,      # [B]
    temperature: float = 4.0,
) -> torch.Tensor:
    """Frame-level KL divergence between Teacher and Student CTC distributions.

    KL(P_student || P_teacher) summed over valid frames, averaged over batch.

    Args:
        student_logits: raw CTC logits from student (before log_softmax)
        teacher_logits: raw CTC logits from teacher (before softmax)
        student_lens:   valid frame counts for each item in the batch
        temperature:    temperature T for softening distributions

    Returns:
        scalar KD loss
    """
    B, T_s, V = student_logits.shape

    # Align teacher to student frame count
    teacher_aligned = align_teacher_to_student(teacher_logits, T_s)  # [B, T_s, V]

    return _masked_kl(
        student_logits=student_logits,
        teacher_logits=teacher_aligned,
        valid_lens=student_lens,
        temperature=temperature,
    )


def rnnt_kd_loss(
    student_am_logits: torch.Tensor,  # [B, T_s, V]
    student_lm_logits: torch.Tensor,  # [B, U_s, V]
    teacher_am_logits: torch.Tensor,  # [B, T_t, V]
    teacher_lm_logits: torch.Tensor,  # [B, U_t, V]
    student_enc_lens: torch.Tensor,   # [B]
    target_lens: torch.Tensor,        # [B] (without SOS)
    temperature: float = 4.0,
    ignore_blank_id: Optional[int] = None,
    skip_am_branch: bool = False,
) -> torch.Tensor:
    """RNNT KD via simple AM/LM branch logit distillation.

    We distill both RNNT branches used by rnnt_loss_smoothed:
      - AM branch: encoder -> simple_am_proj
      - LM branch: decoder -> simple_lm_proj

    Teacher AM is time-aligned to student AM. LM uses SOS-padded target
    length U = target_len + 1 and is aligned only if U differs.

    WARNING: Some JIT teacher models have a degenerate simple_am_proj
    (e.g., bias[<unk>] >> all other biases, causing near-100% <unk>
    prediction regardless of input). In that case, set skip_am_branch=True
    to distill only the LM branch and avoid corrupting the student encoder.
    Use joiner_diag_kd_loss instead for proper acoustic-level distillation.
    """
    _, T_s, _ = student_am_logits.shape
    _, U_s, _ = student_lm_logits.shape

    lm_lens = (target_lens.to(dtype=torch.long) + 1).clamp(max=U_s)
    teacher_lm_aligned = _align_time(teacher_lm_logits, U_s)
    lm_loss = _masked_kl(
        student_logits=student_lm_logits,
        teacher_logits=teacher_lm_aligned,
        valid_lens=lm_lens,
        temperature=temperature,
        ignore_blank_id=ignore_blank_id,
    )

    if skip_am_branch:
        return lm_loss

    teacher_am_aligned = align_teacher_to_student(teacher_am_logits, T_s)
    am_loss = _masked_kl(
        student_logits=student_am_logits,
        teacher_logits=teacher_am_aligned,
        valid_lens=student_enc_lens,
        temperature=temperature,
        ignore_blank_id=ignore_blank_id,
    )

    return 0.5 * (am_loss + lm_loss)


def joiner_diag_kd_loss(
    student_joiner_logits: torch.Tensor,  # [B, T_s, V]
    teacher_joiner_logits: torch.Tensor,  # [B, T_t, V]
    student_enc_lens: torch.Tensor,       # [B]
    temperature: float = 4.0,
    ignore_blank_id: Optional[int] = None,
) -> torch.Tensor:
    """KD on joiner logits sampled along a monotonic diagonal path.

    Teacher joiner logits are time-aligned to student frame count before KL.
    """
    _, T_s, _ = student_joiner_logits.shape
    teacher_aligned = align_teacher_to_student(teacher_joiner_logits, T_s)

    return _masked_kl(
        student_logits=student_joiner_logits,
        teacher_logits=teacher_aligned,
        valid_lens=student_enc_lens,
        temperature=temperature,
        ignore_blank_id=ignore_blank_id,
    )


def fitnet_loss(
    student_feat: torch.Tensor,   # [B, T_s, D_student]  e.g. D_student=512
    teacher_feat: torch.Tensor,   # [B, T_t, D_teacher]  e.g. D_teacher=256
    proj_head: nn.Module,         # Linear(D_student -> D_teacher), on same device
    student_lens: torch.Tensor,   # [B] valid frame counts
) -> torch.Tensor:
    """FitNet-style L2 feature distillation.

    Projects student features to teacher dimensionality, aligns time axes,
    then computes MSE over valid frames.

    Args:
        student_feat: intermediate features from student Zipformer stack
        teacher_feat: intermediate features from teacher Zipformer stack (detached)
        proj_head:    learnable Linear projection (student_dim -> teacher_dim)
        student_lens: valid output frame counts per batch item

    Returns:
        scalar loss
    """
    B, T_s, _ = student_feat.shape

    projected = proj_head(student_feat)                       # [B, T_s, D_teacher]
    teacher_aligned = _align_time(teacher_feat.detach(), T_s)  # [B, T_s, D_teacher]

    diff = (projected - teacher_aligned) ** 2                 # [B, T_s, D_teacher]
    loss_per_frame = diff.mean(-1)                            # [B, T_s]

    mask = torch.arange(T_s, device=student_lens.device).unsqueeze(0) < student_lens.unsqueeze(1)
    loss_per_frame = loss_per_frame * mask.float()

    total_frames = student_lens.sum().clamp(min=1)
    return loss_per_frame.sum() / total_frames


def mgd_loss(
    student_feat: torch.Tensor,   # [B, T_s, D_student]  e.g. D_student=256
    teacher_feat: torch.Tensor,   # [B, T_t, D_teacher]  e.g. D_teacher=256
    proj_head: nn.Module,         # Linear(D_student -> D_teacher)
    student_lens: torch.Tensor,   # [B] valid frame counts
    mask_ratio: float = 0.75,
) -> torch.Tensor:
    """Masked Generative Distillation feature loss.

    Randomly selects mask_ratio fraction of valid frames and computes L2 only
    at those positions.  This encourages the student to encode context rather
    than relying on every individual frame's direct supervision.

    Args:
        student_feat: intermediate features from student Zipformer stack
        teacher_feat: intermediate features from teacher Zipformer stack (detached)
        proj_head:    learnable Linear projection (student_dim -> teacher_dim)
        student_lens: valid output frame counts per batch item
        mask_ratio:   fraction of frames selected for loss (default 0.75)

    Returns:
        scalar loss
    """
    B, T_s, _ = student_feat.shape

    projected = proj_head(student_feat)                        # [B, T_s, D_teacher]
    teacher_aligned = _align_time(teacher_feat.detach(), T_s)  # [B, T_s, D_teacher]

    # Valid-frame mask
    valid = torch.arange(T_s, device=student_lens.device).unsqueeze(0) < student_lens.unsqueeze(1)

    # Random selection mask — independently per batch item
    rand = torch.rand(B, T_s, device=student_feat.device)
    selected = rand < mask_ratio                               # [B, T_s]

    combined = valid & selected                                # [B, T_s]

    diff = (projected - teacher_aligned) ** 2                 # [B, T_s, D_teacher]
    loss_per_frame = diff.mean(-1) * combined.float()         # [B, T_s]

    total = combined.float().sum().clamp(min=1)
    return loss_per_frame.sum() / total
