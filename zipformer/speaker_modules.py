"""
Speaker modules for multi-speaker RNNT.

Routing-only build:
  - PyAnnoteSegmentationWrapper
  - SpeakerKernel
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PyAnnoteSegmentationWrapper(nn.Module):
    """Pretrained pyannote segmentation wrapper for diarization."""

    def __init__(
        self,
        weight_dir: str = "weight",
        num_speakers: int = 3,
        freeze: bool = True,
        chunk_duration_s: float = 10.0,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.S = num_speakers
        self.sample_rate = sample_rate
        self.chunk_duration_s = chunk_duration_s
        self.chunk_samples = int(chunk_duration_s * sample_rate)

        from diarizers import SegmentationModel

        seg_model = SegmentationModel.from_pretrained(weight_dir)
        self.pyannet = seg_model.model
        self.powerset = self.pyannet.powerset

        if freeze:
            for p in self.pyannet.parameters():
                p.requires_grad = False
            self.pyannet.eval()

        with torch.no_grad():
            try:
                p = next(self.pyannet.parameters())
                device = p.device
            except StopIteration:
                device = torch.device("cpu")
            dummy = torch.randn(1, 1, self.chunk_samples, device=device)
            dummy_out = self.pyannet(dummy)
            self._frames_per_chunk = max(int(dummy_out.shape[1]), 1)
        self._frame_duration = chunk_duration_s / self._frames_per_chunk

    def train(self, mode: bool = True):
        """Keep pyannote in eval mode when frozen."""
        super().train(mode)
        if not any(p.requires_grad for p in self.pyannet.parameters()):
            self.pyannet.eval()
        return self

    def _run_pyannet(self, raw_audio: Tensor) -> Tensor:
        """Run pyannote on raw audio and return multilabel activity."""
        x = raw_audio.unsqueeze(1) if raw_audio.dim() == 2 else raw_audio
        logits = self.pyannet(x)
        activity = self.powerset.to_multilabel(logits.exp())
        return activity[:, :, : self.S]

    @staticmethod
    def _resample_to(activity: Tensor, target_len: int) -> Tensor:
        """Resample [B, T_in, S] -> [B, target_len, S] via nearest interp."""
        if activity.shape[1] == target_len:
            return activity
        return F.interpolate(
            activity.transpose(1, 2),
            size=target_len,
            mode="nearest",
        ).transpose(1, 2)

    def resample_activity(self, activity: Tensor, target_len: int) -> Tensor:
        """Public resampling helper used by streaming wrappers."""
        return self._resample_to(activity, target_len)

    def _sliding_window(self, raw_audio: Tensor) -> Tensor:
        """Sliding-window inference for long audio."""
        B = raw_audio.shape[0]
        T_samples = raw_audio.shape[1]
        step_samples = self.chunk_samples

        total_pya_frames = int(T_samples / self.sample_rate / self._frame_duration)
        accum = raw_audio.new_zeros(B, total_pya_frames, self.S)
        count = raw_audio.new_zeros(total_pya_frames)

        start = 0
        while start + self.chunk_samples <= T_samples:
            chunk = raw_audio[:, start : start + self.chunk_samples]
            act = self._run_pyannet(chunk)

            frame_start = int(start / self.sample_rate / self._frame_duration)
            frame_end = min(frame_start + act.shape[1], total_pya_frames)
            n = frame_end - frame_start

            accum[:, frame_start:frame_end, :] += act[:, :n, :]
            count[frame_start:frame_end] += 1
            start += step_samples

        remainder = T_samples - start
        if remainder > 0:
            chunk = F.pad(raw_audio[:, start:], (0, self.chunk_samples - remainder))
            act = self._run_pyannet(chunk)
            valid = int(remainder / self.sample_rate / self._frame_duration)
            valid = max(1, min(valid, act.shape[1]))

            frame_start = int(start / self.sample_rate / self._frame_duration)
            frame_end = min(frame_start + valid, total_pya_frames)
            n = frame_end - frame_start
            accum[:, frame_start:frame_end, :] += act[:, :n, :]
            count[frame_start:frame_end] += 1

        count = count.clamp(min=1)
        return accum / count.unsqueeze(0).unsqueeze(-1)

    def forward_native(self, raw_audio: Tensor) -> Tensor:
        """Predict activity in pyannote's native frame rate (no resampling)."""
        T_samples = raw_audio.shape[-1] if raw_audio.dim() >= 2 else raw_audio.shape[0]

        if T_samples <= self.chunk_samples:
            pad_len = self.chunk_samples - T_samples
            audio_padded = F.pad(
                raw_audio if raw_audio.dim() == 2 else raw_audio.unsqueeze(0),
                (0, pad_len),
            )
            activity = self._run_pyannet(audio_padded)
            valid_frames = int(T_samples / self.chunk_samples * activity.shape[1])
            valid_frames = max(1, valid_frames)
            activity = activity[:, :valid_frames, :]
        else:
                activity = self._sliding_window(raw_audio)

        return activity

    def forward(self, raw_audio: Tensor, target_len: int) -> Tensor:
        """Predict per-speaker activity from raw audio."""
        activity = self.forward_native(raw_audio)
        return self._resample_to(activity, target_len)

    def forward_logits(self, raw_audio: Tensor, target_len: int) -> Tensor:
        """Return pseudo-logits for API compatibility."""
        activity = self.forward(raw_audio, target_len)
        return activity * 12.0 - 6.0

    def get_init_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, Tensor]:
        """No internal state, keep API compatibility."""
        return {"_dummy": torch.zeros(1, device=device)}

    def streaming_forward_logits(
        self,
        raw_audio_chunk: Tensor,
        state: Optional[Dict[str, Tensor]],
        target_len: int,
        e_mix: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        del e_mix
        if state is None:
            state = self.get_init_state(raw_audio_chunk.shape[0], raw_audio_chunk.device)
        logits = self.forward_logits(raw_audio_chunk, target_len)
        return logits, state

    def streaming_forward(
        self,
        raw_audio_chunk: Tensor,
        state: Optional[Dict[str, Tensor]],
        target_len: int,
        e_mix: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        del e_mix
        if state is None:
            state = self.get_init_state(raw_audio_chunk.shape[0], raw_audio_chunk.device)
        activity = self.forward(raw_audio_chunk, target_len)
        return activity, state


class SpeakerKernel(nn.Module):
    """NVIDIA SSA Eq.1+2 kernel: X + ff(X*spk_mask) + ff(X*bg_mask)."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.5,
        add_bg_kernel: bool = True,
        bg_threshold: float = 0.5,
    ):
        super().__init__()
        self.add_bg_kernel = add_bg_kernel
        self.bg_threshold = bg_threshold

        self.spk_kernel = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        if add_bg_kernel:
            self.bg_kernel = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )

    @staticmethod
    def solve_length_mismatch(
        x: Tensor,
        mask: Optional[Tensor],
        default_value: float = 1.0,
    ) -> Tensor:
        """Pad/truncate mask to match x time length (NeMo-style)."""
        if mask is None:
            return torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype) * default_value

        if mask.dim() == x.dim():
            t_mask = mask.shape[-2]
        else:
            t_mask = mask.shape[-1]
        t_x = x.shape[-2]

        if t_mask < t_x:
            pad_len = t_x - t_mask
            if mask.dim() == x.dim():
                mask = F.pad(mask, (0, 0, pad_len, 0), value=default_value)
            else:
                mask = F.pad(mask, (pad_len, 0), value=default_value)
        elif t_mask > t_x:
            if mask.dim() == x.dim():
                mask = mask[..., -t_x:, :]
            else:
                mask = mask[..., -t_x:]
        return mask

    @staticmethod
    def _apply_mask(
        x: Tensor,
        mask: Optional[Tensor],
        default_value: float = 1.0,
    ) -> Tensor:
        mask = SpeakerKernel.solve_length_mismatch(x, mask, default_value)
        if mask.dim() == x.dim() - 1:
            mask = mask.unsqueeze(-1)
        return x * mask

    def build_bg_mask(self, activity: Tensor, target_slot: int) -> Tensor:
        """Build background mask from all non-target speaker activity."""
        S = activity.shape[-1]
        other_ids = [j for j in range(S) if j != target_slot]
        if not other_ids:
            return torch.zeros_like(activity[..., 0])
        inactive = activity[..., other_ids]
        return (inactive > self.bg_threshold).sum(-1).gt(0).float()

    def forward(
        self,
        x: Tensor,
        spk_mask: Tensor,
        bg_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply speaker/background residual kernels."""
        x_spk = self.spk_kernel(self._apply_mask(x, spk_mask, default_value=1.0))
        x = x + x_spk

        if self.add_bg_kernel and bg_mask is not None:
            x_bg = self.bg_kernel(self._apply_mask(x, bg_mask, default_value=0.0))
            x = x + x_bg
        return x
