"""Python Teacher for intermediate-feature Knowledge Distillation.

Wraps the 30M JIT model's encoder_embed + simple_am_proj while using a
Python Zipformer2 instance (causal=True, chunk_size=[-1]) for full-context
encoding.  This gives access to per-stack intermediate features via
forward hooks, which cannot be obtained from the opaque JIT graph.

Weight loading:
    JIT state dict has keys like  encoder.encoder.encoders.*
    Python Zipformer2 expects      encoders.*
    → strip prefix "encoder.encoder." before loading.

Architecture (verified in arch_inspection.txt, 0 missing / 0 unexpected):
    Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=(1, 2, 4, 8, 4, 2),
        encoder_dim=(192, 256, 256, 256, 256, 256),
        num_encoder_layers=(2, 2, 2, 2, 2, 2),
        encoder_unmasked_dim=(192, 256, 256, 256, 256, 256),
        num_heads=(4, 4, 4, 8, 4, 4),
        query_head_dim=32, pos_head_dim=4, value_head_dim=12,
        feedforward_dim=(512, 768, 768, 768, 768, 768),
        cnn_module_kernel=(31, 31, 15, 15, 15, 31),
        pos_dim=48, causal=True, chunk_size=[-1], left_context_frames=[-1],
    )

Distillation hooks:
    encoders[3]  → stack3_feat  [B, T, 256]   (bottleneck, 8-head attention)
    encoders[5]  → stack5_feat  [B, T, 256]   (near-output, ASR-relevant)
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio

logger = logging.getLogger(__name__)

# JIT Zipformer2 state-dict prefix for the encoder stacks
_ENCODER_PREFIX = "encoder.encoder."


def _build_teacher_zipformer2() -> nn.Module:
    """Instantiate a Python Zipformer2 with the confirmed teacher config."""
    from zipformer import Zipformer2
    return Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=(1, 2, 4, 8, 4, 2),
        encoder_dim=(192, 256, 256, 256, 256, 256),
        num_encoder_layers=(2, 2, 2, 2, 2, 2),
        encoder_unmasked_dim=(192, 256, 256, 256, 256, 256),
        num_heads=(4, 4, 4, 8, 4, 4),
        query_head_dim=32,
        pos_head_dim=4,
        value_head_dim=12,
        feedforward_dim=(512, 768, 768, 768, 768, 768),
        cnn_module_kernel=(31, 31, 15, 15, 15, 31),
        pos_dim=48,
        causal=True,
        chunk_size=[-1],
        left_context_frames=[-1],
    )


class TeacherPy(nn.Module):
    """Python Teacher exposing intermediate Zipformer2 stack features.

    Args:
        jit_path:  Path to the 30M JIT .pt file.
        device:    Target device.
    """

    def __init__(self, jit_path: str, device: torch.device):
        super().__init__()

        logger.info(f"Loading Teacher JIT for weight extraction from {jit_path}")
        jit = torch.jit.load(jit_path, map_location=device)
        jit.eval()
        for p in jit.parameters():
            p.requires_grad_(False)
        # Keep JIT for encoder_embed and simple_am_proj calls
        object.__setattr__(self, "_jit", jit)
        object.__setattr__(self, "_device", device)

        # ---- Build Python Zipformer2 & load weights ----
        zipformer = _build_teacher_zipformer2()
        self._load_zipformer_weights(jit, zipformer)
        zipformer = zipformer.to(device)   # match device of JIT / input tensors
        zipformer.eval()
        for p in zipformer.parameters():
            p.requires_grad_(False)
        self.zipformer = zipformer

        # ---- Intermediate-feature storage (filled by hooks) ----
        self._stack3_feat: Optional[torch.Tensor] = None
        self._stack5_feat: Optional[torch.Tensor] = None

        # Register hooks on encoders[3] and encoders[5].
        # DownsampledZipformer2Encoder.forward() returns a plain Tensor
        # of shape (T_full, B, encoder_dim[i]).
        def _hook3(module, inp, output):
            # output: [T_full, B, 256]  →  store as [B, T_full, 256]
            self._stack3_feat = output.detach().permute(1, 0, 2)

        def _hook5(module, inp, output):
            self._stack5_feat = output.detach().permute(1, 0, 2)

        self.zipformer.encoders[3].register_forward_hook(_hook3)
        self.zipformer.encoders[5].register_forward_hook(_hook5)

        logger.info("TeacherPy ready — hooks on encoders[3] and encoders[5].")

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_zipformer_weights(jit, zipformer: nn.Module) -> None:
        """Copy encoder.encoder.* weights from JIT into Python Zipformer2."""
        jit_sd = {}
        for k, v in jit.named_parameters():
            jit_sd[k] = v
        for k, v in jit.named_buffers():
            jit_sd[k] = v

        prefix_len = len(_ENCODER_PREFIX)
        zip_sd = {
            k[prefix_len:]: v
            for k, v in jit_sd.items()
            if k.startswith(_ENCODER_PREFIX)
        }

        missing, unexpected = zipformer.load_state_dict(zip_sd, strict=True)
        if missing or unexpected:
            logger.warning(
                f"  Zipformer2 load: {len(missing)} missing, {len(unexpected)} unexpected"
            )
            for k in missing[:5]:
                logger.warning(f"    missing: {k}")
            for k in unexpected[:5]:
                logger.warning(f"    unexpected: {k}")
        else:
            n = sum(p.numel() for p in zipformer.parameters())
            logger.info(
                f"  Zipformer2 weights loaded OK — {n/1e6:.3f}M params, 0 missing, 0 unexpected"
            )

    # ------------------------------------------------------------------
    # Mel extraction (identical to TeacherJIT)
    # ------------------------------------------------------------------
    @staticmethod
    def _audio_to_mel(
        audio: torch.Tensor,       # [B, T_samples]
        audio_lens: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_list = []
        mel_lens = []
        for wav, length in zip(audio, audio_lens):
            wav_cpu = wav[: int(length)].cpu().unsqueeze(0)
            feat = torchaudio.compliance.kaldi.fbank(
                wav_cpu,
                num_mel_bins=80,
                frame_length=25.0,
                frame_shift=10.0,
                sample_frequency=16000,
                dither=0.0,
                energy_floor=1e-10,
            )
            mel_list.append(feat)
            mel_lens.append(feat.shape[0])

        max_t = max(mel_lens)
        B = len(mel_list)
        mel_padded = torch.zeros(B, max_t, 80, dtype=mel_list[0].dtype)
        for i, feat in enumerate(mel_list):
            mel_padded[i, : feat.shape[0]] = feat

        return mel_padded, torch.tensor(mel_lens, dtype=torch.long)

    # ------------------------------------------------------------------
    # Main inference API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_features_and_logits(
        self,
        audio: torch.Tensor,         # [B, T_samples]
        padding_mask: torch.Tensor,  # [B, T_samples]  True = padded
    ) -> Tuple[
        torch.Tensor,                # logits    [B, T_t, 2000]  raw (before log_softmax)
        torch.Tensor,                # stack3    [B, T_t, 256]
        torch.Tensor,                # stack5    [B, T_t, 256]
        torch.Tensor,                # out_lens  [B]
    ]:
        """Run full-context Teacher forward pass.

        Returns:
            logits:    raw CTC logits  [B, T_t, vocab_size]
            stack3:    encoders[3] output features  [B, T_t, 256]
            stack5:    encoders[5] output features  [B, T_t, 256]
            out_lens:  valid frame counts  [B]
        """
        jit = object.__getattribute__(self, "_jit")
        device = object.__getattribute__(self, "_device")

        audio_lens = (~padding_mask).long().sum(dim=1)

        # 1. Mel features (CPU computation, then move to device)
        mel, mel_lens = self._audio_to_mel(audio, audio_lens)
        mel = mel.to(device)
        mel_lens = mel_lens.to(device)

        # 2. Subsampling embed: [B, T_mel, 80] → [B, T', 192], lens [B]
        x, x_lens = jit.encoder_embed(mel, mel_lens)

        # 3. Python Zipformer2 in TBD layout — hooks fire during this call
        x_tbd = x.permute(1, 0, 2).contiguous()  # [T', B, 192]
        out_tbd, out_lens = self.zipformer(x_tbd, x_lens)

        # 4. CTC projection via JIT's simple_am_proj
        out_btd = out_tbd.permute(1, 0, 2).contiguous()  # [B, T'', 256]
        logits = jit.simple_am_proj(out_btd)               # [B, T'', 2000]

        assert self._stack3_feat is not None, "Hook on encoders[3] did not fire"
        assert self._stack5_feat is not None, "Hook on encoders[5] did not fire"

        return logits, self._stack3_feat, self._stack5_feat, out_lens
