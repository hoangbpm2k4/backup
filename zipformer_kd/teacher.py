"""Teacher JIT wrapper for CTC/RNNT logit distillation.

Wraps the 30M Zipformer JIT model and exposes methods:
        get_ctc_logits(audio, audio_lens) -> Tensor[B, T, vocab_size]
        get_rnnt_simple_logits(audio, padding_mask, decoder_input_ids)
            -> (am_logits[B, T, vocab_size], lm_logits[B, U, vocab_size])

The pipeline inside the JIT model:
    audio [B, T_samples]
    -> kaldi FBANK mel [B, T_mel, 80]  (100fps, 10ms/frame)
    -> encoder_embed (Conv2dSubsampling, 2x) -> [B, T_mel/2, 192]
    -> zipformer (internal 2x) -> [T_mel/4, B, 256]   (~25fps)
    -> simple_am_proj -> [B, T_mel/4, 2000]

Student outputs at ~50fps (HuBERT 320x from 16kHz, Zipformer symmetric).
Frame-rate ratio Teacher:Student = 1:2 -> interpolation handled in kd_loss.py.

Performance notes:
  - _audio_to_mel uses ThreadPoolExecutor to parallelize per-sample kaldi fbank
    extraction across CPU cores, reducing the GPU-idle window.
  - _encode_shared caches encoder output keyed on audio.data_ptr() so that when
    multiple KD branches (CTC / RNNT / Joiner) all run on the same batch the
    expensive Zipformer encoder forward only executes once per step.
"""
from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)


def _load_token_id_to_piece(path: Path) -> Optional[Dict[int, str]]:
    """Load token table from files formatted as: <piece> <id>."""
    if not path.is_file():
        return None

    id2piece: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                continue
            piece, idx_str = parts
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            id2piece[idx] = piece

    return id2piece if id2piece else None


class TeacherJIT(nn.Module):
    """Frozen JIT Teacher wrapper for CTC and RNNT simple-logit KD.

    Args:
        jit_path: Path to the JIT-compiled .pt file.
        device:   Device to run Teacher on (same as Student by default).
    """

    def __init__(
        self,
        jit_path: str,
        device: torch.device,
        student_tokens_path: Optional[str] = None,
        teacher_tokens_path: Optional[str] = None,
    ):
        super().__init__()
        logger.info(f"Loading Teacher JIT from {jit_path}")
        model = torch.jit.load(jit_path, map_location=device)
        model.eval()
        # Register as a buffer-less attribute (not nn.Module) so it doesn't
        # appear in Student's parameters / optimizer.
        object.__setattr__(self, "_jit", model)
        object.__setattr__(self, "_device", device)

        # Freeze everything — Teacher is never trained.
        for p in model.parameters():
            p.requires_grad_(False)

        object.__setattr__(
            self,
            "_student_to_teacher_vocab",
            self._build_vocab_mapping(
                jit_path=jit_path,
                student_tokens_path=student_tokens_path,
                teacher_tokens_path=teacher_tokens_path,
            ),
        )

        # Encoder output cache — invalidated when audio.data_ptr() changes.
        object.__setattr__(self, "_enc_cache_key",  None)
        object.__setattr__(self, "_enc_cache_out",  None)
        object.__setattr__(self, "_enc_cache_lens", None)

        logger.info("Teacher JIT loaded and frozen.")

    @staticmethod
    def _build_vocab_mapping(
        jit_path: str,
        student_tokens_path: Optional[str],
        teacher_tokens_path: Optional[str],
    ) -> Optional[torch.Tensor]:
        """Build student->teacher vocab-id mapping for logit reordering."""
        if not student_tokens_path:
            logger.warning("KD vocab mapping disabled: student_tokens_path is empty")
            return None

        student_path = Path(student_tokens_path)
        student_id2piece = _load_token_id_to_piece(student_path)
        if student_id2piece is None:
            logger.warning(
                "KD vocab mapping disabled: cannot parse student tokens from %s",
                student_path,
            )
            return None

        teacher_candidates = []
        if teacher_tokens_path:
            teacher_candidates.append(Path(teacher_tokens_path))

        jit_dir = Path(jit_path).resolve().parent
        teacher_candidates.extend([
            jit_dir / "tokens.txt",
            jit_dir / "config.json",  # Some exported bundles store tokens in config.json
        ])

        teacher_id2piece = None
        teacher_source = None
        for p in teacher_candidates:
            teacher_id2piece = _load_token_id_to_piece(p)
            if teacher_id2piece is not None:
                teacher_source = p
                break

        if teacher_id2piece is None:
            logger.warning(
                "KD vocab mapping disabled: cannot find parseable teacher token file near %s",
                jit_path,
            )
            return None

        teacher_piece2id = {piece: idx for idx, piece in teacher_id2piece.items()}
        teacher_unk = teacher_piece2id.get("<unk>", 0)

        max_student_id = max(student_id2piece.keys())
        mapping = []
        matched = 0
        for sid in range(max_student_id + 1):
            piece = student_id2piece.get(sid, "<unk>")
            tid = teacher_piece2id.get(piece, teacher_unk)
            if piece in teacher_piece2id:
                matched += 1
            mapping.append(tid)

        logger.info(
            "KD vocab mapping built: matched %s/%s student pieces using teacher table %s",
            matched,
            max_student_id + 1,
            teacher_source,
        )
        return torch.tensor(mapping, dtype=torch.long)

    def map_to_student_vocab(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Reorder teacher logits to student vocab-id order when mapping is available."""
        mapping = object.__getattribute__(self, "_student_to_teacher_vocab")
        if mapping is None:
            return teacher_logits

        idx = mapping.to(device=teacher_logits.device)
        return teacher_logits.index_select(dim=-1, index=idx)

    # ------------------------------------------------------------------
    # Mel feature extraction (kaldi FBANK, matches icefall training)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_one_mel(args: Tuple) -> torch.Tensor:
        wav, length = args
        wav_cpu = wav[: int(length)].cpu().unsqueeze(0)  # [1, T]
        return torchaudio.compliance.kaldi.fbank(
            wav_cpu,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            sample_frequency=16000,
            dither=0.0,
            energy_floor=1e-10,
        )  # [T_mel, 80]

    @staticmethod
    def _audio_to_mel(
        audio: torch.Tensor,      # [B, T_samples]
        audio_lens: torch.Tensor, # [B] number of valid samples per item
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw waveform batch to kaldi FBANK features.

        Samples are processed in parallel using a thread pool to reduce the
        GPU-idle window caused by serial per-sample CPU computation.

        Returns:
            mel:      [B, T_mel, 80]
            mel_lens: [B]
        """
        items = list(zip(audio, audio_lens))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            mel_list = list(pool.map(TeacherJIT._extract_one_mel, items))

        mel_lens = [f.shape[0] for f in mel_list]
        max_t = max(mel_lens)
        B = len(mel_list)
        mel_padded = torch.zeros(B, max_t, 80, dtype=mel_list[0].dtype)
        for i, feat in enumerate(mel_list):
            mel_padded[i, : feat.shape[0]] = feat

        return mel_padded, torch.tensor(mel_lens, dtype=torch.long)

    # ------------------------------------------------------------------
    # Shared encoder (cached per batch to avoid redundant computation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_shared(
        self,
        audio: torch.Tensor,        # [B, T_samples]
        padding_mask: torch.Tensor, # [B, T_samples] True=padded
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run mel extraction + Zipformer encoder, caching result for the batch.

        When multiple KD branches (CTC, RNNT, Joiner) are all active in a single
        training step they call this method on the same audio tensor.  The result
        is cached keyed on (audio.data_ptr(), audio.shape) so the expensive
        encoder forward runs only once per step regardless of how many branches
        consume it.

        Returns:
            enc_out:  [B, T_enc, D_enc]
            out_lens: [B]
        """
        cache_key = (audio.data_ptr(), tuple(audio.shape))
        cached_key = object.__getattribute__(self, "_enc_cache_key")
        if cache_key == cached_key:
            return (
                object.__getattribute__(self, "_enc_cache_out"),
                object.__getattribute__(self, "_enc_cache_lens"),
            )

        jit    = object.__getattribute__(self, "_jit")
        device = object.__getattribute__(self, "_device")

        audio_lens = (~padding_mask).long().sum(dim=1)
        mel, mel_lens = self._audio_to_mel(audio, audio_lens)
        mel      = mel.to(device)
        mel_lens = mel_lens.to(device)

        x, x_lens   = jit.encoder_embed(mel, mel_lens)
        x_tbd        = x.permute(1, 0, 2).contiguous()
        out_tbd, out_lens = jit.encoder.encoder(x_tbd, x_lens)
        enc_out      = out_tbd.permute(1, 0, 2).contiguous()  # [B, T, D]

        object.__setattr__(self, "_enc_cache_key",  cache_key)
        object.__setattr__(self, "_enc_cache_out",  enc_out)
        object.__setattr__(self, "_enc_cache_lens", out_lens)

        return enc_out, out_lens

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_rnnt_logits(
        self,
        audio: torch.Tensor,         # [B, T_samples]
        padding_mask: torch.Tensor,  # [B, T_samples] True=padded
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RNNT frame-level logits at null predictor state for KD.

        Returns am + lm_null instead of just am (as in get_ctc_logits),
        capturing both acoustic and language model knowledge of the teacher.

        Returns:
            logits:   [B, T_teacher, vocab_size]  raw (not log-softmax)
            out_lens: [B]
        """
        jit    = object.__getattribute__(self, "_jit")
        device = object.__getattribute__(self, "_device")

        out_btd, out_lens = self._encode_shared(audio, padding_mask)

        am = jit.simple_am_proj(out_btd)  # [B, T, vocab]

        B = out_btd.shape[0]
        blank_id  = int(jit.decoder.blank_id)
        blank_ids = torch.full((B, 1), blank_id, dtype=torch.long, device=device)
        decoder_out = jit.decoder(blank_ids, need_pad=True)  # [B, 1, decoder_dim]
        lm = jit.simple_lm_proj(decoder_out)                  # [B, 1, vocab]

        return am + lm, out_lens

    @torch.no_grad()
    def get_ctc_logits(
        self,
        audio: torch.Tensor,         # [B, T_samples]
        padding_mask: torch.Tensor,  # [B, T_samples] True=padded
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Teacher forward and return raw CTC logits (before softmax).

        Returns:
            logits:    [B, T_teacher, vocab_size]   raw (not log-softmax)
            out_lens:  [B]
        """
        jit = object.__getattribute__(self, "_jit")

        out_btd, out_lens = self._encode_shared(audio, padding_mask)
        logits = jit.simple_am_proj(out_btd)  # [B, T'', 2000]

        return logits, out_lens

    @torch.no_grad()
    def get_rnnt_simple_logits(
        self,
        audio: torch.Tensor,             # [B, T_samples]
        padding_mask: torch.Tensor,      # [B, T_samples] True=padded
        decoder_input_ids: torch.Tensor, # [B, U] (SOS-padded target ids)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return RNNT simple-branch logits from teacher.

        Args:
            audio: raw waveform batch.
            padding_mask: waveform padding mask.
            decoder_input_ids: SOS-padded token ids for teacher decoder.

        Returns:
            am_logits: [B, T_teacher, vocab_size]
            lm_logits: [B, U, vocab_size]
            out_lens:  [B] encoder valid lengths
        """
        jit    = object.__getattribute__(self, "_jit")
        device = object.__getattribute__(self, "_device")

        out_btd, out_lens = self._encode_shared(audio, padding_mask)

        am_logits = jit.simple_am_proj(out_btd)

        dec_in      = decoder_input_ids.to(device=device, dtype=torch.long)
        decoder_out = jit.decoder(dec_in)
        lm_logits   = jit.simple_lm_proj(decoder_out)

        return am_logits, lm_logits, out_lens

    @torch.no_grad()
    def get_joiner_diag_logits(
        self,
        audio: torch.Tensor,              # [B, T_samples]
        padding_mask: torch.Tensor,       # [B, T_samples] True=padded
        decoder_input_ids: torch.Tensor,  # [B, U] (SOS-padded target ids)
        target_lens: torch.Tensor,        # [B] (without SOS)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return teacher joiner logits sampled along a monotonic diagonal path.

        The path maps each encoder frame t to a decoder step u proportionally:
            u = floor(t * valid_u / valid_t), clamped to [0, valid_u-1].
        """
        jit    = object.__getattribute__(self, "_jit")
        device = object.__getattribute__(self, "_device")

        if not hasattr(jit, "joiner"):
            raise RuntimeError("Teacher JIT does not expose joiner module for joiner KD")

        enc_out, out_lens = self._encode_shared(audio, padding_mask)

        dec_in  = decoder_input_ids.to(device=device, dtype=torch.long)
        dec_out = jit.decoder(dec_in)  # [B, U, D_dec]

        B, T, _ = enc_out.shape
        U = dec_out.shape[1]

        valid_t = out_lens.to(dtype=torch.long).clamp(max=T)
        valid_u = (target_lens.to(device=device, dtype=torch.long) + 1).clamp(max=U)

        t_index = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        denom_t = valid_t.unsqueeze(1).clamp(min=1)
        u_index = (t_index * valid_u.unsqueeze(1)) // denom_t
        u_index = torch.minimum(u_index, (valid_u.unsqueeze(1) - 1).clamp(min=0))
        u_index = u_index.clamp(min=0, max=U - 1)

        dec_diag = dec_out.gather(
            dim=1,
            index=u_index.unsqueeze(-1).expand(-1, -1, dec_out.size(-1)),
        )

        joiner = jit.joiner
        if (
            hasattr(joiner, "encoder_proj")
            and hasattr(joiner, "decoder_proj")
            and hasattr(joiner, "output_linear")
        ):
            logits = joiner.output_linear(
                torch.tanh(joiner.encoder_proj(enc_out) + joiner.decoder_proj(dec_diag))
            )
        else:
            try:
                logits = joiner(enc_out, dec_diag, True)
            except Exception:
                logits = joiner(enc_out, dec_diag)

        return logits, out_lens
