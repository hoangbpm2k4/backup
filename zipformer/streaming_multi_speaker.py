"""
Chunked streaming prototype for multi-speaker RNNT (PHASE 3+).

This module supports streaming end-to-end from raw audio to per-speaker
token sequences:
  1. StreamingConvFrontend: waveform -> conv features (with cache)
  2. HuBERT projections: layer_norm + post_extract_proj + dropout
  3. SpeakerKernel pre-ASR stream routing: expand + route per speaker
  4. Zipformer2.streaming_forward: conv features -> encoder output
  5. External/pyannote activity: diarization
  6. RNNT greedy_decode: per-speaker token sequences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# â”€â”€ Lightweight beam-search primitives (no k2 dependency) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class _Hypothesis:
    """A single beam-search hypothesis (k2-free).

    Attributes:
        ys: full token sequence including the initial context prefix
            ([-1]*(context_size-1) + [blank_id]).  Real tokens start at
            index context_size.
        log_prob: scalar Tensor â€” cumulative log-probability (AM only).
        timestamp: global frame indices for each real token emitted.
        lm_score: accumulated LM log-probability (natural log, **unscaled**).
            The beam ranking uses ``log_prob + lm_scale * lm_score``.
        lm_state: opaque KenLM ``State`` object (None when no LM is used).
    """
    ys: List[int]
    log_prob: Tensor
    timestamp: List[int] = field(default_factory=list)
    lm_score: float = 0.0
    lm_state: Any = None

    @property
    def key(self) -> str:
        return "_".join(map(str, self.ys))


class _HypothesisList:
    """Sorted set of _Hypothesis objects keyed by token sequence (k2-free).

    Duplicate sequences are merged via log-sum-exp.
    """

    def __init__(self) -> None:
        self._data: Dict[str, _Hypothesis] = {}

    def add(self, hyp: _Hypothesis) -> None:
        key = hyp.key
        if key in self._data:
            old = self._data[key]
            # P2-C fix: preserve timestamp + lm_state from higher-prob path.
            if hyp.log_prob.item() > old.log_prob.item():
                old.timestamp = hyp.timestamp
                old.lm_score = hyp.lm_score
                old.lm_state = hyp.lm_state
            old.log_prob = torch.logaddexp(old.log_prob, hyp.log_prob)
        else:
            self._data[key] = hyp

    @staticmethod
    def _sort_key(
        h: _Hypothesis, length_norm: bool, lm_scale: float
    ) -> float:
        score = h.log_prob.item() + lm_scale * h.lm_score
        if length_norm:
            score /= max(len(h.ys), 1)
        return score

    def topk(
        self, k: int, length_norm: bool = False, lm_scale: float = 0.0,
    ) -> "_HypothesisList":
        items = sorted(
            self._data.values(),
            key=lambda h: self._sort_key(h, length_norm, lm_scale),
            reverse=True,
        )[:k]
        hl = _HypothesisList()
        for h in items:
            hl._data[h.key] = h
        return hl

    def get_most_probable(
        self, length_norm: bool = False, lm_scale: float = 0.0,
    ) -> _Hypothesis:
        return max(
            self._data.values(),
            key=lambda h: self._sort_key(h, length_norm, lm_scale),
        )

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)


class BeamSearchStreamingState:
    """Per-stream RNNT beam search state that persists across chunks.

    Attributes:
        beams: List[_HypothesisList] of length N â€” one beam set per stream
            (N = B * S).  Each list holds up to `beam` hypotheses.
        frame_offsets: List[int] of length N â€” cumulative valid frames
            processed per stream (used for global timestamps).
    """

    def __init__(
        self,
        beams: List[_HypothesisList],
        frame_offsets: Optional[List[int]] = None,
    ) -> None:
        self.beams = beams
        N = len(beams)
        self.frame_offsets = frame_offsets if frame_offsets is not None else [0] * N


class DecoderStreamingState:
    """Per-stream RNNT decoder state that persists across chunks.

    Attributes:
        decoder_context: [N, context_size] last context_size token IDs per stream.
            Initialized to [-1, ..., -1, blank_id] following icefall convention.
        hyps: list of N token lists (accumulated, never reset between chunks).
        timestamps: list of N frame-index lists (global, not per-chunk).
        frame_offsets: list of N ints â€” per-stream total valid frames processed.
            Each stream advances by its own h_lens_chunk[n], not T_chunk.
    """

    def __init__(
        self,
        decoder_context: Tensor,
        hyps: List[List[int]],
        timestamps: List[List[int]],
        frame_offsets: Optional[List[int]] = None,
    ):
        self.decoder_context = decoder_context
        self.hyps = hyps
        self.timestamps = timestamps
        N = decoder_context.shape[0]
        self.frame_offsets = frame_offsets if frame_offsets is not None else [0] * N


class StreamingState:
    """Holds chunked streaming state for the multi-speaker encoder path."""

    def __init__(
        self,
        conv_frontend_state: Optional[Tensor],
        zipformer_states: List[Tensor],
        decoder_state: Optional[DecoderStreamingState] = None,
        processed_frames: int = 0,
        processed_lens_base: Optional[Tensor] = None,
        # Phase 5: Beam search state
        beam_search_state: Optional[BeamSearchStreamingState] = None,
        # Phase 6: Precision Streaming State (P1.7)
        cum_valid_samples: Optional[Tensor] = None,
        cum_valid_feat_frames: Optional[Tensor] = None,
        # Active Zipformer left-context configuration for this state
        left_context_frames: Optional[int] = None,
    ):
        self.conv_frontend_state = conv_frontend_state
        self.zipformer_states = zipformer_states
        self.decoder_state = decoder_state
        self.processed_frames = processed_frames
        self.processed_lens_base = processed_lens_base
        # Phase 5
        self.beam_search_state = beam_search_state
        # Phase 6
        self.cum_valid_samples = cum_valid_samples
        self.cum_valid_feat_frames = cum_valid_feat_frames
        # Active Zipformer left context for mask/cache consistency
        self.left_context_frames = left_context_frames


@dataclass
class PreparedAudioChunk:
    """Prepared ASR-feature branch output for one streaming chunk."""

    features: Optional[Tensor]
    x_lens: Optional[Tensor]
    conv_frontend_state: Any
    batch_size: int
    activity_trim: int = 0


class StreamingMultiSpeakerEncoder(nn.Module):
    """
    Streaming wrapper for multi-speaker RNNT.

    Supports two modes:
      1. Feature-level streaming: caller does offline frontend, passes features
         via streaming_forward_chunk().
      2. Audio-level streaming: caller passes raw audio chunks via
         streaming_forward_from_audio() â€” uses StreamingConvFrontend.
    """

    def __init__(self, ms_model: nn.Module, allow_unsafe_streaming: bool = False):
        super().__init__()
        self.ms_model = ms_model
        self.asr_model = ms_model.asr_model
        self.pyannote_diar = getattr(ms_model, "pyannote_diar", None)
        self.S = ms_model.S

        self._hubert = self.asr_model.encoder
        self._zipformer = self._hubert.encoder
        
        if not getattr(self._zipformer, "causal", False):
            raise RuntimeError("Streaming encoder requires encoder.causal=True")

        odf = getattr(self._zipformer, "output_downsampling_factor", 1)
        if odf != 1:
            raise RuntimeError(
                f"Streaming inference requires output_downsampling_factor=1, got {odf}. "
                f"processed_lens_base tracking assumes 1:1 frame rate."
            )

        # Build streaming frontend wrapper (shares weights with original)
        from wav2vec2_module import StreamingConvFrontend
        self._streaming_frontend = StreamingConvFrontend(
            self._hubert.feature_extractor
        )

        # Backward-compat: this flag used to guard GroupNorm frontends.
        # GroupNorm is now handled by CumulativeInstanceNorm inside
        # StreamingConvFrontend, so the flag is intentionally ignored.
        _ = allow_unsafe_streaming

    @property
    def causal_left_trim(self) -> int:
        """Extra leading feature frames from zero-cache init."""
        return self._streaming_frontend.causal_left_trim

    def init_state(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        init_decode: bool = False,
        left_context_frames: int = -1,
    ) -> StreamingState:
        """Initialize full streaming state.

        Args:
            batch_size: B (number of utterances)
            device: torch device
            init_decode: if True, also initialize decoder state for B*S streams
            left_context_frames: active Zipformer left context in encoder
                frames. Use -1 to keep model default.
        """
        conv_frontend_state = self._streaming_frontend.get_init_state(
            batch_size, device
        )
        configured_left_context = int(
            getattr(self._zipformer, "left_context_frames", [0])[0]
        )
        active_left_context_frames = (
            configured_left_context
            if left_context_frames < 0
            else int(left_context_frames)
        )
        if active_left_context_frames < 0:
            raise ValueError(
                f"left_context_frames must be >= 0, got {active_left_context_frames}"
            )
        for ds in getattr(self._zipformer, "downsampling_factor", (1,)):
            if active_left_context_frames % int(ds) != 0:
                raise ValueError(
                    "left_context_frames must be divisible by every Zipformer "
                    f"downsampling factor; got {active_left_context_frames} "
                    f"with ds={ds}."
                )

        # In pre-ASR routing mode, Zipformer caches are maintained per speaker
        # stream (N = B * S) because speaker-conditioned features are fed into
        # the encoder before decoding.
        zip_batch_size = (
            batch_size * self.S
            if bool(getattr(self.ms_model, "enable_pre_asr_stream_routing", False))
            else batch_size
        )

        try:
            zipformer_states = self._zipformer.get_init_states(
                batch_size=zip_batch_size,
                device=device,
                left_context_frames=active_left_context_frames,
            )
        except TypeError:
            # Backward compatibility for lightweight test mocks / legacy wrappers
            # exposing get_init_states(batch_size, device) only.
            zipformer_states = self._zipformer.get_init_states(
                batch_size=zip_batch_size, device=device
            )

        decoder_state = None
        if init_decode:
            decoder_state = self.init_decoder_state(
                num_streams=batch_size * self.S, device=device
            )

        # Always init processed_lens_base â€” needed for Zipformer2 left-context
        # mask on both injection and non-injection paths.
        processed_lens_batch = (
            zip_batch_size
            if bool(getattr(self.ms_model, "enable_pre_asr_stream_routing", False))
            else batch_size
        )
        processed_lens_base = torch.zeros(
            processed_lens_batch, dtype=torch.long, device=device
        )

        return StreamingState(
            conv_frontend_state=conv_frontend_state,
            zipformer_states=zipformer_states,
            decoder_state=decoder_state,
            processed_frames=0,
            processed_lens_base=processed_lens_base,
            beam_search_state=None,
            cum_valid_samples=conv_frontend_state[3]["cum_valid_samples"],
            cum_valid_feat_frames=conv_frontend_state[3]["cum_valid_feat_frames"],
            left_context_frames=active_left_context_frames,
        )

    def _apply_hubert_projections(self, raw_features: Tensor) -> Tensor:
        """Apply HuBERT's layer_norm + post_extract_proj + dropout.

        Args:
            raw_features: [B, C, T] from conv frontend
        Returns:
            features: [B, T, D_input] ready for Zipformer2
        """
        features = raw_features.transpose(1, 2)  # [B, T, C]
        features = self._hubert.layer_norm(features)
        if self._hubert.post_extract_proj is not None:
            features = self._hubert.post_extract_proj(features)
        features = self._hubert.dropout_input(features)
        return features

    def extract_frontend_features(self, source: Tensor) -> Tuple[Tensor, Tensor]:
        """Run frontend conv + HuBERT projections OFFLINE (legacy path)."""
        features = self._hubert.forward_features(source)
        features = features.transpose(1, 2)
        features = self._hubert.layer_norm(features)

        batch_size, t_feat = features.shape[0], features.shape[1]
        padding_mask = torch.zeros(
            batch_size, t_feat, dtype=torch.bool, device=source.device
        )

        if self._hubert.post_extract_proj is not None:
            features = self._hubert.post_extract_proj(features)

        features = self._hubert.dropout_input(features)
        return features, padding_mask

    @staticmethod
    def _nearest_resample_activity(activity: Tensor, target_len: int) -> Tensor:
        """Resample [B, T_in, S] -> [B, target_len, S] with nearest interpolation."""
        if activity.shape[1] == target_len:
            return activity
        out = F.interpolate(
            activity.transpose(1, 2).float(),
            size=target_len,
            mode="nearest",
        ).transpose(1, 2)
        return out.to(activity.dtype)

    @staticmethod
    def _infer_activity_for_prepared(
        audio_chunk: Tensor,
        prepared: "PreparedAudioChunk",
        pyannote_diar: nn.Module,
    ) -> Optional[Tensor]:
        """Infer diar activity after ASR features are prepared (sequential fallback)."""
        feat_len = 0 if prepared.features is None else int(prepared.features.shape[1])
        diar_target_len = feat_len + int(prepared.activity_trim)
        if diar_target_len <= 0:
            return None
        with torch.no_grad():
            return pyannote_diar(audio_chunk, target_len=diar_target_len)

    def _prepare_hubert_and_auto_activity(
        self,
        audio_chunk: Tensor,
        state: StreamingState,
        pyannote_diar: nn.Module,
        sample_lens: Optional[Tensor] = None,
    ) -> Tuple["PreparedAudioChunk", Optional[Tensor]]:
        """Run ASR + pyannote branches with real parallelism when possible.

        On CUDA and in no-grad mode, run:
          - ASR feature branch on one CUDA stream
          - pyannote native branch on another CUDA stream
        then resample pyannote output to ASR-aligned target length.
        """
        can_parallel = bool(
            audio_chunk.is_cuda
            and (not torch.is_grad_enabled())
            and hasattr(pyannote_diar, "forward_native")
        )

        if not can_parallel:
            prepared = self.prepare_hubert_chunk_from_audio(
                audio_chunk=audio_chunk,
                state=state,
                sample_lens=sample_lens,
            )
            inferred_activity = StreamingMultiSpeakerEncoder._infer_activity_for_prepared(
                audio_chunk=audio_chunk,
                prepared=prepared,
                pyannote_diar=pyannote_diar,
            )
            return prepared, inferred_activity

        device = audio_chunk.device
        default_stream = torch.cuda.current_stream(device=device)
        asr_stream = torch.cuda.Stream(device=device)
        diar_stream = torch.cuda.Stream(device=device)

        prepared_holder: Dict[str, Any] = {}
        native_holder: Dict[str, Any] = {}

        with torch.cuda.stream(asr_stream):
            prepared_holder["value"] = self.prepare_hubert_chunk_from_audio(
                audio_chunk=audio_chunk,
                state=state,
                sample_lens=sample_lens,
            )

        with torch.cuda.stream(diar_stream):
            with torch.no_grad():
                native_holder["value"] = pyannote_diar.forward_native(audio_chunk)

        default_stream.wait_stream(asr_stream)
        default_stream.wait_stream(diar_stream)

        prepared = prepared_holder["value"]
        native_activity = native_holder["value"]
        feat_len = 0 if prepared.features is None else int(prepared.features.shape[1])
        diar_target_len = feat_len + int(prepared.activity_trim)

        inferred_activity = None
        if diar_target_len > 0:
            if hasattr(pyannote_diar, "resample_activity"):
                inferred_activity = pyannote_diar.resample_activity(
                    native_activity, diar_target_len
                )
            else:
                inferred_activity = StreamingMultiSpeakerEncoder._nearest_resample_activity(
                    native_activity, diar_target_len
                )

        return prepared, inferred_activity

    def streaming_forward_from_audio(
        self,
        audio_chunk: Tensor,
        spk_activity_chunk: Optional[Tensor],
        state: StreamingState,
        sample_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, StreamingState]:
        """
        Process a raw audio chunk end-to-end through the streaming pipeline.

        Pipeline: conv frontend -> HuBERT proj -> diar (optional) -> pre-ASR routing -> Zipformer2

        Args:
            audio_chunk: [B, T_samples] raw audio
            spk_activity_chunk: [B, T_act, S] pre-computed speaker activity.
                If None and pyannote_diar is available, activity is inferred
                from audio_chunk in no_grad mode. On CUDA, pyannote and HuBERT
                branches are executed on separate streams for true overlap.
            state: current streaming state
            sample_lens: [B] valid audio samples in chunk (for padding mask)

        Returns:
            h_all_chunk: [B*S, T_out, D] per-speaker encoder output
            h_lens_chunk: [B*S] per-sample-per-speaker valid frame count
            new_state: updated state
        """
        pyannote_diar = getattr(self, "pyannote_diar", None)
        if spk_activity_chunk is None and pyannote_diar is not None:
            prepared, spk_activity_chunk = (
                StreamingMultiSpeakerEncoder._prepare_hubert_and_auto_activity(
                    self,
                    audio_chunk=audio_chunk,
                    state=state,
                    pyannote_diar=pyannote_diar,
                    sample_lens=sample_lens,
                )
            )
        else:
            prepared = self.prepare_hubert_chunk_from_audio(
                audio_chunk=audio_chunk, state=state, sample_lens=sample_lens
            )
        return self.merge_prepared_chunk(
            prepared=prepared, spk_activity_chunk=spk_activity_chunk, state=state
        )

    def prepare_hubert_chunk_from_audio(
        self,
        audio_chunk: Tensor,
        state: StreamingState,
        sample_lens: Optional[Tensor] = None,
    ) -> PreparedAudioChunk:
        """Run the ASR feature branch (conv frontend + HuBERT projections)."""
        batch_size = audio_chunk.shape[0]

        raw_features, new_conv_state, x_lens = self._streaming_frontend.streaming_forward(
            audio_chunk,
            state.conv_frontend_state,
            sample_lens=sample_lens,
            return_lens=True,
        )  # raw_features: [B, C, T_feat]

        activity_trim = 0
        wav_cache = state.conv_frontend_state[0]
        is_first_chunk = wav_cache.shape[2] == 0
        trim = self._streaming_frontend.causal_left_trim
        if trim > 0 and is_first_chunk:
            raw_features = raw_features[:, :, trim:]
            x_lens = (x_lens - trim).clamp(min=0)
            activity_trim = trim

        if raw_features.shape[2] == 0:
            return PreparedAudioChunk(
                features=None,
                x_lens=None,
                conv_frontend_state=new_conv_state,
                batch_size=batch_size,
                activity_trim=activity_trim,
            )

        features = self._apply_hubert_projections(raw_features)  # [B, T_feat, D]
        x_lens = x_lens.clamp(min=0, max=features.shape[1])

        return PreparedAudioChunk(
            features=features,
            x_lens=x_lens,
            conv_frontend_state=new_conv_state,
            batch_size=batch_size,
            activity_trim=activity_trim,
        )

    def merge_prepared_chunk(
        self,
        prepared: PreparedAudioChunk,
        spk_activity_chunk: Optional[Tensor],
        state: StreamingState,
    ) -> Tuple[Tensor, Tensor, StreamingState]:
        """Merge prepared features with activity and run the streaming encoder."""
        new_conv_state = prepared.conv_frontend_state
        batch_size = prepared.batch_size

        if spk_activity_chunk is not None and prepared.features is not None:
            feat_len = int(prepared.features.shape[1])
            target_len = feat_len + int(prepared.activity_trim)

            if target_len > 0 and spk_activity_chunk.shape[1] != target_len:
                if hasattr(self.pyannote_diar, "resample_activity"):
                    spk_activity_chunk = self.pyannote_diar.resample_activity(
                        spk_activity_chunk, target_len
                    )
                else:
                    spk_activity_chunk = (
                        StreamingMultiSpeakerEncoder._nearest_resample_activity(
                            spk_activity_chunk, target_len
                        )
                    )

            if prepared.x_lens is not None and target_len > 0:
                valid = (
                    prepared.x_lens.to(device=spk_activity_chunk.device, dtype=torch.long)
                    + int(prepared.activity_trim)
                ).clamp(min=0, max=target_len)
                idx = torch.arange(target_len, device=spk_activity_chunk.device).unsqueeze(0)
                invalid = idx >= valid.unsqueeze(1)
                spk_activity_chunk = spk_activity_chunk.masked_fill(
                    invalid.unsqueeze(-1), 0.0
                )

        if prepared.activity_trim > 0 and spk_activity_chunk is not None:
            trim = int(prepared.activity_trim)
            if spk_activity_chunk.shape[1] > trim:
                spk_activity_chunk = spk_activity_chunk[:, trim:, :]
            else:
                spk_activity_chunk = spk_activity_chunk[:, :0, :]

        if prepared.features is None:
            # Not enough audio to produce any conv frames yet
            new_state = StreamingState(
                conv_frontend_state=new_conv_state,
                zipformer_states=state.zipformer_states,
                decoder_state=state.decoder_state,
                processed_frames=state.processed_frames,
                processed_lens_base=state.processed_lens_base,
                beam_search_state=state.beam_search_state,
                cum_valid_samples=new_conv_state[3]["cum_valid_samples"],
                cum_valid_feat_frames=new_conv_state[3]["cum_valid_feat_frames"],
                left_context_frames=state.left_context_frames,
            )
            device = new_conv_state[0].device
            D = max(self._zipformer.encoder_dim)
            return (
                torch.zeros(batch_size * self.S, 0, D, device=device),
                torch.zeros(batch_size * self.S, dtype=torch.long, device=device),
                new_state,
            )

        feat_state = StreamingState(
            conv_frontend_state=None,
            zipformer_states=state.zipformer_states,
            decoder_state=state.decoder_state,
            processed_frames=state.processed_frames,
            processed_lens_base=state.processed_lens_base,
            beam_search_state=state.beam_search_state,
            left_context_frames=state.left_context_frames,
        )
        h_all_chunk, h_lens_chunk, feat_new_state = self.streaming_forward_chunk(
            prepared.features, spk_activity_chunk, feat_state, x_lens=prepared.x_lens
        )

        new_state = StreamingState(
            conv_frontend_state=new_conv_state,
            zipformer_states=feat_new_state.zipformer_states,
            decoder_state=state.decoder_state,
            processed_frames=feat_new_state.processed_frames,
            processed_lens_base=feat_new_state.processed_lens_base,
            beam_search_state=state.beam_search_state,
            cum_valid_samples=new_conv_state[3]["cum_valid_samples"],
            cum_valid_feat_frames=new_conv_state[3]["cum_valid_feat_frames"],
            left_context_frames=feat_new_state.left_context_frames,
        )
        return h_all_chunk, h_lens_chunk, new_state

    @staticmethod
    def _find_active_speakers(activity_chunk: Tensor,
                              threshold: float = 0.5) -> Tensor:
        """Find active speakers per batch item. Returns active_indices [N] in [0, B*S).

        Matches NeMo's _find_active_speakers: max(activity over time) > threshold.
        """
        B, T, S = activity_chunk.shape
        max_activity = activity_chunk.max(dim=1).values  # [B, S]
        is_active = max_activity > threshold  # [B, S]
        # Build flat indices into [0, B*S)
        active_indices = []
        for b in range(B):
            for s in range(S):
                if is_active[b, s]:
                    active_indices.append(b * S + s)
        if not active_indices:
            # Fallback: always keep at least speaker 0 per batch
            active_indices = [b * S for b in range(B)]
        return torch.tensor(active_indices, dtype=torch.long,
                            device=activity_chunk.device)

    @staticmethod
    def _select_stream_states(states: List[Tensor],
                              indices: Tensor) -> List[Tensor]:
        """Select batch indices from zipformer state list."""
        selected = []
        for i, s in enumerate(states):
            if s.dim() == 0:
                selected.append(s)
                continue
            pos = i % 6
            if pos in (4, 5) or s.dim() == 1:  # conv caches or compact mocks: batch dim 0
                selected.append(s.index_select(0, indices))
            else:  # attention caches: batch dim 1
                selected.append(s.index_select(1, indices))
        return selected

    @staticmethod
    def _scatter_stream_states(full_states: List[Tensor],
                               active_states: List[Tensor],
                               indices: Tensor) -> List[Tensor]:
        """Put active states back into full_states at given indices."""
        new_states = []
        for i in range(len(full_states)):
            s_full = full_states[i].clone()
            if s_full.dim() == 0:
                new_states.append(s_full)
                continue
            pos = i % 6
            if pos in (4, 5) or s_full.dim() == 1:
                s_full[indices] = active_states[i]
            else:
                s_full[:, indices] = active_states[i]
            new_states.append(s_full)
        return new_states

    def streaming_forward_chunk(
        self,
        features_chunk: Tensor,
        spk_activity_chunk: Optional[Tensor],
        state: StreamingState,
        x_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, StreamingState]:
        """
        Process one feature chunk through pre-ASR routing + Zipformer.

        Args:
            features_chunk: [B, T_chunk, D]
            spk_activity_chunk: [B, T_chunk, S] pre-computed speaker activity
            state: current streaming state
            x_lens: [B] valid feature frames (None = all valid)
        Returns:
            h_all_chunk: [B*S, T_out, D]
            h_lens_chunk: [B*S]
            new_state: updated caches
        """
        batch_size, t_chunk, _ = features_chunk.shape
        device = features_chunk.device
        ms = self.ms_model

        if x_lens is None:
            x_lens = torch.full(
                (batch_size,),
                t_chunk,
                dtype=torch.long,
                device=device,
            )

        use_pre_asr_stream_routing = bool(
            getattr(ms, "enable_pre_asr_stream_routing", False)
        )
        speaker_kernel = getattr(ms, "speaker_kernel", None)
        if use_pre_asr_stream_routing and speaker_kernel is None:
            raise RuntimeError(
                "enable_pre_asr_stream_routing=True but speaker_kernel is missing."
            )

        if not use_pre_asr_stream_routing:
            raise RuntimeError(
                "Legacy post-encoder speaker split path is removed. "
                "Set enable_pre_asr_stream_routing=True."
            )
        if spk_activity_chunk is None:
            raise ValueError(
                "spk_activity_chunk is required when pre-ASR stream routing is enabled."
            )

        activity_chunk = spk_activity_chunk
        if activity_chunk.shape[1] != t_chunk:
            activity_chunk = ms._resample_activity(activity_chunk, t_chunk)
        if x_lens is not None and t_chunk > 0:
            idx = torch.arange(t_chunk, device=device).unsqueeze(0)
            invalid = idx >= x_lens.unsqueeze(1)
            activity_chunk = activity_chunk.masked_fill(invalid.unsqueeze(-1), 0.0)

        # Pre-ASR routing: replicate features for all S speakers
        routed_features = features_chunk.unsqueeze(1).expand(
            batch_size, self.S, t_chunk, -1
        ).reshape(batch_size * self.S, t_chunk, -1)

        # Build stacked targets [B*S, T]
        spk_targets = torch.cat(
            [activity_chunk[:, :, s] for s in range(self.S)], dim=0
        )
        bg_targets = torch.cat(
            [speaker_kernel.build_bg_mask(activity_chunk, s)
             for s in range(self.S)], dim=0
        )

        # Find active speakers (NeMo-style: only encode active streams)
        active_idx = self._find_active_speakers(
            activity_chunk, threshold=getattr(ms, "pre_asr_active_threshold", 0.5)
        )
        N_full = batch_size * self.S

        # Select active features and targets
        active_features = routed_features[active_idx]  # [N_active, T, D]
        active_spk = spk_targets[active_idx]
        active_bg = bg_targets[active_idx]

        # Set targets — callbacks will apply kernel inside encoder
        ms.set_speaker_targets(active_spk, active_bg)

        x_routed = active_features.transpose(0, 1)  # [T, N_active, D]
        x_lens_routed = x_lens.unsqueeze(1).expand(
            batch_size, self.S
        ).reshape(-1)[active_idx]

        idx_routed = torch.arange(t_chunk, device=device).unsqueeze(0)
        src_key_padding_mask_routed = idx_routed >= x_lens_routed.unsqueeze(1)
        processed_lens_cur = (
            state.processed_lens_base
            if state.processed_lens_base is not None
            else torch.zeros(N_full, dtype=torch.long, device=device)
        )
        L_base = (
            int(state.left_context_frames)
            if state.left_context_frames is not None
            else int(getattr(self._zipformer, "left_context_frames", [0])[0])
        )
        active_processed_lens = processed_lens_cur[active_idx]
        valid_lc = active_processed_lens.clamp(max=L_base)
        invalid_lc = L_base - valid_lc
        lc_mask = (
            torch.arange(L_base, device=device).unsqueeze(0)
            < invalid_lc.unsqueeze(1)
        )  # [N_active, L_base]
        full_mask = torch.cat([lc_mask, src_key_padding_mask_routed], dim=1)

        # Select active zipformer states
        active_zip_states = self._select_stream_states(
            state.zipformer_states, active_idx
        )

        try:
            encoder_out_tbd, encoder_out_lens, new_active_zip_states = (
                self._zipformer.streaming_forward(
                    x_routed,
                    x_lens_routed,
                    active_zip_states,
                    full_mask,
                    left_context_frames=L_base,
                )
            )
        except TypeError:
            encoder_out_tbd, encoder_out_lens, new_active_zip_states = (
                self._zipformer.streaming_forward(
                    x_routed, x_lens_routed, active_zip_states, full_mask
                )
            )
        ms.clear_speaker_targets()

        # Scatter active results back to full [B*S] layout
        T_out = encoder_out_tbd.shape[0]
        D_out = encoder_out_tbd.shape[2]
        h_all_chunk = torch.zeros(
            N_full, T_out, D_out, device=device, dtype=encoder_out_tbd.dtype
        )
        h_all_chunk[active_idx] = encoder_out_tbd.transpose(0, 1)

        full_encoder_out_lens = torch.zeros(
            N_full, dtype=encoder_out_lens.dtype, device=device
        )
        full_encoder_out_lens[active_idx] = encoder_out_lens
        h_lens_chunk = full_encoder_out_lens

        # Scatter zipformer states back
        new_zipformer_states = self._scatter_stream_states(
            state.zipformer_states, new_active_zip_states, active_idx
        )

        new_processed_lens = processed_lens_cur.clone()
        new_processed_lens[active_idx] = (
            new_processed_lens[active_idx] + encoder_out_lens
        )

        new_state = StreamingState(
            conv_frontend_state=state.conv_frontend_state,
            zipformer_states=new_zipformer_states,
            decoder_state=state.decoder_state,
            processed_frames=state.processed_frames + t_chunk,
            processed_lens_base=new_processed_lens,
            beam_search_state=state.beam_search_state,
            left_context_frames=state.left_context_frames,
        )
        return h_all_chunk, h_lens_chunk, new_state

    def streaming_forward(
        self,
        source: Tensor,
        spk_activity: Optional[Tensor] = None,
        chunk_frames: int = 32,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Run the chunked prototype on the whole segment (offline frontend).

        Args:
            source: [B, T_samples]
            spk_activity: [B, T_act, S] or None if an internal diar branch is enabled
            chunk_frames: feature frames per chunk
        Returns:
            h_all_chunks: list of [B*S, T_out_i, D]
            h_lens_chunks: list of [B*S]
        """
        batch_size = source.shape[0]
        device = source.device

        features, _ = self.extract_frontend_features(source)
        t_feat = features.shape[1]

        activity = None
        pyannote_diar = getattr(self, "pyannote_diar", None)
        if spk_activity is not None:
            activity = self.ms_model._resample_activity(spk_activity, t_feat)
        elif pyannote_diar is not None:
            with torch.no_grad():
                activity = pyannote_diar(source, target_len=t_feat)
        else:
            raise ValueError(
                "spk_activity must be provided when no internal diar branch is present."
            )

        state = self.init_state(batch_size, device)
        h_all_chunks: List[Tensor] = []
        h_lens_chunks: List[Tensor] = []

        for start in range(0, t_feat, chunk_frames):
            end = min(start + chunk_frames, t_feat)
            feat_chunk = features[:, start:end, :]
            act_chunk = None if activity is None else activity[:, start:end, :]

            h_all_chunk, h_lens_chunk, state = self.streaming_forward_chunk(
                feat_chunk, act_chunk, state
            )
            h_all_chunks.append(h_all_chunk)
            h_lens_chunks.append(h_lens_chunk)

        return h_all_chunks, h_lens_chunks

    def init_decoder_state(
        self,
        num_streams: int,
        device: torch.device,
    ) -> DecoderStreamingState:
        """Initialize decoder streaming state for N = B*S streams.

        Args:
            num_streams: number of parallel decode streams (B * S)
            device: torch device
        Returns:
            DecoderStreamingState with blank-initialized context
        """
        decoder = self.asr_model.decoder
        blank_id = decoder.blank_id
        context_size = decoder.context_size

        # icefall convention: [-1, ..., -1, blank_id]
        # The -1 entries are handled by Decoder.forward via clamp(min=0)*(y>=0)
        ctx = torch.full(
            (num_streams, context_size), -1, dtype=torch.long, device=device
        )
        ctx[:, -1] = blank_id

        return DecoderStreamingState(
            decoder_context=ctx,
            hyps=[[] for _ in range(num_streams)],
            timestamps=[[] for _ in range(num_streams)],
            frame_offsets=[0] * num_streams,
        )

    def streaming_greedy_decode_chunk(
        self,
        h_all_chunk: Tensor,
        h_lens_chunk: Tensor,
        decoder_state: DecoderStreamingState,
        max_sym_per_frame: int = 1,
    ) -> DecoderStreamingState:
        """Stateful greedy decode on one chunk of per-speaker encoder output.

        Processes encoder frames sequentially, allowing up to
        `max_sym_per_frame` non-blank symbols per frame.
        Updates decoder_state in-place (context, hyps, timestamps, offset)
        and returns the updated state.

        Args:
            h_all_chunk: [N, T_chunk, D] encoder output for this chunk
            h_lens_chunk: [N] valid frame counts in this chunk
            decoder_state: state from previous chunk (or init_decoder_state)
            max_sym_per_frame: max symbols to emit per encoder frame
        Returns:
            Updated DecoderStreamingState with new tokens appended to hyps
        """
        decoder = self.asr_model.decoder
        joiner = self.asr_model.joiner
        blank_id = decoder.blank_id
        unk_id = getattr(self.asr_model, "unk_id", blank_id)
        context_size = decoder.context_size

        N, T_chunk, _ = h_all_chunk.shape

        if T_chunk == 0:
            return decoder_state

        # Pre-project encoder output for efficiency (same as greedy_search_batch)
        enc_proj = joiner.encoder_proj(h_all_chunk)  # [N, T_chunk, joiner_dim]

        # Current decoder projection (will be re-computed only on emission)
        dec_out = decoder(decoder_state.decoder_context, need_pad=False)
        dec_proj = joiner.decoder_proj(dec_out)  # [N, 1, joiner_dim]

        for t in range(T_chunk):
            # Current encoder frame: [N, 1, 1, joiner_dim]
            enc_t = enc_proj[:, t:t+1, :].unsqueeze(2)

            has_blanked = [False] * N
            for sym_idx in range(max_sym_per_frame):
                # Joiner with pre-projected inputs
                logits = joiner(
                    enc_t, dec_proj.unsqueeze(1), project_input=False
                )  # [N, 1, 1, vocab_size]
                logits = logits.squeeze(1).squeeze(1)  # [N, vocab_size]

                tokens = logits.argmax(dim=-1)  # [N]

                emitted_streams: List[int] = []
                for n in range(N):
                    if t >= h_lens_chunk[n] or has_blanked[n]:
                        continue

                    tok = tokens[n].item()
                    if tok in (blank_id, unk_id):
                        has_blanked[n] = True
                    else:
                        decoder_state.hyps[n].append(tok)
                        decoder_state.timestamps[n].append(
                            decoder_state.frame_offsets[n] + t
                        )
                        # In-place context shift avoids small per-token allocations.
                        decoder_state.decoder_context[n, :-1] = (
                            decoder_state.decoder_context[n, 1:]
                        )
                        decoder_state.decoder_context[n, -1] = tok
                        emitted_streams.append(n)

                if emitted_streams:
                    # Re-compute decoder projection only for streams that emitted.
                    ctx_emitted = decoder_state.decoder_context[emitted_streams]
                    dec_proj[emitted_streams] = joiner.decoder_proj(
                        decoder(ctx_emitted, need_pad=False)
                    )

                if all(has_blanked) or not emitted_streams:
                    break

        # Per-stream offset: advance by valid frames only
        for n in range(N):
            valid = min(int(h_lens_chunk[n].item()), T_chunk)
            decoder_state.frame_offsets[n] += valid
        return decoder_state

    def streaming_greedy_decode(
        self,
        h_all: Tensor,
        h_lens: Tensor,
    ) -> List[List[int]]:
        """Non-streaming greedy decode (backward compat).

        Processes entire encoder output at once. Equivalent to calling
        streaming_greedy_decode_chunk on the full tensor.

        Args:
            h_all: [B*S, T, D] per-speaker encoder output
            h_lens: [B*S] valid frame counts
        Returns:
            hyps: list of B*S token lists
        """
        N = h_all.shape[0]
        dec_state = self.init_decoder_state(N, h_all.device)
        dec_state = self.streaming_greedy_decode_chunk(h_all, h_lens, dec_state)
        return dec_state.hyps

    # â”€â”€ Beam search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def init_beam_search_state(
        self,
        num_streams: int,
        device: torch.device,
        beam: int = 4,
        lm_scorer: Optional[Any] = None,
    ) -> BeamSearchStreamingState:
        """Initialise beam search state for N = B*S streams.

        Each stream starts with a single blank-seeded hypothesis with
        log_prob = 0 (following icefall convention).

        Args:
            num_streams: number of parallel streams (B * S).
            device: torch device for log_prob tensors.
            beam: kept for documentation; does not prune at init.
            lm_scorer: optional ``KenLmScorer`` â€” if provided, each initial
                hypothesis is seeded with a beginning-of-sentence LM state.
        Returns:
            BeamSearchStreamingState with one hypothesis per stream.
        """
        decoder = self.asr_model.decoder
        blank_id = decoder.blank_id
        context_size = decoder.context_size

        init_lm_state = lm_scorer.get_init_state() if lm_scorer is not None else None

        beams: List[_HypothesisList] = []
        for _ in range(num_streams):
            hl = _HypothesisList()
            hl.add(_Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
                lm_score=0.0,
                lm_state=init_lm_state,
            ))
            beams.append(hl)
        return BeamSearchStreamingState(beams=beams)

    def streaming_beam_search_chunk(
        self,
        h_all_chunk: Tensor,
        h_lens_chunk: Tensor,
        bs_state: BeamSearchStreamingState,
        beam: int = 4,
        temperature: float = 1.0,
        blank_penalty: float = 0.0,
        lm_scorer: Optional[Any] = None,
        lm_scale: float = 0.0,
        max_sym_per_frame: int = 1,
    ) -> BeamSearchStreamingState:
        """Stateful modified beam search on one chunk of per-speaker encoder output.

        Time-synchronous. Pure PyTorch â€” no k2 dependency.

        Args:
            h_all_chunk: [N, T_chunk, D] encoder output for this chunk.
            h_lens_chunk: [N] valid frame counts (streams with t >= h_lens
                are skipped at that frame).
            bs_state: beam state from previous chunk (or init_beam_search_state).
            beam: number of active hypotheses per stream.
            temperature: softmax temperature.
            blank_penalty: subtracted from blank logit before softmax.
            lm_scorer: optional ``KenLmScorer`` for n-gram shallow fusion.
            lm_scale: interpolation weight for LM scores
                (``total = am_score + lm_scale * lm_score``).
        Returns:
            Updated BeamSearchStreamingState.
        """
        decoder = self.asr_model.decoder
        joiner = self.asr_model.joiner
        blank_id = decoder.blank_id
        unk_id = getattr(self.asr_model, "unk_id", blank_id)
        context_size = decoder.context_size

        N, T_chunk, _ = h_all_chunk.shape
        device = h_all_chunk.device

        if T_chunk == 0:
            return bs_state

        # Pre-project encoder output once: [N, T_chunk, joiner_dim]
        enc_proj = joiner.encoder_proj(h_all_chunk)

        beams: List[_HypothesisList] = [bs_state.beams[n] for n in range(N)]
        frame_offsets: List[int] = list(bs_state.frame_offsets)

        for t in range(T_chunk):
            active = [t < int(h_lens_chunk[n].item()) for n in range(N)]
            if not any(active):
                break

            # Inner loop: allow up to max_sym_per_frame non-blank emissions
            for _sym in range(max_sym_per_frame):
                # Flatten hypotheses from active streams
                A: List[List[_Hypothesis]] = [list(beams[n]) for n in range(N)]
                all_hyps: List[_Hypothesis] = []
                hyp_start: List[int] = [0] * N
                hyp_count: List[int] = [0] * N
                stream_indices: List[int] = []
                offset = 0
                for n in range(N):
                    hyp_start[n] = offset
                    if active[n]:
                        hyp_count[n] = len(A[n])
                        for hyp in A[n]:
                            all_hyps.append(hyp)
                            stream_indices.append(n)
                        offset += hyp_count[n]

                num_hyps = len(all_hyps)
                if num_hyps == 0:
                    break

                # Batch decoder forward: [num_hyps, context_size]
                decoder_input = torch.tensor(
                    [hyp.ys[-context_size:] for hyp in all_hyps],
                    device=device, dtype=torch.long,
                )
                # [num_hyps, 1, decoder_dim] â†’ [num_hyps, 1, 1, joiner_dim]
                dec_out = decoder(decoder_input, need_pad=False).unsqueeze(1)
                dec_proj = joiner.decoder_proj(dec_out)

                # Encoder frame for each hyp: index by stream, then by t
                stream_idx_t = torch.tensor(stream_indices, device=device, dtype=torch.long)
                # [num_hyps, joiner_dim] â†’ [num_hyps, 1, 1, joiner_dim]
                enc_t = enc_proj[stream_idx_t, t, :].unsqueeze(1).unsqueeze(1)

                # Joiner: [num_hyps, 1, 1, vocab_size]
                logits = joiner(enc_t, dec_proj, project_input=False)
                logits = logits.squeeze(1).squeeze(1)  # [num_hyps, vocab_size]
                vocab_size = logits.size(-1)

                if blank_penalty != 0.0:
                    logits[:, blank_id] -= blank_penalty

                log_probs = (logits / temperature).log_softmax(dim=-1)  # [num_hyps, V]

                # Add cumulative hypothesis scores: [num_hyps]
                ys_lp = torch.cat([hyp.log_prob.reshape(1) for hyp in all_hyps])
                log_probs = log_probs + ys_lp.unsqueeze(1)  # [num_hyps, V]

                # Build new beams
                new_beams: List[_HypothesisList] = [_HypothesisList() for _ in range(N)]
                # Track whether any stream emitted a non-blank as best token
                any_nonblank = False

                for n in range(N):
                    if not active[n]:
                        new_beams[n] = beams[n]
                        continue

                    s = hyp_start[n]
                    c = hyp_count[n]
                    # [c * vocab_size] â€” flat scores for this stream
                    flat = log_probs[s:s + c].reshape(-1)
                    k = min(beam, flat.numel())
                    topk_vals, topk_idx = flat.topk(k)

                    for i in range(k):
                        hyp_idx = int(topk_idx[i] // vocab_size)
                        tok = int(topk_idx[i] % vocab_size)
                        hyp = A[n][hyp_idx]

                        new_ys = hyp.ys[:]
                        new_ts = hyp.timestamp[:]
                        new_lm_score = hyp.lm_score
                        new_lm_state = hyp.lm_state
                        if tok not in (blank_id, unk_id):
                            new_ys.append(tok)
                            new_ts.append(frame_offsets[n] + t)
                            any_nonblank = True
                            # KenLM shallow fusion: advance n-gram state
                            if lm_scorer is not None and new_lm_state is not None:
                                lm_delta, new_lm_state = lm_scorer.score_token(
                                    tok, new_lm_state
                                )
                                new_lm_score = new_lm_score + lm_delta

                        new_beams[n].add(_Hypothesis(
                            ys=new_ys,
                            log_prob=topk_vals[i].reshape(1),
                            timestamp=new_ts,
                            lm_score=new_lm_score,
                            lm_state=new_lm_state,
                        ))

                    new_beams[n] = new_beams[n].topk(beam, lm_scale=lm_scale)

                beams = new_beams

                # If no stream emitted non-blank, no point continuing inner loop
                if not any_nonblank:
                    break

        # Advance per-stream frame offsets by valid frames in this chunk
        for n in range(N):
            frame_offsets[n] += min(int(h_lens_chunk[n].item()), T_chunk)

        return BeamSearchStreamingState(beams=beams, frame_offsets=frame_offsets)

    def streaming_beam_search(
        self,
        h_all: Tensor,
        h_lens: Tensor,
        beam: int = 4,
        temperature: float = 1.0,
        blank_penalty: float = 0.0,
        length_norm: bool = True,
        lm_scorer: Optional[Any] = None,
        lm_scale: float = 0.0,
        max_sym_per_frame: int = 1,
    ) -> List[List[int]]:
        """Non-streaming beam search on full encoder output (offline convenience).

        Equivalent to init_beam_search_state + streaming_beam_search_chunk once.

        Args:
            h_all: [N, T, D] per-speaker encoder output.
            h_lens: [N] valid frame counts.
            beam: beam size.
            temperature: softmax temperature.
            blank_penalty: penalty for blank token.
            length_norm: if True, pick best hypothesis by length-normalised score.
            lm_scorer: optional ``KenLmScorer`` for n-gram shallow fusion.
            lm_scale: interpolation weight for LM scores.
        Returns:
            hyps: list of N token lists (context prefix stripped).
        """
        N = h_all.shape[0]
        bs_state = self.init_beam_search_state(
            N, h_all.device, beam, lm_scorer=lm_scorer
        )
        bs_state = self.streaming_beam_search_chunk(
            h_all, h_lens, bs_state,
            beam=beam, temperature=temperature, blank_penalty=blank_penalty,
            lm_scorer=lm_scorer, lm_scale=lm_scale,
            max_sym_per_frame=max_sym_per_frame,
        )
        context_size = self.asr_model.decoder.context_size
        return [
            bs_state.beams[n].get_most_probable(
                length_norm=length_norm, lm_scale=lm_scale
            ).ys[context_size:]
            for n in range(N)
        ]

    def streaming_step(
        self,
        audio_chunk: Tensor,
        state: StreamingState,
        spk_activity_chunk: Optional[Tensor] = None,
        sample_lens: Optional[Tensor] = None,
        decode: bool = True,
        beam: int = 1,
        temperature: float = 1.0,
        blank_penalty: float = 0.0,
        max_sym_per_frame: int = 1,
        lm_scorer: Optional[Any] = None,
        lm_scale: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Optional[List[List[int]]], StreamingState]:
        """Unified streaming inference step: audio â†’ encoder â†’ optional decode.

        This is the primary API for real-time streaming inference.
        Wires streaming_forward_from_audio() + streaming_greedy_decode_chunk()
        (or streaming_beam_search_chunk when beam > 1) into a single call
        with persistent state.
        When spk_activity_chunk is omitted and pyannote is available, this API
        auto-runs diarization and HuBERT feature extraction in parallel on CUDA.

        Args:
            audio_chunk: [B, T_samples] raw audio chunk
            state: StreamingState from previous step (or init_state())
            spk_activity_chunk: [B, T_act, S] pre-computed speaker activity.
                If None and pyannote_diar is available, activity is inferred
                from audio_chunk in no_grad mode (with CUDA stream overlap when possible).
            sample_lens: [B] valid samples in audio_chunk (optional)
            decode: if True, also run RNNT decode and return token deltas
            beam: 1 = greedy (default, lowest latency); > 1 = modified beam
                search with `beam` active hypotheses per stream.
            temperature: softmax temperature for beam search.
            blank_penalty: penalty subtracted from blank logit.
            max_sym_per_frame: max non-blank symbols emitted per encoder frame
                (both greedy and beam search). Set to 2+ for tonal languages
                (e.g. Vietnamese) where burst tokens are common.
                Default 1 matches original behaviour.
            lm_scorer: optional ``KenLmScorer`` for n-gram shallow fusion
                during beam search (beam > 1 only; ignored for greedy).
            lm_scale: interpolation weight for LM scores.

        Returns:
            h_all_chunk: [B*S, T_out, D] per-speaker encoder output
            h_lens_chunk: [B*S] valid frame counts
            token_deltas: list of B*S lists of NEW tokens emitted this step
                          (None if decode=False).  For beam > 1, reports
                          delta from the current best hypothesis â€” tokens
                          may shift if a different beam becomes best in a
                          later chunk.
            new_state: updated StreamingState (includes decoder / beam state)
        """
        # Encoder step
        pyannote_diar = getattr(self, "pyannote_diar", None)
        if spk_activity_chunk is None and pyannote_diar is not None:
            # Auto diar branch: run pyannote and HuBERT feature branch in
            # parallel on CUDA when available, then align activity to ASR time.
            prepared, inferred_activity = (
                StreamingMultiSpeakerEncoder._prepare_hubert_and_auto_activity(
                    self,
                    audio_chunk=audio_chunk,
                    state=state,
                    pyannote_diar=pyannote_diar,
                    sample_lens=sample_lens,
                )
            )
            h_all_chunk, h_lens_chunk, new_state = self.merge_prepared_chunk(
                prepared=prepared,
                spk_activity_chunk=inferred_activity,
                state=state,
            )
        elif sample_lens is None:
            h_all_chunk, h_lens_chunk, new_state = self.streaming_forward_from_audio(
                audio_chunk, spk_activity_chunk, state
            )
        else:
            try:
                h_all_chunk, h_lens_chunk, new_state = (
                    self.streaming_forward_from_audio(
                        audio_chunk,
                        spk_activity_chunk,
                        state,
                        sample_lens=sample_lens,
                    )
                )
            except TypeError:
                # Backward compatibility for lightweight test shims / legacy
                # wrappers exposing streaming_forward_from_audio(audio, act, state).
                h_all_chunk, h_lens_chunk, new_state = (
                    self.streaming_forward_from_audio(
                        audio_chunk, spk_activity_chunk, state
                    )
                )

        token_deltas = None
        if decode:
            N = h_all_chunk.shape[0]  # B * S
            device = h_all_chunk.device

            if beam > 1:
                # â”€â”€ Beam search path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if new_state.beam_search_state is None:
                    new_state.beam_search_state = self.init_beam_search_state(
                        N, device, beam, lm_scorer=lm_scorer,
                    )

                context_size = self.asr_model.decoder.context_size
                # Record real-token count in current best hypothesis per stream
                prev_best_len = [
                    max(0, len(new_state.beam_search_state.beams[n]
                                .get_most_probable(lm_scale=lm_scale).ys)
                        - context_size)
                    for n in range(N)
                ]

                new_state.beam_search_state = self.streaming_beam_search_chunk(
                    h_all_chunk, h_lens_chunk, new_state.beam_search_state,
                    beam=beam, temperature=temperature, blank_penalty=blank_penalty,
                    lm_scorer=lm_scorer, lm_scale=lm_scale,
                    max_sym_per_frame=max_sym_per_frame,
                )

                # Delta = new tokens appended to current best hypothesis
                token_deltas = [
                    new_state.beam_search_state.beams[n]
                    .get_most_probable(lm_scale=lm_scale)
                    .ys[context_size + prev_best_len[n]:]
                    for n in range(N)
                ]
            else:
                # â”€â”€ Greedy path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if new_state.decoder_state is None:
                    new_state.decoder_state = self.init_decoder_state(N, device)

                prev_lens = [len(h) for h in new_state.decoder_state.hyps]

                new_state.decoder_state = self.streaming_greedy_decode_chunk(
                    h_all_chunk, h_lens_chunk, new_state.decoder_state,
                    max_sym_per_frame=max_sym_per_frame,
                )

                token_deltas = [
                    new_state.decoder_state.hyps[n][prev_lens[n]:]
                    for n in range(len(prev_lens))
                ]

        return h_all_chunk, h_lens_chunk, token_deltas, new_state

    def streaming_forward_full(
        self,
        source: Tensor,
        spk_activity: Optional[Tensor] = None,
        chunk_frames: int = 32,
    ) -> Tuple[Tensor, Tensor]:
        """Concatenate chunked outputs into single tensors for comparison/debug."""
        h_all_chunks, h_lens_chunks = self.streaming_forward(
            source, spk_activity, chunk_frames
        )

        if len(h_all_chunks) == 0:
            batch_size = source.shape[0]
            D = max(self._zipformer.encoder_dim)
            return (
                torch.zeros(
                    batch_size * self.S, 0, D, device=source.device,
                ),
                torch.zeros(batch_size * self.S, dtype=torch.long, device=source.device),
            )

        h_all = torch.cat(h_all_chunks, dim=1)
        h_lens = sum(h_lens_chunks)
        return h_all, h_lens
