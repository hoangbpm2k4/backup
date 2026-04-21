"""
Multi-speaker RNNT model wrapper with NeMo-style pre-ASR stream routing.

Wraps an existing AsrModel (which contains HubertModel as encoder)
and uses SpeakerKernel-based pre-ASR stream routing to produce
per-speaker acoustic streams before RNNT loss.

Key design:
  - Reuses asr_model.encoder (HubertModel) for shared frontend + Zipformer2.
  - Reuses asr_model.forward_transducer() for RNNT loss computation.
  - Does NOT modify any existing module's public API.
  - Pre-ASR routing: SpeakerKernel applies target-speaker masks before
    Zipformer2 encoding (NVIDIA SSA / NeMo style).
  - Each speaker stream is routed independently via block_callbacks.
  - Legacy injection paths are removed in this routing-only build.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import k2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from speaker_modules import PyAnnoteSegmentationWrapper, SpeakerKernel


def _detach_state(state):
    """Recursively detach tensors in a nested dict/list/tuple state.

    Preserves container type: lists stay lists, tuples stay tuples
    (important because conv_state uses tuples for its nested structure).
    """
    if isinstance(state, Tensor):
        return state.detach()
    if isinstance(state, dict):
        return {k: _detach_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        detached = [_detach_state(v) for v in state]
        return type(state)(detached)
    return state


class MultiSpeakerRnntModel(nn.Module):
    """
    Multi-speaker RNNT wrapper around a pretrained AsrModel.

    Architecture (pre-ASR stream routing):
        source [B, T_samples]
            â†’ StreamingConvFrontend â†’ HuBERT proj
            â†’ SpeakerKernel (target-speaker mask via block_callbacks)
            â†’ Zipformer2.streaming_forward (per-stream)
            â†’ asr_model.forward_transducer(H_selected, y_all) â†’ losses

    Args:
        asr_model: pretrained AsrModel instance (contains HubertModel encoder,
                   decoder, joiner, simple_am_proj, simple_lm_proj).
        num_speakers: maximum number of simultaneous speakers (S).
    """

    def __init__(
        self,
        asr_model: nn.Module,
        num_speakers: int = 3,
        # Pretrained pyannote segmentation as diar head
        use_pyannote_diar: bool = False,
        pyannote_weight_dir: str = "weight",
        freeze_pyannote: bool = True,
        # Paper-style: split by target-speaker mask before ASR encoder
        enable_pre_asr_stream_routing: bool = False,
        pre_asr_active_threshold: float = 0.5,
        # SpeakerKernel config (NeMo-matching)
        spk_kernel_dropout: float = 0.5,
        bg_threshold: float = 0.5,
        **legacy_kwargs,
    ):
        super().__init__()

        legacy_keys = {
            "adapter_dim",
            "adapter_layers",
            "adapter_conv_kernel",
            "adapter_nhead",
            "enable_mid_injection",
            "injection_split_idx",
            "enable_pre_injection",
            "injector_cond_dim",
            "injector_adapter_layers",
            "slot_dim",
            "ema_decay",
            "update_tau",
            "use_slot_memory",
            "slot_update_mode",
            "use_pit",
            "use_feature_pit",
            "use_slot_cross_attn",
        }
        unknown_legacy = sorted(set(legacy_kwargs.keys()) - legacy_keys)
        if unknown_legacy:
            raise TypeError(
                "Unexpected keyword arguments for MultiSpeakerRnntModel: "
                + ", ".join(unknown_legacy)
            )
        if bool(legacy_kwargs.get("enable_mid_injection", False)) or bool(
            legacy_kwargs.get("enable_pre_injection", False)
        ):
            raise RuntimeError(
                "Legacy mid/pre injection paths were removed. "
                "Use enable_pre_asr_stream_routing=True."
            )

        self.asr_model = asr_model
        self.S = num_speakers

        # Infer encoder_dim from asr_model.simple_am_proj
        # simple_am_proj: Linear(encoder_dim â†’ vocab_size)
        encoder_dim = self._infer_encoder_dim(asr_model)
        self._encoder_dim = encoder_dim

        # Diarization head: pyannote (pretrained, raw audio) or oracle/external
        # activity.
        self.pyannote_diar = None
        self._use_pyannote_diar = use_pyannote_diar

        if use_pyannote_diar:
            self.pyannote_diar = PyAnnoteSegmentationWrapper(
                weight_dir=pyannote_weight_dir,
                num_speakers=num_speakers,
                freeze=freeze_pyannote,
            )

        # Pre-encode injection also needs pyannote (activity must be available
        # before encoder runs â€" only pyannote provides this since it takes raw
        # audio, not encoder output).
        # Cache StreamingConvFrontend once at construction time so that
        # _compute_causal_trim() (which runs a dummy conv forward) is never
        # repeated per batch-forward and never runs on CPU when weights are
        # already on CUDA. Falls back to lazy creation if the model doesn't
        # expose feature_extractor yet (e.g. lightweight test mocks).
        from wav2vec2_module import StreamingConvFrontend
        try:
            self._streaming_frontend: Optional[StreamingConvFrontend] = (
                StreamingConvFrontend(self.asr_model.encoder.feature_extractor)
            )
        except AttributeError:
            self._streaming_frontend = None

        self.enable_pre_asr_stream_routing = enable_pre_asr_stream_routing
        self.pre_asr_active_threshold = pre_asr_active_threshold

        # Paper Eq.1+2: SpeakerKernel for pre-ASR stream routing
        # Matches NeMo SpeakerKernelMixin: spk_kernel + bg_kernel, no gate.
        # Multi-layer injection via Zipformer2 block_callbacks (NeMo-style).
        self._spk_targets: Optional[Tensor] = None
        self._bg_spk_targets: Optional[Tensor] = None
        self._spk_kernel_layers: List[int] = [0]
        self.spk_kernels = nn.ModuleDict()
        # NOTE: speaker_kernel is stored as a plain attribute (not nn.Module
        # registration) to avoid double-registering the same SpeakerKernel
        # instance that already lives inside spk_kernels ModuleDict.
        # We use object.__setattr__ to bypass nn.Module.__setattr__.
        object.__setattr__(self, "speaker_kernel", None)
        if enable_pre_asr_stream_routing:
            spk_kernel_layers = [0]  # default: inject at block 0 only
            self._spk_kernel_layers = spk_kernel_layers
            try:
                zipformer = self.asr_model.encoder.encoder  # Zipformer2
                for layer_idx in spk_kernel_layers:
                    enc_dims = getattr(zipformer, "encoder_dim", None)
                    if enc_dims is not None and isinstance(enc_dims, (list, tuple)):
                        dim_at_layer = enc_dims[layer_idx]
                    else:
                        dim_at_layer = encoder_dim
                    self.spk_kernels[str(layer_idx)] = SpeakerKernel(
                        dim=dim_at_layer, dropout=spk_kernel_dropout,
                        add_bg_kernel=True, bg_threshold=bg_threshold,
                    )
                # Ensure block_callbacks exists on the zipformer
                if not hasattr(zipformer, "block_callbacks"):
                    zipformer.block_callbacks = {}
                self._register_kernel_callbacks(zipformer)
            except AttributeError:
                # Fallback for mocks without encoder.encoder
                self.spk_kernels["0"] = SpeakerKernel(
                    dim=encoder_dim, dropout=spk_kernel_dropout,
                    add_bg_kernel=True, bg_threshold=bg_threshold,
                )
            # Backward-compat: speaker_kernel points to first kernel
            _first_kernel = (
                self.spk_kernels["0"]
                if "0" in self.spk_kernels
                else next(iter(self.spk_kernels.values()))
            )
            object.__setattr__(self, "speaker_kernel", _first_kernel)

        # Legacy injection modules were removed in routing-only mode.

    @staticmethod
    def _infer_encoder_dim(asr_model: nn.Module) -> int:
        """Infer encoder output dim from the real AsrModel or lightweight mocks."""
        simple_am_proj = getattr(asr_model, "simple_am_proj", None)
        if simple_am_proj is not None and hasattr(simple_am_proj, "weight"):
            return int(simple_am_proj.weight.shape[1])

        hubert = getattr(asr_model, "encoder", None)
        zipformer = getattr(hubert, "encoder", None)
        encoder_dim = getattr(zipformer, "encoder_dim", None)

        if isinstance(encoder_dim, (list, tuple)) and len(encoder_dim) > 0:
            return int(max(encoder_dim))
        if isinstance(encoder_dim, int):
            return int(encoder_dim)

        raise AttributeError(
            "Unable to infer encoder_dim from asr_model. Expected either "
            "asr_model.simple_am_proj.weight or "
            "asr_model.encoder.encoder.encoder_dim."
        )

    @property
    def encoder(self) -> nn.Module:
        return self.asr_model.encoder

    def set_speaker_targets(self, spk_targets: Optional[Tensor],
                            bg_spk_targets: Optional[Tensor] = None) -> None:
        """Set speaker targets for callback-based kernel injection."""
        self._spk_targets = spk_targets
        self._bg_spk_targets = bg_spk_targets

    def clear_speaker_targets(self) -> None:
        """Clear speaker targets after encoder forward."""
        self._spk_targets = None
        self._bg_spk_targets = None

    def _register_kernel_callbacks(self, zipformer) -> None:
        """Register speaker kernel callbacks on Zipformer2 encoder stacks."""
        if not hasattr(zipformer, "block_callbacks"):
            zipformer.block_callbacks = {}
        for layer_idx in self._spk_kernel_layers:
            kernel = self.spk_kernels[str(layer_idx)]
            def make_callback(k):
                def callback(x):
                    # x: [T, B, D] — time-first (Zipformer convention)
                    if self._spk_targets is None:
                        return x
                    xt = x.transpose(0, 1)  # [B, T, D]
                    xt = k(xt, self._spk_targets, self._bg_spk_targets)
                    return xt.transpose(0, 1)  # [T, B, D]
                return callback
            zipformer.block_callbacks[layer_idx] = make_callback(kernel)

    def _get_streaming_frontend(self):
        if self._streaming_frontend is None:
            from wav2vec2_module import StreamingConvFrontend

            self._streaming_frontend = StreamingConvFrontend(
                self.asr_model.encoder.feature_extractor
            )
        return self._streaming_frontend

    def _compute_streaming_feature_chunk_widths(
        self,
        total_samples: int,
        chunk_samples: int,
        streaming_frontend,
    ) -> List[int]:
        widths: List[int] = []
        cache_len = 0
        is_first_chunk = True
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk_len = end - start
            full_len = cache_len + chunk_len
            total_frames = streaming_frontend._frames_from_samples(full_len)
            cache_frames = streaming_frontend._frames_from_samples(cache_len)
            emitted = total_frames - cache_frames
            if is_first_chunk:
                emitted = max(emitted - streaming_frontend.causal_left_trim, 0)
                is_first_chunk = False
            widths.append(emitted)
            next_start = total_frames * streaming_frontend.total_stride
            cache_len = max(full_len - next_start, 0)
        return widths

    @staticmethod
    def sort_speakers_by_onset(
        activity: Tensor,
        y_per_speaker: Optional[list] = None,
    ) -> Tuple[Tensor, Optional[list], List[List[int]]]:
        """Sort GT speakers by onset time per sample (arrival-order).

        This canonicalizes the speaker ordering so slot 0 always gets the
        speaker that speaks first, slot 1 the second, etc. Applied BEFORE
        training to stabilize slot assignment and reduce the need for PIT.

        Args:
            activity: [B, T, S] ground truth speaker activity.
            y_per_speaker: optional list of S k2.RaggedTensor transcripts.

        Returns:
            sorted_activity: [B, T, S] with speakers reordered by onset.
            sorted_y: list of S k2.RaggedTensor (or None) with same reorder.
            sort_perms: list of B lists, each length S â€" the applied permutation.
                sort_perms[b][new_slot] = original_speaker_idx.
        """
        B, T, S = activity.shape
        device = activity.device
        eps = 1e-8

        # Per-sample per-speaker onset: first frame where activity > 0.5
        # For inactive speakers, onset = T (pushed to the end)
        active = activity > 0.5  # [B, T, S]
        # onset_frame[b, s] = first t where active[b, t, s] is True
        onset_frames = torch.full((B, S), T, dtype=torch.long, device=device)
        for s in range(S):
            col = active[:, :, s]  # [B, T]
            # argmax on bool gives first True; if no True, gives 0
            has_any = col.any(dim=1)  # [B]
            first_t = col.float().argmax(dim=1)  # [B]
            onset_frames[:, s] = torch.where(has_any, first_t, torch.tensor(T, device=device))

        # Sort by onset (stable sort preserves order for ties)
        sort_idx = onset_frames.argsort(dim=1, stable=True)  # [B, S]

        sort_perms = sort_idx.tolist()

        # Reorder activity
        sorted_activity = activity.gather(
            2, sort_idx.unsqueeze(1).expand(B, T, S)
        )

        # Reorder transcripts if provided
        sorted_y = None
        if y_per_speaker is not None:
            import k2
            sorted_y = []
            for s in range(S):
                all_values = []
                all_row_splits = [0]
                offset = 0
                for b in range(B):
                    src_s = sort_perms[b][s]
                    rs = y_per_speaker[src_s].shape.row_splits(1)
                    start = rs[b]
                    end = rs[b + 1]
                    vals = y_per_speaker[src_s].values[start:end]
                    all_values.append(vals)
                    offset += (end - start).item()
                    all_row_splits.append(offset)

                ydev = y_per_speaker[0].values.device
                values_cat = torch.cat(all_values, dim=0).to(ydev)
                rs_tensor = torch.tensor(all_row_splits, dtype=torch.int32, device=ydev)
                ragged = k2.RaggedTensor(
                    k2.ragged.create_ragged_shape2(
                        row_splits=rs_tensor, cached_tot_size=values_cat.shape[0]
                    ),
                    values_cat,
                )
                sorted_y.append(ragged)

        return sorted_activity, sorted_y, sort_perms

    def _apply_per_sample_perm(
        self,
        y_per_speaker: List,
        perms: List[List[int]],
        B: int,
        S: int,
    ) -> List:
        """Re-order y_per_speaker according to per-sample permutations.

        Args:
            y_per_speaker: list of S k2.RaggedTensor, each with B entries.
            perms: list of B lists, each length S. perms[b][slot] = text_idx.
            B: batch size.
            S: number of speakers.

        Returns:
            y_permuted: list of S k2.RaggedTensor, each with B entries,
                where entry b of speaker slot s comes from
                y_per_speaker[perms[b][s]][b].
        """
        import k2

        y_permuted = []
        for s in range(S):
            all_values = []
            all_row_splits = [0]
            offset = 0
            for b in range(B):
                src_s = perms[b][s]
                rs = y_per_speaker[src_s].shape.row_splits(1)
                start = rs[b]
                end = rs[b + 1]
                vals = y_per_speaker[src_s].values[start:end]
                all_values.append(vals)
                offset += (end - start).item()
                all_row_splits.append(offset)

            device = y_per_speaker[0].values.device
            values_cat = torch.cat(all_values, dim=0).to(device)
            rs_tensor = torch.tensor(all_row_splits, dtype=torch.int32, device=device)
            ragged = k2.RaggedTensor(
                k2.ragged.create_ragged_shape2(
                    row_splits=rs_tensor, cached_tot_size=values_cat.shape[0]
                ),
                values_cat,
            )
            y_permuted.append(ragged)
        return y_permuted

    def _permute_activity_per_sample(
        self,
        A: Tensor,
        perms: List[List[int]],
    ) -> Tensor:
        """Permute activity [B, T, S] per-sample along speaker dim.

        Args:
            A: [B, T, S] activity tensor.
            perms: list of B lists, each length S. perms[b][slot] = src_speaker.

        Returns:
            A_perm: [B, T, S] with A_perm[b, :, slot] = A[b, :, perms[b][slot]].
        """
        B, T, S = A.shape
        perm_idx = torch.tensor(perms, device=A.device, dtype=torch.long)  # [B, S]
        perm_idx = perm_idx.unsqueeze(1).expand(B, T, S)  # [B, T, S]
        return A.gather(2, perm_idx)

    def _resample_activity(
        self, A: Tensor, target_len: int
    ) -> Tensor:
        """
        Resample speaker activity to match encoder output frame rate.

        Uses F.interpolate(mode='nearest') for deterministic, causal-friendly
        resampling that preserves binary activity semantics. Unlike
        adaptive_avg_pool1d which blurs binary values, nearest interpolation
        keeps 0/1 boundaries intact.

        Consistent between offline and chunked streaming paths.

        Args:
            A: [B, T_act, S]
        Returns:
            [B, target_len, S]
        """
        T_act = A.shape[1]
        if T_act == target_len:
            return A

        # F.interpolate expects [B, C, T], activity is [B, T, S]
        out = F.interpolate(
            A.transpose(1, 2).float(),
            size=target_len,
            mode="nearest",
        ).transpose(1, 2)
        return out.to(A.dtype)

    @staticmethod
    def _samples_to_feature_lengths(
        valid_samples: Tensor, feature_extractor: nn.Module
    ) -> Tensor:
        """Compute conv-frontend output lengths from valid waveform samples."""
        if not hasattr(feature_extractor, "conv_layers"):
            raise AttributeError(
                "feature_extractor must expose conv_layers to compute feature lengths."
            )
        lengths = valid_samples.to(dtype=torch.long)
        for block in feature_extractor.conv_layers:
            conv = block[0]
            k = int(conv.kernel_size[0])
            s = int(conv.stride[0])
            lengths = torch.where(
                lengths >= k,
                (lengths - k) // s + 1,
                torch.zeros_like(lengths),
            )
        return lengths

    def forward_train_routing(
        self,
        source: Tensor,
        y_per_speaker: List[k2.RaggedTensor],
        spk_activity: Tensor,
        padding_mask: Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.25,
        target_speaker_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Full-sequence training path with causal chunk mask in Zipformer.forward().

        Training flow:
          1) Offline HuBERT conv frontend on full sequence (GroupNorm/original path)
          2) Pre-ASR routing over full sequence using SpeakerKernel
          3) Zipformer.forward() (causal chunk mask is applied internally)
          4) RNNT/CTC losses on routed streams
        """
        if not self.enable_pre_asr_stream_routing:
            raise RuntimeError(
                "forward_train_routing() requires enable_pre_asr_stream_routing=True."
            )
        if spk_activity is None:
            raise RuntimeError("spk_activity is required for routing training.")

        hubert = self.asr_model.encoder
        zipformer = hubert.encoder
        speaker_kernel = getattr(self, "speaker_kernel", None)
        if speaker_kernel is None:
            raise RuntimeError(
                "enable_pre_asr_stream_routing=True but speaker_kernel is missing."
            )

        B = source.shape[0]
        device = source.device
        if padding_mask is None:
            padding_mask = torch.zeros_like(source, dtype=torch.bool, device=device)

        valid_samples = (~padding_mask).sum(dim=1).to(dtype=torch.long, device=device)

        # Step 1: offline HuBERT frontend over full sequence.
        raw_features = hubert.feature_extractor(source)  # [B, C, T_feat]
        T_feat = int(raw_features.shape[2])
        x_lens_batch = self._samples_to_feature_lengths(
            valid_samples, hubert.feature_extractor
        ).clamp(min=0, max=T_feat)

        features = raw_features.transpose(1, 2)  # [B, T_feat, D_hubert]
        features = hubert.layer_norm(features)
        if hubert.post_extract_proj is not None:
            features = hubert.post_extract_proj(features)
        features = hubert.dropout_input(features)

        # Step 2: resample activity to feature rate.
        activity_full = self._resample_activity(spk_activity, T_feat)

        # Step 3: resolve routed streams on full sequence.
        stream_batch_idx, stream_slot_idx = self._resolve_pre_asr_stream_pairs(
            activity_full=activity_full,
            target_speaker_ids=target_speaker_ids,
            active_threshold=self.pre_asr_active_threshold,
        )
        N = int(stream_batch_idx.numel())
        if N == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        y_streams, y_lens_streams = self._make_targets_for_stream_pairs(
            y_per_speaker=y_per_speaker,
            stream_batch_idx=stream_batch_idx,
            stream_slot_idx=stream_slot_idx,
            device=device,
        )

        routed_features = features.index_select(0, stream_batch_idx)  # [N, T, D]
        routed_activity = activity_full.index_select(0, stream_batch_idx)  # [N, T, S]

        slot_index = stream_slot_idx.view(N, 1, 1).expand(-1, T_feat, 1)
        spk_targets = routed_activity.gather(2, slot_index).squeeze(-1)  # [N, T]
        bg_targets = torch.stack(
            [
                speaker_kernel.build_bg_mask(activity_full[b : b + 1], int(s))[0]
                for b, s in zip(stream_batch_idx.tolist(), stream_slot_idx.tolist())
            ],
            dim=0,
        )  # [N, T]

        routed_features = speaker_kernel(routed_features, spk_targets, bg_targets)
        x_lens_stream = x_lens_batch.index_select(0, stream_batch_idx)
        src_key_padding_mask = (
            torch.arange(T_feat, device=device).unsqueeze(0)
            >= x_lens_stream.unsqueeze(1)
        )  # [N, T_feat]

        # Step 4: full-sequence Zipformer.forward() with internal causal masking.
        x = routed_features.transpose(0, 1)  # [T_feat, N, D]
        enc_out_tbd, enc_lens = zipformer(
            x, x_lens_stream, src_key_padding_mask=src_key_padding_mask
        )
        enc_out = enc_out_tbd.transpose(0, 1)  # [N, T_out, D]

        # Step 5: scatter to [B*S, T_out, D] layout (inactive streams remain zero).
        T_out = int(enc_out.shape[1])
        D_out = int(enc_out.shape[2])
        flat_stream_idx = stream_batch_idx * int(self.S) + stream_slot_idx  # [N]
        H_all = torch.zeros(
            B * int(self.S), T_out, D_out, device=device, dtype=enc_out.dtype
        )
        enc_out_lens_all = torch.zeros(
            B * int(self.S), dtype=enc_lens.dtype, device=device
        )
        H_all[flat_stream_idx] = enc_out
        enc_out_lens_all[flat_stream_idx] = enc_lens

        # Loss path uses the routed order to match y_streams rows.
        H_loss = H_all.index_select(0, flat_stream_idx)
        enc_lens_loss = enc_out_lens_all.index_select(0, flat_stream_idx)

        # Step 6: RNNT + optional CTC.
        simple_loss, pruned_loss = self.asr_model.forward_transducer(
            encoder_out=H_loss,
            encoder_out_lens=enc_lens_loss,
            y=y_streams,
            y_lens=y_lens_streams,
            prune_range=prune_range,
            am_scale=am_scale,
            lm_scale=lm_scale,
            delay_penalty=getattr(self, "_delay_penalty", 0.0),
        )

        if getattr(self.asr_model, "use_ctc", False) and hasattr(
            self.asr_model, "ctc_output"
        ):
            ctc_loss = self.asr_model.forward_ctc(
                encoder_out=H_loss,
                encoder_out_lens=enc_lens_loss,
                targets=y_streams.values,
                target_lengths=y_lens_streams,
            )
        else:
            ctc_loss = torch.tensor(0.0, device=device)

        return simple_loss, pruned_loss, ctc_loss

    def forward(
        self,
        source: Tensor,
        y_per_speaker: List[k2.RaggedTensor],
        spk_activity: Optional[Tensor] = None,
        target_speaker_ids: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        raw_audio: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        del (
            source,
            y_per_speaker,
            spk_activity,
            target_speaker_ids,
            padding_mask,
            prune_range,
            am_scale,
            lm_scale,
            raw_audio,
        )
        raise RuntimeError(
            "forward() is no longer supported. "
            "Use forward_train_routing() for training."
        )

    def chunked_forward_train(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "chunked_forward_train() is no longer supported. "
            "Use chunked_forward_train_streaming() instead."
        )

    @property
    def conv_norm_mode(self) -> str:
        """Detect conv frontend norm mode for debug and test visibility."""
        conv_model = self.asr_model.encoder.feature_extractor
        first_block = conv_model.conv_layers[0]
        for layer in first_block:
            if isinstance(layer, nn.GroupNorm):
                return "group_norm"
            if isinstance(layer, nn.Sequential):
                for sub in layer:
                    if isinstance(sub, nn.LayerNorm):
                        return "layer_norm"
        return "none"

    def chunked_forward_train_streaming(
        self,
        source: Tensor,
        y_per_speaker: List[k2.RaggedTensor],
        target_speaker_ids: Optional[Tensor] = None,
        chunk_samples: int = 5120,
        chunk_size_frames: Optional[int] = None,
        left_context_frames: Optional[int] = None,
        spk_activity: Optional[Tensor] = None,
        teacher_activity: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        allow_unsafe_streaming: bool = False,
        activity_teacher_forcing_ratio: float = 0.0,
        raw_audio: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        del chunk_size_frames, allow_unsafe_streaming, activity_teacher_forcing_ratio, raw_audio

        B = source.shape[0]
        device = source.device
        T_total = source.shape[1]
        teacher_activity = teacher_activity if teacher_activity is not None else spk_activity

        hubert = self.asr_model.encoder
        zipformer = hubert.encoder
        if not getattr(zipformer, "causal", False):
            raise RuntimeError("Streaming training requires encoder.causal=True")

        odf = getattr(zipformer, "output_downsampling_factor", 1)
        assert odf == 1, (
            "Multi-speaker streaming requires output_downsampling_factor=1, "
            f"got {odf}."
        )

        active_left_context_frames = (
            int(left_context_frames)
            if left_context_frames is not None and left_context_frames >= 0
            else int(getattr(zipformer, "left_context_frames", [0])[0])
        )
        if active_left_context_frames < 0:
            raise ValueError(
                f"left_context_frames must be >= 0, got {active_left_context_frames}"
            )

        if not self.enable_pre_asr_stream_routing:
            raise RuntimeError(
                "chunked_forward_train_streaming() supports only pre-ASR stream routing. "
                "Set enable_pre_asr_stream_routing=True."
            )
        if teacher_activity is None:
            raise RuntimeError(
                "teacher_activity/spk_activity is required for pre-ASR routing training."
            )

        streaming_frontend = self._get_streaming_frontend()
        teacher_chunk_widths = self._compute_streaming_feature_chunk_widths(
            total_samples=T_total,
            chunk_samples=chunk_samples,
            streaming_frontend=streaming_frontend,
        )
        teacher_total_frames = sum(teacher_chunk_widths)
        teacher_activity_full = self._resample_activity(
            teacher_activity, teacher_total_frames
        )

        if padding_mask is None:
            padding_mask = torch.zeros(
                B, T_total, dtype=torch.bool, device=device
            )

        return self._chunked_forward_pre_asr_stream_routing(
            source=source,
            y_per_speaker=y_per_speaker,
            padding_mask=padding_mask,
            chunk_samples=chunk_samples,
            active_left_context_frames=active_left_context_frames,
            teacher_activity_full=teacher_activity_full,
            teacher_chunk_widths=teacher_chunk_widths,
            target_speaker_ids=target_speaker_ids,
            prune_range=prune_range,
            am_scale=am_scale,
            lm_scale=lm_scale,
            streaming_frontend=streaming_frontend,
            hubert=hubert,
            zipformer=zipformer,
        )

    def _make_batched_targets(
        self, y_per_speaker: List[k2.RaggedTensor], device: torch.device
    ) -> Tuple[k2.RaggedTensor, Tensor]:
        """
        Flatten S per-speaker ragged tensors into one ragged tensor
        with B*S entries, matching the [B*S, T, D] layout of H_all.

        Layout: [b0_s0, b0_s1, ..., b0_sS-1, b1_s0, b1_s1, ...]

        Args:
            y_per_speaker: list of S k2.RaggedTensor, each with B entries.
        Returns:
            y_all: k2.RaggedTensor with B*S entries
            y_lens_all: [B*S] tensor of target lengths
        """
        import k2

        S = len(y_per_speaker)
        B = y_per_speaker[0].dim0

        all_values = []
        all_row_splits = [0]
        offset = 0

        for b in range(B):
            for s in range(S):
                # Get row_splits for speaker s
                rs = y_per_speaker[s].shape.row_splits(1)
                start = rs[b]
                end = rs[b + 1]
                vals = y_per_speaker[s].values[start:end]
                all_values.append(vals)
                offset += (end - start).item()
                all_row_splits.append(offset)

        if len(all_values) > 0:
            values_cat = torch.cat(all_values, dim=0).to(device)
        else:
            values_cat = torch.zeros(0, dtype=torch.int32, device=device)
        row_splits_tensor = torch.tensor(
            all_row_splits, dtype=torch.int32, device=device
        )
        y_all = k2.RaggedTensor(
            k2.ragged.create_ragged_shape2(
                row_splits=row_splits_tensor, cached_tot_size=values_cat.shape[0]
            ),
            values_cat,
        )

        # Compute lengths
        y_lens_all = torch.diff(row_splits_tensor).to(device)

        return y_all, y_lens_all

    def _make_selected_targets(
        self,
        y_per_speaker: List[k2.RaggedTensor],
        target_speaker_ids: torch.Tensor,
        device: torch.device,
    ) -> Tuple[k2.RaggedTensor, Tensor]:
        """
        Build one target sequence per sample using selected speaker ids.

        Args:
            y_per_speaker: list of S k2.RaggedTensor, each with B entries.
            target_speaker_ids: [B] tensor; selected speaker slot for each sample.
        Returns:
            y_selected: k2.RaggedTensor with B entries.
            y_lens: [B] target lengths.
        """
        import k2

        S = len(y_per_speaker)
        B = y_per_speaker[0].dim0
        if target_speaker_ids.numel() != B:
            raise ValueError(
                f"target_speaker_ids must have {B} elements, got {target_speaker_ids.numel()}"
            )

        all_values = []
        all_row_splits = [0]
        offset = 0
        for b in range(B):
            s = int(target_speaker_ids[b].item())
            if s < 0 or s >= S:
                raise ValueError(
                    f"target_speaker_ids[{b}]={s} is out of range [0, {S - 1}]"
                )
            rs = y_per_speaker[s].shape.row_splits(1)
            start = rs[b]
            end = rs[b + 1]
            vals = y_per_speaker[s].values[start:end]
            all_values.append(vals)
            offset += int((end - start).item())
            all_row_splits.append(offset)

        if len(all_values) > 0:
            values_cat = torch.cat(all_values, dim=0).to(device)
        else:
            values_cat = torch.zeros(0, dtype=torch.int32, device=device)
        row_splits_tensor = torch.tensor(
            all_row_splits, dtype=torch.int32, device=device
        )
        y_selected = k2.RaggedTensor(
            k2.ragged.create_ragged_shape2(
                row_splits=row_splits_tensor, cached_tot_size=values_cat.shape[0]
            ),
            values_cat,
        )
        y_lens = torch.diff(row_splits_tensor).to(device)
        return y_selected, y_lens

    def _resolve_pre_asr_stream_pairs(
        self,
        activity_full: Tensor,
        target_speaker_ids: Optional[Tensor],
        active_threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """Resolve which (sample, speaker-slot) streams to run before ASR.

        If target_speaker_ids is provided (SSA single-target mode), returns one
        stream per sample. Otherwise returns all active speaker slots per sample.
        A sample with no active slot falls back to slot 0.

        Args:
            activity_full: [B, T, S] activity mask used for routing.
            target_speaker_ids: [B] optional selected slot ids.
            active_threshold: slot is active iff max_t activity > threshold.

        Returns:
            stream_batch_idx: [N] sample index for each routed stream.
            stream_slot_idx: [N] speaker-slot index for each routed stream.
        """
        B, _, S = activity_full.shape
        device = activity_full.device

        if target_speaker_ids is not None:
            target_speaker_ids = target_speaker_ids.to(device=device, dtype=torch.long)
            if target_speaker_ids.numel() != B:
                raise ValueError(
                    f"target_speaker_ids must have {B} elements, got {target_speaker_ids.numel()}"
                )
            if torch.any(target_speaker_ids < 0) or torch.any(target_speaker_ids >= S):
                raise ValueError(f"target_speaker_ids must be in [0, {S - 1}]")
            stream_batch_idx = torch.arange(B, device=device, dtype=torch.long)
            stream_slot_idx = target_speaker_ids
            return stream_batch_idx, stream_slot_idx

        active = activity_full.amax(dim=1) > float(active_threshold)  # [B, S]
        stream_batch: List[int] = []
        stream_slot: List[int] = []
        for b in range(B):
            slots = torch.nonzero(active[b], as_tuple=False).flatten().tolist()
            if len(slots) == 0:
                slots = [0]
            for s in slots:
                stream_batch.append(b)
                stream_slot.append(int(s))

        if len(stream_batch) == 0:
            # Safety fallback (should not happen due per-sample fallback above).
            stream_batch = list(range(B))
            stream_slot = [0] * B

        stream_batch_idx = torch.tensor(stream_batch, device=device, dtype=torch.long)
        stream_slot_idx = torch.tensor(stream_slot, device=device, dtype=torch.long)
        return stream_batch_idx, stream_slot_idx

    def _make_targets_for_stream_pairs(
        self,
        y_per_speaker: List[k2.RaggedTensor],
        stream_batch_idx: Tensor,
        stream_slot_idx: Tensor,
        device: torch.device,
    ) -> Tuple[k2.RaggedTensor, Tensor]:
        """Build targets for routed streams (one ragged row per stream pair).

        Args:
            y_per_speaker: list of S k2.RaggedTensor, each with B rows.
            stream_batch_idx: [N] sample index for each stream.
            stream_slot_idx: [N] speaker-slot index for each stream.
        Returns:
            y_streams: k2.RaggedTensor with N rows.
            y_lens: [N] target lengths.
        """
        import k2

        if stream_batch_idx.numel() != stream_slot_idx.numel():
            raise ValueError(
                "stream_batch_idx and stream_slot_idx must have same length"
            )

        N = int(stream_batch_idx.numel())
        S = len(y_per_speaker)
        B = y_per_speaker[0].dim0

        all_values = []
        all_row_splits = [0]
        offset = 0

        for i in range(N):
            b = int(stream_batch_idx[i].item())
            s = int(stream_slot_idx[i].item())
            if b < 0 or b >= B:
                raise ValueError(f"stream_batch_idx[{i}]={b} out of range [0, {B - 1}]")
            if s < 0 or s >= S:
                raise ValueError(f"stream_slot_idx[{i}]={s} out of range [0, {S - 1}]")
            rs = y_per_speaker[s].shape.row_splits(1)
            start = rs[b]
            end = rs[b + 1]
            vals = y_per_speaker[s].values[start:end]
            all_values.append(vals)
            offset += int((end - start).item())
            all_row_splits.append(offset)

        if len(all_values) > 0:
            values_cat = torch.cat(all_values, dim=0).to(device)
        else:
            values_cat = torch.zeros(0, dtype=torch.int32, device=device)
        row_splits_tensor = torch.tensor(
            all_row_splits, dtype=torch.int32, device=device
        )
        y_streams = k2.RaggedTensor(
            k2.ragged.create_ragged_shape2(
                row_splits=row_splits_tensor, cached_tot_size=values_cat.shape[0]
            ),
            values_cat,
        )
        y_lens = torch.diff(row_splits_tensor).to(device)
        return y_streams, y_lens

    def _chunked_forward_pre_asr_stream_routing(
        self,
        source: Tensor,
        y_per_speaker: List[k2.RaggedTensor],
        padding_mask: Tensor,
        chunk_samples: int,
        active_left_context_frames: int,
        teacher_activity_full: Tensor,
        teacher_chunk_widths: List[int],
        target_speaker_ids: Optional[Tensor],
        prune_range: int,
        am_scale: float,
        lm_scale: float,
        streaming_frontend,
        hubert: nn.Module,
        zipformer: nn.Module,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Paper-style pre-ASR stream routing with dynamic active stream count.

        Routing is done before Zipformer (NVIDIA SSA-style):
          - select stream pairs (sample, speaker-slot)
          - apply target-speaker activity mask via SpeakerKernel
          - run Zipformer only on selected streams (N <= B*S)
        """
        B = source.shape[0]
        device = source.device
        T_total = source.shape[1]

        speaker_kernel = getattr(self, "speaker_kernel", None)
        if speaker_kernel is None:
            raise RuntimeError(
                "enable_pre_asr_stream_routing=True but speaker_kernel is missing."
            )

        stream_batch_idx, stream_slot_idx = self._resolve_pre_asr_stream_pairs(
            activity_full=teacher_activity_full,
            target_speaker_ids=target_speaker_ids,
            active_threshold=self.pre_asr_active_threshold,
        )
        N = int(stream_batch_idx.numel())
        if N == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        y_streams, y_lens_streams = self._make_targets_for_stream_pairs(
            y_per_speaker=y_per_speaker,
            stream_batch_idx=stream_batch_idx,
            stream_slot_idx=stream_slot_idx,
            device=device,
        )

        conv_state = streaming_frontend.get_init_state(B, device)
        try:
            zip_states = zipformer.get_init_states(
                N, device, left_context_frames=active_left_context_frames
            )
        except TypeError:
            zip_states = zipformer.get_init_states(N, device)

        processed_lens_base = torch.zeros(N, dtype=torch.long, device=device)

        enc_chunks: List[Tensor] = []
        enc_lens_list: List[Tensor] = []
        teacher_width_index = 0
        teacher_feat_offset = 0

        for start in range(0, T_total, chunk_samples):
            end = min(start + chunk_samples, T_total)
            audio_chunk = source[:, start:end]
            is_first_chunk = conv_state[0].shape[2] == 0

            chunk_pad_mask = padding_mask[:, start:end]
            valid_samples = (end - start) - chunk_pad_mask.sum(dim=1).long()
            valid_samples = valid_samples.clamp(min=0)

            raw_features, conv_state, new_feat_lens = streaming_frontend.streaming_forward(
                audio_chunk,
                conv_state,
                sample_lens=valid_samples,
                return_lens=True,
            )

            if is_first_chunk and streaming_frontend.causal_left_trim > 0:
                raw_features = raw_features[:, :, streaming_frontend.causal_left_trim :]

            t_feat = raw_features.shape[2]
            if t_feat == 0:
                teacher_width_index += 1
                conv_state = _detach_state(conv_state)
                continue

            x_lens_batch = new_feat_lens.clamp(min=0, max=t_feat)
            expected_width = teacher_chunk_widths[teacher_width_index]
            if expected_width != t_feat:
                raise RuntimeError(
                    f"Teacher activity chunk width mismatch: expected {expected_width}, got {t_feat}"
                )

            activity_chunk = teacher_activity_full[
                :, teacher_feat_offset : teacher_feat_offset + t_feat, :
            ]
            teacher_feat_offset += t_feat
            teacher_width_index += 1

            features = raw_features.transpose(1, 2)
            features = hubert.layer_norm(features)
            if hubert.post_extract_proj is not None:
                features = hubert.post_extract_proj(features)
            features = hubert.dropout_input(features)

            routed_features = features.index_select(0, stream_batch_idx)
            routed_activity = activity_chunk.index_select(0, stream_batch_idx)

            slot_index = stream_slot_idx.view(N, 1, 1).expand(-1, t_feat, 1)
            spk_targets = routed_activity.gather(2, slot_index).squeeze(-1)

            bg_targets = torch.stack(
                [
                    speaker_kernel.build_bg_mask(
                        activity_chunk[b : b + 1], int(s)
                    )[0]
                    for b, s in zip(stream_batch_idx.tolist(), stream_slot_idx.tolist())
                ],
                dim=0,
            )

            x_lens_stream = x_lens_batch.index_select(0, stream_batch_idx)

            self.set_speaker_targets(spk_targets, bg_targets)

            x_t = routed_features.transpose(0, 1)
            feat_pad = torch.arange(t_feat, device=device).unsqueeze(0) >= x_lens_stream.unsqueeze(1)

            L_base = active_left_context_frames
            valid_lc = processed_lens_base.clamp(max=L_base)
            invalid_lc = L_base - valid_lc
            lc_mask = (
                torch.arange(L_base, device=device).unsqueeze(0)
                < invalid_lc.unsqueeze(1)
            )
            full_mask = torch.cat([lc_mask, feat_pad], dim=1)

            try:
                enc_out_tbd, enc_lens, zip_states = zipformer.streaming_forward(
                    x_t,
                    x_lens_stream,
                    zip_states,
                    full_mask,
                    left_context_frames=active_left_context_frames,
                )
            except TypeError:
                enc_out_tbd, enc_lens, zip_states = zipformer.streaming_forward(
                    x_t, x_lens_stream, zip_states, full_mask
                )
            finally:
                self.clear_speaker_targets()

            enc_chunks.append(enc_out_tbd.transpose(0, 1))
            enc_lens_list.append(enc_lens)
            processed_lens_base = processed_lens_base + enc_lens

            conv_state = _detach_state(conv_state)
            zip_states = [s.detach() for s in zip_states]

        if len(enc_chunks) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        H_all = torch.cat(enc_chunks, dim=1)
        enc_out_lens = torch.stack(enc_lens_list, dim=0).sum(dim=0)

        simple_loss, pruned_loss = self.asr_model.forward_transducer(
            encoder_out=H_all,
            encoder_out_lens=enc_out_lens,
            y=y_streams,
            y_lens=y_lens_streams,
            prune_range=prune_range,
            am_scale=am_scale,
            lm_scale=lm_scale,
            delay_penalty=getattr(self, "_delay_penalty", 0.0),
        )

        if getattr(self.asr_model, "use_ctc", False) and hasattr(self.asr_model, "ctc_output"):
            ctc_loss = self.asr_model.forward_ctc(
                encoder_out=H_all,
                encoder_out_lens=enc_out_lens,
                targets=y_streams.values,
                target_lengths=y_lens_streams,
            )
        else:
            ctc_loss = torch.tensor(0.0, device=device)

        return simple_loss, pruned_loss, ctc_loss
