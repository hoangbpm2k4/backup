# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import Fp32GroupNorm, Fp32LayerNorm, TransposeLast


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class CumulativeInstanceNorm(nn.Module):
    """Streaming-exact replacement for GroupNorm(C, C) / InstanceNorm.

    Standard GroupNorm(C,C) normalizes each channel over the full time dim,
    making it chunk-dependent.  This module uses Welford's online algorithm
    to maintain per-channel running mean and variance.  Each frame is
    normalized with cumulative stats from frames [0 … t], so the output
    is identical regardless of how the input is chunked.

    Properties:
      - Chunk-boundary invariant (streaming-exact).
      - Causal: frame t uses only frames [0 … t].
      - NOT identical to offline GroupNorm (which uses global stats).
        The difference shrinks as sequence length grows.

    State: tuple(count [B, C], mean [B, C], M2 [B, C])  — Welford accumulators.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @staticmethod
    def from_group_norm(gn: nn.Module) -> "CumulativeInstanceNorm":
        """Create from an existing nn.GroupNorm(C, C) / Fp32GroupNorm."""
        nc = gn.num_channels
        cin = CumulativeInstanceNorm(nc, eps=gn.eps, affine=gn.affine)
        if gn.affine:
            cin.weight.data.copy_(gn.weight.data)
            cin.bias.data.copy_(gn.bias.data)
        return cin

    def get_init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple:
        C = self.num_channels
        return (
            torch.zeros(batch_size, C, device=device),  # count
            torch.zeros(batch_size, C, device=device),  # mean
            torch.zeros(batch_size, C, device=device),  # M2
        )

    def streaming_forward(
        self, x: Tensor, state: Tuple,
    ) -> Tuple[Tensor, Tuple]:
        """
        Args:
            x: [B, C, T] — feature map (only NEW frames, no cache overlap).
            state: (count, mean, M2)
        Returns:
            (x_norm, new_state)

        Vectorized Welford: uses cumsum to compute running mean/variance
        across T frames without a Python loop.  Produces identical output
        to the frame-by-frame version but leverages GPU parallelism.
        """
        x_float = x.float()
        B, C, T = x_float.shape

        if T == 0:
            return x_float.type_as(x), state

        count0 = state[0]  # [B, C]
        mean0 = state[1]   # [B, C]
        M2_0 = state[2]    # [B, C]

        # Running counts: count0+1, count0+2, ..., count0+T
        offsets = torch.arange(1, T + 1, device=x.device, dtype=x_float.dtype)
        counts = count0.unsqueeze(-1) + offsets.view(1, 1, T)  # [B, C, T]

        # Running mean via cumsum
        cumsum = x_float.cumsum(dim=2)  # [B, C, T]
        prev_total = (mean0 * count0).unsqueeze(-1)  # [B, C, 1]
        running_mean = (prev_total + cumsum) / counts  # [B, C, T]

        # Running M2 via parallel Welford merge:
        #   M2_total = M2_0 + M2_new + delta^2 * n0 * n_new / n_total
        cumsum_sq = (x_float ** 2).cumsum(dim=2)  # [B, C, T]
        new_counts = offsets.view(1, 1, T)  # [1, 1, T]
        new_mean = cumsum / new_counts  # [B, C, T]
        M2_new = cumsum_sq - new_counts * (new_mean ** 2)  # [B, C, T]

        n0 = count0.unsqueeze(-1)  # [B, C, 1]
        delta_means = new_mean - mean0.unsqueeze(-1)  # [B, C, T]
        merge_term = (delta_means ** 2) * n0 * new_counts / counts.clamp(min=1)
        M2_total = M2_0.unsqueeze(-1) + M2_new + merge_term  # [B, C, T]

        # Running variance and normalize
        running_var = M2_total / counts.clamp(min=1)  # [B, C, T]
        out = (x_float - running_mean) / torch.sqrt(running_var + self.eps)

        if self.weight is not None:
            out = out * self.weight.float().view(1, C, 1) + self.bias.float().view(1, C, 1)

        # Final state = last column values
        new_count = counts[:, :, -1]       # [B, C]
        new_mean_f = running_mean[:, :, -1]  # [B, C]
        new_M2 = M2_total[:, :, -1]        # [B, C]

        return out.type_as(x), (new_count, new_mean_f, new_M2)

    def forward(self, x: Tensor) -> Tensor:
        """Offline: standard instance-norm (same as GroupNorm(C,C))."""
        x_float = x.float()
        mean = x_float.mean(dim=2, keepdim=True)
        var = x_float.var(dim=2, keepdim=True, unbiased=False)
        out = (x_float - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight.float().view(1, -1, 1) + self.bias.float().view(1, -1, 1)
        return out.type_as(x)


class StreamingConvFrontend(nn.Module):
    """
    Streaming wrapper for ConvFeatureExtractionModel.

    Uses waveform-level input caching for exact equivalence with offline:
    each invocation processes (cache + new_audio) through the full conv stack,
    drops frames attributable to the cache-only prefix, and keeps the exact
    unconsumed waveform suffix needed for the next stride-aligned output.

    If the original model uses GroupNorm (mode='default'), the GroupNorm is
    automatically replaced with CumulativeInstanceNorm in streaming mode,
    making the output chunk-boundary invariant.

    Reuses the EXACT SAME weights from the original ConvFeatureExtractionModel.

    Usage:
        streaming = StreamingConvFrontend(conv_model)
        state = streaming.get_init_state(batch_size, device)
        for audio_chunk in chunks:
            features, state = streaming.streaming_forward(audio_chunk, state)
    """

    def __init__(self, conv_model: ConvFeatureExtractionModel):
        super().__init__()
        self.conv_model = conv_model

        # Compute total stride and receptive field of the conv stack
        total_stride = 1
        receptive_field = 1
        for block in conv_model.conv_layers:
            conv = block[0]
            k = conv.kernel_size[0]
            s = conv.stride[0]
            receptive_field = receptive_field + (k - 1) * total_stride
            total_stride *= s

        self.total_stride = total_stride
        self.samples_per_frame = int(total_stride)
        self.subsampling_factor_in_samples = int(total_stride)
        self.receptive_field = receptive_field
        self._base_cache_size = receptive_field - total_stride

        # Detect GroupNorm and build streaming-safe mirror
        self._gn_layer_idx: Optional[int] = None  # index within first block
        self._cum_norm: Optional[CumulativeInstanceNorm] = None

        first_block = conv_model.conv_layers[0]
        for i, layer in enumerate(first_block):
            if isinstance(layer, nn.GroupNorm):
                self._gn_layer_idx = i
                self._cum_norm = CumulativeInstanceNorm.from_group_norm(layer)
                break

        self.causal_left_trim = self._compute_causal_trim()

    @property
    def streaming_safe(self) -> bool:
        """Always True — GroupNorm is handled by CumulativeInstanceNorm."""
        return True

    @property
    def has_groupnorm(self) -> bool:
        """True if the underlying conv model has GroupNorm (informational)."""
        return self._gn_layer_idx is not None

    def _compute_causal_trim(self) -> int:
        """Compute fixed leading-frame offset between offline and streaming."""
        device = next(self.conv_model.parameters()).device
        dummy = torch.zeros(1, 4000, device=device)
        with torch.no_grad():
            offline_out = self.conv_model(dummy)
            state = self.get_init_state(1, device)
            stream_out, _ = self.streaming_forward(dummy, state)
        return stream_out.shape[2] - offline_out.shape[2]

    def get_init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple:
        """Returns (wav_cache, norm_state, block0_feat_cache, exact_len_state)."""
        wav_cache = torch.zeros(batch_size, 1, 0, device=device)
        norm_state = None
        if self._cum_norm is not None:
            norm_state = self._cum_norm.get_init_state(batch_size, device)
        exact_len_state = {
            "cum_valid_samples": torch.zeros(
                batch_size, dtype=torch.long, device=device
            ),
            "cum_valid_feat_frames": torch.zeros(
                batch_size, dtype=torch.long, device=device
            ),
        }
        return (wav_cache, norm_state, None, exact_len_state)

    def _frames_from_samples_block0_conv(self, n_samples: int) -> int:
        """Compute output frames from block 0's first conv sublayer only."""
        block = self.conv_model.conv_layers[0]
        conv = block[0]
        k = conv.kernel_size[0]
        s = conv.stride[0]
        if n_samples < k:
            return 0
        return (n_samples - k) // s + 1

    def streaming_forward(
        self,
        x: Tensor,
        state: Tuple,
        sample_lens: Optional[Tensor] = None,
        return_lens: bool = False,
    ) -> Tuple[Tensor, Tuple]:
        """
        Process an audio chunk through the conv stack with waveform caching.

        For GroupNorm frontends:
          1. Run block0 Conv+Dropout on cache+new.
          2. Split at cache boundary.
          3. CumulativeInstanceNorm + GELU on new frames only.
          4. Substitute cache frames with saved block0 output from prev call.
          5. Stitch, run blocks 1-2, drop final cache frames.

        Args:
            x: [B, T_samples] raw audio chunk
            state: (wav_cache, norm_state, block0_feat_cache, exact_len_state)
            sample_lens: [B] valid samples in the current chunk.
            return_lens: If True, also return exact per-sample emitted feature
                lengths for this chunk.

        Returns:
            features: [B, C, T_out] conv features for this chunk only
            new_state: updated state tuple
            feature_lens: [B] exact emitted feature lengths when return_lens=True
        """
        wav_cache = state[0]
        norm_state = state[1]
        block0_feat_cache = state[2]
        if len(state) >= 4 and state[3] is not None:
            exact_len_state = state[3]
        else:
            batch_size = x.shape[0]
            device = x.device
            exact_len_state = {
                "cum_valid_samples": torch.zeros(
                    batch_size, dtype=torch.long, device=device
                ),
                "cum_valid_feat_frames": torch.zeros(
                    batch_size, dtype=torch.long, device=device
                ),
            }

        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        # Concatenate cache + new audio
        x_full = torch.cat([wav_cache, x], dim=2)
        cache_len = wav_cache.shape[2]

        # Guard: if total input is smaller than receptive field, no output
        # is possible. Keep everything in cache for next chunk.
        if x_full.shape[2] < self.receptive_field:
            B = x.shape[0]
            C = self.conv_model.conv_layers[-1][0].out_channels
            empty_feat = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)
            new_state = (x_full.detach(), norm_state, block0_feat_cache, exact_len_state)
            if return_lens:
                return empty_feat, new_state, torch.zeros(B, dtype=torch.long, device=x.device)
            return empty_feat, new_state

        if self._gn_layer_idx is not None:
            # ── GroupNorm path ──
            block = self.conv_model.conv_layers[0]
            skip_n = self._frames_from_samples_block0_conv(cache_len)

            # Step 1: Run pre-norm sublayers (Conv, Dropout) on full cache+new
            pre_norm = x_full
            for i, layer in enumerate(block):
                if i == self._gn_layer_idx:
                    break
                pre_norm = layer(pre_norm)
            # pre_norm shape: [B, C, T_block0_conv]

            # Step 2: Split at cache boundary
            pre_norm_new = pre_norm[:, :, skip_n:]  # only new frames

            # Step 3: CumulativeInstanceNorm on new frames only
            normed_new, norm_state = self._cum_norm.streaming_forward(
                pre_norm_new, norm_state
            )

            # Step 4: Apply post-norm sublayers (GELU etc.) on new frames
            post_new = normed_new
            for i, layer in enumerate(block):
                if i > self._gn_layer_idx:
                    post_new = layer(post_new)

            # Step 5: Build full block0 output by stitching cache + new
            if skip_n > 0 and block0_feat_cache is not None:
                # Use saved post-block0 output for cache frames
                cache_portion = block0_feat_cache[:, :, -skip_n:]
                out_block0 = torch.cat([cache_portion, post_new], dim=2)
            else:
                out_block0 = post_new

            # Save full block0 output for next call's cache overlap
            new_block0_feat_cache = out_block0.detach()

            # Step 6: Run remaining blocks on stitched output
            out = out_block0
            for idx in range(1, len(self.conv_model.conv_layers)):
                out = self.conv_model.conv_layers[idx](out)
        else:
            # ── No GroupNorm: standard path ──
            out = x_full
            for conv_block in self.conv_model.conv_layers:
                out = conv_block(out)
            new_block0_feat_cache = None

        # Drop cache-overlap frames (already emitted)
        cache_only_out = self._frames_from_samples(cache_len)
        features = out[:, :, cache_only_out:]

        # Cache unconsumed waveform suffix
        n_out_total = out.shape[2]
        next_start = n_out_total * self.total_stride
        new_wav_cache = x_full[:, :, next_start:].detach()
        new_exact_len_state = {
            "cum_valid_samples": exact_len_state["cum_valid_samples"],
            "cum_valid_feat_frames": exact_len_state["cum_valid_feat_frames"],
        }

        feature_lens = None
        if sample_lens is not None:
            sample_lens = sample_lens.to(
                device=x.device, dtype=torch.long
            ).clamp(min=0, max=x.shape[2])
            cum_valid_samples = (
                exact_len_state["cum_valid_samples"] + sample_lens
            )
            feature_lens = torch.zeros_like(cum_valid_samples)
            trim = self.causal_left_trim
            for b in range(cum_valid_samples.shape[0]):
                curr_total = max(
                    self._frames_from_samples(int(cum_valid_samples[b].item())) - trim,
                    0,
                )
                prev_total = int(
                    exact_len_state["cum_valid_feat_frames"][b].item()
                )
                feature_lens[b] = curr_total - prev_total

            feature_lens = feature_lens.clamp(
                min=0, max=features.shape[2]
            )
            new_exact_len_state = {
                "cum_valid_samples": cum_valid_samples,
                "cum_valid_feat_frames": (
                    exact_len_state["cum_valid_feat_frames"] + feature_lens
                ),
            }

        new_state = (
            new_wav_cache,
            norm_state,
            new_block0_feat_cache,
            new_exact_len_state,
        )
        if return_lens:
            if feature_lens is None:
                feature_lens = torch.full(
                    (features.shape[0],),
                    features.shape[2],
                    dtype=torch.long,
                    device=features.device,
                )
            return features, new_state, feature_lens

        return features, new_state

    def _frames_from_samples(self, n_samples: int) -> int:
        """Compute how many output frames the conv stack produces for n_samples."""
        t = n_samples
        for block in self.conv_model.conv_layers:
            conv = block[0]
            k = conv.kernel_size[0]
            s = conv.stride[0]
            if t < k:
                return 0
            t = (t - k) // s + 1
        return t

    def forward(self, x: Tensor) -> Tensor:
        """Offline forward — delegates to original model."""
        return self.conv_model(x)
