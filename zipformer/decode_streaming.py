#!/usr/bin/env python3
"""
Streaming multi-speaker decode with cpWER evaluation.

Decoding methods (matching icefall streaming_decode.py):
  - greedy_search: argmax per frame, max_sym_per_frame configurable
  - modified_beam_search: keep top-K hypotheses per frame

Usage:
  python zipformer/decode_streaming.py \
    --exp-dir ./zipformer/exp \
    --epoch 30 --avg 5 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --test-manifest data/manifests/test.jsonl.gz \
    --decode-chunk-samples 5120 \
    --decoding-method greedy_search \
    --num-speakers 3

Decode flow per chunk (explicit decoder + joiner calls):
  ┌─────────────────────────────────────────────────────────┐
  │ audio_chunk [1, T_samples]                              │
  │      ↓                                                  │
  │ StreamingMultiSpeakerEncoder.streaming_forward_from_audio│
  │      ↓                                                  │
  │ h_all [B*S, T_enc, D] per-speaker encoder output        │
  │      ↓                                                  │
  │ joiner.encoder_proj(h_all) → enc_proj [B*S, T_enc, J]  │
  │      ↓                                                  │
  │ for t in range(T_enc):                                  │
  │   enc_t = enc_proj[:, t:t+1, :]     ← 1 frame          │
  │   dec_out = decoder(context)         ← stateless        │
  │   dec_proj = joiner.decoder_proj(dec_out)               │
  │   logits = joiner(enc_t, dec_proj)   ← combine          │
  │   if greedy: token = argmax(logits)                     │
  │   if beam:   expand top-K hypotheses                    │
  │   if token != blank: append to hyps, update context     │
  └─────────────────────────────────────────────────────────┘

cpWER evaluation:
  For each utterance with S speakers:
    1. Accumulate tokens across all chunks → per-speaker token lists
    2. BPE decode → per-speaker word lists
    3. Try all S! permutations of hyp↔ref speaker assignment
    4. Pick permutation with LOWEST total edit distance (= cpWER)
"""

import argparse
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn

from finetune import add_model_arguments, get_model, get_params

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Streaming multi-speaker decode with cpWER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--exp-dir", type=str, default="zipformer/exp")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--avg", type=int, default=1)
    parser.add_argument("--bpe-model", type=str, default="data/lang_bpe_500/bpe.model")
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument(
        "--decode-chunk-samples", type=int, default=5120,
        help="Audio chunk size in samples (5120 = 320ms at 16kHz)",
    )
    parser.add_argument(
        "--decoding-method", type=str, default="greedy_search",
        choices=["greedy_search", "modified_beam_search"],
    )
    parser.add_argument(
        "--num-active-paths", type=int, default=4,
        help="Beam width for modified_beam_search",
    )
    parser.add_argument("--max-sym-per-frame", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--blank-penalty", type=float, default=0.0)
    parser.add_argument("--num-speakers", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)

    add_model_arguments(parser)
    return parser


# ═══════════════════════════════════════════════════════════
# Greedy search (explicit decoder + joiner, matching icefall)
# ═══════════════════════════════════════════════════════════


class GreedyStreamState:
    """Per-stream state for greedy decode."""
    def __init__(self, blank_id: int, context_size: int, device: torch.device):
        self.hyps: List[int] = [blank_id] * context_size
        self.context_size = context_size
        self.blank_id = blank_id
        self.device = device


def greedy_search_one_chunk(
    decoder: nn.Module,
    joiner: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    states: List[GreedyStreamState],
    max_sym_per_frame: int = 1,
) -> None:
    """Greedy search on one chunk — updates states in-place.

    This mirrors icefall streaming_beam_search.greedy_search() exactly:
      1. Pre-project encoder output with joiner.encoder_proj
      2. For each frame t:
         - Get decoder output from last context_size tokens
         - Combine with joiner: logits = joiner(enc_proj, dec_proj)
         - argmax → if non-blank, append to hyps and re-run decoder
         - Repeat up to max_sym_per_frame times

    Args:
        decoder: stateless RNNT decoder (Embedding + Conv1d)
        joiner: RNNT joiner (encoder_proj + decoder_proj + output_linear)
        encoder_out: [N, T, D] encoder output for this chunk
        encoder_out_lens: [N] valid frame counts
        states: list of N GreedyStreamState objects
        max_sym_per_frame: max non-blank symbols per encoder frame
    """
    assert len(states) == encoder_out.size(0)
    N, T, _ = encoder_out.shape
    blank_id = decoder.blank_id
    context_size = decoder.context_size
    device = encoder_out.device

    # Step 1: Pre-project encoder output (done once per chunk, not per frame)
    enc_proj = joiner.encoder_proj(encoder_out)  # [N, T, joiner_dim]

    # Step 2: Initial decoder projection from current context
    decoder_input = torch.tensor(
        [s.hyps[-context_size:] for s in states],
        device=device, dtype=torch.int64,
    )  # [N, context_size]
    dec_out = decoder(decoder_input, need_pad=False)  # [N, 1, decoder_dim]
    dec_proj = joiner.decoder_proj(dec_out)  # [N, 1, joiner_dim]

    # Step 3: Frame-by-frame decode
    for t in range(T):
        # Current encoder frame: [N, 1, 1, joiner_dim] (for joiner broadcast)
        enc_t = enc_proj[:, t:t+1, :].unsqueeze(2)

        for _ in range(max_sym_per_frame):
            # Joiner: combine encoder + decoder projections
            logits = joiner(
                enc_t,                    # [N, 1, 1, joiner_dim]
                dec_proj.unsqueeze(1),    # [N, 1, 1, joiner_dim]
                project_input=False,      # already projected
            )
            logits = logits.squeeze(1).squeeze(1)  # [N, vocab_size]

            # Greedy: argmax
            tokens = logits.argmax(dim=-1)  # [N]

            emitted = False
            for n in range(N):
                if t >= encoder_out_lens[n]:
                    continue
                tok = tokens[n].item()
                if tok != blank_id:
                    states[n].hyps.append(tok)
                    emitted = True

            if not emitted:
                break  # all streams emitted blank → next frame

            if emitted:
                # Re-compute decoder projection for streams that emitted
                decoder_input = torch.tensor(
                    [s.hyps[-context_size:] for s in states],
                    device=device, dtype=torch.int64,
                )
                dec_out = decoder(decoder_input, need_pad=False)
                dec_proj = joiner.decoder_proj(dec_out)


# ═══════════════════════════════════════════════════════════
# Modified beam search (explicit decoder + joiner)
# ═══════════════════════════════════════════════════════════


class Hypothesis:
    """Single beam hypothesis."""
    def __init__(self, tokens: List[int], log_prob: float = 0.0):
        self.tokens = list(tokens)
        self.log_prob = log_prob


class BeamStreamState:
    """Per-stream state for modified beam search."""
    def __init__(self, blank_id: int, context_size: int, num_active_paths: int):
        self.context_size = context_size
        self.blank_id = blank_id
        self.num_active_paths = num_active_paths
        # Start with single hypothesis: context_size blanks
        init_tokens = [blank_id] * context_size
        self.hyps = [Hypothesis(tokens=init_tokens, log_prob=0.0)]


def modified_beam_search_one_chunk(
    decoder: nn.Module,
    joiner: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    states: List[BeamStreamState],
    temperature: float = 1.0,
) -> None:
    """Modified beam search on one chunk — updates states in-place.

    For each frame, each stream expands its top-K hypotheses:
      1. For each hyp, compute decoder(context) → dec_proj
      2. logits = joiner(enc_proj, dec_proj) → log_softmax
      3. Expand: blank extends same hyp, non-blank forks new hyp
      4. Keep top-K by cumulative log_prob

    Args:
        decoder: stateless RNNT decoder
        joiner: RNNT joiner
        encoder_out: [N, T, D]
        encoder_out_lens: [N]
        states: list of N BeamStreamState
        temperature: softmax temperature
    """
    N, T, _ = encoder_out.shape
    device = encoder_out.device
    blank_id = decoder.blank_id
    context_size = decoder.context_size
    enc_proj = joiner.encoder_proj(encoder_out)  # [N, T, joiner_dim]

    for t in range(T):
        for n in range(N):
            if t >= encoder_out_lens[n]:
                continue

            enc_t = enc_proj[n, t:t+1, :]  # [1, joiner_dim]
            K = states[n].num_active_paths
            current_hyps = states[n].hyps

            # Build decoder input for all hypotheses
            dec_input = torch.tensor(
                [h.tokens[-context_size:] for h in current_hyps],
                device=device, dtype=torch.int64,
            )  # [num_hyps, context_size]
            dec_out = decoder(dec_input, need_pad=False)  # [num_hyps, 1, dec_dim]
            dec_proj = joiner.decoder_proj(dec_out)  # [num_hyps, 1, joiner_dim]

            # Expand encoder frame for all hyps
            enc_expanded = enc_t.unsqueeze(0).expand(len(current_hyps), -1, -1)
            # [num_hyps, 1, joiner_dim]

            logits = joiner(
                enc_expanded.unsqueeze(2),  # [num_hyps, 1, 1, joiner_dim]
                dec_proj.unsqueeze(1),      # [num_hyps, 1, 1, joiner_dim]
                project_input=False,
            )
            logits = logits.squeeze(1).squeeze(1)  # [num_hyps, vocab_size]

            if temperature != 1.0:
                logits = logits / temperature
            log_probs = logits.log_softmax(dim=-1)  # [num_hyps, vocab_size]

            # Expand hypotheses
            new_hyps: List[Hypothesis] = []
            for h_idx, hyp in enumerate(current_hyps):
                # Blank extension (same tokens, updated log_prob)
                blank_log_prob = hyp.log_prob + log_probs[h_idx, blank_id].item()
                new_hyps.append(Hypothesis(
                    tokens=hyp.tokens,
                    log_prob=blank_log_prob,
                ))

                # Non-blank extensions: top-K most likely tokens
                topk_log_probs, topk_ids = log_probs[h_idx].topk(K)
                for k in range(K):
                    tok = topk_ids[k].item()
                    if tok == blank_id:
                        continue
                    new_log_prob = hyp.log_prob + topk_log_probs[k].item()
                    new_hyps.append(Hypothesis(
                        tokens=hyp.tokens + [tok],
                        log_prob=new_log_prob,
                    ))

            # Prune: keep top-K by log_prob
            new_hyps.sort(key=lambda h: h.log_prob, reverse=True)
            states[n].hyps = new_hyps[:K]


# ═══════════════════════════════════════════════════════════
# cpWER: Concatenated minimum-permutation Word Error Rate
# ═══════════════════════════════════════════════════════════
#
# Multi-speaker ASR produces S transcripts but we don't know which
# hypothesis speaker matches which reference speaker. cpWER solves this:
#
#   For each of S! permutations of hypothesis speakers:
#     1. Concatenate all ref speakers' words: ref_0 + ref_1 + ... + ref_{S-1}
#     2. Concatenate hyp speakers in permuted order: hyp_π(0) + hyp_π(1) + ...
#     3. Compute standard WER(ref_concat, hyp_concat)
#   Return the permutation π* with LOWEST WER (minimum errors).
#
# Example with S=2:
#   ref = {spk_A: "hello world", spk_B: "goodbye"}
#   hyp = {slot_0: "goodbye",    slot_1: "hello word"}
#
#   Perm (0→A, 1→B): concat_hyp = "goodbye hello word"
#                     concat_ref = "hello world goodbye"
#                     WER = 3/3 = 1.0
#
#   Perm (0→B, 1→A): concat_hyp = "hello word goodbye"
#                     concat_ref = "hello world goodbye"
#                     WER = 1/3 = 0.33  ← BEST → cpWER = 0.33


def _edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Standard DP edit distance. Returns (subs, ins, dels, ref_len)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])

    # Backtrack to decompose into S, I, D
    i, j = n, m
    subs = ins = dels = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    return subs, ins, dels, n


def compute_cpwer(
    ref_per_speaker: List[List[str]],
    hyp_per_speaker: List[List[str]],
) -> Tuple[float, int, int, int, int, int, Tuple[int, ...]]:
    """Concatenated minimum-permutation WER.

    Returns:
        (wer, total_errors, total_ref_words, subs, ins, dels, best_perm)
        best_perm: the permutation tuple mapping hyp slots → ref slots
                   that gives the lowest WER
    """
    S_ref = len(ref_per_speaker)
    S_hyp = len(hyp_per_speaker)
    max_s = max(S_ref, S_hyp)
    ref_padded = ref_per_speaker + [[] for _ in range(max_s - S_ref)]
    hyp_padded = hyp_per_speaker + [[] for _ in range(max_s - S_hyp)]

    # Fixed ref concatenation order
    ref_concat = []
    for s in range(max_s):
        ref_concat.extend(ref_padded[s])

    best_errors = float("inf")
    best_stats = (0, 0, 0, 0)
    best_perm = tuple(range(max_s))

    # Try all S! permutations — find the one with MINIMUM errors
    for perm in itertools.permutations(range(max_s)):
        hyp_concat = []
        for s in range(max_s):
            hyp_concat.extend(hyp_padded[perm[s]])

        s, i, d, n = _edit_distance(ref_concat, hyp_concat)
        total_err = s + i + d
        if total_err < best_errors:
            best_errors = total_err
            best_stats = (s, i, d, n)
            best_perm = perm

    s, i, d, n = best_stats
    wer = (s + i + d) / n if n > 0 else (0.0 if best_errors == 0 else float("inf"))
    return wer, s + i + d, n, s, i, d, best_perm


# ═══════════════════════════════════════════════════════════
# Streaming decode loop
# ═══════════════════════════════════════════════════════════


def decode_one_utterance(
    streamer,
    decoder: nn.Module,
    joiner: nn.Module,
    audio: torch.Tensor,
    spk_activity: Optional[torch.Tensor],
    chunk_samples: int,
    num_speakers: int,
    decoding_method: str = "greedy_search",
    num_active_paths: int = 4,
    max_sym_per_frame: int = 1,
    temperature: float = 1.0,
) -> List[List[int]]:
    """Stream-decode one utterance with explicit decoder + joiner.

    Pipeline per chunk:
      1. streaming_forward_from_audio(audio_chunk) → h_all [B*S, T, D]
      2. greedy_search / modified_beam_search on h_all using decoder + joiner
      3. Accumulate emitted tokens per speaker across chunks

    Args:
        streamer: StreamingMultiSpeakerEncoder (encoder only, no decode)
        decoder: RNNT decoder module
        joiner: RNNT joiner module
        audio: [1, T_samples]
        spk_activity: [1, T_frames, S] or None
        chunk_samples: samples per chunk
        num_speakers: S
        decoding_method: "greedy_search" or "modified_beam_search"

    Returns:
        tokens_per_speaker: list of S lists of BPE token IDs (full utterance)
    """
    device = audio.device
    B = 1
    S = num_speakers
    N = B * S  # number of decoder streams
    blank_id = decoder.blank_id
    context_size = decoder.context_size

    # Init encoder streaming state
    enc_state = streamer.init_state(batch_size=B, device=device, init_decode=False)

    # Init decode states per speaker stream
    if decoding_method == "greedy_search":
        decode_states = [
            GreedyStreamState(blank_id, context_size, device)
            for _ in range(N)
        ]
    else:
        decode_states = [
            BeamStreamState(blank_id, context_size, num_active_paths)
            for _ in range(N)
        ]

    total_samples = audio.shape[1]
    offset = 0

    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        audio_chunk = audio[:, offset:end]

        # Activity chunk
        act_chunk = None
        if spk_activity is not None:
            frac = spk_activity.shape[1] / total_samples
            f_start = int(offset * frac)
            f_end = min(int(end * frac) + 1, spk_activity.shape[1])
            if f_end <= f_start:
                f_end = f_start + 1
            act_chunk = spk_activity[:, f_start:f_end, :]

        # === ENCODER STEP ===
        # streaming_forward_from_audio runs: conv_frontend → zipformer.streaming_forward
        # → speaker_kernel/adapter → per-speaker h_all [B*S, T_enc, D]
        h_all, h_lens, enc_state = streamer.streaming_forward_from_audio(
            audio_chunk, act_chunk, enc_state
        )
        # h_all: [N, T_enc, D] where N = B*S
        # h_lens: [N] valid frames per stream

        if h_all.shape[1] == 0:
            offset = end
            continue

        # === DECODE STEP (explicit decoder + joiner) ===
        if decoding_method == "greedy_search":
            greedy_search_one_chunk(
                decoder=decoder,
                joiner=joiner,
                encoder_out=h_all,
                encoder_out_lens=h_lens,
                states=decode_states,
                max_sym_per_frame=max_sym_per_frame,
            )
        else:
            modified_beam_search_one_chunk(
                decoder=decoder,
                joiner=joiner,
                encoder_out=h_all,
                encoder_out_lens=h_lens,
                states=decode_states,
                temperature=temperature,
            )

        offset = end

    # Extract final tokens per speaker
    tokens_per_speaker = []
    for s in range(S):
        if decoding_method == "greedy_search":
            # Remove initial blank context
            tokens = decode_states[s].hyps[context_size:]
        else:
            # Best hypothesis
            best_hyp = decode_states[s].hyps[0]
            tokens = best_hyp.tokens[context_size:]
        tokens_per_speaker.append(tokens)

    return tokens_per_speaker


def extract_ref_transcripts(cut, num_speakers: int) -> List[List[str]]:
    """Extract per-speaker reference transcripts from a lhotse Cut.

    Supports:
    1. cut.supervisions with `speaker` field → group by speaker, sorted by time
    2. cut.custom["ref_per_speaker"] → direct list of texts
    3. Single supervision → speaker 0 only
    """
    ref_per_speaker = [[] for _ in range(num_speakers)]

    if hasattr(cut, "supervisions") and len(cut.supervisions) > 0:
        spk_to_idx = {}
        next_idx = 0
        for sup in sorted(cut.supervisions, key=lambda s: s.start):
            spk = getattr(sup, "speaker", None) or "0"
            if spk not in spk_to_idx:
                if next_idx < num_speakers:
                    spk_to_idx[spk] = next_idx
                    next_idx += 1
                else:
                    continue
            idx = spk_to_idx[spk]
            text = sup.text or ""
            ref_per_speaker[idx].extend(text.strip().split())
    elif hasattr(cut, "custom") and "ref_per_speaker" in (cut.custom or {}):
        refs = cut.custom["ref_per_speaker"]
        for s, text in enumerate(refs):
            if s < num_speakers:
                ref_per_speaker[s] = text.strip().split() if isinstance(text, str) else text

    return ref_per_speaker


@torch.no_grad()
def decode_dataset(
    streamer,
    decoder: nn.Module,
    joiner: nn.Module,
    sp: spm.SentencePieceProcessor,
    cuts,
    params,
) -> Dict[str, Any]:
    """Decode all cuts with streaming and compute cpWER."""
    S = params.num_speakers
    device = next(streamer.ms_model.parameters()).device

    per_utt_results = []
    total_errors = 0
    total_ref_words = 0
    total_subs = 0
    total_ins = 0
    total_dels = 0

    log_interval = 50

    for idx, cut in enumerate(cuts):
        audio_np = cut.load_audio()
        if audio_np.ndim == 2:
            audio_np = audio_np[0]
        audio = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)

        # Load speaker activity if available
        spk_activity = None
        if hasattr(cut, "custom") and cut.custom and "spk_activity" in cut.custom:
            act = cut.custom["spk_activity"]
            if isinstance(act, (list, np.ndarray)):
                spk_activity = torch.tensor(act, dtype=torch.float32).unsqueeze(0).to(device)

        # Streaming decode with explicit decoder + joiner
        tokens_per_speaker = decode_one_utterance(
            streamer=streamer,
            decoder=decoder,
            joiner=joiner,
            audio=audio,
            spk_activity=spk_activity,
            chunk_samples=params.decode_chunk_samples,
            num_speakers=S,
            decoding_method=params.decoding_method,
            num_active_paths=params.num_active_paths,
            max_sym_per_frame=params.max_sym_per_frame,
            temperature=params.temperature,
        )

        # BPE decode → words per speaker
        hyp_per_speaker = []
        for s in range(S):
            text = sp.decode(tokens_per_speaker[s]) if tokens_per_speaker[s] else ""
            hyp_per_speaker.append(text.strip().split())

        ref_per_speaker = extract_ref_transcripts(cut, S)

        # cpWER: try all S! permutations, keep the one with LOWEST errors
        wer, errs, n_ref, subs, ins, dels, best_perm = compute_cpwer(
            ref_per_speaker, hyp_per_speaker
        )

        total_errors += errs
        total_ref_words += n_ref
        total_subs += subs
        total_ins += ins
        total_dels += dels

        per_utt_results.append({
            "cut_id": cut.id,
            "cpwer": round(wer, 4),
            "errors": errs,
            "ref_words": n_ref,
            "best_perm": list(best_perm),
            "ref_per_speaker": [" ".join(r) for r in ref_per_speaker],
            "hyp_per_speaker": [" ".join(h) for h in hyp_per_speaker],
        })

        if (idx + 1) % log_interval == 0:
            running_wer = total_errors / max(total_ref_words, 1)
            logging.info(
                f"Decoded {idx + 1} utterances, "
                f"running cpWER: {running_wer:.4f} "
                f"({total_errors}/{total_ref_words})"
            )

    overall_cpwer = total_errors / max(total_ref_words, 1)
    return {
        "per_utt": per_utt_results,
        "summary": {
            "cpwer": overall_cpwer,
            "total_errors": total_errors,
            "total_ref_words": total_ref_words,
            "substitutions": total_subs,
            "insertions": total_ins,
            "deletions": total_dels,
            "num_utterances": len(per_utt_results),
            "decoding_method": params.decoding_method,
        },
    }


def save_results(params, results: Dict, output_dir: Path):
    """Save decode results and cpWER summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = results["summary"]
    per_utt = results["per_utt"]

    # Summary
    with open(output_dir / "cpwer_summary.txt", "w") as f:
        f.write(f"Decoding method: {summary['decoding_method']}\n")
        f.write(f"cpWER: {summary['cpwer']:.4f}\n")
        f.write(f"Total errors: {summary['total_errors']}\n")
        f.write(f"Total ref words: {summary['total_ref_words']}\n")
        f.write(f"Substitutions: {summary['substitutions']}\n")
        f.write(f"Insertions: {summary['insertions']}\n")
        f.write(f"Deletions: {summary['deletions']}\n")
        f.write(f"Utterances: {summary['num_utterances']}\n")

    # Per-utterance JSON
    with open(output_dir / "per_utt_results.json", "w") as f:
        json.dump(per_utt, f, indent=2, ensure_ascii=False)

    # Human-readable transcripts with best permutation
    with open(output_dir / "transcripts.txt", "w") as f:
        for utt in per_utt:
            f.write(f"=== {utt['cut_id']} (cpWER={utt['cpwer']}) "
                    f"perm={utt['best_perm']} ===\n")
            S = len(utt["ref_per_speaker"])
            perm = utt["best_perm"]
            for s in range(S):
                hyp_idx = perm[s] if s < len(perm) else s
                f.write(f"  REF[spk{s}]: {utt['ref_per_speaker'][s]}\n")
                hyp_text = utt["hyp_per_speaker"][hyp_idx] if hyp_idx < len(utt["hyp_per_speaker"]) else ""
                f.write(f"  HYP[slot{hyp_idx}→spk{s}]: {hyp_text}\n")
            f.write("\n")

    logging.info(
        f"\n{'='*60}\n"
        f"  Method: {summary['decoding_method']}\n"
        f"  cpWER = {summary['cpwer']:.4f}  "
        f"(S={summary['substitutions']} I={summary['insertions']} D={summary['deletions']})\n"
        f"  {summary['total_errors']} errors / {summary['total_ref_words']} words "
        f"over {summary['num_utterances']} utterances\n"
        f"  Results: {output_dir}\n"
        f"{'='*60}"
    )


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    exp_dir = Path(params.exp_dir)
    assert exp_dir.is_dir(), f"{exp_dir} does not exist"

    output_dir = Path(params.output_dir) if params.output_dir else (
        exp_dir / "streaming_decode"
        / f"epoch{params.epoch}_avg{params.avg}_{params.decoding_method}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load BPE model
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    if params.blank_id == -1:
        params.blank_id = 0
    params.vocab_size = sp.get_piece_size()

    # Load model
    logging.info("Loading model...")
    model = get_model(params)

    # Load checkpoint(s)
    if params.avg > 1:
        checkpoints = []
        for i in range(params.epoch - params.avg + 1, params.epoch + 1):
            ckpt_path = exp_dir / f"epoch-{i}.pt"
            if ckpt_path.exists():
                checkpoints.append(ckpt_path)
        assert len(checkpoints) > 0, f"No checkpoints found in {exp_dir}"
        logging.info(f"Averaging {len(checkpoints)} checkpoints")

        avg_state = None
        for ckpt_path in checkpoints:
            state = torch.load(ckpt_path, map_location="cpu")
            model_state = state.get("model", state)
            if avg_state is None:
                avg_state = {k: v.float() for k, v in model_state.items()}
            else:
                for k in avg_state:
                    if k in model_state:
                        avg_state[k] += model_state[k].float()
        for k in avg_state:
            avg_state[k] /= len(checkpoints)
        model.load_state_dict(avg_state, strict=False)
    else:
        ckpt_path = exp_dir / f"epoch-{params.epoch}.pt"
        assert ckpt_path.exists(), f"{ckpt_path} not found"
        logging.info(f"Loading {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model_state = state.get("model", state)
        model.load_state_dict(model_state, strict=False)

    model.to(device)
    model.eval()

    # Extract decoder + joiner from model
    # MultiSpeakerRnntModel → asr_model → decoder, joiner
    asr_model = model.asr_model
    decoder = asr_model.decoder
    joiner = asr_model.joiner

    # Build streaming encoder (encoder-only, no decode)
    from streaming_multi_speaker import StreamingMultiSpeakerEncoder
    streamer = StreamingMultiSpeakerEncoder(model)

    # Load test data
    from lhotse import CutSet
    logging.info(f"Loading test manifest: {params.test_manifest}")
    cuts = CutSet.from_file(params.test_manifest)
    logging.info(f"Loaded {len(cuts)} cuts")

    # Decode + evaluate
    results = decode_dataset(streamer, decoder, joiner, sp, cuts, params)

    # Save
    save_results(params, results, output_dir)


if __name__ == "__main__":
    main()
