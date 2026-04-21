"""Batch eval on VLSP2023TestPB using the SAME forward path as test_asr_quick.py
(speaker kernel applied with spk_mask=ones, bg_mask=zeros).

Usage:
    conda activate voice_to_text
    python eval_vlsp_with_kernel.py \
        --checkpoint zipformer_kd/exp_hybrid_v3/epoch-3.pt \
        --cuts data/lhotse_cutsets_vlsp_testpb/val.all.cuts_clean.jsonl.gz \
        --out-dir eval_out/v3_ep3
"""

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zipformer"))

import numpy as np
import soundfile as sf
import torch

from test_asr_quick import build_model, encode_full_sequence, ctc_greedy_decode
from tokenizer import TokenEncoder
from icefall.decode import ctc_prefix_beam_search


def levenshtein(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[m]


def load_audio(source_path, sr_target=16000):
    data, sr = sf.read(source_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sr_target:
        length_target = int(len(data) * sr_target / sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, length_target),
            np.arange(len(data)),
            data,
        )
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cuts", required=True)
    ap.add_argument("--tokens", default="zipformer/tokens.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--decode-chunk-size", type=int, default=32)
    ap.add_argument("--left-context-frames", type=int, default=128)
    ap.add_argument("--use-model", action="store_true",
                    help="Use 'model' weights (latest) rather than model_avg")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-cuts", type=int, default=None)
    ap.add_argument("--decoding", choices=["greedy", "prefix_beam"], default="greedy")
    ap.add_argument("--beam", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = TokenEncoder(args.tokens)
    print(f"Vocab: {tokenizer.get_piece_size()}")

    ms_model, params = build_model(
        args.checkpoint, device, use_model_avg=not args.use_model
    )
    assert getattr(ms_model, "speaker_kernel", None) is not None, \
        "speaker_kernel missing — this script expects MultiSpeakerRnntModel"

    # Load cuts
    cuts = []
    with gzip.open(args.cuts, "rt") as f:
        for line in f:
            cuts.append(json.loads(line))
    if args.max_cuts:
        cuts = cuts[: args.max_cuts]
    print(f"Cuts: {len(cuts)}")

    recogs_path = os.path.join(args.out_dir, "recogs.txt")
    wer_path = os.path.join(args.out_dir, "wer.txt")

    total_ref_words = 0
    total_err = 0
    total_sub = 0
    total_ins = 0
    total_del = 0
    t0 = time.time()

    with open(recogs_path, "w", encoding="utf-8") as rf:
        for i, cut in enumerate(cuts):
            sup = cut["supervisions"][0]
            gt = sup["text"].strip()
            src = cut["recording"]["sources"][0]["source"]
            cut_start = cut.get("start", 0.0)
            cut_dur = cut.get("duration")

            data, sr = sf.read(src, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                length_target = int(len(data) * 16000 / sr)
                data = np.interp(
                    np.linspace(0, len(data) - 1, length_target),
                    np.arange(len(data)),
                    data,
                )
                sr = 16000
            # Trim to cut window
            s = int(cut_start * sr)
            e = s + int(cut_dur * sr) if cut_dur else len(data)
            data = data[s:e]

            waveform = torch.from_numpy(data).unsqueeze(0)
            waveform = torch.nn.functional.layer_norm(
                waveform, (waveform.shape[-1],)
            )
            waveform = waveform.to(device)

            with torch.no_grad():
                enc_out, enc_lens = encode_full_sequence(
                    ms_model, waveform, device,
                    decode_chunk_size=args.decode_chunk_size,
                    left_context_frames=args.left_context_frames,
                )
                if args.decoding == "greedy":
                    ctc_tokens, _, _ = ctc_greedy_decode(ms_model, enc_out, enc_lens)
                else:
                    ctc_log = ms_model.asr_model.ctc_output(enc_out)  # [B, T, V]
                    hyps = ctc_prefix_beam_search(
                        ctc_output=ctc_log,
                        encoder_out_lens=enc_lens,
                        beam=args.beam,
                        blank_id=0,
                    )
                    ctc_tokens = hyps[0]

            hyp_text = tokenizer.decode(ctc_tokens) if ctc_tokens else ""

            ref_words = gt.lower().split()
            hyp_words = hyp_text.lower().split()
            d = levenshtein(ref_words, hyp_words)
            total_ref_words += len(ref_words)
            total_err += d

            rf.write(f"{cut['id']}:\tref={gt}\n")
            rf.write(f"{cut['id']}:\thyp={hyp_text}\n")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                wer_running = 100.0 * total_err / max(1, total_ref_words)
                print(f"  [{i+1}/{len(cuts)}] wer={wer_running:.2f}% "
                      f"elapsed={elapsed:.0f}s")

    wer = 100.0 * total_err / max(1, total_ref_words)
    msg = (
        f"Checkpoint: {args.checkpoint}\n"
        f"Chunk: {args.decode_chunk_size}  LC: {args.left_context_frames}\n"
        f"use_model={args.use_model}\n"
        f"Cuts: {len(cuts)}  Ref words: {total_ref_words}  Errors: {total_err}\n"
        f"WER: {wer:.2f}%\n"
    )
    print("\n" + msg)
    with open(wer_path, "w") as f:
        f.write(msg)


if __name__ == "__main__":
    main()
