"""Eval on directory of wav+txt pairs using speaker-kernel forward path.

Usage:
    python eval_dir_with_kernel.py \
        --checkpoint zipformer_kd/exp_hybrid_v9/epoch-30.pt \
        --dir data/vlsp2020/VLSP2020Test1 \
        --out-dir eval_out/v9_ep30_vlsp2020test1
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zipformer"))

import numpy as np
import soundfile as sf
import torch

from test_asr_quick import build_model, encode_full_sequence, ctc_greedy_decode
from tokenizer import TokenEncoder


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tokens", default="zipformer/tokens.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--decode-chunk-size", type=int, default=32)
    ap.add_argument("--left-context-frames", type=int, default=128)
    ap.add_argument("--use-model", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = TokenEncoder(args.tokens)
    ms_model, params = build_model(args.checkpoint, device, use_model_avg=not args.use_model)
    assert getattr(ms_model, "speaker_kernel", None) is not None

    wavs = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if args.max_files:
        wavs = wavs[: args.max_files]
    print(f"Files: {len(wavs)}")

    recogs_path = os.path.join(args.out_dir, "recogs.txt")
    wer_path = os.path.join(args.out_dir, "wer.txt")

    total_ref, total_err = 0, 0
    t0 = time.time()

    with open(recogs_path, "w", encoding="utf-8") as rf:
        for i, wav_path in enumerate(wavs):
            txt_path = os.path.splitext(wav_path)[0] + ".txt"
            if not os.path.exists(txt_path):
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                gt = f.read().strip()

            data, sr = sf.read(wav_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                length_target = int(len(data) * 16000 / sr)
                data = np.interp(
                    np.linspace(0, len(data) - 1, length_target),
                    np.arange(len(data)),
                    data,
                )

            waveform = torch.from_numpy(data).unsqueeze(0)
            waveform = torch.nn.functional.layer_norm(waveform, (waveform.shape[-1],))
            waveform = waveform.to(device)

            with torch.no_grad():
                enc_out, enc_lens = encode_full_sequence(
                    ms_model, waveform, device,
                    decode_chunk_size=args.decode_chunk_size,
                    left_context_frames=args.left_context_frames,
                )
                ctc_tokens, _, _ = ctc_greedy_decode(ms_model, enc_out, enc_lens)

            hyp_text = tokenizer.decode(ctc_tokens) if ctc_tokens else ""

            ref_words = gt.lower().split()
            hyp_words = hyp_text.lower().split()
            d = levenshtein(ref_words, hyp_words)
            total_ref += len(ref_words)
            total_err += d

            name = os.path.basename(wav_path)
            rf.write(f"{name}:\tref={gt}\n")
            rf.write(f"{name}:\thyp={hyp_text}\n")

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                wer = 100.0 * total_err / max(1, total_ref)
                print(f"  [{i+1}/{len(wavs)}] wer={wer:.2f}% elapsed={elapsed:.0f}s")

    wer = 100.0 * total_err / max(1, total_ref)
    msg = (
        f"Checkpoint: {args.checkpoint}\n"
        f"Dir: {args.dir}\n"
        f"Files: {len(wavs)}  Ref words: {total_ref}  Errors: {total_err}\n"
        f"WER: {wer:.2f}%\n"
    )
    print("\n" + msg)
    with open(wer_path, "w") as f:
        f.write(msg)


if __name__ == "__main__":
    main()
