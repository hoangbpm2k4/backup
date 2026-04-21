#!/usr/bin/env python3
"""Extract audio from HuggingFace parquet datasets and create ASR manifest JSONL.

Reads parquet files with embedded audio bytes, writes wav files, and creates
a manifest JSONL compatible with generate_multidataset_overlap.py.

Usage:
    python extract_parquet_to_manifest.py \
        --parquet-dir /home/vietnam/voice_to_text/data/fpt_fosd/data \
        --dataset-name fpt_fosd \
        --output-audio-dir /home/vietnam/voice_to_text/data/fpt_fosd/wav \
        --output-manifest /home/vietnam/voice_to_text/data/fpt_fosd/manifest.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract parquet audio to wav + manifest JSONL.")
    p.add_argument("--parquet-dir", type=Path, required=True,
                    help="Directory containing *.parquet files.")
    p.add_argument("--dataset-name", type=str, required=True,
                    help="Dataset name to use in manifest (e.g. fpt_fosd).")
    p.add_argument("--output-audio-dir", type=Path, required=True,
                    help="Directory to write extracted wav files.")
    p.add_argument("--output-manifest", type=Path, required=True,
                    help="Path to output manifest JSONL file.")
    p.add_argument("--split", type=str, default="train",
                    help="Split name for manifest entries.")
    p.add_argument("--audio-col", type=str, default="audio",
                    help="Column name containing audio dict with 'bytes' and 'path' keys.")
    p.add_argument("--text-col", type=str, default="transcription",
                    help="Column name containing transcription text.")
    p.add_argument("--target-sr", type=int, default=16000,
                    help="Target sample rate for output wav files.")
    p.add_argument("--limit", type=int, default=0,
                    help="If >0, only process first N rows (for testing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_audio_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(args.parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {args.parquet_dir}")

    print(f"[INFO] Found {len(parquet_files)} parquet files in {args.parquet_dir}")

    written = 0
    skipped = 0

    with args.output_manifest.open("w", encoding="utf-8", newline="\n") as mf:
        for pf in parquet_files:
            print(f"[INFO] Reading {pf.name}...", flush=True)
            df = pd.read_parquet(pf)
            print(f"[INFO]   {len(df)} rows", flush=True)

            for idx, row in df.iterrows():
                if 0 < args.limit <= written:
                    break

                audio_data = row[args.audio_col]
                text = str(row[args.text_col]).strip()

                if not text:
                    skipped += 1
                    continue

                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get("bytes", b"")
                    orig_path = audio_data.get("path", "")
                else:
                    skipped += 1
                    continue

                if not audio_bytes:
                    skipped += 1
                    continue

                # Derive filename from original path or index
                if orig_path:
                    stem = Path(orig_path).stem
                else:
                    stem = f"{args.dataset_name}_{written:08d}"

                utt_id = f"{args.dataset_name}__{args.split}__{stem}"
                wav_path = args.output_audio_dir / f"{stem}.wav"

                # Decode audio bytes and write as wav
                try:
                    buf = io.BytesIO(audio_bytes)
                    audio, sr = sf.read(buf, dtype="float32")
                    if audio.size == 0:
                        skipped += 1
                        continue
                    # Resample if needed
                    if sr != args.target_sr:
                        import numpy as np
                        src_n = audio.shape[0]
                        dst_n = max(1, int(round(src_n * args.target_sr / sr)))
                        x_old = np.linspace(0, 1, src_n, endpoint=False)
                        x_new = np.linspace(0, 1, dst_n, endpoint=False)
                        if audio.ndim == 1:
                            audio = np.interp(x_new, x_old, audio).astype(np.float32)
                        else:
                            audio = np.column_stack([
                                np.interp(x_new, x_old, audio[:, c]).astype(np.float32)
                                for c in range(audio.shape[1])
                            ])
                        sr = args.target_sr
                    # Mono
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    sf.write(str(wav_path), audio, sr)
                except Exception as e:
                    print(f"[WARN] Failed to decode audio for {stem}: {e}")
                    skipped += 1
                    continue

                # Clean text: remove null bytes and control chars
                text = text.replace("\x00", "").replace("\n", " ").replace("\r", " ").strip()
                if not text:
                    skipped += 1
                    continue

                manifest_row = {
                    "utt_id": utt_id,
                    "dataset": args.dataset_name,
                    "split": args.split,
                    "audio_path": str(wav_path.resolve()),
                    "text": text,
                }
                mf.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
                mf.flush()
                written += 1

                if written % 5000 == 0:
                    print(f"[PROGRESS] written={written:,} skipped={skipped}", flush=True)

            if 0 < args.limit <= written:
                break

    print(f"[DONE] written={written:,} skipped={skipped}")
    print(f"  Audio dir: {args.output_audio_dir}")
    print(f"  Manifest:  {args.output_manifest}")


if __name__ == "__main__":
    main()
