#!/usr/bin/env python3
"""Convert accepted_manifest_combined.csv (new YT 1000h) to input JSONL manifest.

Output JSONL fields: utt_id, dataset, split, audio_path, text
- dataset=yt_1900h
- split=train
- utt_id=yt1900h__train__<stem>__<hash8>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/data_label_auto_yt_1900h/manifests/accepted_manifest_combined.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/overlap_multidataset_yt1000h/input_manifest_yt1000h.jsonl"))
    ap.add_argument("--dataset-name", default="yt_1900h")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_skip = 0
    csv.field_size_limit(10_000_000)
    with args.csv.open("r", encoding="utf-8") as rf, args.out.open("w", encoding="utf-8") as wf:
        reader = csv.DictReader(rf)
        for row in reader:
            n_in += 1
            audio_path = (row.get("final_wav") or "").strip()
            text = (row.get("final_text") or "").strip()
            if not audio_path or not text:
                n_skip += 1
                continue
            stem = Path(audio_path).stem
            h = hashlib.md5(audio_path.encode("utf-8")).hexdigest()[:8]
            utt_id = f"{args.dataset_name}__{args.split}__{stem}__{h}"
            obj = {
                "utt_id": utt_id,
                "dataset": args.dataset_name,
                "split": args.split,
                "audio_path": audio_path,
                "text": text,
            }
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Read: {n_in}  Written: {n_out}  Skipped: {n_skip}")
    print(f"Out: {args.out}")


if __name__ == "__main__":
    main()
