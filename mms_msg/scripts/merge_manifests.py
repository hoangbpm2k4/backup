#!/usr/bin/env python3
"""Merge all per-dataset manifest.jsonl into one combined manifest.
Also prints stats per dataset.

Usage:
    python merge_manifests.py
    python merge_manifests.py --output /path/to/output.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("/home/vietnam/voice_to_text/data")
DEFAULT_OUTPUT = DATA_ROOT / "asr_manifest_full_all_audio.jsonl"

DATASETS = [
    "fpt_fosd",
    "vais1000",
    "vivos",
    "vlsp2020_vinai_100h",
    "viet_bud500",
    "phoaudiobook",
    "viVoice",
    "vietspeech",
    "gigaspeech2_vi",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-exists", action="store_true",
                        help="Only include entries where audio file exists")
    args = parser.parse_args()

    stats = defaultdict(int)
    total = 0
    skipped = 0

    with open(args.output, 'w', encoding='utf-8') as out:
        for ds in DATASETS:
            manifest = DATA_ROOT / ds / "manifest.jsonl"
            if not manifest.exists():
                print(f"  SKIP {ds}: no manifest.jsonl")
                continue

            count = 0
            with open(manifest, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if args.verify_exists:
                        try:
                            obj = json.loads(line)
                            audio_path = obj.get("audio_path", "")
                            if not Path(audio_path).exists():
                                skipped += 1
                                continue
                        except:
                            skipped += 1
                            continue
                    out.write(line + "\n")
                    count += 1

            stats[ds] = count
            total += count
            print(f"  {ds}: {count:,} entries")

    print(f"\n  TOTAL: {total:,} entries → {args.output}")
    if skipped:
        print(f"  Skipped (missing audio): {skipped:,}")


if __name__ == "__main__":
    main()
