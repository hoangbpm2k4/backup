#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate total audio duration (hours) from ASR manifest JSONL"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/asr_manifest_extracted_16k.jsonl"),
        help="Path to JSONL manifest containing fields: dataset, split, audio_path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker threads for reading wav headers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Rows per chunk to process in one thread-pool cycle",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Debug only: limit number of manifest rows (0 = full)",
    )
    parser.add_argument(
        "--dedupe",
        choices=["auto", "none", "utt_id", "audio_path"],
        default="auto",
        help=(
            "Duplicate handling before duration sum: "
            "auto=utt_id if present else audio_path; "
            "none=keep all rows."
        ),
    )
    return parser.parse_args()


def read_manifest_rows(
    manifest_path: Path, max_rows: int, dedupe_mode: str
) -> Tuple[List[Tuple[str, str, Path]], int]:
    rows: List[Tuple[str, str, Path]] = []
    seen_keys: Set[Tuple[str, str]] = set()
    dup_rows = 0

    def dedupe_key(obj: dict, audio_path: str) -> Optional[Tuple[str, str]]:
        if dedupe_mode == "none":
            return None
        if dedupe_mode in ("auto", "utt_id"):
            utt_id = str(obj.get("utt_id", "")).strip()
            if utt_id:
                return ("utt_id", utt_id)
            if dedupe_mode == "utt_id":
                return None
        return ("audio_path", audio_path)

    with manifest_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if max_rows and idx > max_rows:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio_path = str(obj.get("audio_path", ""))
            if "://" in audio_path:
                # Skip non-file references like parquet:// or arrow://
                continue
            if not audio_path.lower().endswith(".wav"):
                continue
            dataset = str(obj.get("dataset", "unknown"))
            split = str(obj.get("split", "unknown"))

            key = dedupe_key(obj, audio_path)
            if key is not None:
                if key in seen_keys:
                    dup_rows += 1
                    continue
                seen_keys.add(key)

            rows.append((dataset, split, Path(audio_path)))
    return rows, dup_rows


def get_wav_duration_sec(path: Path) -> Optional[float]:
    try:
        with wave.open(path.as_posix(), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


def chunks(items: List[Tuple[str, str, Path]], n: int) -> Iterable[List[Tuple[str, str, Path]]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows, dup_rows = read_manifest_rows(manifest_path, args.max_rows, args.dedupe)
    total_rows = len(rows)
    if total_rows == 0:
        print("No local .wav entries found in manifest.")
        return

    by_dataset_sec: Dict[str, float] = defaultdict(float)
    by_dataset_split_sec: Dict[Tuple[str, str], float] = defaultdict(float)
    bad_files = 0

    batch_size = max(1, args.batch_size)
    processed = 0

    for batch in chunks(rows, batch_size):
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            durs = list(ex.map(lambda x: get_wav_duration_sec(x[2]), batch))

        for (dataset, split, _path), dur in zip(batch, durs):
            if dur is None:
                bad_files += 1
                continue
            by_dataset_sec[dataset] += dur
            by_dataset_split_sec[(dataset, split)] += dur

        processed += len(batch)
        print(f"[PROGRESS] {processed:,}/{total_rows:,}", flush=True)

    total_sec = sum(by_dataset_sec.values())
    total_hours = total_sec / 3600.0

    print("\n=== TOTAL ===")
    print(f"rows_wav: {total_rows:,}")
    print(f"dup_rows_skipped: {dup_rows:,}")
    print(f"bad_files: {bad_files:,}")
    print(f"total_hours: {total_hours:,.2f}")

    print("\n=== BY DATASET ===")
    for dataset, sec in sorted(by_dataset_sec.items(), key=lambda x: x[0]):
        print(f"{dataset}\t{sec / 3600.0:,.2f} h")

    print("\n=== BY DATASET/SPLIT ===")
    for (dataset, split), sec in sorted(by_dataset_split_sec.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"{dataset}/{split}\t{sec / 3600.0:,.2f} h")


if __name__ == "__main__":
    main()
