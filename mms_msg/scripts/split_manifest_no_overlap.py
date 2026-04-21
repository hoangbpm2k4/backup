#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split ASR manifest into disjoint pools for single/overlap2/overlap3.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--single-ratio", type=float, default=0.65)
    p.add_argument("--ov2-ratio", type=float, default=0.25)
    p.add_argument("--ov3-ratio", type=float, default=0.06)
    return p.parse_args()


def normalized_split(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s == "train":
        return "train"
    if s in {"val", "validation", "dev"}:
        return "val"
    if s == "test":
        return "test"
    return "train"


def h01(key: str) -> float:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    s = max(0.0, args.single_ratio)
    o2 = max(0.0, args.ov2_ratio)
    o3 = max(0.0, args.ov3_ratio)
    # Match generator behavior: remaining budget goes to overlap2.
    o2 += max(0.0, 1.0 - (s + o2 + o3))
    if s + o2 + o3 <= 0:
        raise ValueError("Invalid ratios: total <= 0")
    z = s + o2 + o3
    s, o2, o3 = s / z, o2 / z, o3 / z

    p_single = args.out_dir / "manifest_single.jsonl"
    p_ov2 = args.out_dir / "manifest_overlap2.jsonl"
    p_ov3 = args.out_dir / "manifest_overlap3.jsonl"

    counts = defaultdict(int)
    split_counts = {k: defaultdict(int) for k in ["single", "overlap2", "overlap3"]}

    with (
        args.manifest.open("r", encoding="utf-8") as rf,
        p_single.open("w", encoding="utf-8", newline="\n") as ws,
        p_ov2.open("w", encoding="utf-8", newline="\n") as w2,
        p_ov3.open("w", encoding="utf-8", newline="\n") as w3,
    ):
        for line in rf:
            if not line.strip():
                continue
            obj = json.loads(line)
            audio_path = str(obj.get("audio_path", "")).strip()
            utt_id = str(obj.get("utt_id", "")).strip()
            if not audio_path:
                continue
            key = utt_id if utt_id else audio_path
            x = h01(key)
            split = normalized_split(str(obj.get("split", "")))
            if x < s:
                ws.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counts["single"] += 1
                split_counts["single"][split] += 1
            elif x < s + o2:
                w2.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counts["overlap2"] += 1
                split_counts["overlap2"][split] += 1
            else:
                w3.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counts["overlap3"] += 1
                split_counts["overlap3"][split] += 1

    suggest = {
        "single": {k: int(split_counts["single"][k]) for k in ["train", "val", "test"]},
        "overlap2": {k: int(split_counts["overlap2"][k] // 2) for k in ["train", "val", "test"]},
        "overlap3": {k: int(split_counts["overlap3"][k] // 3) for k in ["train", "val", "test"]},
    }
    summary = {
        "source_manifest": args.manifest.as_posix(),
        "ratios_normalized": {"single": s, "overlap2": o2, "overlap3": o3},
        "counts_rows": dict(counts),
        "split_counts_rows": {
            mt: {k: int(split_counts[mt][k]) for k in ["train", "val", "test"]} for mt in ["single", "overlap2", "overlap3"]
        },
        "suggest_target_mixtures": suggest,
        "files": {
            "single": p_single.as_posix(),
            "overlap2": p_ov2.as_posix(),
            "overlap3": p_ov3.as_posix(),
        },
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

