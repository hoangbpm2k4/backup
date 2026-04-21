#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            c += 1
    return c


def expected_from_shards(shard_dir: Path, mix_type: str) -> Dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0}
    for mf in sorted(shard_dir.glob("shard_*.jsonl")):
        by_split = {"train": 0, "val": 0, "test": 0}
        with mf.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                s = str(obj.get("split", "train")).strip().lower()
                if s == "train":
                    by_split["train"] += 1
                elif s in {"validation", "val", "dev"}:
                    by_split["val"] += 1
                elif s == "test":
                    by_split["test"] += 1
                else:
                    by_split["train"] += 1
        for sp in ["train", "val", "test"]:
            n = by_split[sp]
            if mix_type == "single":
                out[sp] += n
            elif mix_type == "overlap2":
                out[sp] += n // 2
            elif mix_type == "overlap3":
                out[sp] += n // 3
    return out


def done_from_outputs(out_dir: Path) -> Dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0}
    for sp in ["train", "val", "test"]:
        for mf in out_dir.rglob(f"{sp}.manifest.jsonl"):
            out[sp] += count_lines(mf)
    return out


def pct(done: int, exp: int) -> float:
    if exp <= 0:
        return 0.0
    return 100.0 * done / exp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("/home/vietnam/voice_to_text/data/overlap_runs"))
    args = p.parse_args()
    base = args.run_dir.resolve()

    cfg = {
        "single": (base / "manifests" / "shards_single", base / "single"),
        "overlap2": (base / "manifests" / "shards_overlap2", base / "overlap2"),
        "overlap3": (base / "manifests" / "shards_overlap3", base / "overlap3"),
    }

    grand_done = 0
    grand_exp = 0
    rows = []
    for mt in ["single", "overlap2", "overlap3"]:
        shard_dir, out_dir = cfg[mt]
        exp = expected_from_shards(shard_dir, mt)
        done = done_from_outputs(out_dir)
        e = sum(exp.values())
        d = sum(done.values())
        grand_done += d
        grand_exp += e
        rows.append((mt, d, e, pct(d, e), done, exp))

    for mt, d, e, p1, done, exp in rows:
        print(f"{mt:8s} {d}/{e} ({p1:.2f}%)")
        print(f"  train {done['train']}/{exp['train']}  val {done['val']}/{exp['val']}  test {done['test']}/{exp['test']}")
    print(f"TOTAL    {grand_done}/{grand_exp} ({pct(grand_done, grand_exp):.2f}%)")


if __name__ == "__main__":
    main()

