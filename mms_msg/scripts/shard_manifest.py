#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shard a jsonl manifest into N disjoint files.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    return p.parse_args()


def shard_id(key: str, n: int) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


def main() -> None:
    args = parse_args()
    n = max(1, int(args.num_shards))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outs = [(args.out_dir / f"shard_{i:03d}.jsonl").open("w", encoding="utf-8", newline="\n") for i in range(n)]
    counts = [0] * n
    try:
        with args.manifest.open("r", encoding="utf-8") as rf:
            for line in rf:
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = str(obj.get("utt_id") or obj.get("audio_path") or "")
                sid = shard_id(key, n)
                outs[sid].write(json.dumps(obj, ensure_ascii=False) + "\n")
                counts[sid] += 1
    finally:
        for f in outs:
            f.close()

    summary = {
        "manifest": args.manifest.as_posix(),
        "num_shards": n,
        "counts_per_shard": counts,
        "total": sum(counts),
        "out_dir": args.out_dir.as_posix(),
    }
    (args.out_dir / "shard_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

