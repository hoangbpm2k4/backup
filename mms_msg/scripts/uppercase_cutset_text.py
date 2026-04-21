#!/usr/bin/env python3
"""Uppercase all supervision text in Lhotse CutSet jsonl.gz files.

Reads each .jsonl.gz, uppercases the 'text' field in all supervisions,
writes to a temp file then replaces the original.

Usage:
    python -u uppercase_cutset_text.py [--workers 8]
"""

import argparse
import gzip
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


CUTSET_DIR = Path("/home/vietnam/voice_to_text/data/lhotse_cutsets")


def uppercase_one_file(filepath: str) -> str:
    """Read jsonl.gz, uppercase supervision text, write back."""
    fp = Path(filepath)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl.gz", dir=str(fp.parent))
    os.close(tmp_fd)

    count = 0
    changed = 0
    try:
        with gzip.open(str(fp), "rt", encoding="utf-8") as fin, \
             gzip.open(tmp_path, "wt", encoding="utf-8") as fout:
            for line in fin:
                count += 1
                obj = json.loads(line)
                modified = False

                # Handle supervisions in cuts
                if "supervisions" in obj:
                    for sup in obj["supervisions"]:
                        if "text" in sup and sup["text"] != sup["text"].upper():
                            sup["text"] = sup["text"].upper()
                            modified = True

                # Handle MixedCut: tracks → cuts → supervisions
                if "tracks" in obj:
                    for track in obj["tracks"]:
                        cut = track.get("cut", {})
                        for sup in cut.get("supervisions", []):
                            if "text" in sup and sup["text"] != sup["text"].upper():
                                sup["text"] = sup["text"].upper()
                                modified = True

                if modified:
                    changed += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Replace original
        os.replace(tmp_path, str(fp))
        return f"OK {fp.name}: {count:,} lines, {changed:,} uppercased"
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return f"FAIL {fp.name}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dir", type=str, default=str(CUTSET_DIR))
    args = parser.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob("*.jsonl.gz"))
    print(f"Found {len(files)} files in {d}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(uppercase_one_file, str(f)): f.name for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            print(f"[{i}/{len(files)}] {result}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
