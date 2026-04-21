#!/usr/bin/env python3
"""Copy segments_denoised/* files referenced by accepted_manifest_combined.csv
into accepted/, and write a unified CSV with all paths under accepted/.

Safe: never touches source dir, skips if destination exists.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-in", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/data_label_auto_yt_1900h/manifests/accepted_manifest_combined.csv"))
    ap.add_argument("--csv-out", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/data_label_auto_yt_1900h/manifests/accepted_manifest_combined.unified.csv"))
    ap.add_argument("--accepted-dir", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/data_label_auto_yt_1900h/accepted"))
    ap.add_argument("--segments-dir", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/data_label_auto_yt_1900h/segments_denoised"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv.field_size_limit(10_000_000)
    args.accepted_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_copy = 0
    n_skip_exists = 0
    n_missing_src = 0
    n_already_accepted = 0
    total_dur = 0.0

    with args.csv_in.open("r", encoding="utf-8") as rf, \
         args.csv_out.open("w", encoding="utf-8", newline="") as wf:
        reader = csv.DictReader(rf)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            n_rows += 1
            src_path = Path(row["final_wav"])
            try:
                total_dur += float(row.get("duration_sec") or 0.0)
            except ValueError:
                pass

            if src_path.parent == args.accepted_dir:
                n_already_accepted += 1
                writer.writerow(row)
                continue

            # row points somewhere else (segments_denoised) -> copy to accepted
            dst_path = args.accepted_dir / src_path.name
            if dst_path.exists():
                n_skip_exists += 1
            elif not src_path.exists():
                n_missing_src += 1
                print(f"MISSING SRC: {src_path}", file=sys.stderr)
                continue
            else:
                if args.dry_run:
                    pass
                else:
                    shutil.copy2(src_path, dst_path)
                n_copy += 1

            row["final_wav"] = str(dst_path)
            writer.writerow(row)

            if (n_copy + n_skip_exists) % 2000 == 0 and (n_copy + n_skip_exists) > 0:
                print(f"  progress: copied={n_copy} existed={n_skip_exists} missing={n_missing_src}")

    print("=" * 60)
    print(f"Total rows:          {n_rows}")
    print(f"Already in accepted: {n_already_accepted}")
    print(f"Copied:              {n_copy}")
    print(f"Already existed at dst: {n_skip_exists}")
    print(f"Missing source:      {n_missing_src}")
    print(f"Total duration (h):  {total_dur/3600:.2f}")
    print(f"Unified CSV:         {args.csv_out}")


if __name__ == "__main__":
    main()
