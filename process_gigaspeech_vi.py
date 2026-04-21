#!/usr/bin/env python3
"""
Process GigaSpeech2 Vietnamese subset (gigaspeech2_vi).

Functions:
1) Optional extraction of tar.gz audio archives with skip-existing behavior.
2) Build normalized shard manifests compatible with normalized_3sets layout:
   manifests/shards/gigaspeech2_vi/{split}/00000.jsonl

After building manifests, you can rebuild merged manifests with:
python normalize_3datasets.py --out-root normalized_3sets --rebuild-merged-only --clean-merged-manifests
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process GigaSpeech2 Vietnamese subset")
    parser.add_argument("--data-root", type=Path, default=Path("gigaspeech2_vi/data/vi"))
    parser.add_argument("--out-root", type=Path, default=Path("normalized_3sets"))
    parser.add_argument(
        "--train-transcript",
        type=str,
        default="train_refined.tsv",
        choices=["train_refined.tsv", "train_raw.tsv"],
        help="Train transcript file to use",
    )
    parser.add_argument("--extract", action="store_true", help="Extract dev/test/train tar.gz archives")
    parser.add_argument(
        "--delete-archive-on-success",
        action="store_true",
        help="Delete each tar.gz archive only after extract state is marked done and saved.",
    )
    parser.add_argument(
        "--extract-state",
        type=Path,
        default=None,
        help=(
            "Path to extraction state json. Default: <data-root>/extract_state.json. "
            "Used to skip archives that were fully processed in previous runs."
        ),
    )
    parser.add_argument(
        "--ignore-extract-state",
        action="store_true",
        help="Ignore extract state and re-scan all archives.",
    )
    parser.add_argument(
        "--reset-extract-state",
        action="store_true",
        help="Delete existing extract state before extraction.",
    )
    parser.add_argument("--build-manifest", action="store_true", help="Build JSONL manifests")
    parser.add_argument(
        "--check-audio-exists",
        action="store_true",
        help="Validate each audio path exists while building manifest (slower)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Debug only: limit lines per split manifest (0 = full)",
    )
    return parser.parse_args()


def iter_tar_paths(data_root: Path) -> List[Path]:
    train_tars = sorted((data_root / "train").glob("*.tar.gz"), key=lambda p: p.name)
    return [data_root / "dev.tar.gz", data_root / "test.tar.gz", *train_tars]


def load_extract_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"version": 1, "archives": {}}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 1, "archives": {}}
        if "archives" not in data or not isinstance(data["archives"], dict):
            data["archives"] = {}
        return data
    except Exception:
        return {"version": 1, "archives": {}}


def save_extract_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def archive_key(data_root: Path, tar_path: Path) -> str:
    # Keep key based on archive path under data_root without dereferencing symlinks.
    return tar_path.absolute().relative_to(data_root.absolute()).as_posix()


def archive_signature(tar_path: Path) -> Dict[str, int]:
    st = tar_path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def is_archive_done(state: Dict[str, Any], key: str, sig: Dict[str, int]) -> bool:
    entry = state.get("archives", {}).get(key)
    if not isinstance(entry, dict):
        return False
    return (
        bool(entry.get("done"))
        and int(entry.get("size", -1)) == sig["size"]
        and int(entry.get("mtime_ns", -1)) == sig["mtime_ns"]
    )


def mark_archive_done(
    state: Dict[str, Any], key: str, sig: Dict[str, int], extracted: int, skipped: int
) -> None:
    archives = state.setdefault("archives", {})
    archives[key] = {
        "done": True,
        "size": sig["size"],
        "mtime_ns": sig["mtime_ns"],
        "extracted": int(extracted),
        "skipped_existing": int(skipped),
    }


def mark_archive_deleted(state: Dict[str, Any], key: str, deleted: bool) -> None:
    archives = state.setdefault("archives", {})
    item = archives.setdefault(key, {})
    if isinstance(item, dict):
        item["archive_deleted"] = bool(deleted)


def extract_tar_skip_existing(tar_path: Path, dest_root: Path) -> Tuple[int, int]:
    extracted = 0
    skipped = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            target = dest_root / m.name
            if target.exists():
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                skipped += 1
                continue
            with src, target.open("wb") as out_f:
                shutil.copyfileobj(src, out_f, length=1024 * 1024)
            extracted += 1
    return extracted, skipped


def parse_tsv_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.rstrip("\n")
    if not line:
        return None
    parts = line.split("\t", 1)
    if len(parts) != 2:
        return None
    utt_id = parts[0].strip()
    text = parts[1].strip()
    if not utt_id or not text:
        return None
    return utt_id, text


def audio_path_for_utt(data_root: Path, split: str, utt_id: str) -> Path:
    toks = utt_id.split("-")
    if split in {"dev", "test"}:
        if len(toks) < 2:
            return data_root / split / f"{utt_id}.wav"
        return data_root / split / toks[0] / f"{utt_id}.wav"
    # train: expected format a-b-c
    if len(toks) < 3:
        return data_root / "release1" / "train" / toks[0] / f"{utt_id}.wav"
    return data_root / "release1" / "train" / toks[0] / toks[1] / f"{utt_id}.wav"


def build_manifest_for_split(
    *,
    split: str,
    tsv_path: Path,
    data_root: Path,
    out_manifest_path: Path,
    check_audio_exists: bool,
    max_lines: int,
) -> Dict[str, int]:
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    missing_audio = 0
    bad_rows = 0

    with tsv_path.open("r", encoding="utf-8", errors="ignore") as in_f, out_manifest_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as out_f:
        for i, raw in enumerate(in_f):
            parsed = parse_tsv_line(raw)
            if parsed is None:
                bad_rows += 1
                continue

            utt_id_raw, text = parsed
            audio_path = audio_path_for_utt(data_root, split, utt_id_raw)

            total += 1
            if max_lines and total > max_lines:
                break

            if check_audio_exists and (not audio_path.exists()):
                missing_audio += 1
                continue

            record = {
                "utt_id": f"gigaspeech2_vi__{split}__{utt_id_raw}",
                "dataset": "gigaspeech2_vi",
                "split": split,
                "audio_path": audio_path.resolve().as_posix(),
                "text": text,
                "speaker_id": None,
                "recording_id": f"gigaspeech2_vi::{utt_id_raw}",
                "start_sec": 0.0,
                "end_sec": None,
                "duration_sec": None,
                "sample_rate": None,
                "num_samples": None,
                "source_file": tsv_path.resolve().as_posix(),
                "source_row": i,
                "source_audio_path": audio_path.as_posix(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written and written % 200000 == 0:
                print(f"[PROGRESS] {split}: written={written:,}", flush=True)

    return {
        "rows_total": total,
        "rows_written": written,
        "rows_bad": bad_rows,
        "rows_missing_audio": missing_audio,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    state_path = args.extract_state.resolve() if args.extract_state else (data_root / "extract_state.json")

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    if not args.extract and not args.build_manifest:
        raise ValueError("Nothing to do. Use --extract and/or --build-manifest")

    if args.extract:
        if args.reset_extract_state and state_path.exists():
            state_path.unlink(missing_ok=True)
            print(f"[STATE] reset extract state: {state_path}", flush=True)
        state = load_extract_state(state_path)

        tar_paths = iter_tar_paths(data_root)
        for tar_path in tar_paths:
            if not tar_path.exists():
                print(f"[WARN] Missing archive: {tar_path}")
                continue
            key = archive_key(data_root, tar_path)
            sig = archive_signature(tar_path)

            if not args.ignore_extract_state and is_archive_done(state, key, sig):
                entry = state["archives"][key]
                print(
                    f"[SKIP_STATE] {tar_path.name}: "
                    f"extracted={entry.get('extracted', 0):,}, "
                    f"skipped_existing={entry.get('skipped_existing', 0):,}",
                    flush=True,
                )
                if args.delete_archive_on_success and tar_path.exists():
                    try:
                        tar_path.unlink()
                        mark_archive_deleted(state, key, True)
                        save_extract_state(state_path, state)
                        print(f"[DEL] {tar_path}", flush=True)
                    except Exception as e:
                        print(f"[WARN] Failed to delete archive after skip-state: {tar_path} ({e})", flush=True)
                continue

            print(f"[EXTRACT] {tar_path.name}", flush=True)
            extracted, skipped = extract_tar_skip_existing(tar_path, data_root)
            print(
                f"[DONE] {tar_path.name}: extracted={extracted:,}, skipped_existing={skipped:,}",
                flush=True,
            )
            mark_archive_done(state, key, sig, extracted, skipped)
            save_extract_state(state_path, state)
            if args.delete_archive_on_success:
                try:
                    tar_path.unlink()
                    mark_archive_deleted(state, key, True)
                    save_extract_state(state_path, state)
                    print(f"[DEL] {tar_path}", flush=True)
                except Exception as e:
                    mark_archive_deleted(state, key, False)
                    save_extract_state(state_path, state)
                    print(f"[WARN] Failed to delete archive after done-state: {tar_path} ({e})", flush=True)

        print(f"[STATE] saved: {state_path}", flush=True)

    if args.build_manifest:
        split_to_tsv = {
            "dev": data_root / "dev.tsv",
            "test": data_root / "test.tsv",
            "train": data_root / args.train_transcript,
        }

        for split, tsv in split_to_tsv.items():
            if not tsv.exists():
                raise FileNotFoundError(f"Missing transcript file: {tsv}")
            out_manifest = (
                out_root
                / "manifests"
                / "shards"
                / "gigaspeech2_vi"
                / split
                / "00000.jsonl"
            )
            print(f"[MANIFEST] split={split}, transcript={tsv.name}", flush=True)
            stats = build_manifest_for_split(
                split=split,
                tsv_path=tsv,
                data_root=data_root,
                out_manifest_path=out_manifest,
                check_audio_exists=args.check_audio_exists,
                max_lines=args.max_lines,
            )
            print(f"[DONE] {split}: {stats}", flush=True)

    print("[DONE] process_gigaspeech_vi", flush=True)


if __name__ == "__main__":
    main()
