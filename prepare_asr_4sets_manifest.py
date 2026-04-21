#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified ASR manifest from data/audio layout")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/asr_manifest_4sets.jsonl"))
    parser.add_argument("--out-tsv", type=Path, default=Path("data/asr_audio_text_4sets.tsv"))
    parser.add_argument(
        "--include",
        nargs="*",
        default=["vlsp2020_vinai_100h", "phoaudiobook", "viet_bud500", "gigaspeech2_vi", "vietspeech"],
    )
    parser.add_argument("--max-rows-per-dataset", type=int, default=0, help="0 = full")
    parser.add_argument("--check-gigaspeech-audio-exists", action="store_true")
    return parser.parse_args()


def split_from_filename(path: Path) -> str:
    n = path.name
    if n.startswith("train-"):
        return "train"
    if n.startswith("validation-"):
        return "validation"
    if n.startswith("test-"):
        return "test"
    return "unknown"


def text_field_for_dataset(dataset: str) -> str:
    return "text" if dataset == "phoaudiobook" else "transcription"


def iter_parquet_rows(dataset_name: str, parquet_dir: Path) -> Iterator[Dict[str, object]]:
    files = sorted(parquet_dir.glob("*.parquet"))
    text_field = text_field_for_dataset(dataset_name)

    for p_idx, parquet_path in enumerate(files):
        split = split_from_filename(parquet_path)
        pf = pq.ParquetFile(parquet_path)
        row_in_file = 0

        for batch in pf.iter_batches(batch_size=1024, columns=["audio", text_field]):
            for r in batch.to_pylist():
                audio_obj = r.get("audio")
                text = (r.get(text_field) or "").strip()
                if not isinstance(audio_obj, dict) or not text:
                    row_in_file += 1
                    continue

                utt_id = f"{dataset_name}__{split}__p{p_idx:05d}__r{row_in_file:08d}"
                audio_path = f"parquet://{parquet_path.resolve().as_posix()}#row={row_in_file}"
                source_audio_name = audio_obj.get("path") if isinstance(audio_obj.get("path"), str) else None

                yield {
                    "utt_id": utt_id,
                    "dataset": dataset_name,
                    "split": split,
                    "audio_path": audio_path,
                    "text": text,
                    "text_path": f"jsonl://manifest#utt_id={utt_id}",
                    "source_file": parquet_path.resolve().as_posix(),
                    "source_row": row_in_file,
                    "source_audio_path": source_audio_name,
                }
                row_in_file += 1


def parse_tsv_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.rstrip("\n")
    if not line:
        return None
    parts = line.split("\t", 1)
    if len(parts) != 2:
        return None
    utt_id, text = parts[0].strip(), parts[1].strip()
    if not utt_id or not text:
        return None
    return utt_id, text


def gigaspeech_wav_path(audio_root: Path, split: str, utt_id: str) -> Path:
    toks = utt_id.split("-")
    if split in {"dev", "test"}:
        if len(toks) < 2:
            return audio_root / split / f"{utt_id}.wav"
        return audio_root / split / toks[0] / f"{utt_id}.wav"
    if len(toks) < 3:
        return audio_root / "release1" / "train" / toks[0] / f"{utt_id}.wav"
    return audio_root / "release1" / "train" / toks[0] / toks[1] / f"{utt_id}.wav"


def iter_gigaspeech_rows(tsv_root: Path, audio_root: Path, check_exists: bool) -> Iterator[Dict[str, object]]:
    split_to_tsv = {
        "train": tsv_root / "train_refined.tsv",
        "dev": tsv_root / "dev.tsv",
        "test": tsv_root / "test.tsv",
    }

    for split, tsv_path in split_to_tsv.items():
        if not tsv_path.exists():
            continue
        with tsv_path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, raw in enumerate(f):
                parsed = parse_tsv_line(raw)
                if parsed is None:
                    continue
                utt_raw, text = parsed
                wav_path = gigaspeech_wav_path(audio_root, split, utt_raw)
                if check_exists and (not wav_path.exists()):
                    continue
                utt_id = f"gigaspeech2_vi__{split}__{utt_raw}"
                yield {
                    "utt_id": utt_id,
                    "dataset": "gigaspeech2_vi",
                    "split": split,
                    "audio_path": wav_path.resolve().as_posix(),
                    "text": text,
                    "text_path": f"jsonl://manifest#utt_id={utt_id}",
                    "source_file": tsv_path.resolve().as_posix(),
                    "source_row": i,
                    "source_audio_path": wav_path.as_posix(),
                }


def iter_vietspeech_arrow_rows(arrow_root: Path) -> Iterator[Dict[str, object]]:
    arrow_files = sorted(arrow_root.glob("data-*-of-*.arrow"))
    for f_idx, arrow_path in enumerate(arrow_files):
        with arrow_path.open("rb") as f:
            reader = ipc.open_stream(f)
            row_in_file = 0
            for batch in reader:
                rows = batch.to_pylist()
                for r in rows:
                    audio_obj = r.get("audio")
                    text = (r.get("text") or "").strip()
                    sid = (r.get("sid") or "").strip()
                    if not isinstance(audio_obj, dict) or not text:
                        row_in_file += 1
                        continue

                    utt_key = sid if sid else f"f{f_idx:05d}_r{row_in_file:08d}"
                    utt_id = f"vietspeech__train__{utt_key}"
                    audio_path = f"arrow://{arrow_path.resolve().as_posix()}#row={row_in_file}"
                    source_audio_name = audio_obj.get("path") if isinstance(audio_obj.get("path"), str) else None

                    yield {
                        "utt_id": utt_id,
                        "dataset": "vietspeech",
                        "split": "train",
                        "audio_path": audio_path,
                        "text": text,
                        "text_path": f"jsonl://manifest#utt_id={utt_id}",
                        "source_file": arrow_path.resolve().as_posix(),
                        "source_row": row_in_file,
                        "source_audio_path": source_audio_name,
                    }
                    row_in_file += 1


def write_outputs(records: Iterable[Dict[str, object]], out_jsonl: Path, out_tsv: Path) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as jf, out_tsv.open("w", encoding="utf-8", newline="") as tf:
        writer = csv.writer(tf, delimiter="\t", lineterminator="\n")
        writer.writerow(["audio_path", "text"])
        for rec in records:
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.writerow([rec["audio_path"], rec["text"]])
            n += 1
    return n


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out_jsonl = args.out_jsonl if args.out_jsonl.is_absolute() else root / args.out_jsonl
    out_tsv = args.out_tsv if args.out_tsv.is_absolute() else root / args.out_tsv

    include = set(args.include)
    generators: List[Tuple[str, Iterator[Dict[str, object]]]] = []

    if "vlsp2020_vinai_100h" in include:
        generators.append(
            (
                "vlsp2020_vinai_100h",
                iter_parquet_rows(
                    "vlsp2020_vinai_100h",
                    root / "data" / "audio" / "vlsp2020_vinai_100h" / "parquet",
                ),
            )
        )
    if "phoaudiobook" in include:
        generators.append(
            (
                "phoaudiobook",
                iter_parquet_rows("phoaudiobook", root / "data" / "audio" / "phoaudiobook" / "parquet"),
            )
        )
    if "viet_bud500" in include:
        generators.append(
            (
                "viet_bud500",
                iter_parquet_rows("viet_bud500", root / "data" / "audio" / "viet_bud500" / "parquet"),
            )
        )
    if "gigaspeech2_vi" in include:
        generators.append(
            (
                "gigaspeech2_vi",
                iter_gigaspeech_rows(
                    root / "data" / "meta" / "gigaspeech2_vi",
                    root / "data" / "audio" / "gigaspeech2_vi",
                    args.check_gigaspeech_audio_exists,
                ),
            )
        )
    if "vietspeech" in include:
        generators.append(
            (
                "vietspeech",
                iter_vietspeech_arrow_rows(root / "data" / "audio" / "vietspeech"),
            )
        )

    def merged() -> Iterator[Dict[str, object]]:
        counts: Dict[str, int] = {}
        for ds_name, gen in generators:
            counts[ds_name] = 0
            for rec in gen:
                counts[ds_name] += 1
                if args.max_rows_per_dataset and counts[ds_name] > args.max_rows_per_dataset:
                    break
                yield rec
            print(f"[DONE] {ds_name}: {counts[ds_name]:,} records", flush=True)

    total = write_outputs(merged(), out_jsonl, out_tsv)
    print(f"[DONE] Wrote total {total:,} records")
    print(f"[OUT] JSONL: {out_jsonl}")
    print(f"[OUT] TSV:   {out_tsv}")


if __name__ == "__main__":
    main()
