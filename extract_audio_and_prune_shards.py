#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio bytes from parquet/arrow shards, append manifest, then optionally delete processed shard."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/asr_manifest_extracted.jsonl"))
    parser.add_argument("--out-tsv", type=Path, default=Path("data/asr_audio_text_extracted.tsv"))
    parser.add_argument("--state-path", type=Path, default=Path("data/extract_state.json"))
    parser.add_argument(
        "--include",
        nargs="*",
        default=[
            "vlsp2020_vinai_100h",
            "phoaudiobook",
            "viet_bud500",
            "vietspeech",
            "fpt_fosd",
            "vais1000",
            "viVoice",
        ],
    )
    parser.add_argument("--max-shards-per-dataset", type=int, default=0, help="0 means full")
    parser.add_argument("--delete-source-on-success", action="store_true")
    parser.add_argument("--target-sample-rate", type=int, default=16000, help="0 keeps original sample rate")
    parser.add_argument("--uppercase-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1024, help="PyArrow parquet batch size when iterating rows")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Audio decode/write workers. 0=auto, 1=single-threaded",
    )
    parser.add_argument("--reset-state", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else (root / p)


def find_parquet_shards(audio_root: Path, dataset: str) -> List[Path]:
    dataset_root = audio_root / dataset
    candidates = [dataset_root / "parquet", dataset_root / "data", dataset_root]
    shards: List[Path] = []
    seen = set()

    for c in candidates:
        if not c.exists() or (not c.is_dir()):
            continue
        for p in sorted(c.glob("*.parquet")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            shards.append(p)

    # Fallback for unexpected nested layouts from some snapshots.
    # Do not recurse whole dataset_root to avoid scanning huge extracted wav trees.
    if not shards:
        for base in (dataset_root / "parquet", dataset_root / "data"):
            if not base.exists() or (not base.is_dir()):
                continue
            for p in sorted(base.rglob("*.parquet")):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                shards.append(p)

    return shards


def find_arrow_shards(audio_root: Path, dataset: str) -> List[Path]:
    dataset_root = audio_root / dataset
    candidates = [dataset_root, dataset_root / "data"]
    shards: List[Path] = []
    seen = set()

    for c in candidates:
        if not c.exists() or (not c.is_dir()):
            continue
        for pat in ("data-*-of-*.arrow", "*.arrow"):
            for p in sorted(c.glob(pat)):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                shards.append(p)

    if not shards:
        for base in (dataset_root / "data", dataset_root / "arrow"):
            if not base.exists() or (not base.is_dir()):
                continue
            for p in sorted(base.rglob("*.arrow")):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                shards.append(p)

    return shards


def load_state(path: Path, reset: bool) -> Dict[str, object]:
    if reset or (not path.exists()):
        return {"version": 1, "datasets": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 1, "datasets": {}}
        if "datasets" not in data or not isinstance(data["datasets"], dict):
            data["datasets"] = {}
        return data
    except Exception:
        return {"version": 1, "datasets": {}}


def save_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def split_from_parquet_name(name: str) -> str:
    if name.startswith("train-"):
        return "train"
    if name.startswith("validation-"):
        return "validation"
    if name.startswith("test-"):
        return "test"
    return "train"


def sanitize_name(s: str, max_len: int = 80) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    if not s:
        s = "audio"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def ensure_wav_name(name: Optional[str], fallback: str) -> str:
    if isinstance(name, str) and name.strip():
        base = Path(name).name
        if not base.lower().endswith(".wav"):
            base = f"{base}.wav"
        return sanitize_name(base)
    return f"{sanitize_name(fallback)}.wav"


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if audio.size == 0:
        return audio
    src_n = int(audio.shape[0])
    dst_n = max(1, int(round(src_n * float(dst_sr) / float(src_sr))))
    x_old = np.linspace(0.0, 1.0, num=src_n, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=dst_n, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)


def normalize_text(text: str, uppercase: bool) -> str:
    s = (text or "").strip()
    if uppercase:
        s = s.upper()
    return s


def write_audio_bytes(
    audio_bytes: bytes,
    target: Path,
    *,
    source_audio_name: Optional[str],
    target_sample_rate: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    src_ext = Path(source_audio_name).suffix if source_audio_name else ""
    if not src_ext:
        src_ext = ".wav"
    tmp_src = target.with_suffix(target.suffix + ".src" + src_ext.lower())
    with tmp_src.open("wb") as f:
        f.write(audio_bytes)
    try:
        audio, sr = sf.read(str(tmp_src), dtype="float32")
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            mono = arr
        else:
            mono = np.mean(arr, axis=1) if arr.shape[1] <= 8 else np.mean(arr, axis=-1)
            if mono.ndim != 1:
                mono = mono.reshape(-1)
        out_sr = int(sr)
        if target_sample_rate > 0 and out_sr != target_sample_rate:
            mono = resample_linear(mono, out_sr, target_sample_rate)
            out_sr = target_sample_rate
        sf.write(str(target), mono, out_sr, subtype="PCM_16")
    except Exception:
        # Fallback to raw bytes when decode/resample fails.
        tmp_raw = target.with_suffix(target.suffix + ".tmp")
        with tmp_raw.open("wb") as f:
            f.write(audio_bytes)
        tmp_raw.replace(target)
    finally:
        try:
            if tmp_src.exists():
                tmp_src.unlink()
        except Exception:
            pass


def iter_parquet_rows(shard_path: Path, text_field: str, batch_size: int) -> Iterator[Dict[str, object]]:
    pf = pq.ParquetFile(shard_path)
    row_idx = 0
    for batch in pf.iter_batches(batch_size=max(1, int(batch_size)), columns=["audio", text_field]):
        for row in batch.to_pylist():
            audio_obj = row.get("audio")
            text = (row.get(text_field) or "").strip()
            if not isinstance(audio_obj, dict) or (not text):
                row_idx += 1
                continue
            audio_bytes = audio_obj.get("bytes")
            if not isinstance(audio_bytes, (bytes, bytearray)):
                row_idx += 1
                continue
            yield {
                "row_idx": row_idx,
                "text": text,
                "audio_bytes": bytes(audio_bytes),
                "source_audio_name": audio_obj.get("path"),
            }
            row_idx += 1


def iter_arrow_rows(shard_path: Path) -> Iterator[Dict[str, object]]:
    with shard_path.open("rb") as f:
        reader = ipc.open_stream(f)
        row_idx = 0
        for batch in reader:
            for row in batch.to_pylist():
                audio_obj = row.get("audio")
                text = (row.get("text") or "").strip()
                sid = (row.get("sid") or "").strip()
                if not isinstance(audio_obj, dict) or (not text):
                    row_idx += 1
                    continue
                audio_bytes = audio_obj.get("bytes")
                if not isinstance(audio_bytes, (bytes, bytearray)):
                    row_idx += 1
                    continue
                yield {
                    "row_idx": row_idx,
                    "text": text,
                    "audio_bytes": bytes(audio_bytes),
                    "source_audio_name": audio_obj.get("path"),
                    "sid": sid,
                }
                row_idx += 1


def open_manifest_files(jsonl_path: Path, tsv_path: Path) -> tuple:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_f = jsonl_path.open("a", encoding="utf-8", newline="\n")
    need_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
    tsv_f = tsv_path.open("a", encoding="utf-8", newline="")
    tsv_writer = csv.writer(tsv_f, delimiter="\t", lineterminator="\n")
    if need_header:
        tsv_writer.writerow(["audio_path", "text"])
    return jsonl_f, tsv_f, tsv_writer


def append_record(jsonl_f, tsv_writer, rec: Dict[str, object]) -> None:
    jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tsv_writer.writerow([rec["audio_path"], rec["text"]])


def resolve_workers(raw_workers: int) -> int:
    if raw_workers > 0:
        return raw_workers
    cpu = os.cpu_count() or 1
    return max(1, min(32, cpu))


def process_parquet_row(
    *,
    dataset: str,
    split: str,
    shard_id: str,
    out_dir: Path,
    shard_path: Path,
    item: Dict[str, object],
    target_sample_rate: int,
    uppercase_text: bool,
) -> Dict[str, object]:
    row_idx = int(item["row_idx"])
    src_name = ensure_wav_name(item.get("source_audio_name"), fallback=f"{dataset}_{row_idx}")
    file_name = f"{row_idx:08d}_{src_name}"
    wav_path = out_dir / file_name
    write_audio_bytes(
        item["audio_bytes"],
        wav_path,
        source_audio_name=item.get("source_audio_name"),
        target_sample_rate=target_sample_rate,
    )

    utt_id = f"{dataset}__{split}__{shard_id}__r{row_idx:08d}"
    return {
        "utt_id": utt_id,
        "dataset": dataset,
        "split": split,
        "audio_path": wav_path.resolve().as_posix(),
        "text": normalize_text(item["text"], uppercase=uppercase_text),
        "text_path": f"jsonl://manifest#utt_id={utt_id}",
        "source_file": shard_path.resolve().as_posix(),
        "source_row": row_idx,
        "source_audio_path": item.get("source_audio_name"),
    }


def process_vietspeech_row(
    *,
    dataset: str,
    split: str,
    out_dir: Path,
    shard_path: Path,
    item: Dict[str, object],
    target_sample_rate: int,
    uppercase_text: bool,
) -> Dict[str, object]:
    row_idx = int(item["row_idx"])
    sid = sanitize_name(item.get("sid") or "", max_len=120)
    src_name = ensure_wav_name(item.get("source_audio_name"), fallback=f"{dataset}_{sid}_{row_idx}")
    file_name = f"{row_idx:08d}_{src_name}"
    wav_path = out_dir / file_name
    write_audio_bytes(
        item["audio_bytes"],
        wav_path,
        source_audio_name=item.get("source_audio_name"),
        target_sample_rate=target_sample_rate,
    )

    sid_part = sid if sid else f"r{row_idx:08d}"
    utt_id = f"{dataset}__{split}__{sid_part}"
    return {
        "utt_id": utt_id,
        "dataset": dataset,
        "split": split,
        "audio_path": wav_path.resolve().as_posix(),
        "text": normalize_text(item["text"], uppercase=uppercase_text),
        "text_path": f"jsonl://manifest#utt_id={utt_id}",
        "source_file": shard_path.resolve().as_posix(),
        "source_row": row_idx,
        "source_audio_path": item.get("source_audio_name"),
    }


def dataset_done_set(state: Dict[str, object], dataset: str) -> set:
    datasets = state.setdefault("datasets", {})
    item = datasets.setdefault(dataset, {"done_shards": []})
    done = item.setdefault("done_shards", [])
    return set(done)


def mark_dataset_done(state: Dict[str, object], dataset: str, shard_rel: str) -> None:
    datasets = state.setdefault("datasets", {})
    item = datasets.setdefault(dataset, {"done_shards": []})
    done = item.setdefault("done_shards", [])
    if shard_rel not in done:
        done.append(shard_rel)


def process_parquet_dataset(
    *,
    dataset: str,
    shard_paths: List[Path],
    text_field: str,
    audio_root: Path,
    jsonl_f,
    tsv_writer,
    state: Dict[str, object],
    state_path: Path,
    project_root: Path,
    max_shards: int,
    delete_on_success: bool,
    target_sample_rate: int,
    uppercase_text: bool,
    batch_size: int,
    workers: int,
) -> None:
    done = dataset_done_set(state, dataset)
    processed = 0
    for shard_path in shard_paths:
        shard_rel = shard_path.resolve().relative_to(project_root.resolve()).as_posix()
        if shard_rel in done:
            continue
        if max_shards and processed >= max_shards:
            break

        split = split_from_parquet_name(shard_path.name)
        shard_id = sanitize_name(shard_path.stem, max_len=120)
        out_dir = audio_root / dataset / "wav" / split / shard_id
        rows_written = 0

        if workers <= 1:
            for item in iter_parquet_rows(shard_path, text_field=text_field, batch_size=batch_size):
                rec = process_parquet_row(
                    dataset=dataset,
                    split=split,
                    shard_id=shard_id,
                    out_dir=out_dir,
                    shard_path=shard_path,
                    item=item,
                    target_sample_rate=target_sample_rate,
                    uppercase_text=uppercase_text,
                )
                append_record(jsonl_f, tsv_writer, rec)
                rows_written += 1
        else:
            pending: set = set()
            max_pending = max(workers * 4, workers)
            with cf.ThreadPoolExecutor(max_workers=workers) as executor:
                for item in iter_parquet_rows(shard_path, text_field=text_field, batch_size=batch_size):
                    fut = executor.submit(
                        process_parquet_row,
                        dataset=dataset,
                        split=split,
                        shard_id=shard_id,
                        out_dir=out_dir,
                        shard_path=shard_path,
                        item=item,
                        target_sample_rate=target_sample_rate,
                        uppercase_text=uppercase_text,
                    )
                    pending.add(fut)
                    if len(pending) >= max_pending:
                        finished_futures, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                        for d in finished_futures:
                            rec = d.result()
                            append_record(jsonl_f, tsv_writer, rec)
                            rows_written += 1

                for d in cf.as_completed(pending):
                    rec = d.result()
                    append_record(jsonl_f, tsv_writer, rec)
                    rows_written += 1

        mark_dataset_done(state, dataset, shard_rel)
        save_state(state_path, state)
        processed += 1
        print(f"[OK] {dataset} shard={shard_path.name} rows={rows_written:,}", flush=True)

        if delete_on_success:
            shard_path.unlink()
            print(f"[DEL] {shard_path}", flush=True)


def process_vietspeech_arrow(
    *,
    dataset: str,
    shard_paths: List[Path],
    audio_root: Path,
    jsonl_f,
    tsv_writer,
    state: Dict[str, object],
    state_path: Path,
    project_root: Path,
    max_shards: int,
    delete_on_success: bool,
    target_sample_rate: int,
    uppercase_text: bool,
    workers: int,
) -> None:
    done = dataset_done_set(state, dataset)
    processed = 0
    for shard_path in shard_paths:
        shard_rel = shard_path.resolve().relative_to(project_root.resolve()).as_posix()
        if shard_rel in done:
            continue
        if max_shards and processed >= max_shards:
            break

        split = "train"
        shard_id = sanitize_name(shard_path.stem, max_len=120)
        out_dir = audio_root / dataset / "wav" / split / shard_id
        rows_written = 0

        if workers <= 1:
            for item in iter_arrow_rows(shard_path):
                rec = process_vietspeech_row(
                    dataset=dataset,
                    split=split,
                    out_dir=out_dir,
                    shard_path=shard_path,
                    item=item,
                    target_sample_rate=target_sample_rate,
                    uppercase_text=uppercase_text,
                )
                append_record(jsonl_f, tsv_writer, rec)
                rows_written += 1
        else:
            pending: set = set()
            max_pending = max(workers * 4, workers)
            with cf.ThreadPoolExecutor(max_workers=workers) as executor:
                for item in iter_arrow_rows(shard_path):
                    fut = executor.submit(
                        process_vietspeech_row,
                        dataset=dataset,
                        split=split,
                        out_dir=out_dir,
                        shard_path=shard_path,
                        item=item,
                        target_sample_rate=target_sample_rate,
                        uppercase_text=uppercase_text,
                    )
                    pending.add(fut)
                    if len(pending) >= max_pending:
                        finished_futures, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                        for d in finished_futures:
                            rec = d.result()
                            append_record(jsonl_f, tsv_writer, rec)
                            rows_written += 1

                for d in cf.as_completed(pending):
                    rec = d.result()
                    append_record(jsonl_f, tsv_writer, rec)
                    rows_written += 1

        mark_dataset_done(state, dataset, shard_rel)
        save_state(state_path, state)
        processed += 1
        print(f"[OK] {dataset} shard={shard_path.name} rows={rows_written:,}", flush=True)

        if delete_on_success:
            shard_path.unlink()
            print(f"[DEL] {shard_path}", flush=True)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    audio_root = resolve_path(root, args.audio_root)
    out_jsonl = resolve_path(root, args.out_jsonl)
    out_tsv = resolve_path(root, args.out_tsv)
    state_path = resolve_path(root, args.state_path)

    state = load_state(state_path, reset=args.reset_state)
    workers = resolve_workers(args.workers)

    jsonl_f, tsv_f, tsv_writer = open_manifest_files(out_jsonl, out_tsv)
    try:
        print(f"[INFO] workers={workers} batch_size={max(1, int(args.batch_size))}", flush=True)
        include = set(args.include)

        if "vlsp2020_vinai_100h" in include:
            shards = find_parquet_shards(audio_root, "vlsp2020_vinai_100h")
            print(f"[INFO] vlsp2020_vinai_100h source_shards={len(shards)}", flush=True)
            process_parquet_dataset(
                dataset="vlsp2020_vinai_100h",
                shard_paths=shards,
                text_field="transcription",
                audio_root=audio_root,
                jsonl_f=jsonl_f,
                tsv_writer=tsv_writer,
                state=state,
                state_path=state_path,
                project_root=root,
                max_shards=args.max_shards_per_dataset,
                delete_on_success=args.delete_source_on_success,
                target_sample_rate=args.target_sample_rate,
                uppercase_text=args.uppercase_text,
                batch_size=args.batch_size,
                workers=workers,
            )

        if "phoaudiobook" in include:
            shards = find_parquet_shards(audio_root, "phoaudiobook")
            print(f"[INFO] phoaudiobook source_shards={len(shards)}", flush=True)
            process_parquet_dataset(
                dataset="phoaudiobook",
                shard_paths=shards,
                text_field="text",
                audio_root=audio_root,
                jsonl_f=jsonl_f,
                tsv_writer=tsv_writer,
                state=state,
                state_path=state_path,
                project_root=root,
                max_shards=args.max_shards_per_dataset,
                delete_on_success=args.delete_source_on_success,
                target_sample_rate=args.target_sample_rate,
                uppercase_text=args.uppercase_text,
                batch_size=args.batch_size,
                workers=workers,
            )

        if "viet_bud500" in include:
            shards = find_parquet_shards(audio_root, "viet_bud500")
            print(f"[INFO] viet_bud500 source_shards={len(shards)}", flush=True)
            process_parquet_dataset(
                dataset="viet_bud500",
                shard_paths=shards,
                text_field="transcription",
                audio_root=audio_root,
                jsonl_f=jsonl_f,
                tsv_writer=tsv_writer,
                state=state,
                state_path=state_path,
                project_root=root,
                max_shards=args.max_shards_per_dataset,
                delete_on_success=args.delete_source_on_success,
                target_sample_rate=args.target_sample_rate,
                uppercase_text=args.uppercase_text,
                batch_size=args.batch_size,
                workers=workers,
            )

        if "vietspeech" in include:
            shards = find_arrow_shards(audio_root, "vietspeech")
            print(f"[INFO] vietspeech source_shards={len(shards)}", flush=True)
            process_vietspeech_arrow(
                dataset="vietspeech",
                shard_paths=shards,
                audio_root=audio_root,
                jsonl_f=jsonl_f,
                tsv_writer=tsv_writer,
                state=state,
                state_path=state_path,
                project_root=root,
                max_shards=args.max_shards_per_dataset,
                delete_on_success=args.delete_source_on_success,
                target_sample_rate=args.target_sample_rate,
                uppercase_text=args.uppercase_text,
                workers=workers,
            )

        if "fpt_fosd" in include:
            shards = find_parquet_shards(audio_root, "fpt_fosd")
            print(f"[INFO] fpt_fosd source_shards={len(shards)}", flush=True)
            process_parquet_dataset(
                dataset="fpt_fosd",
                shard_paths=shards,
                text_field="transcription",
                audio_root=audio_root,
                jsonl_f=jsonl_f,
                tsv_writer=tsv_writer,
                state=state,
                state_path=state_path,
                project_root=root,
                max_shards=args.max_shards_per_dataset,
                delete_on_success=args.delete_source_on_success,
                target_sample_rate=args.target_sample_rate,
                uppercase_text=args.uppercase_text,
                batch_size=args.batch_size,
                workers=workers,
            )

        if "vais1000" in include:
            shards = find_parquet_shards(audio_root, "vais1000")
            print(f"[INFO] vais1000 source_shards={len(shards)}", flush=True)
            process_parquet_dataset(
                dataset="vais1000",
                shard_paths=shards,
                text_field="transcription",
                audio_root=audio_root,
                jsonl_f=jsonl_f,
                tsv_writer=tsv_writer,
                state=state,
                state_path=state_path,
                project_root=root,
                max_shards=args.max_shards_per_dataset,
                delete_on_success=args.delete_source_on_success,
                target_sample_rate=args.target_sample_rate,
                uppercase_text=args.uppercase_text,
                batch_size=args.batch_size,
                workers=workers,
            )

        if "viVoice" in include:
            shards = find_parquet_shards(audio_root, "viVoice")
            print(f"[INFO] viVoice source_shards={len(shards)}", flush=True)
            process_parquet_dataset(
                dataset="viVoice",
                shard_paths=shards,
                text_field="text",
                audio_root=audio_root,
                jsonl_f=jsonl_f,
                tsv_writer=tsv_writer,
                state=state,
                state_path=state_path,
                project_root=root,
                max_shards=args.max_shards_per_dataset,
                delete_on_success=args.delete_source_on_success,
                target_sample_rate=args.target_sample_rate,
                uppercase_text=args.uppercase_text,
                batch_size=args.batch_size,
                workers=workers,
            )

        save_state(state_path, state)
        print(f"[DONE] Manifest JSONL: {out_jsonl}", flush=True)
        print(f"[DONE] Manifest TSV:   {out_tsv}", flush=True)
        print(f"[DONE] State file:     {state_path}", flush=True)
    finally:
        jsonl_f.close()
        tsv_f.close()


if __name__ == "__main__":
    main()
