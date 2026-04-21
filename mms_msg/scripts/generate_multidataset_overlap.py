#!/usr/bin/env python3
"""Generate single/overlap2/overlap3 data from multi-dataset ASR manifest.

Architecture:
- Single (65%): separate pipeline, NO audio I/O, just manifest + labels.
  Source audio is NEVER deleted for singles (manifest points to original path).
- Overlap 2spk (28%): batch processing with multiprocessing.
  Source audio deleted only AFTER entire batch is verified on disk.
- Overlap 3spk (7%): batch processing with multiprocessing.
  Source audio deleted only AFTER entire batch is verified on disk.
- Dry-run mode: test logic without audio I/O (--dry-run).

Performance:
- O(1) pool removal via index dict (not O(n) linear scan)
- Random pick with retry (not list filter)
- Multiprocessing for audio read/mix/write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    from lhotse import CutSet, Recording, SupervisionSegment
    from lhotse.cut import MonoCut
except Exception:  # pragma: no cover
    CutSet = None
    Recording = None
    SupervisionSegment = None
    MonoCut = None


DEFAULT_DATASET_HOURS = {
    "gigaspeech2_vi": 6058.25,
    "phoaudiobook": 1490.99,
    "vais1000": 1.62,
    "viVoice": 1016.97,
    "viet_bud500": 462.08,
    "vietspeech": 721.45,
    "vivos": 15.67,
    "vlsp2020_vinai_100h": 101.38,
    "fpt_fosd": 100.0,
}


@dataclass
class SourceRow:
    utt_id: str
    dataset: str
    split: str
    speaker_id: str
    audio_path: Path
    text: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multi-dataset overlap corpus.")
    p.add_argument("--manifest", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/asr_manifest_full_all_audio.jsonl"))
    p.add_argument("--output-dir", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/overlap_multidataset"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--single-ratio", type=float, default=0.65)
    p.add_argument("--ov2-ratio", type=float, default=0.28)
    p.add_argument("--ov3-ratio", type=float, default=0.07)
    p.add_argument("--train-ratio", type=float, default=0.90)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--target-train-mixtures", type=int, default=0)
    p.add_argument("--target-val-mixtures", type=int, default=0)
    p.add_argument("--target-test-mixtures", type=int, default=0)
    p.add_argument("--preview-mixes", type=int, default=0)
    p.add_argument("--max-tries-per-mix", type=int, default=80)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--mix-id-prefix", type=str, default="")
    p.add_argument("--verify-audio-exists", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--write-cutset", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--only-splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    p.add_argument("--delete-used-source", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--workers", type=int, default=64,
                    help="Number of parallel workers for audio mixing.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--min-disk-gb", type=float, default=10.0,
                    help="Stop when free disk drops below this (GB).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalized_split(raw_split: str) -> str:
    s = (raw_split or "").strip().lower()
    if s in {"train"}:
        return "train"
    if s in {"validation", "val", "dev"}:
        return "val"
    if s in {"test"}:
        return "test"
    return "train"


def hash01(text: str) -> float:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def hash_bucket(text: str, num_buckets: int) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(1, int(num_buckets))


def parse_speaker_id(dataset: str, utt_id: str, audio_path: Path) -> str:
    ds = dataset.lower()
    uid = utt_id
    stem = audio_path.stem.upper()
    if ds == "vietspeech":
        parts = uid.split("__")
        if len(parts) >= 3 and parts[2].startswith("VI_"):
            toks = parts[2].split("_")
            if len(toks) >= 2:
                return "_".join(toks[:2]).upper()
        idx = stem.find("VI_")
        if idx >= 0:
            toks = stem[idx:].split("_")
            if len(toks) >= 2:
                return "_".join(toks[:2]).upper()
    if ds == "vivos":
        parts = uid.split("__")
        token = parts[-1] if parts else stem
        return token.split("_")[0].upper()
    if ds == "gigaspeech2_vi":
        parts = uid.split("__")
        token = parts[-1] if parts else stem
        toks = token.split("-")
        if len(toks) >= 2:
            return f"{toks[0]}-{toks[1]}"
        return token
    return uid if uid else stem


def load_rows(
    manifest: Path, num_shards: int = 1, shard_id: int = 0,
    verify_audio_exists: bool = False,
) -> Tuple[List[SourceRow], int]:
    rows: List[SourceRow] = []
    total_seen = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_seen += 1
            obj = json.loads(line)
            audio_path = Path(str(obj.get("audio_path", "")))
            if verify_audio_exists and (not audio_path.exists()):
                continue
            utt_id = str(obj.get("utt_id", audio_path.stem))
            if num_shards > 1 and hash_bucket(utt_id, num_shards) != shard_id:
                continue
            text = str(obj.get("text", "")).strip()
            if not text:
                continue
            dataset = str(obj.get("dataset", "")).strip()
            if not dataset:
                continue
            split = normalized_split(str(obj.get("split", "")))
            speaker = parse_speaker_id(dataset, utt_id, audio_path)
            rows.append(SourceRow(utt_id=utt_id, dataset=dataset, split=split,
                                  speaker_id=speaker, audio_path=audio_path, text=text))
    return rows, total_seen


def split_train_only_dataset(
    rows: List[SourceRow], train_ratio: float, val_ratio: float,
    test_ratio: float, speaker_disjoint: bool,
) -> List[SourceRow]:
    if not rows:
        return rows
    has_non_train = any(r.split in {"val", "test"} for r in rows)
    if has_non_train:
        return rows
    if speaker_disjoint:
        speakers = sorted(set(r.speaker_id for r in rows))
        speaker_bucket = {}
        for spk in speakers:
            x = hash01(spk)
            if x < train_ratio:
                speaker_bucket[spk] = "train"
            elif x < train_ratio + val_ratio:
                speaker_bucket[spk] = "val"
            else:
                speaker_bucket[spk] = "test"
        for r in rows:
            r.split = speaker_bucket[r.speaker_id]
    else:
        for r in rows:
            x = hash01(r.utt_id)
            if x < train_ratio:
                r.split = "train"
            elif x < train_ratio + val_ratio:
                r.split = "val"
            else:
                r.split = "test"
    return rows


def check_disk_free_gb(path: Path = Path("/home/vietnam/voice_to_text/data")) -> float:
    stat = os.statvfs(str(path))
    return (stat.f_bavail * stat.f_frsize) / (1024**3)


def try_delete_source(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
        return True
    except Exception:
        return False


def choose_dataset_weighted(rng: random.Random, datasets: List[str], weights: Dict[str, float]) -> str:
    ws = [max(1e-8, float(weights.get(d, 1.0))) for d in datasets]
    return rng.choices(datasets, weights=ws, k=1)[0]


# ---------------------------------------------------------------------------
# Fast Pool — O(1) removal via swap-and-pop with index dict
# ---------------------------------------------------------------------------
class FastPool:
    """Pool of SourceRows with O(1) random pick and O(1) removal by utt_id."""

    def __init__(self, rows: List[SourceRow]):
        self.rows = list(rows)
        # utt_id -> index in self.rows
        self.idx: Dict[str, int] = {r.utt_id: i for i, r in enumerate(self.rows)}

    def __len__(self):
        return len(self.rows)

    def pick_random(self, rng: random.Random) -> Optional[SourceRow]:
        if not self.rows:
            return None
        idx = rng.randrange(len(self.rows))
        return self.rows[idx]

    def pick_random_excluding(self, rng: random.Random, exclude_utt_ids: set,
                               exclude_speakers: set, max_retries: int = 50) -> Optional[SourceRow]:
        """Pick random row not in exclude sets. Fast for large pools with small excludes."""
        n = len(self.rows)
        if n == 0:
            return None
        for _ in range(max_retries):
            idx = rng.randrange(n)
            r = self.rows[idx]
            if r.utt_id not in exclude_utt_ids and r.speaker_id not in exclude_speakers:
                return r
        return None

    def remove(self, utt_id: str) -> bool:
        """O(1) removal via swap-and-pop."""
        idx = self.idx.pop(utt_id, None)
        if idx is None:
            return False
        last = len(self.rows) - 1
        if idx != last:
            # Swap with last element
            self.rows[idx] = self.rows[last]
            self.idx[self.rows[idx].utt_id] = idx
        self.rows.pop()
        return True

    def remove_many(self, utt_ids: List[str]) -> int:
        """Remove multiple utt_ids efficiently."""
        removed = 0
        for uid in utt_ids:
            if self.remove(uid):
                removed += 1
        return removed


# ---------------------------------------------------------------------------
# Resume: scan existing output to find already-done mixes
# ---------------------------------------------------------------------------
def scan_done_mixes(out_dir: Path, mix_type: str, split_name: str) -> Tuple[int, set, int]:
    """Scan existing metadata.json files to find already-completed mixes.

    Uses os.scandir for speed on large directories (200K+ entries).

    Returns:
        (num_done, used_utt_ids, next_mix_idx)
    """
    search_dir = out_dir / mix_type / split_name
    if not search_dir.exists():
        return 0, set(), 0

    used_utt_ids: set = set()
    max_idx = -1
    num_done = 0
    scan_errors = 0

    print(f"[RESUME] Scanning {search_dir} ...", flush=True)
    try:
        entries = list(os.scandir(str(search_dir)))
    except OSError:
        return 0, set(), 0

    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        meta_file = os.path.join(entry.path, "metadata.json")
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for src in meta.get("sources", []):
                uid = src.get("utt_id", "")
                if uid:
                    used_utt_ids.add(uid)
            # Extract mix index from dir name like "train_00000042_overlap2"
            name = entry.name  # e.g. "train_00000042_overlap2"
            # Split: prefix_XXXXXXXX_mixtype
            parts = name.split("_")
            # Find the numeric part (8 digits)
            for p in parts:
                if len(p) == 8 and p.isdigit():
                    idx = int(p)
                    if idx > max_idx:
                        max_idx = idx
                    break
            num_done += 1
        except Exception:
            scan_errors += 1
            continue

    if num_done % 10000 == 0 or num_done > 0:
        print(f"[RESUME] Scanned {len(entries):,} dirs: {num_done:,} valid mixes, "
              f"{len(used_utt_ids):,} used utt_ids, {scan_errors} errors", flush=True)

    next_mix_idx = max_idx + 1 if max_idx >= 0 else 0
    return num_done, used_utt_ids, next_mix_idx


# ---------------------------------------------------------------------------
# Multiprocessing worker for mixing audio
# ---------------------------------------------------------------------------
def _mix_worker(args) -> Optional[dict]:
    """Worker: read source audio → mix → write mix.wav → return metadata.
    Runs in separate process for parallelism.
    """
    (mix_id, mix_type, mix_dir_str, source_paths_str, rng_seed) = args
    try:
        rng = random.Random(rng_seed)
        source_paths = [Path(p) for p in source_paths_str]
        mix_dir = Path(mix_dir_str)
        mix_path = mix_dir / "mix.wav"

        # Read all source audio
        audio_cache = []
        sr_list = []
        for sp in source_paths:
            audio, sr = sf.read(str(sp), dtype="float32")
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=1)
            if arr.size == 0:
                return None
            audio_cache.append(arr)
            sr_list.append(int(sr))

        # Resample to 16kHz
        target_sr = 16000
        for i, (a, s) in enumerate(zip(audio_cache, sr_list)):
            if s != target_sr:
                src_n = a.shape[0]
                dst_n = max(1, int(round(src_n * target_sr / s)))
                x_old = np.linspace(0, 1, src_n, endpoint=False, dtype=np.float64)
                x_new = np.linspace(0, 1, dst_n, endpoint=False, dtype=np.float64)
                audio_cache[i] = np.interp(x_new, x_old, a.astype(np.float64)).astype(np.float32)

        sr = target_sr
        durations_sec = [len(a) / float(sr) for a in audio_cache]

        # Mix
        if mix_type == "overlap2":
            a0, a1 = audio_cache
            ov = int(min(len(a0), len(a1)) * rng.uniform(0.30, 0.70))
            starts = [0, max(0, len(a0) - ov)]
        elif mix_type == "overlap3":
            a0 = audio_cache[0]
            starts = [0, int(len(a0) * rng.uniform(0.30, 0.60)),
                      int(len(a0) * rng.uniform(0.45, 0.75))]
        else:
            return None

        max_len = max(st + len(a) for st, a in zip(starts, audio_cache))
        mix = np.zeros(max_len, dtype=np.float32)
        for st, a in zip(starts, audio_cache):
            mix[st: st + len(a)] += a * rng.uniform(0.8, 1.0)
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.99 and peak > 0:
            mix *= 0.99 / peak

        # Overlap metrics
        n = max_len
        act = np.zeros((len(audio_cache), n), dtype=np.int8)
        for i, (st, a) in enumerate(zip(starts, audio_cache)):
            act[i, st: st + len(a)] = (np.abs(a) > 1e-6).astype(np.int8)
        c = np.sum(act, axis=0)
        ov2_ratio = float(np.mean(c >= 2))
        ov3_ratio = float(np.mean(c >= 3))
        maxc = int(np.max(c))

        # Validate overlap
        if mix_type == "overlap2" and maxc < 2:
            return None
        if mix_type == "overlap3" and maxc < 3:
            return None

        # Write
        mix_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(mix_path), mix, sr)

        return {
            "mix_id": mix_id,
            "mix_path": str(mix_path),
            "sr": sr,
            "starts": starts,
            "durations_sec": durations_sec,
            "ov2_ratio": ov2_ratio,
            "ov3_ratio": ov3_ratio,
            "maxc": maxc,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SINGLE pipeline — no audio I/O, just manifest + labels
# ---------------------------------------------------------------------------
def process_singles(
    split_name: str,
    single_rows: List[SourceRow],
    out_dir: Path,
    seed: int,
    write_cutset: bool,
    dry_run: bool,
    progress_every: int,
    mix_id_prefix: str,
    min_disk_gb: float,
) -> Dict[str, object]:
    rng = random.Random(seed)
    rng.shuffle(single_rows)

    split_dir = out_dir
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / f"{split_name}.single.manifest.jsonl"
    sup_text_path = split_dir / f"{split_name}.single.supervisions.text"
    cutset_path = split_dir / f"{split_name}.single.cuts.jsonl.gz"

    cuts = []
    written = 0
    dataset_draws: Dict[str, int] = defaultdict(int)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as mf, \
         sup_text_path.open("w", encoding="utf-8", newline="\n") as sf_text:
        for idx, row in enumerate(single_rows):
            if idx % 10000 == 0 and idx > 0:
                free_gb = check_disk_free_gb()
                if free_gb < min_disk_gb:
                    print(f"[SINGLE] DISK LOW: {free_gb:.1f}GB. "
                          f"Stopping at {written}/{len(single_rows)}", flush=True)
                    break
            mix_id = f"{mix_id_prefix}{split_name}_{idx:08d}_single"
            seg_id = f"{mix_id}_seg01"

            duration_sec = 0.0
            sr = 16000
            if not dry_run:
                try:
                    info = sf.info(str(row.audio_path))
                    if info.frames <= 0 or info.samplerate <= 0:
                        continue
                    duration_sec = float(info.frames) / float(info.samplerate)
                    sr = int(info.samplerate)
                except Exception:
                    continue

            out_row = {
                "mix_id": mix_id,
                "dataset": "multidataset_overlap",
                "split": split_name,
                "mix_type": "single",
                "num_speakers": 1,
                "audio_path": row.audio_path.as_posix(),
                "text": row.text,
            }
            mf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            sf_text.write(f"{seg_id} {row.text}\n")
            dataset_draws[row.dataset] += 1

            if write_cutset and CutSet is not None and not dry_run:
                try:
                    recording = Recording.from_file(str(row.audio_path), recording_id=mix_id)
                    sup = SupervisionSegment(
                        id=seg_id, recording_id=mix_id,
                        start=0.0, duration=duration_sec,
                        text=row.text, speaker=row.speaker_id,
                        channel=0, language="vi",
                    )
                    cuts.append(MonoCut(
                        id=mix_id, start=0.0, duration=recording.duration,
                        channel=0, recording=recording, supervisions=[sup],
                    ))
                except Exception:
                    pass

            written += 1
            if progress_every > 0 and written % progress_every == 0:
                print(f"[SINGLE] split={split_name} done={written:,}/{len(single_rows):,}", flush=True)

    if write_cutset and CutSet is not None and cuts:
        CutSet.from_cuts(cuts).to_file(str(cutset_path))

    return {
        "type": "single", "split": split_name, "written": written,
        "total_source": len(single_rows), "dataset_draws": dict(dataset_draws),
        "manifest_path": manifest_path.as_posix(), "deleted_source_files": 0,
    }


# ---------------------------------------------------------------------------
# OVERLAP pipeline — batch + multiprocessing
# ---------------------------------------------------------------------------
def process_overlaps_batched(
    split_name: str,
    source_split_name: str,
    mix_type: str,
    target_count: int,
    pools: Dict[str, FastPool],   # dataset_name -> FastPool
    dataset_weights: Dict[str, float],
    out_dir: Path,
    seed: int,
    max_tries: int,
    write_cutset: bool,
    delete_used_source: bool,
    batch_size: int,
    num_workers: int,
    dry_run: bool,
    progress_every: int,
    mix_id_prefix: str,
    min_disk_gb: float,
    resume_mix_idx: int = 0,
    resume_written: int = 0,
) -> Dict[str, object]:
    """Process overlap mixes in batches with multiprocessing.

    Flow per batch:
    1. Plan batch_size mixes (pick source rows from pool)
    2. Submit audio mixing to process pool (parallel)
    3. Verify mix files exist
    4. Write manifest
    5. Remove used rows from pool + delete source audio
    """
    n_spk = 2 if mix_type == "overlap2" else 3
    rng = random.Random(seed)

    split_dir = out_dir
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / f"{split_name}.{mix_type}.manifest.jsonl"
    sup_text_path = split_dir / f"{split_name}.{mix_type}.supervisions.text"
    cutset_path = split_dir / f"{split_name}.{mix_type}.cuts.jsonl.gz"

    # Resume: open in append mode if resuming
    file_mode = "a" if resume_written > 0 else "w"

    all_cuts = []
    written_total = resume_written
    deleted_sources = 0
    dataset_draws: Dict[str, int] = defaultdict(int)
    mix_idx = resume_mix_idx

    def get_avail_datasets():
        return [d for d, p in pools.items() if len(p) > 0]

    # Persistent executor — reuse across all batches (avoid create/destroy overhead)
    executor = ProcessPoolExecutor(max_workers=num_workers) if not dry_run else None

    try:
      with manifest_path.open(file_mode, encoding="utf-8", newline="\n") as mf, \
           sup_text_path.open(file_mode, encoding="utf-8", newline="\n") as sf_text:

        while written_total < target_count:
            # Disk check (every batch, not every mix)
            free_gb = check_disk_free_gb()
            if free_gb < min_disk_gb:
                print(f"[{mix_type.upper()}] DISK LOW: {free_gb:.1f}GB. "
                      f"Stopping at {written_total}/{target_count}", flush=True)
                break

            avail_datasets = get_avail_datasets()
            if not avail_datasets:
                print(f"[{mix_type.upper()}] Pool exhausted at {written_total}/{target_count}", flush=True)
                break

            non_viet = [d for d in avail_datasets if d != "vietspeech"]

            # --- Step 1: Plan batch (pick rows, no audio I/O) ---
            planned = []  # list of (mix_id, mix_dir, chosen_rows: [(ds, row), ...])
            batch_used_utt_ids: set = set()
            batch_attempts = 0
            max_batch_attempts = batch_size * max_tries

            while len(planned) < batch_size and written_total + len(planned) < target_count:
                batch_attempts += 1
                if batch_attempts > max_batch_attempts:
                    break

                avail_datasets = get_avail_datasets()
                if not avail_datasets:
                    break
                non_viet = [d for d in avail_datasets if d != "vietspeech"]

                chosen_rows: List[Tuple[str, SourceRow]] = []
                used_spk: set = set()

                # Anchor
                anchor = None
                if "vietspeech" in pools and len(pools["vietspeech"]) > 0:
                    anchor = pools["vietspeech"].pick_random_excluding(
                        rng, batch_used_utt_ids, set(), max_retries=20)
                    if anchor:
                        chosen_rows.append(("vietspeech", anchor))

                if not anchor:
                    ds = choose_dataset_weighted(rng, avail_datasets, dataset_weights)
                    if ds in pools:
                        anchor = pools[ds].pick_random_excluding(
                            rng, batch_used_utt_ids, set(), max_retries=20)
                        if anchor:
                            chosen_rows.append((ds, anchor))
                if not chosen_rows:
                    continue

                used_spk.add(chosen_rows[0][1].speaker_id)

                # Remaining speakers
                ok = True
                for _ in range(max_tries):
                    if len(chosen_rows) >= n_spk:
                        break
                    ds_pool = non_viet if non_viet else avail_datasets
                    if not ds_pool:
                        ok = False
                        break
                    ds = rng.choice(ds_pool)
                    if ds not in pools or len(pools[ds]) == 0:
                        continue
                    cand = pools[ds].pick_random_excluding(
                        rng, batch_used_utt_ids, used_spk, max_retries=10)
                    if cand:
                        chosen_rows.append((ds, cand))
                        used_spk.add(cand.speaker_id)

                if len(chosen_rows) < n_spk:
                    continue

                mix_id = f"{mix_id_prefix}{split_name}_{mix_idx:08d}_{mix_type}"
                mix_dir = split_dir / mix_type / split_name / mix_id

                planned.append((mix_id, str(mix_dir), chosen_rows))
                for _, row in chosen_rows:
                    batch_used_utt_ids.add(row.utt_id)
                mix_idx += 1

            if not planned:
                print(f"[{mix_type.upper()}] Could not plan any more mixes, stopping", flush=True)
                break

            prev_written = written_total

            # --- Step 2: Submit mixing to process pool (or dry-run) ---
            if not dry_run:
                # Build worker args
                worker_args = []
                for mix_id, mix_dir_str, chosen_rows in planned:
                    source_paths = [str(r.audio_path) for _, r in chosen_rows]
                    worker_args.append((
                        mix_id, mix_type, mix_dir_str, source_paths, rng.randint(0, 2**31)
                    ))

                # Run in parallel (reuse persistent executor)
                results = list(executor.map(_mix_worker, worker_args, chunksize=64))
            else:
                # Dry-run: fake results
                results = [{"mix_id": p[0], "mix_path": str(Path(p[1]) / "mix.wav"),
                            "sr": 16000, "starts": [0] * n_spk,
                            "durations_sec": [1.0] * n_spk,
                            "ov2_ratio": 0.5, "ov3_ratio": 0.0, "maxc": n_spk}
                           for p in planned]

            # --- Step 3+4: Verify, write manifest, remove pool, delete source ---
            verified_utt_ids_to_remove: List[str] = []
            verified_source_paths: set = set()

            for (mix_id, mix_dir_str, chosen_rows), result in zip(planned, results):
                if result is None:
                    continue

                mix_path = Path(result["mix_path"])
                meta_path = Path(mix_dir_str) / "metadata.json"
                sr = result["sr"]
                starts = result["starts"]
                durations_sec = result["durations_sec"]

                # Verify file exists
                if not dry_run:
                    if not mix_path.exists() or mix_path.stat().st_size == 0:
                        continue

                # Build metadata
                rows_only = [r for _, r in chosen_rows]
                src_meta = []
                sup_lines = []
                for j, (row, st) in enumerate(zip(rows_only, starts)):
                    seg_id = f"{mix_id}_seg{j+1:02d}"
                    dur = durations_sec[j] if j < len(durations_sec) else 0.0
                    st_sec = st / float(sr)
                    src_meta.append({
                        "seg_id": seg_id, "dataset": row.dataset,
                        "speaker_id": row.speaker_id, "utt_id": row.utt_id,
                        "audio_path": row.audio_path.as_posix(),
                        "start_sec": st_sec, "duration_sec": dur,
                        "end_sec": st_sec + dur, "text": row.text,
                    })
                    sup_lines.append(f"{seg_id} {row.text}")
                    dataset_draws[row.dataset] += 1

                src_meta = sorted(src_meta, key=lambda x: x["start_sec"])
                text_merged = " | ".join(s["text"] for s in src_meta)

                metadata = {
                    "mix_id": mix_id, "split": split_name, "mix_type": mix_type,
                    "num_speakers": n_spk, "sample_rate": sr,
                    "mix_path": mix_path.as_posix(),
                    "overlap_ratio_ge_2": result["ov2_ratio"],
                    "overlap_ratio_ge_3": result["ov3_ratio"],
                    "max_concurrent_observed": result["maxc"],
                    "sources": src_meta,
                }
                if not dry_run:
                    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

                out_row = {
                    "mix_id": mix_id, "dataset": "multidataset_overlap",
                    "split": split_name, "mix_type": mix_type,
                    "num_speakers": n_spk, "audio_path": mix_path.as_posix(),
                    "metadata_path": meta_path.as_posix(), "text": text_merged,
                }
                mf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                for sl in sup_lines:
                    sf_text.write(sl + "\n")

                # CutSet
                if write_cutset and CutSet is not None and not dry_run:
                    try:
                        recording = Recording.from_file(str(mix_path), recording_id=mix_id)
                        sups = [
                            SupervisionSegment(
                                id=s["seg_id"], recording_id=mix_id,
                                start=float(s["start_sec"]), duration=float(s["duration_sec"]),
                                text=s["text"], speaker=s["speaker_id"],
                                channel=0, language="vi",
                            )
                            for s in src_meta
                        ]
                        all_cuts.append(MonoCut(
                            id=mix_id, start=0.0, duration=recording.duration,
                            channel=0, recording=recording, supervisions=sups,
                        ))
                    except Exception:
                        pass

                written_total += 1

                # Collect for removal
                for _, row in chosen_rows:
                    verified_utt_ids_to_remove.append(row.utt_id)
                    verified_source_paths.add(row.audio_path)

            mf.flush()
            sf_text.flush()

            # Remove VERIFIED (successful) rows from pools
            for uid in verified_utt_ids_to_remove:
                for p in pools.values():
                    if p.remove(uid):
                        break

            # Also remove FAILED rows from pools so they don't get picked again
            # (e.g. source audio was deleted by another process)
            failed_utt_ids = []
            for (mix_id_p, mix_dir_str_p, chosen_rows_p), result in zip(planned, results):
                if result is None:
                    for _, row in chosen_rows_p:
                        if row.utt_id not in batch_used_utt_ids or row.utt_id not in {u for u in verified_utt_ids_to_remove}:
                            failed_utt_ids.append(row.utt_id)
            if failed_utt_ids:
                removed_failed = 0
                for uid in failed_utt_ids:
                    for p in pools.values():
                        if p.remove(uid):
                            removed_failed += 1
                            break
                if removed_failed > 0:
                    print(f"[{mix_type.upper()}] Removed {removed_failed} failed/missing-audio rows from pool", flush=True)

            # Delete source audio
            if delete_used_source and not dry_run:
                for sp in verified_source_paths:
                    if try_delete_source(sp):
                        deleted_sources += 1

            # Stall detection: if no new mixes written in this batch, count stalls
            batch_new = written_total - prev_written
            if batch_new == 0:
                stall_count = getattr(process_overlaps_batched, '_stall', 0) + 1
                process_overlaps_batched._stall = stall_count
                print(f"[{mix_type.upper()}] STALL #{stall_count}: batch planned {len(planned)} but 0 succeeded", flush=True)
                if stall_count >= 3:
                    print(f"[{mix_type.upper()}] Too many stalls, stopping at {written_total}/{target_count}", flush=True)
                    break
            else:
                process_overlaps_batched._stall = 0

            if progress_every > 0:
                print(f"[{mix_type.upper()}] split={split_name} done={written_total:,}/{target_count:,} "
                      f"({100.0*written_total/max(1,target_count):.1f}%) deleted={deleted_sources}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    if write_cutset and CutSet is not None and all_cuts:
        CutSet.from_cuts(all_cuts).to_file(str(cutset_path))

    return {
        "type": mix_type, "split": split_name, "target": target_count,
        "written": written_total, "dataset_draws": dict(dataset_draws),
        "deleted_source_files": deleted_sources, "manifest_path": manifest_path.as_posix(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading manifest: {args.manifest}", flush=True)
    rows, total_seen = load_rows(
        args.manifest.resolve(),
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        verify_audio_exists=args.verify_audio_exists,
    )
    if not rows:
        raise RuntimeError("No valid rows loaded from manifest.")
    print(f"[INFO] Loaded {len(rows):,} rows from {total_seen:,} scanned", flush=True)

    # Split train-only datasets
    by_dataset = defaultdict(list)
    for r in rows:
        by_dataset[r.dataset].append(r)
    for ds, ds_rows in by_dataset.items():
        speaker_disjoint = ds == "vietspeech"
        by_dataset[ds] = split_train_only_dataset(
            ds_rows, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, speaker_disjoint=speaker_disjoint,
        )

    merged_rows: List[SourceRow] = []
    for ds_rows in by_dataset.values():
        merged_rows.extend(ds_rows)

    # Assign rows to single/ov2/ov3 pools by hash
    single_pool: Dict[str, List[SourceRow]] = defaultdict(list)
    ov2_pool_rows: Dict[str, Dict[str, List[SourceRow]]] = {
        s: defaultdict(list) for s in ["train", "val", "test"]}
    ov3_pool_rows: Dict[str, Dict[str, List[SourceRow]]] = {
        s: defaultdict(list) for s in ["train", "val", "test"]}

    s_ratio = max(0.0, args.single_ratio)
    o2_ratio = max(0.0, args.ov2_ratio)
    o3_ratio = max(0.0, args.ov3_ratio)
    s_weight = s_ratio * 1
    o2_weight = o2_ratio * 2
    o3_weight = o3_ratio * 3
    total_weight = s_weight + o2_weight + o3_weight
    if total_weight <= 0:
        raise ValueError("Ratios sum to 0")
    s_norm = s_weight / total_weight
    o2_norm = o2_weight / total_weight

    for r in merged_rows:
        x = hash01(r.utt_id + "__pool_assign")
        if x < s_norm:
            single_pool[r.split].append(r)
        elif x < s_norm + o2_norm:
            ov2_pool_rows[r.split][r.dataset].append(r)
        else:
            ov3_pool_rows[r.split][r.dataset].append(r)

    # Dataset weights
    all_datasets = sorted({r.dataset for r in merged_rows})
    dataset_weights = {d: float(DEFAULT_DATASET_HOURS.get(d, 1.0)) for d in all_datasets}
    z = sum(dataset_weights.values())
    if z > 0:
        dataset_weights = {k: v / z for k, v in dataset_weights.items()}

    # Stats
    for split in ["train", "val", "test"]:
        n_s = len(single_pool.get(split, []))
        n_o2 = sum(len(v) for v in ov2_pool_rows[split].values())
        n_o3 = sum(len(v) for v in ov3_pool_rows[split].values())
        print(f"[POOL] {split}: single={n_s:,}  ov2_sources={n_o2:,} (→{n_o2//2:,} mixes)  "
              f"ov3_sources={n_o3:,} (→{n_o3//3:,} mixes)", flush=True)

    mix_id_prefix = args.mix_id_prefix
    if not mix_id_prefix and args.num_shards > 1:
        mix_id_prefix = f"s{args.shard_id:02d}_"

    # Save plan
    plan = {
        "source_manifest": args.manifest.as_posix(),
        "num_rows_loaded": len(merged_rows),
        "ratios": {"single": args.single_ratio, "overlap2": args.ov2_ratio, "overlap3": args.ov3_ratio},
        "batch_size": args.batch_size, "workers": args.workers,
        "dry_run": args.dry_run, "delete_used_source": args.delete_used_source,
        "min_disk_gb": args.min_disk_gb,
        "datasets": all_datasets,
    }
    (out_dir / "sampling_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    run_summary: Dict[str, object] = {"plan": plan, "splits": {}}

    for split in ["train", "val", "test"]:
        if split not in set(args.only_splits):
            continue

        split_results: Dict[str, object] = {}

        # --- SINGLE ---
        s_rows = single_pool.get(split, [])
        single_manifest = out_dir / f"{split}.single.manifest.jsonl"
        if s_rows and single_manifest.exists() and single_manifest.stat().st_size > 0:
            existing_lines = sum(1 for _ in single_manifest.open("r", encoding="utf-8"))
            print(f"[RESUME] single/{split}: manifest already has {existing_lines:,} lines, skipping", flush=True)
            split_results["single"] = {"type": "single", "split": split, "written": existing_lines, "resumed": True}
        elif s_rows:
            print(f"\n{'='*60}", flush=True)
            print(f"[SINGLE] Processing {len(s_rows):,} singles for {split}", flush=True)
            summary_s = process_singles(
                split_name=split, single_rows=list(s_rows), out_dir=out_dir,
                seed=args.seed, write_cutset=args.write_cutset, dry_run=args.dry_run,
                progress_every=args.progress_every, mix_id_prefix=mix_id_prefix,
                min_disk_gb=args.min_disk_gb,
            )
            split_results["single"] = summary_s
            print(f"[SINGLE] {split} done: {summary_s['written']:,} written", flush=True)

        # --- OVERLAP 2 ---
        # Build FastPools for this split
        ov2_fast: Dict[str, FastPool] = {
            ds: FastPool(rows) for ds, rows in ov2_pool_rows[split].items() if rows
        }
        n_ov2_sources = sum(len(p) for p in ov2_fast.values())
        target_ov2 = n_ov2_sources // 2

        # Resume: scan existing output
        ov2_done, ov2_used_uids, ov2_next_idx = scan_done_mixes(out_dir, "overlap2", split)
        if ov2_done > 0:
            print(f"[RESUME] overlap2/{split}: found {ov2_done:,} existing mixes, "
                  f"{len(ov2_used_uids):,} used utt_ids, next_idx={ov2_next_idx}", flush=True)
            # Remove already-used utt_ids from pools
            for p in ov2_fast.values():
                p.remove_many(list(ov2_used_uids))
            n_ov2_remaining = sum(len(p) for p in ov2_fast.values())
            print(f"[RESUME] overlap2/{split}: pool reduced to {n_ov2_remaining:,} sources", flush=True)

        if target_ov2 > ov2_done:
            print(f"\n{'='*60}", flush=True)
            print(f"[OVERLAP2] Processing {target_ov2:,} mixes for {split} "
                  f"(resuming from {ov2_done:,}, batch={args.batch_size}, workers={args.workers})", flush=True)
            summary_o2 = process_overlaps_batched(
                split_name=split, source_split_name=split, mix_type="overlap2",
                target_count=target_ov2, pools=ov2_fast,
                dataset_weights=dataset_weights, out_dir=out_dir,
                seed=args.seed + 100, max_tries=args.max_tries_per_mix,
                write_cutset=args.write_cutset, delete_used_source=args.delete_used_source,
                batch_size=args.batch_size, num_workers=args.workers,
                dry_run=args.dry_run, progress_every=args.progress_every,
                mix_id_prefix=mix_id_prefix, min_disk_gb=args.min_disk_gb,
                resume_mix_idx=ov2_next_idx, resume_written=ov2_done,
            )
            split_results["overlap2"] = summary_o2
            print(f"[OVERLAP2] {split} done: {summary_o2['written']:,}/{target_ov2:,}, "
                  f"deleted={summary_o2['deleted_source_files']}", flush=True)
        elif ov2_done > 0:
            print(f"[OVERLAP2] {split}: already complete ({ov2_done:,}/{target_ov2:,})", flush=True)

        # --- OVERLAP 3 ---
        ov3_fast: Dict[str, FastPool] = {
            ds: FastPool(rows) for ds, rows in ov3_pool_rows[split].items() if rows
        }
        n_ov3_sources = sum(len(p) for p in ov3_fast.values())
        target_ov3 = n_ov3_sources // 3

        # Resume: scan existing output
        ov3_done, ov3_used_uids, ov3_next_idx = scan_done_mixes(out_dir, "overlap3", split)
        if ov3_done > 0:
            print(f"[RESUME] overlap3/{split}: found {ov3_done:,} existing mixes, "
                  f"{len(ov3_used_uids):,} used utt_ids, next_idx={ov3_next_idx}", flush=True)
            for p in ov3_fast.values():
                p.remove_many(list(ov3_used_uids))
            n_ov3_remaining = sum(len(p) for p in ov3_fast.values())
            print(f"[RESUME] overlap3/{split}: pool reduced to {n_ov3_remaining:,} sources", flush=True)

        if target_ov3 > ov3_done:
            print(f"\n{'='*60}", flush=True)
            print(f"[OVERLAP3] Processing {target_ov3:,} mixes for {split} "
                  f"(resuming from {ov3_done:,}, batch={args.batch_size}, workers={args.workers})", flush=True)
            summary_o3 = process_overlaps_batched(
                split_name=split, source_split_name=split, mix_type="overlap3",
                target_count=target_ov3, pools=ov3_fast,
                dataset_weights=dataset_weights, out_dir=out_dir,
                seed=args.seed + 200, max_tries=args.max_tries_per_mix,
                write_cutset=args.write_cutset, delete_used_source=args.delete_used_source,
                batch_size=args.batch_size, num_workers=args.workers,
                dry_run=args.dry_run, progress_every=args.progress_every,
                mix_id_prefix=mix_id_prefix, min_disk_gb=args.min_disk_gb,
                resume_mix_idx=ov3_next_idx, resume_written=ov3_done,
            )
            split_results["overlap3"] = summary_o3
            print(f"[OVERLAP3] {split} done: {summary_o3['written']:,}/{target_ov3:,}, "
                  f"deleted={summary_o3['deleted_source_files']}", flush=True)
        elif ov3_done > 0:
            print(f"[OVERLAP3] {split}: already complete ({ov3_done:,}/{target_ov3:,})", flush=True)

        run_summary["splits"][split] = split_results

    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
