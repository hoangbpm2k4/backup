#!/usr/bin/env python3
"""
Extract all Vietnamese speech datasets to wav + manifest.jsonl format.
Strategy: process one source file at a time → extract wavs in parallel batches → append manifest → delete source.
This minimizes disk usage since SSD space is limited.

Usage:
    python extract_all_datasets.py                  # extract all
    python extract_all_datasets.py vais1000 vivos   # extract specific datasets
    python extract_all_datasets.py --workers 32      # custom worker count

Manifest format (matching fpt_fosd):
{"utt_id": "dataset__split__filename", "dataset": "...", "split": "train", "audio_path": "/full/path.wav", "text": "..."}
"""

import os
import sys
import json
import io
import re
import gzip
import tarfile
import traceback
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import soundfile as sf

DATA_ROOT = Path("/home/vietnam/voice_to_text/data")
SAMPLE_RATE = 16000  # target sample rate for all wavs
NUM_WORKERS = 16     # default parallel workers
# Config dict so we can update from main() without global issues
CFG = {"batch_size": 512}


def clean_text(text: str) -> str:
    """Normalize text: lowercase, strip, collapse whitespace."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def _convert_one(args):
    """Worker function: convert audio bytes to wav. Runs in separate process.
    Returns (utt_id, dataset, split, wav_path, text) on success, None on failure.
    """
    audio_bytes, out_path, orig_path, utt_id, dataset, split, text = args
    try:
        buf = io.BytesIO(audio_bytes)
        data, sr = sf.read(buf)
        # Convert to mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Resample if needed
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        sf.write(out_path, data, SAMPLE_RATE, subtype='PCM_16')
        return (utt_id, dataset, split, out_path, text)
    except Exception:
        # Fallback: try pydub for mp3/flac/etc
        try:
            from pydub import AudioSegment
            buf = io.BytesIO(audio_bytes)
            fmt = None
            if orig_path:
                ext = Path(orig_path).suffix.lower().lstrip('.')
                if ext in ('mp3', 'flac', 'ogg', 'opus', 'm4a', 'aac'):
                    fmt = ext
            audio = AudioSegment.from_file(buf, format=fmt) if fmt else AudioSegment.from_file(buf)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            audio.export(out_path, format="wav")
            return (utt_id, dataset, split, out_path, text)
        except Exception as e:
            return None


def flush_batch_to_manifest(manifest_path, results):
    """Write batch of successful results to manifest file.
    SAFETY: verify each wav file actually exists on disk before writing manifest entry.
    """
    written = 0
    with open(manifest_path, 'a') as mf:
        for r in results:
            if r is None:
                continue
            utt_id, dataset, split, wav_path, text = r
            # CRITICAL: verify wav file exists and is non-empty before recording
            if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
                print(f"    WARNING: wav missing or empty after convert: {wav_path}")
                continue
            entry = {
                "utt_id": utt_id,
                "dataset": dataset,
                "split": split,
                "audio_path": wav_path,
                "text": clean_text(text),
            }
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
    return written


def load_existing_ids(manifest_path):
    """Load existing utt_ids from manifest for resume support."""
    existing = set()
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            for line in f:
                try:
                    existing.add(json.loads(line)['utt_id'])
                except:
                    pass
        if existing:
            print(f"  Resuming: {len(existing)} entries already in manifest")
    return existing


def check_disk_space():
    """Return free disk space in GB."""
    stat = os.statvfs(str(DATA_ROOT))
    return (stat.f_bavail * stat.f_frsize) / (1024**3)


# ============================================================
# Dataset-specific extractors
# ============================================================

def extract_parquet_dataset(dataset_name, text_col, split="train", num_workers=NUM_WORKERS):
    """Generic extractor for parquet datasets with embedded audio bytes.
    Processes one parquet file at a time, extracts wavs in parallel batches.
    """
    dataset_dir = DATA_ROOT / dataset_name
    data_dir = dataset_dir / "data"
    wav_dir = dataset_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    manifest_path = dataset_dir / "manifest.jsonl"

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"  No parquet files found in {data_dir}")
        return 0

    existing_ids = load_existing_ids(manifest_path)
    total_extracted = 0

    for pi, pfile in enumerate(parquet_files):
        print(f"  [{pi+1}/{len(parquet_files)}] {pfile.name}...", end=" ", flush=True)
        try:
            df = pd.read_parquet(pfile)
        except Exception as e:
            print(f"ERROR reading: {e}")
            continue

        # Determine split from filename
        file_split = split
        if "test" in pfile.name:
            file_split = "test"
        elif "dev" in pfile.name or "valid" in pfile.name:
            file_split = "dev"

        # Prepare batch of work items
        batch = []
        for idx, row in df.iterrows():
            audio_info = row['audio']
            text = row.get(text_col, "")
            if not text or not isinstance(text, str):
                continue

            orig_path = audio_info.get('path', None) if isinstance(audio_info, dict) else None
            if orig_path and str(orig_path) != 'None' and orig_path is not None:
                stem = Path(str(orig_path)).stem
            else:
                stem = f"{pfile.stem}_{idx:06d}"

            utt_id = f"{dataset_name}__{file_split}__{stem}"
            if utt_id in existing_ids:
                continue

            wav_path = str(wav_dir / f"{stem}.wav")
            audio_bytes = audio_info['bytes'] if isinstance(audio_info, dict) else audio_info
            if not audio_bytes:
                continue

            batch.append((audio_bytes, wav_path, str(orig_path) if orig_path else None,
                          utt_id, dataset_name, file_split, text))

        # Process batch in parallel
        extracted_count = 0
        if batch:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_convert_one, batch, chunksize=64))
            extracted_count = flush_batch_to_manifest(manifest_path, results)

        total_extracted += extracted_count

        # SAFETY: only delete source if extraction succeeded
        failed = len(batch) - extracted_count
        if failed > len(batch) * 0.5 and len(batch) > 10:
            print(f"WARNING: {failed}/{len(batch)} failed! NOT deleting {pfile.name}", flush=True)
        else:
            print(f"{extracted_count} extracted ({failed} failed). Deleting...", flush=True)
            pfile.unlink()

    # Cleanup empty data dir
    try:
        if data_dir.exists() and not any(data_dir.iterdir()):
            data_dir.rmdir()
    except:
        pass

    return total_extracted


def extract_vivos(num_workers=NUM_WORKERS):
    """Extract vivos from tar.gz + prompts txt."""
    dataset_name = "vivos"
    dataset_dir = DATA_ROOT / dataset_name
    data_dir = dataset_dir / "data"
    wav_dir = dataset_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    manifest_path = dataset_dir / "manifest.jsonl"

    # Load prompts
    prompts = {}
    for split_name in ['train', 'test']:
        prompt_file = data_dir / f"prompts-{split_name}.txt.gz"
        if prompt_file.exists():
            with gzip.open(prompt_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        prompts[parts[0]] = (parts[1], split_name)
    print(f"  Loaded {len(prompts)} prompts")

    existing_ids = load_existing_ids(manifest_path)

    tar_path = data_dir / "vivos.tar.gz"
    if not tar_path.exists():
        print(f"  tar file not found")
        return 0

    # Read all wav members from tar into memory, then process in parallel batches
    batch = []
    total_extracted = 0

    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar:
            if not member.name.endswith('.wav'):
                continue
            stem = Path(member.name).stem
            parts = member.name.split('/')
            split_name = parts[1] if len(parts) >= 3 else "train"

            utt_id = f"{dataset_name}__{split_name}__{stem}"
            if utt_id in existing_ids:
                continue

            text_info = prompts.get(stem)
            if not text_info:
                continue
            text, _ = text_info

            f_in = tar.extractfile(member)
            if f_in is None:
                continue
            audio_bytes = f_in.read()
            wav_path = str(wav_dir / f"{stem}.wav")

            batch.append((audio_bytes, wav_path, member.name,
                          utt_id, dataset_name, split_name, text))

            if len(batch) >= CFG["batch_size"]:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(executor.map(_convert_one, batch, chunksize=64))
                total_extracted += flush_batch_to_manifest(manifest_path, results)
                batch = []
                print(f"    Progress: {total_extracted}", flush=True)

    # Flush remaining
    if batch:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_convert_one, batch, chunksize=64))
        total_extracted += flush_batch_to_manifest(manifest_path, results)

    print(f"  Extracted {total_extracted}. Deleting source...")
    tar_path.unlink()
    for f in data_dir.glob("prompts-*.txt.gz"):
        f.unlink()

    return total_extracted


def extract_vietspeech(num_workers=NUM_WORKERS):
    """Extract vietspeech from arrow streaming files. Process one arrow at a time."""
    dataset_name = "vietspeech"
    dataset_dir = DATA_ROOT / dataset_name
    wav_dir = dataset_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    manifest_path = dataset_dir / "manifest.jsonl"

    arrow_files = sorted(dataset_dir.glob("*.arrow"))
    if not arrow_files:
        print(f"  No arrow files found")
        return 0

    existing_ids = load_existing_ids(manifest_path)
    total_extracted = 0

    for ai, afile in enumerate(arrow_files):
        if (ai + 1) % 50 == 0 or ai == 0:
            print(f"  [{ai+1}/{len(arrow_files)}] {afile.name}...", flush=True)

        try:
            reader = pa.ipc.open_stream(afile)
            table = reader.read_all()
            data = table.to_pydict()
        except Exception as e:
            print(f"    ERROR reading {afile.name}: {e}")
            continue

        batch = []
        n_rows = len(data.get('text', []))
        for i in range(n_rows):
            audio_info = data['audio'][i]
            text = data['text'][i]
            sid = data.get('sid', [None] * n_rows)[i]

            if not text:
                continue

            orig_path = audio_info.get('path', None) if isinstance(audio_info, dict) else None
            if sid:
                stem = sid
            elif orig_path:
                stem = Path(orig_path).stem
            else:
                stem = f"{afile.stem}_{i:06d}"

            utt_id = f"{dataset_name}__train__{stem}"
            if utt_id in existing_ids:
                continue

            wav_path = str(wav_dir / f"{stem}.wav")
            audio_bytes = audio_info['bytes'] if isinstance(audio_info, dict) else audio_info
            if not audio_bytes:
                continue

            batch.append((audio_bytes, wav_path, str(orig_path) if orig_path else None,
                          utt_id, dataset_name, "train", text))

        # Process batch in parallel
        extracted_count = 0
        if batch:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_convert_one, batch, chunksize=64))
            extracted_count = flush_batch_to_manifest(manifest_path, results)

        total_extracted += extracted_count
        # SAFETY: only delete if extraction was not catastrophically bad
        failed = len(batch) - extracted_count
        if failed > len(batch) * 0.5 and len(batch) > 10:
            print(f"    WARNING: {failed}/{len(batch)} failed! NOT deleting {afile.name}", flush=True)
        else:
            afile.unlink()

        if (ai + 1) % 50 == 0:
            free_gb = check_disk_space()
            print(f"    Total so far: {total_extracted}, Disk free: {free_gb:.1f}GB", flush=True)

    return total_extracted


def extract_gigaspeech2(num_workers=NUM_WORKERS):
    """Extract gigaspeech2_vi from tar.gz files + tsv transcripts.
    Process one tar at a time, extract wavs in parallel batches.
    """
    dataset_name = "gigaspeech2_vi"
    dataset_dir = DATA_ROOT / dataset_name
    vi_dir = dataset_dir / "data" / "vi"
    wav_dir = dataset_dir / "wav"
    wav_dir.mkdir(exist_ok=True)
    manifest_path = dataset_dir / "manifest.jsonl"

    # Load transcripts
    print("  Loading transcripts from TSV...")
    transcripts = {}
    for tsv_name, split_name in [("train_raw.tsv", "train"),
                                  ("train_refined.tsv", "train"),
                                  ("dev.tsv", "dev"),
                                  ("test.tsv", "test")]:
        tsv_path = vi_dir / tsv_name
        if not tsv_path.exists():
            continue
        count = 0
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    uid, text = parts
                    # refined overwrites raw
                    if uid not in transcripts or tsv_name == "train_refined.tsv":
                        transcripts[uid] = (text, split_name)
                        count += 1
        print(f"    {tsv_name}: {count} entries")
    print(f"  Total transcripts: {len(transcripts)}")

    existing_ids = load_existing_ids(manifest_path)

    # Collect tar files
    train_dir = vi_dir / "train"
    tar_files = []
    if train_dir.exists():
        tar_files = sorted(train_dir.glob("*.tar.gz"),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else 999999)
    for extra in ["dev.tar.gz", "test.tar.gz"]:
        p = vi_dir / extra
        if p.exists():
            tar_files.append(p)

    if not tar_files:
        print(f"  No tar.gz files found")
        return 0

    total_extracted = 0
    for ti, tfile in enumerate(tar_files):
        free_gb = check_disk_space()
        print(f"  [{ti+1}/{len(tar_files)}] {tfile.name} (disk free: {free_gb:.1f}GB)...", flush=True)

        if free_gb < 15:
            print(f"  WARNING: Less than 15GB free! Stopping gigaspeech2 extraction.")
            break

        try:
            batch = []
            extracted_this_tar = 0

            with tarfile.open(tfile, 'r:gz') as tar:
                for member in tar:
                    if not member.name.endswith('.wav'):
                        continue
                    stem = Path(member.name).stem
                    text_info = transcripts.get(stem)
                    if not text_info:
                        continue
                    text, split_name = text_info

                    utt_id = f"{dataset_name}__{split_name}__{stem}"
                    if utt_id in existing_ids:
                        continue

                    f_in = tar.extractfile(member)
                    if f_in is None:
                        continue
                    audio_bytes = f_in.read()
                    wav_path = str(wav_dir / f"{stem}.wav")

                    batch.append((audio_bytes, wav_path, member.name,
                                  utt_id, dataset_name, split_name, text))

                    # Process in batches to limit memory usage
                    if len(batch) >= CFG["batch_size"]:
                        with ProcessPoolExecutor(max_workers=num_workers) as executor:
                            results = list(executor.map(_convert_one, batch, chunksize=64))
                        extracted_this_tar += flush_batch_to_manifest(manifest_path, results)
                        batch = []

            # Flush remaining batch
            if batch:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(executor.map(_convert_one, batch, chunksize=64))
                extracted_this_tar += flush_batch_to_manifest(manifest_path, results)

            total_extracted += extracted_this_tar
            # SAFETY: verify before delete
            if extracted_this_tar == 0 and len(batch) == 0:
                # Nothing to extract from this tar (all already done or no matching transcripts)
                print(f"    Nothing new. Deleting {tfile.name}...", flush=True)
                tfile.unlink()
            elif extracted_this_tar > 0:
                print(f"    {extracted_this_tar} extracted. Deleting {tfile.name}...", flush=True)
                tfile.unlink()
            else:
                print(f"    WARNING: 0 extracted but had items! NOT deleting {tfile.name}", flush=True)

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()

    # Cleanup tsv files
    for tsv_name in ["train_refined.tsv", "train_raw.tsv", "dev.tsv", "test.tsv"]:
        tsv_path = vi_dir / tsv_name
        if tsv_path.exists():
            tsv_path.unlink()

    # Cleanup empty dirs
    for d in [train_dir, vi_dir, dataset_dir / "data"]:
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except:
            pass

    return total_extracted


# ============================================================
# Main
# ============================================================

DATASETS = [
    ("vais1000", "parquet", "transcription"),
    ("vivos", "vivos", None),
    ("vlsp2020_vinai_100h", "parquet", "transcription"),
    ("viet_bud500", "parquet", "transcription"),
    ("phoaudiobook", "parquet", "text"),
    ("viVoice", "parquet", "text"),
    ("vietspeech", "vietspeech", None),
    ("gigaspeech2_vi", "gigaspeech2", None),
]


def main():
    parser = argparse.ArgumentParser(description="Extract all datasets to wav + manifest")
    parser.add_argument("datasets", nargs="*", help="Specific datasets to extract (default: all)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for parallel processing")
    args = parser.parse_args()

    num_workers = args.workers
    CFG["batch_size"] = args.batch_size

    selected = set(args.datasets) if args.datasets else None

    print(f"Configuration: workers={num_workers}, batch_size={CFG['batch_size']}")
    print(f"Disk free: {check_disk_space():.1f} GB")
    print()

    for name, dtype, text_col in DATASETS:
        if selected and name not in selected:
            continue

        dataset_dir = DATA_ROOT / name
        if not dataset_dir.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Extracting: {name} (type={dtype})")
        print(f"{'='*60}")

        free_gb = check_disk_space()
        print(f"  Disk free: {free_gb:.1f} GB")

        if free_gb < 10:
            print(f"  CRITICAL: Less than 10GB free! Stopping.")
            break

        try:
            if dtype == "parquet":
                count = extract_parquet_dataset(name, text_col, num_workers=num_workers)
            elif dtype == "vivos":
                count = extract_vivos(num_workers=num_workers)
            elif dtype == "vietspeech":
                count = extract_vietspeech(num_workers=num_workers)
            elif dtype == "gigaspeech2":
                count = extract_gigaspeech2(num_workers=num_workers)
            else:
                print(f"  Unknown type: {dtype}")
                continue
            print(f"  DONE: {name} → {count} entries")
        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_datasets = [n for n, _, _ in DATASETS] + ["fpt_fosd"]
    for name in all_datasets:
        manifest_path = DATA_ROOT / name / "manifest.jsonl"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                count = sum(1 for _ in f)
            print(f"  {name}: {count:,} entries")
        else:
            print(f"  {name}: no manifest")

    print(f"\nDisk free: {check_disk_space():.1f} GB")


if __name__ == "__main__":
    main()
