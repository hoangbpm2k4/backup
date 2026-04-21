#!/usr/bin/env python3
"""Build Lhotse CutSet from overlap manifests + apply RIR/MUSAN augmentation.

Steps:
1. Download MUSAN + RIR_NOISES (if not present)
2. Convert single/overlap2/overlap3 manifests → Lhotse Recording + Supervision
3. Apply augmentation (on-the-fly metadata, NO new wav files):
   - RIR convolution (reverb + far-field simulation)
   - MUSAN noise/music/babble mixing
4. Save clean + augmented CutSets

Output structure:
  data/lhotse_cutsets/
    {split}.{mix_type}.cuts_clean.jsonl.gz       # no augmentation
    {split}.{mix_type}.cuts_rir.jsonl.gz          # + RIR
    {split}.{mix_type}.cuts_noise.jsonl.gz        # + MUSAN noise
    {split}.{mix_type}.cuts_rir_noise.jsonl.gz    # + RIR + MUSAN noise
    {split}.all.cuts_clean.jsonl.gz               # merged all types
    {split}.all.cuts_augmented.jsonl.gz           # merged clean + all augmented
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
from lhotse import (
    CutSet,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)
from lhotse.recipes import (
    download_musan,
    download_rir_noise,
    prepare_musan,
    prepare_rir_noise,
)


# ---------------------------------------------------------------------------
# Step 1: Download MUSAN + RIR
# ---------------------------------------------------------------------------

def ensure_musan(data_dir: Path) -> Dict:
    """Download + prepare MUSAN if needed. Returns manifest dict."""
    musan_dir = data_dir / "musan"
    musan_manifest_dir = data_dir / "musan_lhotse"

    if not musan_dir.exists():
        print("[DOWNLOAD] Downloading MUSAN...", flush=True)
        download_musan(target_dir=str(data_dir))
    else:
        print(f"[MUSAN] Already exists at {musan_dir}", flush=True)

    print("[MUSAN] Preparing manifests...", flush=True)
    manifests = prepare_musan(
        corpus_dir=str(musan_dir),
        output_dir=str(musan_manifest_dir),
        parts=("music", "speech", "noise"),
    )
    return manifests


def ensure_rir(data_dir: Path) -> Dict:
    """Download + prepare RIR_NOISES if needed. Returns manifest dict."""
    rir_dir = data_dir / "RIRS_NOISES"
    rir_manifest_dir = data_dir / "rir_lhotse"

    if not rir_dir.exists():
        print("[DOWNLOAD] Downloading RIR_NOISES...", flush=True)
        download_rir_noise(target_dir=str(data_dir))
    else:
        print(f"[RIR] Already exists at {rir_dir}", flush=True)

    print("[RIR] Preparing manifests...", flush=True)
    manifests = prepare_rir_noise(
        corpus_dir=str(rir_dir),
        output_dir=str(rir_manifest_dir),
    )
    return manifests


# ---------------------------------------------------------------------------
# Step 2: Convert manifests → Lhotse Recording + Supervision
# ---------------------------------------------------------------------------

def _probe_audio(path: str) -> Optional[Tuple[int, int]]:
    """Return (num_samples, sample_rate) or None if file missing."""
    try:
        info = sf.info(path)
        return int(info.frames), int(info.samplerate)
    except Exception:
        return None


def _build_single_worker(line: str) -> Optional[Tuple[dict, dict]]:
    """Worker: parse single manifest line → (recording_dict, supervision_dict)."""
    row = json.loads(line)
    audio_path = row["audio_path"]
    probe = _probe_audio(audio_path)
    if probe is None:
        return None
    num_samples, sr = probe
    duration = num_samples / sr
    rec_id = row["mix_id"]

    rec = {
        "id": rec_id,
        "sources": [{"type": "file", "channels": [0], "source": audio_path}],
        "sampling_rate": sr,
        "num_samples": num_samples,
        "duration": duration,
    }

    # For single: speaker_id from dataset + utt_id
    speaker = row.get("speaker_id", row["mix_id"])
    sup = {
        "id": f"{rec_id}-seg0",
        "recording_id": rec_id,
        "start": 0.0,
        "duration": duration,
        "text": row.get("text", ""),
        "speaker": speaker,
        "channel": 0,
    }
    return rec, sup


def _build_overlap_worker(args: Tuple[str, str]) -> Optional[Tuple[dict, List[dict]]]:
    """Worker: parse overlap manifest line → (recording_dict, [supervision_dicts])."""
    manifest_line, metadata_dir = args
    row = json.loads(manifest_line)
    mix_id = row["mix_id"]
    audio_path = row["audio_path"]

    probe = _probe_audio(audio_path)
    if probe is None:
        return None
    num_samples, sr = probe
    duration = num_samples / sr

    rec = {
        "id": mix_id,
        "sources": [{"type": "file", "channels": [0], "source": audio_path}],
        "sampling_rate": sr,
        "num_samples": num_samples,
        "duration": duration,
    }

    # Read metadata.json for supervision segments
    meta_path = row.get("metadata_path")
    if not meta_path or not os.path.exists(meta_path):
        # Fallback: single supervision with full text
        sup = {
            "id": f"{mix_id}-seg0",
            "recording_id": mix_id,
            "start": 0.0,
            "duration": duration,
            "text": row.get("text", ""),
            "speaker": "unknown",
            "channel": 0,
        }
        return rec, [sup]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    sups = []
    for src in meta.get("sources", []):
        sup = {
            "id": src["seg_id"],
            "recording_id": mix_id,
            "start": src["start_sec"],
            "duration": src["duration_sec"],
            "text": src.get("text", ""),
            "speaker": src["speaker_id"],
            "channel": 0,
        }
        sups.append(sup)

    return rec, sups


def build_cutset_from_manifest(
    manifest_path: Path,
    mix_type: str,
    workers: int = 32,
) -> Optional[CutSet]:
    """Convert our manifest.jsonl → Lhotse CutSet."""
    if not manifest_path.exists():
        print(f"  [SKIP] {manifest_path} not found", flush=True)
        return None

    lines = manifest_path.read_text().strip().split("\n")
    total = len(lines)
    print(f"  Converting {manifest_path.name} ({total:,} entries)...", flush=True)

    recordings = []
    supervisions = []
    failed = 0

    log_every = 100_000 if mix_type == "single" else 50_000

    if mix_type == "single":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_build_single_worker, lines, chunksize=2048)
            for i, result in enumerate(results):
                if result is None:
                    failed += 1
                    continue
                rec_dict, sup_dict = result
                recordings.append(rec_dict)
                supervisions.append(sup_dict)
                if (i + 1) % log_every == 0:
                    print(f"    {i+1:,}/{total:,} ({failed} failed)...", flush=True)
    else:
        # Overlap: need metadata path
        args = [(line, "") for line in lines]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_build_overlap_worker, args, chunksize=1024)
            for i, result in enumerate(results):
                if result is None:
                    failed += 1
                    continue
                rec_dict, sup_list = result
                recordings.append(rec_dict)
                supervisions.extend(sup_list)
                if (i + 1) % log_every == 0:
                    print(f"    {i+1:,}/{total:,} ({failed} failed)...", flush=True)

    if not recordings:
        print(f"  [EMPTY] No valid recordings from {manifest_path.name}", flush=True)
        return None

    print(f"  Done: {len(recordings):,} recordings, {len(supervisions):,} supervisions "
          f"({failed:,} failed)", flush=True)

    rec_set = RecordingSet.from_dicts(recordings)
    sup_set = SupervisionSet.from_dicts(supervisions)
    cuts = CutSet.from_manifests(recordings=rec_set, supervisions=sup_set)
    return cuts


# ---------------------------------------------------------------------------
# Step 3: Augmentation
# ---------------------------------------------------------------------------

def build_musan_cuts(musan_manifests: Dict) -> Dict[str, CutSet]:
    """Build CutSet for each MUSAN part (noise, music, speech/babble)."""
    musan_cuts = {}
    for part_name, part_data in musan_manifests.items():
        recs = part_data["recordings"]
        sups = part_data.get("supervisions")
        if sups is not None:
            cuts = CutSet.from_manifests(recordings=recs, supervisions=sups)
        else:
            cuts = CutSet.from_manifests(recordings=recs)
        musan_cuts[part_name] = cuts
        print(f"  MUSAN {part_name}: {len(cuts)} cuts", flush=True)
    return musan_cuts


def build_rir_recordings(rir_manifests: Dict) -> RecordingSet:
    """Build combined RecordingSet from all RIR parts."""
    all_recs = []
    for part_name, part_data in rir_manifests.items():
        if "recordings" in part_data:
            recs = part_data["recordings"]
            all_recs.append(recs)
            print(f"  RIR {part_name}: {len(recs)} recordings", flush=True)
    if not all_recs:
        return None
    combined = all_recs[0]
    for r in all_recs[1:]:
        combined = combined + r
    print(f"  RIR total: {len(combined)} recordings", flush=True)
    return combined


def apply_augmentation(
    cuts: CutSet,
    rir_recordings: Optional[RecordingSet],
    musan_cuts: Dict[str, CutSet],
    output_dir: Path,
    prefix: str,
    aug_prob: float = 0.5,
    noise_snr: Tuple[float, float] = (5, 20),
    skip_existing: bool = False,
) -> Dict[str, CutSet]:
    """Apply augmentation variants and save CutSets.

    Returns dict of {variant_name: CutSet}.
    """
    results = {}

    def _should_skip(path: Path) -> bool:
        if skip_existing and path.exists() and path.stat().st_size > 0:
            print(f"    [SKIP] {path.name} already exists", flush=True)
            return True
        return False

    # Clean (no augmentation)
    clean_path = output_dir / f"{prefix}.cuts_clean.jsonl.gz"
    if not _should_skip(clean_path):
        cuts.to_file(str(clean_path))
        print(f"    Saved {clean_path.name} ({len(cuts):,} cuts)", flush=True)
    results["clean"] = cuts

    # RIR only (reverb + far-field)
    cuts_rir = None
    if rir_recordings is not None:
        rir_path = output_dir / f"{prefix}.cuts_rir.jsonl.gz"
        if not _should_skip(rir_path):
            print(f"    Applying RIR augmentation...", flush=True)
            cuts_rir = cuts.reverb_rir(
                rir_recordings=rir_recordings,
                normalize_output=True,
                affix_id=True,
            )
            cuts_rir.to_file(str(rir_path))
            print(f"    Saved {rir_path.name} ({len(cuts_rir):,} cuts)", flush=True)
        else:
            cuts_rir = CutSet.from_file(str(rir_path))
        results["rir"] = cuts_rir

    # MUSAN noise only
    noise_parts = []
    for part in ("noise", "music", "speech"):
        if part in musan_cuts:
            noise_parts.append(musan_cuts[part])

    combined_noise = None
    if noise_parts:
        combined_noise = noise_parts[0]
        for np_ in noise_parts[1:]:
            combined_noise = combined_noise + np_

        noise_path = output_dir / f"{prefix}.cuts_noise.jsonl.gz"
        if not _should_skip(noise_path):
            print(f"    Applying MUSAN noise augmentation...", flush=True)
            cuts_noise = cuts.mix(
                cuts=combined_noise,
                snr=list(noise_snr),
                mix_prob=aug_prob,
                seed=42,
                random_mix_offset=True,
            )
            cuts_noise.to_file(str(noise_path))
            print(f"    Saved {noise_path.name} ({len(cuts_noise):,} cuts)", flush=True)
            results["noise"] = cuts_noise
        else:
            results["noise"] = CutSet.from_file(str(noise_path))

    # RIR + MUSAN noise (both)
    if cuts_rir is not None and combined_noise is not None:
        rn_path = output_dir / f"{prefix}.cuts_rir_noise.jsonl.gz"
        if not _should_skip(rn_path):
            print(f"    Applying RIR + MUSAN noise augmentation...", flush=True)
            cuts_rir_noise = cuts_rir.mix(
                cuts=combined_noise,
                snr=list(noise_snr),
                mix_prob=aug_prob,
                seed=42,
                random_mix_offset=True,
            )
            cuts_rir_noise.to_file(str(rn_path))
            print(f"    Saved {rn_path.name} ({len(cuts_rir_noise):,} cuts)", flush=True)
            results["rir_noise"] = cuts_rir_noise
        else:
            results["rir_noise"] = CutSet.from_file(str(rn_path))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Lhotse CutSet + augmentation")
    parser.add_argument("--data-dir", type=str,
                        default="/home/vietnam/voice_to_text/data",
                        help="Root data directory")
    parser.add_argument("--overlap-dir", type=str,
                        default="/home/vietnam/voice_to_text/data/overlap_multidataset",
                        help="Directory with overlap manifests")
    parser.add_argument("--output-dir", type=str,
                        default="/home/vietnam/voice_to_text/data/lhotse_cutsets",
                        help="Output directory for CutSets")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel workers for audio probing")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--mix-types", nargs="+",
                        default=["single", "overlap2", "overlap3"])
    parser.add_argument("--aug-prob", type=float, default=0.5,
                        help="Probability of applying noise augmentation per cut")
    parser.add_argument("--noise-snr-low", type=float, default=5,
                        help="Min SNR for noise mixing (dB)")
    parser.add_argument("--noise-snr-high", type=float, default=20,
                        help="Max SNR for noise mixing (dB)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip MUSAN/RIR download (assume already present)")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Only build clean CutSets, no augmentation")
    parser.add_argument("--merge-all", action="store_true", default=True,
                        help="Merge all mix types per split into combined CutSet")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip CutSet files that already exist on disk")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    overlap_dir = Path(args.overlap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download & prepare MUSAN + RIR ---
    rir_recordings = None
    musan_cuts = {}

    if not args.skip_augment:
        if not args.skip_download:
            musan_manifests = ensure_musan(data_dir)
            rir_manifests = ensure_rir(data_dir)
        else:
            musan_dir = data_dir / "musan"
            rir_dir = data_dir / "RIRS_NOISES"
            musan_manifests = prepare_musan(str(musan_dir),
                                            parts=("music", "speech", "noise"))
            rir_manifests = prepare_rir_noise(str(rir_dir))

        print("\n[MUSAN] Building noise CutSets...", flush=True)
        musan_cuts = build_musan_cuts(musan_manifests)

        print("\n[RIR] Building RIR RecordingSet...", flush=True)
        rir_recordings = build_rir_recordings(rir_manifests)

    # --- Step 2 & 3: Convert manifests + augment ---
    for split in args.splits:
        print(f"\n{'='*60}", flush=True)
        print(f"[{split.upper()}] Processing...", flush=True)
        print(f"{'='*60}", flush=True)

        split_cutsets_clean = []

        for mix_type in args.mix_types:
            manifest_name = f"{split}.{mix_type}.manifest.jsonl"
            manifest_path = overlap_dir / manifest_name
            prefix = f"{split}.{mix_type}"

            print(f"\n--- {prefix} ---", flush=True)

            # Convert to CutSet
            cuts = build_cutset_from_manifest(
                manifest_path, mix_type, workers=args.workers
            )
            if cuts is None:
                continue

            split_cutsets_clean.append(cuts)

            if args.skip_augment:
                # Just save clean
                clean_path = output_dir / f"{prefix}.cuts_clean.jsonl.gz"
                cuts.to_file(str(clean_path))
                print(f"    Saved {clean_path.name} ({len(cuts):,} cuts)", flush=True)
            else:
                # Apply augmentation
                apply_augmentation(
                    cuts=cuts,
                    rir_recordings=rir_recordings,
                    musan_cuts=musan_cuts,
                    output_dir=output_dir,
                    prefix=prefix,
                    aug_prob=args.aug_prob,
                    noise_snr=(args.noise_snr_low, args.noise_snr_high),
                    skip_existing=args.skip_existing,
                )

        # Merge all mix types for this split
        if args.merge_all and split_cutsets_clean:
            print(f"\n[{split.upper()}] Merging all types...", flush=True)
            merged_clean = split_cutsets_clean[0]
            for cs in split_cutsets_clean[1:]:
                merged_clean = merged_clean + cs

            merged_path = output_dir / f"{split}.all.cuts_clean.jsonl.gz"
            merged_clean.to_file(str(merged_path))
            print(f"  Saved {merged_path.name} ({len(merged_clean):,} cuts)", flush=True)

            # Merge augmented (clean + rir + noise + rir_noise)
            if not args.skip_augment:
                all_variants = [merged_clean]
                for variant in ["rir", "noise", "rir_noise"]:
                    variant_parts = []
                    for mix_type in args.mix_types:
                        vpath = output_dir / f"{split}.{mix_type}.cuts_{variant}.jsonl.gz"
                        if vpath.exists():
                            variant_parts.append(CutSet.from_file(str(vpath)))
                    if variant_parts:
                        merged_v = variant_parts[0]
                        for vp in variant_parts[1:]:
                            merged_v = merged_v + vp
                        all_variants.append(merged_v)

                combined = all_variants[0]
                for av in all_variants[1:]:
                    combined = combined + av

                aug_path = output_dir / f"{split}.all.cuts_augmented.jsonl.gz"
                combined.to_file(str(aug_path))
                print(f"  Saved {aug_path.name} ({len(combined):,} cuts)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("DONE!", flush=True)
    print(f"Output: {output_dir}", flush=True)
    ls = sorted(output_dir.glob("*.jsonl.gz"))
    for f in ls:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
