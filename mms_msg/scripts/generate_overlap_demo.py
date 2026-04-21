#!/usr/bin/env python
"""Generate 2-speaker/3-speaker overlap demo mixtures with quality checks.

Example:
python D:\\data_voice\\mms_msg\\scripts\\generate_overlap_demo.py ^
  --audio-dir D:\\data_voice\\test_audio ^
  --output-dir D:\\data_voice\\mms_msg\\test_mix_output ^
  --num-speakers 2 3 ^
  --scan-examples 120
"""

from __future__ import annotations

import argparse
import functools
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import lazy_dataset
import numpy as np
import soundfile as sf

# Allow running this script directly without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mms_msg import keys
from mms_msg.sampling.environment.scaling import UniformScalingSampler
from mms_msg.sampling.pattern.meeting import MeetingSampler
from mms_msg.sampling.pattern.meeting.overlap_sampler import UniformOverlapSampler
from mms_msg.sampling.source_composition import get_composition_dataset
from mms_msg.sampling.source_composition.composition import sample_fast_utterance_composition
from mms_msg.simulation.anechoic import anechoic_scenario_map_fn
from mms_msg.simulation.utils import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate overlap-heavy demo mixtures (2spk/3spk) from an audio folder."
    )
    parser.add_argument("--audio-dir", type=Path, required=True, help="Folder containing source audio.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for generated mixes.")
    parser.add_argument(
        "--num-speakers",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Speaker counts to generate (e.g., 2 3).",
    )
    parser.add_argument(
        "--speaker-mode",
        choices=["virtual", "from_filename"],
        default="virtual",
        help=(
            "virtual: assign pseudo speakers round-robin; "
            "from_filename: parse speaker as first 2 tokens split by '_' (e.g., vi_000000022_0005 -> vi_000000022)."
        ),
    )
    parser.add_argument("--virtual-speakers", type=int, default=6, help="Number of pseudo speakers in virtual mode.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for composition dataset.")
    parser.add_argument("--scan-examples", type=int, default=120, help="How many candidate meetings to scan.")
    parser.add_argument("--meeting-seconds", type=float, default=60.0, help="Target minimum meeting duration (seconds).")
    parser.add_argument("--max-files", type=int, default=0, help="Use first N valid files only (0 = all).")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"],
        help="Audio file extensions to search.",
    )
    parser.add_argument(
        "--overlap-max-seconds",
        type=float,
        default=4.0,
        help="Maximum overlap duration between adjacent segments (seconds).",
    )
    parser.add_argument(
        "--overlap-soft-min-seconds",
        type=float,
        default=1.5,
        help="Soft minimum overlap duration (seconds).",
    )
    parser.add_argument(
        "--overlap-hard-min-seconds",
        type=float,
        default=0.5,
        help="Hard minimum overlap duration (seconds).",
    )
    parser.add_argument(
        "--silence-max-seconds",
        type=float,
        default=0.10,
        help="Maximum silence between adjacent segments (seconds).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=3.0,
        help="UniformScalingSampler max_weight.",
    )
    parser.add_argument(
        "--verify-min-overlap2-2spk",
        type=float,
        default=0.20,
        help="Minimum overlap ratio >=2 for 2spk quality PASS.",
    )
    parser.add_argument(
        "--verify-min-overlap2-3spk",
        type=float,
        default=0.15,
        help="Minimum overlap ratio >=2 for 3spk quality PASS.",
    )
    parser.add_argument(
        "--verify-min-overlap3",
        type=float,
        default=0.02,
        help="Minimum overlap ratio >=3 for 3spk quality PASS.",
    )
    parser.add_argument(
        "--verify-require-max-concurrent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require max concurrent observed to equal target speaker count.",
    )
    return parser.parse_args()


def find_audio_files(audio_dir: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for p in sorted(audio_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)
    return files


def detect_speaker_id(path: Path, mode: str, idx: int, virtual_speakers: int) -> str:
    if mode == "virtual":
        return f"spk_{idx % max(1, virtual_speakers):03d}"

    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}".upper()
    return f"spk_{idx % max(1, virtual_speakers):03d}"


def collect_valid_audio(
    audio_dir: Path,
    extensions: set[str],
    max_files: int,
) -> tuple[list[tuple[Path, int, int]], list[dict[str, str]]]:
    bad_files: list[dict[str, str]] = []
    valid: list[tuple[Path, int, int]] = []
    for p in find_audio_files(audio_dir, extensions):
        try:
            info = sf.info(str(p))
            if info.channels <= 0:
                bad_files.append({"file": str(p), "reason": f"channels={info.channels} (invalid)"})
                continue
            if info.frames <= 0:
                bad_files.append({"file": str(p), "reason": "empty/zero frames"})
                continue
            valid.append((p, int(info.samplerate), int(info.frames)))
            if max_files > 0 and len(valid) >= max_files:
                break
        except Exception as e:
            bad_files.append({"file": str(p), "reason": str(e)})
    return valid, bad_files


def choose_samplerate(valid: list[tuple[Path, int, int]]) -> int:
    counts = Counter(sr for _, sr, _ in valid)
    return counts.most_common(1)[0][0]


def build_input_dataset(
    valid: list[tuple[Path, int, int]],
    target_sr: int,
    speaker_mode: str,
    virtual_speakers: int,
) -> lazy_dataset.core.Dataset:
    examples: list[dict[str, Any]] = []
    for i, (path, sr, frames) in enumerate(valid):
        if sr != target_sr:
            continue
        spk = detect_speaker_id(path, speaker_mode, i, virtual_speakers)
        examples.append(
            {
                "example_id": f"utt_{i:07d}",
                "speaker_id": spk,
                "scenario": spk,
                "audio_path": str(path),
                "num_samples": int(frames),
                "dataset": "input_audio",
            }
        )
    if not examples:
        raise RuntimeError("No valid files left after sample-rate filtering.")
    return lazy_dataset.new(examples)


def simulate(example: dict, sr: int) -> tuple[dict, np.ndarray, list[np.ndarray], list[np.ndarray], dict[str, float]]:
    ex = load_audio(example, keys.ORIGINAL_SOURCE)
    # Support stereo/multichannel input by downmixing each utterance to mono.
    mono_sources = []
    for src in ex["audio_data"][keys.ORIGINAL_SOURCE]:
        arr = np.asarray(src)
        if arr.ndim == 1:
            mono = arr
        else:
            # Prefer averaging a likely channel axis (usually last, sometimes first).
            if arr.shape[-1] <= 8:
                mono = np.mean(arr, axis=-1)
            elif arr.shape[0] <= 8:
                mono = np.mean(arr, axis=0)
            else:
                mono = np.mean(arr, axis=-1)
            if mono.ndim != 1:
                mono = mono.reshape(-1)
        mono_sources.append(mono.astype(np.float32, copy=False))
    ex["audio_data"][keys.ORIGINAL_SOURCE] = mono_sources
    ex = anechoic_scenario_map_fn(ex)

    mix = np.asarray(ex["audio_data"]["observation"], dtype=np.float32)
    speech_image = [np.asarray(x, dtype=np.float32) for x in ex["audio_data"]["speech_image"]]
    speech_source = [np.asarray(x, dtype=np.float32) for x in ex["audio_data"]["speech_source"]]

    active = np.stack([(np.abs(s) > 1e-6).astype(np.int32) for s in speech_source], axis=0)
    active_count = np.sum(active, axis=0)
    metrics = {
        "meeting_duration_sec": len(mix) / float(sr),
        "overlap_ratio_ge_2": float(np.mean(active_count >= 2)),
        "overlap_ratio_ge_3": float(np.mean(active_count >= 3)),
        "max_concurrent_observed": int(active_count.max()),
    }
    return ex, mix, speech_image, speech_source, metrics


def quality_check(
    n_spk: int,
    metrics: dict[str, float],
    min_overlap2_2spk: float,
    min_overlap2_3spk: float,
    min_overlap3: float,
    require_max_concurrent: bool,
) -> dict[str, Any]:
    min_overlap2 = min_overlap2_2spk if n_spk == 2 else min_overlap2_3spk
    checks = {
        "pass_overlap2": metrics["overlap_ratio_ge_2"] >= min_overlap2,
        "pass_overlap3": True if n_spk < 3 else metrics["overlap_ratio_ge_3"] >= min_overlap3,
        "pass_max_concurrent": True
        if not require_max_concurrent
        else int(metrics["max_concurrent_observed"]) >= int(n_spk),
    }
    checks["pass_all"] = bool(all(checks.values()))
    return checks


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {(e if e.startswith(".") else f".{e}").lower() for e in args.extensions}

    valid, bad_files = collect_valid_audio(args.audio_dir, extensions, args.max_files)
    if not valid:
        raise RuntimeError(f"No readable mono audio found in {args.audio_dir}")

    target_sr = choose_samplerate(valid)
    input_ds = build_input_dataset(valid, target_sr, args.speaker_mode, args.virtual_speakers)

    summary: dict[str, Any] = {
        "audio_dir": str(args.audio_dir),
        "output_dir": str(args.output_dir),
        "target_samplerate": target_sr,
        "num_input_valid_files": len(input_ds),
        "num_input_bad_files": len(bad_files),
        "bad_files_preview": bad_files[:30],
        "config": {
            "speaker_mode": args.speaker_mode,
            "virtual_speakers": args.virtual_speakers,
            "seed": args.seed,
            "scan_examples": args.scan_examples,
            "meeting_seconds": args.meeting_seconds,
            "overlap_max_seconds": args.overlap_max_seconds,
            "overlap_soft_min_seconds": args.overlap_soft_min_seconds,
            "overlap_hard_min_seconds": args.overlap_hard_min_seconds,
            "silence_max_seconds": args.silence_max_seconds,
            "max_weight": args.max_weight,
        },
        "results": [],
    }

    for n_spk in args.num_speakers:
        if n_spk < 2:
            continue

        overlap_sampler = UniformOverlapSampler(
            max_concurrent_spk=n_spk,
            p_silence=0.0,
            maximum_silence=int(args.silence_max_seconds * target_sr),
            maximum_overlap=int(args.overlap_max_seconds * target_sr),
            soft_minimum_overlap=int(args.overlap_soft_min_seconds * target_sr),
            hard_minimum_overlap=int(args.overlap_hard_min_seconds * target_sr),
        )
        meeting_sampler = MeetingSampler(
            duration=int(args.meeting_seconds * target_sr),
            overlap_sampler=overlap_sampler,
        )
        try:
            ds = get_composition_dataset(input_ds, num_speakers=n_spk, rng=args.seed)
        except RuntimeError:
            # Fallback for highly imbalanced / very small subsets used in quick tests.
            ds = get_composition_dataset(
                input_ds,
                num_speakers=n_spk,
                rng=args.seed,
                composition_sampler=functools.partial(
                    sample_fast_utterance_composition,
                    length=len(input_ds),
                ),
            )
        ds = ds.map(UniformScalingSampler(max_weight=args.max_weight))
        ds = ds.map(meeting_sampler(input_ds))

        scanned = min(int(args.scan_examples), len(ds))
        best = None
        best_score = None

        for idx in range(scanned):
            ex, mix, speech_image, speech_source, metrics = simulate(ds[idx], target_sr)
            score = (
                int(metrics["max_concurrent_observed"]),
                float(metrics["overlap_ratio_ge_3"]),
                float(metrics["overlap_ratio_ge_2"]),
            )
            if best is None or score > best_score:
                best = (idx, ex, mix, speech_image, speech_source, metrics)
                best_score = score

        if best is None:
            raise RuntimeError(f"Failed to generate candidate for {n_spk} speakers.")

        idx, ex, mix, speech_image, _speech_source, metrics = best
        peak = float(np.max(np.abs(mix)))
        norm = 0.99 / peak if peak > 0 else 1.0

        out_sub = args.output_dir / f"{n_spk}spk"
        out_sub.mkdir(parents=True, exist_ok=True)

        mix_path = out_sub / "mix.wav"
        sf.write(str(mix_path), mix * norm, target_sr)

        source_paths: list[str] = []
        for i, src in enumerate(speech_image, start=1):
            p = out_sub / f"source_{i:02d}.wav"
            sf.write(str(p), src * norm, target_sr)
            source_paths.append(str(p))

        checks = quality_check(
            n_spk=n_spk,
            metrics=metrics,
            min_overlap2_2spk=float(args.verify_min_overlap2_2spk),
            min_overlap2_3spk=float(args.verify_min_overlap2_3spk),
            min_overlap3=float(args.verify_min_overlap3),
            require_max_concurrent=bool(args.verify_require_max_concurrent),
        )

        result = {
            "num_target_speakers": n_spk,
            "selected_from_example_index": int(idx),
            "num_candidates_scanned": int(scanned),
            "samplerate": target_sr,
            "mix_path": str(mix_path),
            "source_paths": source_paths,
            "num_segments_in_meeting": len(ex["source_id"]),
            "meeting_duration_sec": metrics["meeting_duration_sec"],
            "overlap_ratio_ge_2": metrics["overlap_ratio_ge_2"],
            "overlap_ratio_ge_3": metrics["overlap_ratio_ge_3"],
            "max_concurrent_observed": metrics["max_concurrent_observed"],
            "source_ids": list(ex["source_id"]),
            "speaker_ids_sequence": list(ex["speaker_id"]),
            "offset_samples": [int(x) for x in ex["offset"]["original_source"]],
            "log_weights": [float(x) for x in ex["log_weights"]],
            "quality_checks": checks,
        }

        with (out_sub / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        summary["results"].append(result)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
