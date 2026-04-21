#!/usr/bin/env python3
"""Generate single/2spk/3spk overlap corpus from VietSpeech with train/val/test splits.

This script:
1) Reads VietSpeech entries from ASR manifest JSONL.
2) Splits speakers into train/val/test (speaker-disjoint).
3) Generates mixtures by ratio: single/overlap2/overlap3.
4) Writes mix.wav + metadata.json per example.
5) Exports split manifest JSONL, supervisions.text, and optional Lhotse CutSet.

Use preview mode first:
python generate_vietspeech_overlap_dataset.py --preview-mixes 2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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


@dataclass
class SourceUtt:
    utt_id: str
    speaker_id: str
    audio_path: Path
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate overlap dataset from VietSpeech with text + CutSet export."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/vietnam/voice_to_text/data/asr_manifest_full_all_audio.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/vietnam/voice_to_text/data/overlap_vietspeech"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--single-ratio", type=float, default=0.65)
    parser.add_argument("--ov2-ratio", type=float, default=0.28)
    parser.add_argument("--ov3-ratio", type=float, default=0.07)
    parser.add_argument(
        "--target-mixtures-per-split",
        type=int,
        default=0,
        help="0 => use number of source utterances in that split.",
    )
    parser.add_argument(
        "--preview-mixes",
        type=int,
        default=0,
        help="If >0, only generate this many examples in train split for listening test.",
    )
    parser.add_argument("--write-cutset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tries-per-mix", type=int, default=50)
    return parser.parse_args()


def parse_vietspeech_speaker(utt_id: str, audio_path: Path) -> Optional[str]:
    # Preferred: from utt_id -> vietspeech__train__VI_000000003_0000
    parts = utt_id.split("__")
    if len(parts) >= 3 and parts[2].startswith("VI_"):
        toks = parts[2].split("_")
        if len(toks) >= 2:
            return "_".join(toks[:2]).upper()  # VI_000000003

    # Fallback from filename -> ..._VI_000000003_0000.wav
    stem = audio_path.stem.upper()
    idx = stem.find("VI_")
    if idx == -1:
        return None
    tail = stem[idx:].split("_")
    if len(tail) >= 2:
        return "_".join(tail[:2])
    return None


def load_vietspeech_from_manifest(manifest_path: Path) -> List[SourceUtt]:
    rows: List[SourceUtt] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("dataset") != "vietspeech":
                continue
            ap = Path(str(obj.get("audio_path", "")))
            if not ap.exists():
                continue
            speaker_id = parse_vietspeech_speaker(str(obj.get("utt_id", "")), ap)
            if not speaker_id:
                continue
            rows.append(
                SourceUtt(
                    utt_id=str(obj.get("utt_id", "")),
                    speaker_id=speaker_id,
                    audio_path=ap,
                    text=str(obj.get("text", "")),
                )
            )
    return rows


def split_speakers(
    speaker_ids: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, set]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratio must sum to 1.0")
    rng = random.Random(seed)
    spk = sorted(set(speaker_ids))
    rng.shuffle(spk)
    n = len(spk)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return {
        "train": set(spk[:n_train]),
        "val": set(spk[n_train : n_train + n_val]),
        "test": set(spk[n_train + n_val : n_train + n_val + n_test]),
    }


def compute_mix_counts(total: int, single_ratio: float, ov2_ratio: float, ov3_ratio: float) -> Dict[str, int]:
    if abs((single_ratio + ov2_ratio + ov3_ratio) - 1.0) > 1e-6:
        raise ValueError("single/ov2/ov3 ratio must sum to 1.0")
    single = int(round(total * single_ratio))
    ov2 = int(round(total * ov2_ratio))
    ov3 = total - single - ov2
    if ov3 < 0:
        ov3 = 0
    return {"single": single, "overlap2": ov2, "overlap3": ov3}


def read_mono(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        mono = arr
    else:
        mono = np.mean(arr, axis=1) if arr.shape[1] <= 8 else np.mean(arr, axis=-1)
        if mono.ndim != 1:
            mono = mono.reshape(-1)
    return mono, int(sr)


def mix_sources(
    src_audio: List[np.ndarray],
    sr: int,
    mix_type: str,
    rng: random.Random,
) -> Tuple[np.ndarray, List[int]]:
    if mix_type == "single":
        starts = [0]
    elif mix_type == "overlap2":
        a0, a1 = src_audio
        ov = int(min(len(a0), len(a1)) * rng.uniform(0.30, 0.70))
        starts = [0, max(0, len(a0) - ov)]
    elif mix_type == "overlap3":
        a0, a1, a2 = src_audio
        # Force period with 3 concurrent speakers by placing sources around shared region.
        start2 = int(len(a0) * rng.uniform(0.30, 0.60))
        start3 = int(len(a0) * rng.uniform(0.45, 0.75))
        starts = [0, max(0, start2), max(0, start3)]
    else:
        raise ValueError(f"Unsupported mix_type: {mix_type}")

    max_len = 0
    for st, a in zip(starts, src_audio):
        max_len = max(max_len, st + len(a))
    mix = np.zeros(max_len, dtype=np.float32)

    for st, a in zip(starts, src_audio):
        gain = rng.uniform(0.8, 1.0)
        mix[st : st + len(a)] += a * gain

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99 and peak > 0:
        mix = mix * (0.99 / peak)

    return mix, starts


def active_overlap_metrics(starts: List[int], src_audio: List[np.ndarray]) -> Tuple[float, float, int]:
    max_len = max(st + len(a) for st, a in zip(starts, src_audio))
    active = np.zeros((len(src_audio), max_len), dtype=np.int8)
    for i, (st, a) in enumerate(zip(starts, src_audio)):
        active[i, st : st + len(a)] = (np.abs(a) > 1e-6).astype(np.int8)
    c = np.sum(active, axis=0)
    return float(np.mean(c >= 2)), float(np.mean(c >= 3)), int(np.max(c))


def generate_split(
    split_name: str,
    split_rows: List[SourceUtt],
    out_dir: Path,
    counts: Dict[str, int],
    seed: int,
    max_tries_per_mix: int,
    write_cutset: bool,
) -> Dict[str, int]:
    rng = random.Random(seed)
    by_spk: Dict[str, List[SourceUtt]] = defaultdict(list)
    for r in split_rows:
        by_spk[r.speaker_id].append(r)
    speakers = sorted(by_spk.keys())

    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.jsonl"
    supervisions_text_path = split_dir / "supervisions.text"
    summary_path = split_dir / "summary.json"
    cutset_path = split_dir / "cuts.jsonl.gz"

    mix_plan: List[str] = (
        ["single"] * counts["single"] + ["overlap2"] * counts["overlap2"] + ["overlap3"] * counts["overlap3"]
    )
    rng.shuffle(mix_plan)

    cut_objs = []
    written = 0
    by_type_written = defaultdict(int)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as mf, supervisions_text_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as sf_text:
        for idx, mix_type in enumerate(mix_plan):
            n_spk = 1 if mix_type == "single" else (2 if mix_type == "overlap2" else 3)
            ok = False
            for _ in range(max_tries_per_mix):
                if len(speakers) < n_spk:
                    break
                chosen_spk = rng.sample(speakers, n_spk)
                chosen_rows = [rng.choice(by_spk[s]) for s in chosen_spk]

                src_audio = []
                src_sr = []
                bad = False
                for r in chosen_rows:
                    a, sr = read_mono(r.audio_path)
                    if a.size == 0:
                        bad = True
                        break
                    src_audio.append(a)
                    src_sr.append(sr)
                if bad or len(set(src_sr)) != 1:
                    continue

                sr = src_sr[0]
                mix, starts = mix_sources(src_audio, sr, mix_type, rng)
                ov2, ov3, max_conc = active_overlap_metrics(starts, src_audio)
                if mix_type == "overlap2" and max_conc < 2:
                    continue
                if mix_type == "overlap3" and max_conc < 3:
                    continue

                mix_id = f"{split_name}_{idx:07d}_{mix_type}"
                mix_dir = split_dir / mix_type / mix_id
                mix_dir.mkdir(parents=True, exist_ok=True)
                mix_path = mix_dir / "mix.wav"
                metadata_path = mix_dir / "metadata.json"
                sf.write(str(mix_path), mix, sr)

                src_meta = []
                for j, (r, st, a) in enumerate(zip(chosen_rows, starts, src_audio)):
                    seg_id = f"{mix_id}_seg{j+1:02d}"
                    duration = len(a) / float(sr)
                    src_meta.append(
                        {
                            "seg_id": seg_id,
                            "speaker_id": r.speaker_id,
                            "utt_id": r.utt_id,
                            "audio_path": r.audio_path.as_posix(),
                            "start_sec": st / float(sr),
                            "duration_sec": duration,
                            "end_sec": (st + len(a)) / float(sr),
                            "text": r.text,
                        }
                    )
                    sf_text.write(f"{seg_id} {r.text}\n")

                src_meta_sorted = sorted(src_meta, key=lambda x: x["start_sec"])
                merged_text = " | ".join(x["text"] for x in src_meta_sorted)

                metadata = {
                    "mix_id": mix_id,
                    "split": split_name,
                    "mix_type": mix_type,
                    "num_speakers": n_spk,
                    "sample_rate": sr,
                    "mix_path": mix_path.as_posix(),
                    "overlap_ratio_ge_2": ov2,
                    "overlap_ratio_ge_3": ov3,
                    "max_concurrent_observed": max_conc,
                    "sources": src_meta_sorted,
                    "text_merged": merged_text,
                }
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

                row = {
                    "mix_id": mix_id,
                    "dataset": "vietspeech_overlap",
                    "split": split_name,
                    "mix_type": mix_type,
                    "num_speakers": n_spk,
                    "audio_path": mix_path.as_posix(),
                    "metadata_path": metadata_path.as_posix(),
                    "text": merged_text,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")

                if write_cutset and CutSet is not None:
                    recording = Recording.from_file(str(mix_path), recording_id=mix_id)
                    sups = [
                        SupervisionSegment(
                            id=s["seg_id"],
                            recording_id=mix_id,
                            start=float(s["start_sec"]),
                            duration=float(s["duration_sec"]),
                            text=s["text"],
                            speaker=s["speaker_id"],
                            channel=0,
                            language="vi",
                        )
                        for s in src_meta_sorted
                    ]
                    cut_objs.append(
                        MonoCut(
                            id=mix_id,
                            start=0.0,
                            duration=recording.duration,
                            channel=0,
                            recording=recording,
                            supervisions=sups,
                        )
                    )

                written += 1
                by_type_written[mix_type] += 1
                ok = True
                break

            if not ok:
                print(f"[WARN] Failed to generate {split_name}:{mix_type} at idx={idx}")

    if write_cutset and CutSet is not None and cut_objs:
        CutSet.from_cuts(cut_objs).to_file(str(cutset_path))

    summary = {
        "split": split_name,
        "requested": counts,
        "written_total": written,
        "written_by_type": dict(by_type_written),
        "num_source_rows": len(split_rows),
        "num_source_speakers": len(speakers),
        "manifest_path": manifest_path.as_posix(),
        "supervisions_text_path": supervisions_text_path.as_posix(),
        "cutset_path": cutset_path.as_posix() if (write_cutset and CutSet is not None) else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"written_total": written, **{f"written_{k}": v for k, v in by_type_written.items()}}


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_vietspeech_from_manifest(args.manifest.resolve())
    if not rows:
        raise RuntimeError("No VietSpeech rows found in manifest.")

    split_spk = split_speakers(
        [r.speaker_id for r in rows],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_rows: Dict[str, List[SourceUtt]] = {"train": [], "val": [], "test": []}
    for r in rows:
        if r.speaker_id in split_spk["train"]:
            split_rows["train"].append(r)
        elif r.speaker_id in split_spk["val"]:
            split_rows["val"].append(r)
        else:
            split_rows["test"].append(r)

    run_summary = {
        "num_rows_total": len(rows),
        "num_speakers_total": len(set(r.speaker_id for r in rows)),
        "split_source_counts": {k: len(v) for k, v in split_rows.items()},
        "split_speaker_counts": {k: len(split_spk[k]) for k in ["train", "val", "test"]},
        "preview_mode": args.preview_mixes > 0,
        "splits": {},
    }

    if args.preview_mixes > 0:
        preview_counts = compute_mix_counts(
            max(1, args.preview_mixes), args.single_ratio, args.ov2_ratio, args.ov3_ratio
        )
        stats = generate_split(
            split_name="train_preview",
            split_rows=split_rows["train"],
            out_dir=out_dir,
            counts=preview_counts,
            seed=args.seed,
            max_tries_per_mix=args.max_tries_per_mix,
            write_cutset=args.write_cutset,
        )
        run_summary["splits"]["train_preview"] = {"counts": preview_counts, "stats": stats}
    else:
        for i, split in enumerate(["train", "val", "test"]):
            total = args.target_mixtures_per_split if args.target_mixtures_per_split > 0 else len(split_rows[split])
            counts = compute_mix_counts(total, args.single_ratio, args.ov2_ratio, args.ov3_ratio)
            stats = generate_split(
                split_name=split,
                split_rows=split_rows[split],
                out_dir=out_dir,
                counts=counts,
                seed=args.seed + i + 1,
                max_tries_per_mix=args.max_tries_per_mix,
                write_cutset=args.write_cutset,
            )
            run_summary["splits"][split] = {"counts": counts, "stats": stats}

    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
