#!/usr/bin/env python3
"""Generate noise/rir/rir_noise augmented train cuts from existing clean cuts.

Lhotse mix/reverb_rir are lazy (metadata only, no audio I/O at write time) —
dataloader materializes per-batch.

Input:
  data/lhotse_cutsets_new_train/train.{single,overlap2,overlap3}.cuts_clean.jsonl.gz

Output (same dir):
  train.{mix}.cuts_noise.jsonl.gz
  train.{mix}.cuts_rir.jsonl.gz
  train.{mix}.cuts_rir_noise.jsonl.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.recipes import prepare_musan, prepare_rir_noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts-dir", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/lhotse_cutsets_new_train"))
    ap.add_argument("--musan-dir", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/musan"))
    ap.add_argument("--rir-dir", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/RIRS_NOISES"))
    ap.add_argument("--musan-lhotse", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/musan_lhotse"))
    ap.add_argument("--rir-lhotse", type=Path,
                    default=Path("/home/vietnam/voice_to_text/data/rir_lhotse"))
    ap.add_argument("--variants", nargs="+", default=["noise", "rir", "rir_noise"],
                    choices=["noise", "rir", "rir_noise"])
    ap.add_argument("--mix-types", nargs="+", default=["single", "overlap2", "overlap3"])
    ap.add_argument("--aug-prob", type=float, default=0.5)
    ap.add_argument("--snr-low", type=float, default=5.0)
    ap.add_argument("--snr-high", type=float, default=20.0)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--musan-parts", nargs="+", default=["noise"],
                    choices=["music", "speech", "noise"])
    args = ap.parse_args()

    args.musan_lhotse.mkdir(parents=True, exist_ok=True)
    args.rir_lhotse.mkdir(parents=True, exist_ok=True)

    need_noise = "noise" in args.variants or "rir_noise" in args.variants
    need_rir = "rir" in args.variants or "rir_noise" in args.variants

    combined_noise = None
    if need_noise:
        print("[MUSAN] preparing manifests...")
        musan = prepare_musan(
            corpus_dir=str(args.musan_dir),
            output_dir=str(args.musan_lhotse),
            parts=tuple(args.musan_parts),
        )
        parts = []
        for part_name in args.musan_parts:
            pd = musan.get(part_name)
            if not pd:
                continue
            recs = pd["recordings"] if isinstance(pd, dict) else pd
            sups = pd.get("supervisions") if isinstance(pd, dict) else None
            if sups is not None:
                cs = CutSet.from_manifests(recordings=recs, supervisions=sups)
            else:
                cs = CutSet.from_manifests(recordings=recs)
            parts.append(cs)
        combined_noise = parts[0]
        for p in parts[1:]:
            combined_noise = combined_noise + p
        print(f"[MUSAN] combined_noise cuts: {len(combined_noise):,}")

    rir_recordings = None
    if need_rir:
        print("[RIR] preparing manifests...")
        rir = prepare_rir_noise(
            corpus_dir=str(args.rir_dir),
            output_dir=str(args.rir_lhotse),
        )
        all_recs = []
        for part_name, pd in rir.items():
            recs = pd["recordings"] if isinstance(pd, dict) else pd
            for r in recs:
                all_recs.append(r)
        rir_recordings = RecordingSet.from_recordings(all_recs)
        print(f"[RIR] recordings: {len(rir_recordings):,}")

    for mix in args.mix_types:
        clean_path = args.cuts_dir / f"train.{mix}.cuts_clean.jsonl.gz"
        if not clean_path.exists():
            print(f"  [SKIP missing] {clean_path}")
            continue
        print(f"\n=== {mix} ===")
        cuts = CutSet.from_file(clean_path)
        print(f"  clean cuts: {len(cuts):,}")

        cuts_rir = None
        if "rir" in args.variants:
            out = args.cuts_dir / f"train.{mix}.cuts_rir.jsonl.gz"
            if args.skip_existing and out.exists():
                print(f"  [skip existing] {out.name}")
                cuts_rir = CutSet.from_file(out)
            else:
                print(f"  applying RIR...")
                cuts_rir = cuts.reverb_rir(
                    rir_recordings=rir_recordings,
                    normalize_output=True,
                    affix_id=True,
                )
                cuts_rir.to_file(out)
                print(f"  saved {out.name}")

        if "noise" in args.variants:
            out = args.cuts_dir / f"train.{mix}.cuts_noise.jsonl.gz"
            if args.skip_existing and out.exists():
                print(f"  [skip existing] {out.name}")
            else:
                print(f"  applying MUSAN noise (snr={args.snr_low}-{args.snr_high}, prob={args.aug_prob})...")
                cuts_noise = cuts.mix(
                    cuts=combined_noise,
                    snr=[args.snr_low, args.snr_high],
                    mix_prob=args.aug_prob,
                    seed=42,
                    random_mix_offset=False,
                )
                cuts_noise.to_file(out)
                print(f"  saved {out.name}")

        if "rir_noise" in args.variants and cuts_rir is not None and combined_noise is not None:
            out = args.cuts_dir / f"train.{mix}.cuts_rir_noise.jsonl.gz"
            if args.skip_existing and out.exists():
                print(f"  [skip existing] {out.name}")
            else:
                print(f"  applying RIR + MUSAN noise...")
                cuts_rn = cuts_rir.mix(
                    cuts=combined_noise,
                    snr=[args.snr_low, args.snr_high],
                    mix_prob=args.aug_prob,
                    seed=43,
                    random_mix_offset=False,
                )
                cuts_rn.to_file(out)
                print(f"  saved {out.name}")


if __name__ == "__main__":
    main()
