#!/usr/bin/env python3
"""Merge new YT1000h cuts into existing train cuts, renaming IDs to avoid collision.

Before:
  data/lhotse_cutsets_new_train/train.{single,overlap2,overlap3}.cuts_clean.jsonl.gz
  data/overlap_multidataset_yt1000h/train.{single,overlap2,overlap3}.cuts.jsonl.gz

After (in-place update of *_clean.jsonl.gz, old backed up to *.bak.jsonl.gz):
  data/lhotse_cutsets_new_train/train.{single,overlap2,overlap3}.cuts_clean.jsonl.gz
"""
from __future__ import annotations

import shutil
from pathlib import Path

from lhotse import CutSet
from lhotse.cut import MonoCut


OLD_DIR = Path("/home/vietnam/voice_to_text/data/lhotse_cutsets_new_train")
NEW_DIR = Path("/home/vietnam/voice_to_text/data/overlap_multidataset_yt1000h")
PREFIX = "yt1000h_"


def rename_cut(c: MonoCut) -> MonoCut:
    new_id = PREFIX + c.id
    new_rec = c.recording
    if new_rec is not None:
        new_rec = new_rec.with_path_prefix("") if False else new_rec  # keep path
        new_rec = type(new_rec)(
            id=PREFIX + new_rec.id,
            sources=new_rec.sources,
            sampling_rate=new_rec.sampling_rate,
            num_samples=new_rec.num_samples,
            duration=new_rec.duration,
            channel_ids=new_rec.channel_ids,
            transforms=new_rec.transforms,
        )
    new_sups = []
    for s in c.supervisions:
        new_sups.append(s.with_alignment(None) if False else type(s)(
            id=PREFIX + s.id,
            recording_id=PREFIX + s.recording_id,
            start=s.start,
            duration=s.duration,
            channel=s.channel,
            text=(s.text or "").upper(),
            speaker=s.speaker,
            language=s.language,
            gender=s.gender,
            custom=s.custom,
            alignment=s.alignment,
        ))
    return MonoCut(
        id=new_id,
        start=c.start,
        duration=c.duration,
        channel=c.channel,
        recording=new_rec,
        supervisions=new_sups,
        custom=c.custom,
    )


def main():
    for mix in ["single", "overlap2", "overlap3"]:
        old_path = OLD_DIR / f"train.{mix}.cuts_clean.jsonl.gz"
        new_path = NEW_DIR / f"train.{mix}.cuts.jsonl.gz"
        bak_path = OLD_DIR / f"train.{mix}.cuts_clean.bak_pre_yt1000h.jsonl.gz"
        out_path = old_path  # in-place

        assert old_path.exists(), old_path
        assert new_path.exists(), new_path

        if not bak_path.exists():
            shutil.copy2(old_path, bak_path)
            print(f"[backup] {bak_path.name}")

        old_cs = CutSet.from_file(old_path)
        new_cs = CutSet.from_file(new_path)

        renamed_new = CutSet.from_cuts(rename_cut(c) for c in new_cs)
        merged = old_cs + renamed_new

        tmp = out_path.with_suffix(".tmp.jsonl.gz")
        merged.to_file(tmp)
        tmp.replace(out_path)

        n_old = len(old_cs)
        n_new = len(new_cs)
        print(f"[merge] {mix}: {n_old} + {n_new} = {n_old + n_new} → {out_path.name}")


if __name__ == "__main__":
    main()
