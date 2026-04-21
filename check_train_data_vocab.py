#!/usr/bin/env python3
"""
Run a one-pass data health check and vocabulary coverage scan for Zipformer training data.

Checks:
1) Vocabulary coverage on supervision texts (UNK usage from tokens.txt tokenizer)
2) One-pass train dataloader iteration (audio read/collate path) to catch runtime data errors
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch
from lhotse import load_manifest_lazy


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "zipformer"))

from asr_datamodule import LibriSpeechAsrDataModule  # noqa: E402
from tokenizer import TokenEncoder  # noqa: E402


def _iter_train_manifests(train_manifest_dir: Path, augment_types: List[str]) -> Iterable[Path]:
    for mix in ("single", "overlap2", "overlap3"):
        for aug in augment_types:
            p = train_manifest_dir / f"train.{mix}.cuts_{aug}.jsonl.gz"
            if p.exists():
                yield p


def scan_vocab_coverage(
    manifest_paths: Iterable[Path],
    encoder: TokenEncoder,
    max_unk_examples: int,
) -> Dict:
    unk_id = encoder.piece_to_id("<unk>")
    summary = {
        "files": {},
        "total": {
            "cuts": 0,
            "supervisions": 0,
            "nonempty_supervisions": 0,
            "empty_supervisions": 0,
            "total_tokens": 0,
            "unk_tokens": 0,
            "texts_with_unk": 0,
            "texts_with_only_unk_or_empty": 0,
        },
        "unk_examples": [],
    }

    text_cache: Dict[str, tuple[int, int, bool]] = {}
    progress_every_cuts = 20000

    for p in manifest_paths:
        file_stat = defaultdict(int)
        cuts = load_manifest_lazy(p)
        file_cuts = 0
        for cut in cuts:
            file_cuts += 1
            file_stat["cuts"] += 1
            summary["total"]["cuts"] += 1
            for sup in cut.supervisions:
                file_stat["supervisions"] += 1
                summary["total"]["supervisions"] += 1
                text = (getattr(sup, "text", "") or "").strip()
                if not text:
                    file_stat["empty_supervisions"] += 1
                    summary["total"]["empty_supervisions"] += 1
                    continue

                file_stat["nonempty_supervisions"] += 1
                summary["total"]["nonempty_supervisions"] += 1

                cached = text_cache.get(text)
                if cached is None:
                    ids = encoder.encode(text, out_type=int)
                    token_count = len(ids)
                    unk_count = sum(1 for t in ids if t == unk_id)
                    only_unk_or_empty = (token_count == 0 or unk_count == token_count)
                    text_cache[text] = (token_count, unk_count, only_unk_or_empty)
                else:
                    token_count, unk_count, only_unk_or_empty = cached

                file_stat["total_tokens"] += token_count
                summary["total"]["total_tokens"] += token_count
                file_stat["unk_tokens"] += unk_count
                summary["total"]["unk_tokens"] += unk_count

                if unk_count > 0:
                    file_stat["texts_with_unk"] += 1
                    summary["total"]["texts_with_unk"] += 1
                    if len(summary["unk_examples"]) < max_unk_examples:
                        summary["unk_examples"].append(
                            {
                                "file": p.name,
                                "cut_id": str(cut.id),
                                "speaker": getattr(sup, "speaker", None),
                                "unk_count": int(unk_count),
                                "token_count": int(token_count),
                                "text": text[:280],
                            }
                        )

                if only_unk_or_empty:
                    file_stat["texts_with_only_unk_or_empty"] += 1
                    summary["total"]["texts_with_only_unk_or_empty"] += 1

            if file_cuts % progress_every_cuts == 0:
                print(
                    f"[vocab] file={p.name} cuts={file_cuts} "
                    f"supervisions={file_stat['supervisions']} "
                    f"unk_tokens={file_stat['unk_tokens']}",
                    flush=True,
                )

        summary["files"][p.name] = dict(file_stat)

    total_tokens = summary["total"]["total_tokens"]
    unk_tokens = summary["total"]["unk_tokens"]
    summary["total"]["unk_token_rate"] = (
        float(unk_tokens) / float(total_tokens) if total_tokens > 0 else 0.0
    )
    summary["total"]["vocab_covered"] = bool(unk_tokens == 0)
    return summary


def run_dataloader_one_pass(
    manifest_dir: Path,
    train_manifest_dir: Path,
    num_speakers: int,
    ssa_single_target: bool,
    ssa_target_policy: str,
    augment_types: str,
    num_workers: int,
    num_buckets: int,
    max_duration: float,
    progress_every_batches: int,
    max_batches: int,
) -> Dict:
    args = SimpleNamespace(
        full_libri=True,
        manifest_dir=manifest_dir,
        train_manifest_dir=train_manifest_dir,
        max_duration=max_duration,
        bucketing_sampler=True,
        num_buckets=num_buckets,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        do_normalize=True,
        ssa_single_target=ssa_single_target,
        ssa_target_policy=ssa_target_policy,
        augment_types=augment_types,
        num_speakers=num_speakers,
    )

    dm = LibriSpeechAsrDataModule(args)
    train_cuts = dm.train_all_shuf_cuts()
    dl = dm.train_dataloaders(cuts_train=train_cuts, do_normalize=args.do_normalize)

    stats = {
        "ok": True,
        "batches": 0,
        "cuts": 0,
        "audio_samples_valid": 0,
        "audio_hours_valid": 0.0,
        "samples_all_empty_targets": 0,
        "elapsed_sec": 0.0,
        "error": None,
    }

    t0 = time.time()
    try:
        for batch_idx, batch in enumerate(dl):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            audio = batch["audio"]
            padding_mask = batch["padding_mask"]
            B = int(audio.shape[0])
            S = len(batch["y_texts"])

            valid_samples = int((~padding_mask).sum().item())
            stats["batches"] += 1
            stats["cuts"] += B
            stats["audio_samples_valid"] += valid_samples

            empty_targets = 0
            for b in range(B):
                if all((batch["y_texts"][s][b].strip() == "") for s in range(S)):
                    empty_targets += 1
            stats["samples_all_empty_targets"] += empty_targets

            if stats["batches"] % progress_every_batches == 0:
                hrs = stats["audio_samples_valid"] / 16000.0 / 3600.0
                print(
                    f"[dataloader] batches={stats['batches']} cuts={stats['cuts']} "
                    f"hours={hrs:.2f} empty_targets={stats['samples_all_empty_targets']}",
                    flush=True,
                )

    except Exception:
        stats["ok"] = False
        stats["error"] = traceback.format_exc()

    stats["elapsed_sec"] = time.time() - t0
    stats["audio_hours_valid"] = stats["audio_samples_valid"] / 16000.0 / 3600.0
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="One-pass data + vocab checker")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/lhotse_cutsets"),
        help="Base manifest dir (val/test and fallback).",
    )
    parser.add_argument(
        "--train-manifest-dir",
        type=Path,
        default=Path("data/lhotse_cutsets_new_train"),
        help="Train manifest dir.",
    )
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=Path("zipformer/tokens.txt"),
        help="Path to tokens.txt",
    )
    parser.add_argument(
        "--augment-types",
        type=str,
        default="clean",
        help="Comma-separated train augmentation types, e.g. clean,noise,rir",
    )
    parser.add_argument("--num-speakers", type=int, default=3)
    parser.add_argument("--ssa-single-target", type=int, default=1)
    parser.add_argument(
        "--ssa-target-policy",
        type=str,
        default="round_robin",
        choices=["random", "round_robin"],
    )
    parser.add_argument("--num-workers", type=int, default=128)
    parser.add_argument("--num-buckets", type=int, default=80)
    parser.add_argument("--max-duration", type=float, default=700.0)
    parser.add_argument("--progress-every-batches", type=int, default=200)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=-1,
        help="For quick test only. -1 means full one-pass.",
    )
    parser.add_argument("--max-unk-examples", type=int, default=30)
    parser.add_argument(
        "--skip-vocab",
        action="store_true",
        help="Skip vocabulary coverage scan.",
    )
    parser.add_argument(
        "--skip-dataloader",
        action="store_true",
        help="Skip one-pass dataloader scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/checks/train_data_vocab_check.json"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    augment_types = [s.strip() for s in args.augment_types.split(",") if s.strip()]
    if not augment_types:
        augment_types = ["clean"]

    vocab_summary = None
    if not args.skip_vocab:
        print("=== Vocab coverage scan ===", flush=True)
        manifest_paths = list(_iter_train_manifests(args.train_manifest_dir, augment_types))
        if not manifest_paths:
            raise FileNotFoundError(
                f"No train manifests found in {args.train_manifest_dir} for augment_types={augment_types}"
            )
        print("manifests:", [p.name for p in manifest_paths], flush=True)
        encoder = TokenEncoder(args.tokens_path)
        vocab_summary = scan_vocab_coverage(
            manifest_paths=manifest_paths,
            encoder=encoder,
            max_unk_examples=args.max_unk_examples,
        )

    data_summary = None
    if not args.skip_dataloader:
        print("=== One-pass dataloader scan ===", flush=True)
        data_summary = run_dataloader_one_pass(
            manifest_dir=args.manifest_dir,
            train_manifest_dir=args.train_manifest_dir,
            num_speakers=args.num_speakers,
            ssa_single_target=bool(args.ssa_single_target),
            ssa_target_policy=args.ssa_target_policy,
            augment_types=",".join(augment_types),
            num_workers=args.num_workers,
            num_buckets=args.num_buckets,
            max_duration=args.max_duration,
            progress_every_batches=args.progress_every_batches,
            max_batches=args.max_batches,
        )

    final = {
        "config": {
            "manifest_dir": str(args.manifest_dir),
            "train_manifest_dir": str(args.train_manifest_dir),
            "tokens_path": str(args.tokens_path),
            "augment_types": augment_types,
            "num_speakers": args.num_speakers,
            "ssa_single_target": bool(args.ssa_single_target),
            "ssa_target_policy": args.ssa_target_policy,
            "num_workers": args.num_workers,
            "num_buckets": args.num_buckets,
            "max_duration": args.max_duration,
            "max_batches": args.max_batches,
        },
        "vocab_summary": vocab_summary,
        "dataloader_summary": data_summary,
    }

    args.output.write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== Done ===", flush=True)
    print(f"output: {args.output}", flush=True)
    if vocab_summary is not None:
        print(
            f"vocab_covered={vocab_summary['total']['vocab_covered']} "
            f"unk_token_rate={vocab_summary['total']['unk_token_rate']:.8f}",
            flush=True,
        )
    if data_summary is not None:
        print(
            f"dataloader_ok={data_summary['ok']} batches={data_summary['batches']} "
            f"cuts={data_summary['cuts']} hours={data_summary['audio_hours_valid']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
