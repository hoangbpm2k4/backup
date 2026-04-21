#!/usr/bin/env python3
"""
Lightweight local logic test for voting.ipynb cells 9 -> 13.

This script does NOT touch running pipelines. It creates a separate test
workspace, samples a few VAD wav files, and runs:
  - Cell 9 style ASR smoke test (mock recognizers)
  - Cell 10 exact-match accept/reject
  - Cell 11 recognizer init (mock)
  - Cell 12 word-aligned voting salvage
  - Cell 13 combined manifest + summary stats
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 16000
TAIL_PAD_SEC = 0.0
MERGE_GAP_SEC = 0.0


def normalize_text(s: str) -> str:
    s = str(s or "").lower().strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_word_text(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_time(val, default=0.0):
    try:
        t = float(val)
    except Exception:
        return float(default)
    if not np.isfinite(t):
        return float(default)
    return t


def parse_seg_times(seg_path: str) -> tuple[float, float]:
    name = Path(seg_path).name
    m = re.search(r"__seg_\d+_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)\.wav$", name)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def extract_zip_words_with_time(zip_tokens, zip_timestamps):
    words = []
    n = min(len(zip_tokens), len(zip_timestamps))
    if n == 0:
        return words

    cleaned = []
    for i in range(n):
        tok = str(zip_tokens[i] or "").strip()
        if not tok:
            continue
        if tok.startswith("<") and tok.endswith(">"):
            continue
        cleaned.append((i, tok))

    if not cleaned:
        return words

    def add_word(raw_word, start_t, end_t):
        norm_word = normalize_word_text(raw_word)
        if not norm_word:
            return
        end_t = max(float(end_t), float(start_t) + 1e-3)
        words.append(
            {
                "raw": str(raw_word),
                "norm": norm_word,
                "start": float(start_t),
                "end": float(end_t),
            }
        )

    has_spm_boundary = any(tok.startswith("\u2581") for _, tok in cleaned)

    if not has_spm_boundary:
        for k, (i, tok) in enumerate(cleaned):
            t_start = safe_time(zip_timestamps[i], 0.0)
            if k + 1 < len(cleaned):
                next_i = cleaned[k + 1][0]
                t_end = safe_time(zip_timestamps[next_i], t_start + 1e-3)
            elif i + 1 < n:
                t_end = safe_time(zip_timestamps[i + 1], t_start + 0.05)
            else:
                t_end = t_start + 0.05
            add_word(tok, t_start, t_end)
        return words

    cur_pieces = []
    cur_start = None
    cur_end = None

    def flush_word():
        nonlocal cur_pieces, cur_start, cur_end
        if not cur_pieces or cur_start is None or cur_end is None:
            cur_pieces = []
            cur_start = None
            cur_end = None
            return
        raw_word = "".join(cur_pieces).strip()
        add_word(raw_word, cur_start, cur_end)
        cur_pieces = []
        cur_start = None
        cur_end = None

    for i, tok in cleaned:
        starts_new_word = tok.startswith("\u2581")
        piece = tok.lstrip("\u2581")
        if not piece:
            continue

        t_start = safe_time(zip_timestamps[i], cur_end if cur_end is not None else 0.0)
        if i + 1 < n:
            t_end = safe_time(zip_timestamps[i + 1], t_start)
        else:
            t_end = t_start + 0.05

        if starts_new_word and cur_pieces:
            flush_word()

        if cur_start is None:
            cur_start = t_start
        cur_end = max(t_end, t_start + 1e-3)
        cur_pieces.append(piece)

    flush_word()
    return words


def find_consensus_runs_word_alignment(zip_tokens, zip_timestamps, gip_text, tail_pad_sec: float = TAIL_PAD_SEC):
    zip_words = extract_zip_words_with_time(zip_tokens, zip_timestamps)
    gip_words = [w for w in normalize_word_text(gip_text).split() if w]

    if not zip_words or not gip_words:
        return [], 0, 0, 0, [], zip_words, gip_words

    z_norm = [w["norm"] for w in zip_words]
    g_norm = gip_words

    sm = difflib.SequenceMatcher(a=z_norm, b=g_norm, autojunk=False)
    opcodes = sm.get_opcodes()

    runs = []
    events = []
    matched_words = 0
    mismatch_words = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            span = i2 - i1
            matched_words += span
            if span <= 0:
                continue

            start_time = float(zip_words[i1]["start"])
            end_time = float(zip_words[i2 - 1]["end"]) + tail_pad_sec
            end_time = max(end_time, start_time + 1e-3)

            text = " ".join(w["raw"] for w in zip_words[i1:i2]).strip()
            if text:
                runs.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": round(end_time - start_time, 3),
                        "text": text,
                        "start_word_idx": int(i1),
                        "end_word_idx": int(i2 - 1),
                    }
                )

            events.append(
                {
                    "type": "equal",
                    "zip_word_span": [int(i1), int(i2)],
                    "gip_word_span": [int(j1), int(j2)],
                    "text": text,
                }
            )
        else:
            mismatch_words += max(i2 - i1, j2 - j1)
            events.append(
                {
                    "type": "break",
                    "tag": tag,
                    "zip_word_span": [int(i1), int(i2)],
                    "gip_word_span": [int(j1), int(j2)],
                    "zip_words": " ".join(z_norm[i1:i2]),
                    "gip_words": " ".join(g_norm[j1:j2]),
                }
            )

    aligned_words = matched_words + mismatch_words
    return runs, matched_words, mismatch_words, aligned_words, events, zip_words, gip_words


def merge_runs(audio: np.ndarray, sample_rate: int, runs, gap_sec: float = MERGE_GAP_SEC):
    packed = []
    for run in runs:
        s = max(0, int(run["start_time"] * sample_rate))
        e = min(len(audio), int(run["end_time"] * sample_rate))
        if e <= s:
            continue
        clip = audio[s:e]
        if clip.size:
            packed.append((clip.astype(np.float32, copy=False), run["text"], run))

    if not packed:
        return None

    gap = np.zeros(int(gap_sec * sample_rate), dtype=np.float32) if gap_sec > 0 else None
    merged_parts = []
    merged_text_parts = []
    used_runs = []

    for i, (clip, text, run) in enumerate(packed):
        merged_parts.append(clip)
        merged_text_parts.append(text)
        used_runs.append(run)
        if gap is not None and i < len(packed) - 1:
            merged_parts.append(gap)

    merged_audio = np.concatenate(merged_parts)
    merged_text = " ".join(t for t in merged_text_parts if str(t).strip()).strip()
    return merged_audio, merged_text, used_runs


def build_tokens_timestamps(zip_text: str, duration_sec: float):
    words = [w for w in normalize_word_text(zip_text).split() if w]
    if not words:
        return [], []
    n = len(words)
    step = max(duration_sec / max(n, 1), 0.05)
    timestamps = [round(i * step, 3) for i in range(n)]
    return words, timestamps


@dataclass
class MockAsrResult:
    zip_text: str
    gip_text: str
    zip_tokens: list[str]
    zip_timestamps: list[float]


def build_mock_result(idx: int, duration_sec: float) -> MockAsrResult:
    patterns = [
        ("benh nhan on dinh sau dieu tri", "benh nhan on dinh sau dieu tri"),
        (
            "hom nay benh nhan dau dau nhe nhung huyet ap on dinh",
            "hom nay benh nhan dau dau nang nhung huyet ap on dinh",
        ),
        ("", "benh nhan co tien trien tot"),
        ("", ""),
        (
            "khong ghi nhan trieu chung bat thuong",
            "khong ghi nhan them trieu chung bat thuong",
        ),
        ("de nghi tai kham sau mot tuan", "xin cam on quy vi da theo doi"),
    ]
    zip_text, gip_text = patterns[idx % len(patterns)]
    zip_tokens, zip_timestamps = build_tokens_timestamps(zip_text, max(float(duration_sec), 0.5))
    return MockAsrResult(
        zip_text=zip_text,
        gip_text=gip_text,
        zip_tokens=zip_tokens,
        zip_timestamps=zip_timestamps,
    )


def safe_sum_hours(df: pd.DataFrame | None, col="duration_sec") -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum() / 3600.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vad-manifest",
        default="/home/vietnam/voice_to_text/data/data_label_auto_yt_run_full/manifests/vad_segments.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/vietnam/voice_to_text/data/voting_logic_test_from_vad",
    )
    parser.add_argument("--sample-size", type=int, default=12)
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    accept_dir = base_dir / "accepted"
    reject_dir = base_dir / "rejected"
    manifest_dir = base_dir / "manifests"
    accept_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("=== Prepare test sample from VAD ===")
    vad_df = pd.read_csv(args.vad_manifest)
    if "segment_path" not in vad_df.columns:
        raise ValueError("vad_segments.csv missing segment_path")
    sample_df = vad_df.copy()
    sample_df = sample_df[sample_df["segment_path"].astype(str).map(os.path.exists)].head(args.sample_size).copy()
    if sample_df.empty:
        raise RuntimeError("No valid VAD wav found for sample test")

    sample_df["denoised_path"] = sample_df["segment_path"].astype(str)
    if "src" not in sample_df.columns:
        sample_df["src"] = sample_df["denoised_path"].map(lambda x: Path(x).stem)
    if "duration_sec" not in sample_df.columns:
        sample_df["duration_sec"] = np.nan

    starts = []
    ends = []
    for p in sample_df["denoised_path"].astype(str).tolist():
        s, e = parse_seg_times(p)
        starts.append(s)
        ends.append(e)
    sample_df["start_sec"] = starts
    sample_df["end_sec"] = ends

    denoise_manifest = manifest_dir / "denoised_segments.csv"
    sample_df.to_csv(denoise_manifest, index=False)
    print(f"Sample rows: {len(sample_df)}")
    print(f"Saved: {denoise_manifest}")

    mock_map: dict[str, MockAsrResult] = {}
    for i, row in enumerate(sample_df.to_dict("records")):
        wav_path = str(row["denoised_path"])
        mock_map[wav_path] = build_mock_result(i, float(row.get("duration_sec", 0.0) or 0.0))

    # --------------------------
    # CELL 9: ASR function test
    # --------------------------
    print("\n=== CELL 9 (test ASR functions - mock) ===")
    sample_audio = str(sample_df.iloc[0]["denoised_path"])
    print("sample:", sample_audio)
    print("zipformer (mock):", mock_map[sample_audio].zip_text)
    print("gipformer (mock):", mock_map[sample_audio].gip_text)

    # --------------------------
    # CELL 10: all_results + accepted_manifest
    # --------------------------
    print("\n=== CELL 10 (create all_results.csv, accepted_manifest.csv) ===")
    all_results_csv = manifest_dir / "all_results.csv"
    accepted_csv = manifest_dir / "accepted_manifest.csv"
    accepted_jsonl = manifest_dir / "accepted_manifest.jsonl"

    rows_out = []
    run_ok = 0
    run_fail = 0
    for row in sample_df.to_dict("records"):
        wav_path = str(row["denoised_path"])
        m = mock_map[wav_path]
        zip_norm = normalize_text(m.zip_text)
        gip_norm = normalize_text(m.gip_text)
        accept = bool(zip_norm) and (zip_norm == gip_norm)

        if not zip_norm and not gip_norm:
            reason = "both_empty"
        elif not zip_norm:
            reason = "zip_empty"
        elif not gip_norm:
            reason = "gip_empty"
        elif accept:
            reason = "exact"
        else:
            reason = "mismatch"

        out = dict(row)
        out.update(
            {
                "zipformer_text": m.zip_text,
                "gipformer_text": m.gip_text,
                "zipformer_norm": zip_norm,
                "gipformer_norm": gip_norm,
                "decision_reason": reason,
                "accepted": bool(accept),
                "final_wav": wav_path if accept else "",
                "final_text": zip_norm if accept else "",
            }
        )
        rows_out.append(out)
        if accept:
            run_ok += 1
        else:
            run_fail += 1

    all_df = pd.DataFrame(rows_out)
    all_df.to_csv(all_results_csv, index=False)

    keep_cols = ["final_wav", "final_text", "duration_sec", "src", "start_sec", "end_sec"]
    accepted_df = all_df[all_df["accepted"] == True].copy()
    for c in keep_cols:
        if c not in accepted_df.columns:
            accepted_df[c] = np.nan
    accepted_df = accepted_df[keep_cols].copy()
    accepted_df.to_csv(accepted_csv, index=False)
    with open(accepted_jsonl, "w", encoding="utf-8") as f:
        for r in accepted_df.to_dict("records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Saved:", all_results_csv)
    print("Saved:", accepted_csv)
    print("Saved:", accepted_jsonl)
    print("Current run accepted:", run_ok)
    print("Current run rejected:", run_fail)
    print("Reason counts:")
    print(all_df["decision_reason"].value_counts(dropna=False).to_string())

    # --------------------------
    # CELL 11: init recognizers
    # --------------------------
    print("\n=== CELL 11 (init recognizers for voting - mock) ===")
    print("ONNX providers: ['CPUExecutionProvider'] (mock)")
    print("Zipformer provider: cpu (mock)")
    print("Initializing Zipformer... done (mock)")
    print("Initializing Gipformer... done (mock)")
    print("Recognizers are ready.")

    # --------------------------
    # CELL 12: voting salvage
    # --------------------------
    print("\n=== CELL 12 (voting salvage) ===")
    rejected_df = all_df[all_df["accepted"] != True].copy()
    print(f"Start word-aligned break-and-continue on {len(rejected_df)} selected file(s)...")

    all_salvaged_rows = []
    all_run_rows = []
    file_debug_rows = []

    for row in rejected_df.to_dict("records"):
        wav_path = str(row.get("denoised_path", ""))
        if not wav_path or not os.path.exists(wav_path):
            continue

        m = mock_map[wav_path]
        zip_tokens = m.zip_tokens
        zip_timestamps = m.zip_timestamps
        gip_text = m.gip_text

        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        if audio.size == 0:
            continue

        runs, matched_words, mismatch_words, aligned_words, events, zip_words, gip_words = find_consensus_runs_word_alignment(
            zip_tokens,
            zip_timestamps,
            gip_text,
            tail_pad_sec=TAIL_PAD_SEC,
        )

        consensus_text = " ".join(r["text"] for r in runs if r.get("text"))
        file_debug_rows.append(
            {
                "wav": wav_path,
                "zip_words": len(zip_words),
                "gip_words": len(gip_words),
                "aligned_words": aligned_words,
                "matched_words": matched_words,
                "mismatch_words": mismatch_words,
                "num_runs": len(runs),
                "consensus_text": consensus_text,
                "events_json": json.dumps(events, ensure_ascii=False),
            }
        )

        if not runs:
            continue

        merged = merge_runs(audio, SAMPLE_RATE, runs, gap_sec=MERGE_GAP_SEC)
        if not merged:
            continue

        merged_audio, merged_text, used_runs = merged
        merged_duration = len(merged_audio) / SAMPLE_RATE
        if not merged_text or merged_duration <= 0:
            continue

        out_name = f"salvaged_vote_{Path(wav_path).name}"
        out_path = accept_dir / out_name
        sf.write(str(out_path), merged_audio, SAMPLE_RATE)

        new_row = dict(row)
        new_row["final_wav"] = str(out_path)
        new_row["final_text"] = merged_text
        new_row["duration_sec"] = round(merged_duration, 3)
        new_row["accepted"] = True
        new_row["is_salvaged_vote"] = True
        new_row["num_merged_runs"] = len(used_runs)
        new_row["matched_words"] = matched_words
        new_row["mismatch_words"] = mismatch_words
        new_row["merged_runs"] = json.dumps(used_runs, ensure_ascii=False)
        all_salvaged_rows.append(new_row)

        for run_idx, run in enumerate(used_runs):
            all_run_rows.append(
                {
                    "src_wav": wav_path,
                    "out_wav": str(out_path),
                    "run_idx": run_idx,
                    "start_word_idx": run["start_word_idx"],
                    "end_word_idx": run["end_word_idx"],
                    "start_time": run["start_time"],
                    "end_time": run["end_time"],
                    "duration": run["duration"],
                    "text": run["text"],
                }
            )

    salvaged_path = manifest_dir / "salvaged_vote_manifest.csv"
    salvaged_runs_path = manifest_dir / "salvaged_vote_runs_manifest.csv"
    salvaged_debug_path = manifest_dir / "salvaged_vote_debug.csv"

    pd.DataFrame(file_debug_rows).to_csv(salvaged_debug_path, index=False)
    print("Saved debug:", salvaged_debug_path)

    if all_salvaged_rows:
        sal_df = pd.DataFrame(all_salvaged_rows)
        sal_df.to_csv(salvaged_path, index=False)
        pd.DataFrame(all_run_rows).to_csv(salvaged_runs_path, index=False)
        print(f"Salvaged {len(sal_df)} files with word-aligned break-and-continue voting.")
        print("Saved manifest:", salvaged_path)
        print("Saved run details:", salvaged_runs_path)
    else:
        sal_df = pd.DataFrame()
        print("No file produced merged output.")

    # --------------------------
    # CELL 13: combine + stats
    # --------------------------
    print("\n=== CELL 13 (combine final accepted_manifest_combined.csv + stats) ===")
    accepted_combined_path = manifest_dir / "accepted_manifest_combined.csv"
    keep_cols = ["final_wav", "final_text", "duration_sec", "src", "start_sec", "end_sec"]

    accepted_pre_df = all_df[all_df["accepted"] == True].copy()
    for c in keep_cols:
        if c not in accepted_pre_df.columns:
            accepted_pre_df[c] = np.nan
    accepted_pre_df = accepted_pre_df[keep_cols].copy()

    for c in keep_cols:
        if c not in sal_df.columns:
            sal_df[c] = np.nan
    sal_df = sal_df[keep_cols].copy() if not sal_df.empty else pd.DataFrame(columns=keep_cols)

    combined = pd.concat([accepted_pre_df, sal_df], ignore_index=True)
    if not combined.empty and "final_wav" in combined.columns:
        combined = combined.drop_duplicates(subset=["final_wav"], keep="last")
    combined.to_csv(accepted_combined_path, index=False)

    raw_hours = 0.0
    for wav in sample_df["denoised_path"].astype(str).tolist():
        try:
            raw_hours += float(sf.info(wav).duration) / 3600.0
        except Exception:
            pass

    print("=== Pipeline Hours Breakdown ===")
    print("Raw input hours        :", round(raw_hours, 3))
    print("VAD segment hours      :", round(safe_sum_hours(sample_df), 3))
    print("Denoised segment hours :", round(safe_sum_hours(sample_df), 3))
    print("Accepted (pre-salvage) :", round(safe_sum_hours(accepted_pre_df), 3))
    print("Salvaged voting hours  :", round(safe_sum_hours(sal_df), 3))
    print("Accepted (combined)    :", round(safe_sum_hours(combined), 3))

    print("\n=== Acceptance Stats ===")
    print("Total segments :", len(all_df))
    print("Accepted       :", len(combined))
    print("Accept rate    :", round(100 * len(combined) / max(1, len(all_df)), 2), "%")

    print("\nSaved manifests:")
    print("-", accepted_csv)
    print("-", accepted_combined_path)

    if len(combined) and "duration_sec" in combined.columns:
        print("Total accepted hours:", round(float(combined["duration_sec"].sum()) / 3600.0, 3))

    print("\nDone.")
    print("Output base:", base_dir)


if __name__ == "__main__":
    main()
