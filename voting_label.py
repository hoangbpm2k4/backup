#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import sherpa_onnx
import soundfile as sf
from tqdm.auto import tqdm


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


def as_bool_series(val, length: int, index):
    if isinstance(val, pd.Series):
        s = val.copy()
    else:
        s = pd.Series([val] * length, index=index)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def parse_json_list(val):
    if isinstance(val, list):
        return val
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def load_audio_16k(wav_path: str, sr_out: int) -> np.ndarray:
    wav, sr = sf.read(wav_path, dtype="float32")
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != sr_out:
        x = librosa.resample(x, orig_sr=sr, target_sr=sr_out).astype(np.float32)
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.clip(x, -1.0, 1.0)


def ensure_onnxruntime(provider: str, auto_install: bool):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        if not auto_install:
            raise RuntimeError(
                "Missing onnxruntime. Install:\n"
                "  pip install -U onnxruntime-gpu   (CUDA)\n"
                "or\n"
                "  pip install -U onnxruntime       (CPU)"
            )
        pkg = "onnxruntime-gpu" if provider in ("auto", "cuda") else "onnxruntime"
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", pkg], check=False)
        import onnxruntime as ort
    providers = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"Requested cuda but available providers={providers}")
    if provider == "auto":
        provider = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
    return provider, providers


def get_gpu_info():
    if not shutil.which("nvidia-smi"):
        return "", 0.0
    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0].strip()
        mem = float(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0].strip()
        ) / 1024.0
        return name, mem
    except Exception:
        return "", 0.0


def resolve_zip_tokens_path(zip_dir: Path) -> Path:
    # Some repos store token table in `config.json` instead of `tokens.txt`.
    tok = zip_dir / "tokens.txt"
    if tok.exists():
        return tok
    cfg = zip_dir / "config.json"
    if cfg.exists():
        return cfg
    raise FileNotFoundError(f"Missing tokens file in {zip_dir} (need tokens.txt or config.json)")


def init_zip(zip_dir: Path, provider: str, sr: int, num_threads: int):
    tokens_path = resolve_zip_tokens_path(zip_dir)
    bpe_model = zip_dir / "bpe.model"
    extra = {}
    if bpe_model.exists():
        extra["modeling_unit"] = "bpe"
        extra["bpe_vocab"] = str(bpe_model)
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(zip_dir / "encoder-epoch-20-avg-10.onnx"),
        decoder=str(zip_dir / "decoder-epoch-20-avg-10.onnx"),
        joiner=str(zip_dir / "joiner-epoch-20-avg-10.onnx"),
        tokens=str(tokens_path),
        num_threads=num_threads,
        sample_rate=sr,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=provider,
        **extra,
    )


def init_gip(repo: Path, provider: str, num_threads: int, hf_cache_dir: Path):
    repo_s = str(repo.resolve())
    if repo_s not in sys.path:
        sys.path.append(repo_s)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    # Force writable local HF cache to avoid permission issues from mounted disks.
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
    from infer_onnx import create_recognizer, download_model

    model_paths = download_model(quantize="fp32")
    try:
        rec = create_recognizer(model_paths, num_threads=num_threads, provider=provider)
        return rec
    except TypeError:
        return create_recognizer(model_paths, num_threads=num_threads)


def decode_texts_batch(recognizer, audios, sr: int, bs: int):
    if recognizer is None:
        return [""] * len(audios), bs
    texts = [""] * len(audios)
    use_batch = hasattr(recognizer, "decode_streams")
    i = 0
    while i < len(audios):
        j = min(i + max(1, int(bs)), len(audios))
        part = audios[i:j]
        streams = []
        for a in part:
            st = recognizer.create_stream()
            st.accept_waveform(sr, a.astype(np.float32, copy=False))
            streams.append(st)
        try:
            if use_batch:
                recognizer.decode_streams(streams)
            else:
                for st in streams:
                    recognizer.decode_stream(st)
            for k, st in enumerate(streams):
                texts[i + k] = str(getattr(st.result, "text", "") or "").strip()
            i = j
        except Exception as e:
            msg = str(e).lower()
            if bs > 1 and any(x in msg for x in ["cuda", "memory", "alloc", "cublas", "out of"]):
                bs = max(1, bs // 2)
                print(f"[decode] reduce batch -> {bs} ({e})")
            elif bs > 1:
                bs = 1
                print(f"[decode] fallback batch=1 ({e})")
            else:
                texts[i] = ""
                i += 1
    return texts, bs


def decode_zip_meta_batch(recognizer, audios, sr: int, bs: int):
    if recognizer is None:
        return [{"text": "", "tokens": [], "timestamps": []} for _ in audios], bs
    results = [{"text": "", "tokens": [], "timestamps": []} for _ in audios]
    use_batch = hasattr(recognizer, "decode_streams")
    i = 0
    while i < len(audios):
        j = min(i + max(1, int(bs)), len(audios))
        part = audios[i:j]
        streams = []
        for a in part:
            st = recognizer.create_stream()
            st.accept_waveform(sr, a.astype(np.float32, copy=False))
            streams.append(st)
        try:
            if use_batch:
                recognizer.decode_streams(streams)
            else:
                for st in streams:
                    recognizer.decode_stream(st)
            for k, st in enumerate(streams):
                r = st.result
                results[i + k] = {
                    "text": str(getattr(r, "text", "") or "").strip(),
                    "tokens": list(getattr(r, "tokens", []) or []),
                    "timestamps": list(getattr(r, "timestamps", []) or []),
                }
            i = j
        except Exception as e:
            msg = str(e).lower()
            if bs > 1 and any(x in msg for x in ["cuda", "memory", "alloc", "cublas", "out of"]):
                bs = max(1, bs // 2)
                print(f"[decode-zip] reduce batch -> {bs} ({e})")
            elif bs > 1:
                bs = 1
                print(f"[decode-zip] fallback batch=1 ({e})")
            else:
                i += 1
    return results, bs


def run_recognizer(recognizer, audio: np.ndarray, sr: int):
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, audio.astype(np.float32, copy=False))
    recognizer.decode_stream(stream)
    result = stream.result
    return {
        "text": str(getattr(result, "text", "") or "").strip(),
        "tokens": list(getattr(result, "tokens", []) or []),
        "timestamps": list(getattr(result, "timestamps", []) or []),
    }


def extract_zip_words_with_time(tokens, timestamps):
    n = min(len(tokens), len(timestamps))
    cleaned = []
    for i in range(n):
        tok = str(tokens[i] or "").strip()
        if tok and not (tok.startswith("<") and tok.endswith(">")):
            cleaned.append((i, tok))
    words = []
    if not cleaned:
        return words
    has_sp = any(tok.startswith("▁") for _, tok in cleaned)

    def add_word(raw, s, e):
        norm = normalize_word_text(raw)
        if norm:
            words.append({"raw": raw, "norm": norm, "start": float(s), "end": max(float(e), float(s) + 1e-3)})

    if not has_sp:
        for k, (i, tok) in enumerate(cleaned):
            s = float(timestamps[i])
            if k + 1 < len(cleaned):
                e = float(timestamps[cleaned[k + 1][0]])
            elif i + 1 < n:
                e = float(timestamps[i + 1])
            else:
                e = s + 0.05
            add_word(tok, s, e)
        return words

    cur, cur_s, cur_e = [], None, None

    def flush():
        nonlocal cur, cur_s, cur_e
        if cur and cur_s is not None and cur_e is not None:
            add_word("".join(cur).strip(), cur_s, cur_e)
        cur, cur_s, cur_e = [], None, None

    for i, tok in cleaned:
        new_word = tok.startswith("▁")
        piece = tok.lstrip("▁")
        if not piece:
            continue
        s = float(timestamps[i])
        e = float(timestamps[i + 1]) if i + 1 < n else s + 0.05
        if new_word and cur:
            flush()
        if cur_s is None:
            cur_s = s
        cur_e = max(e, s + 1e-3)
        cur.append(piece)
    flush()
    return words


def find_consensus_runs(
    zip_tokens,
    zip_ts,
    gip_text: str,
    tail_pad_sec: float,
    trim_start_sec: float = 0.0,
    trim_end_sec: float = 0.0,
    collect_events: bool = False,
):
    zip_words = extract_zip_words_with_time(zip_tokens, zip_ts)
    gip_words = [w for w in normalize_word_text(gip_text).split() if w]
    if not zip_words or not gip_words:
        return [], 0, 0, 0, []
    z = [w["norm"] for w in zip_words]
    g = gip_words
    sm = difflib.SequenceMatcher(a=z, b=g, autojunk=False)
    runs, events, matched, mismatch = [], [], 0, 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            span = i2 - i1
            matched += span
            if span > 0:
                raw_s = float(zip_words[i1]["start"])
                raw_e = max(float(zip_words[i2 - 1]["end"]) + tail_pad_sec, raw_s + 1e-3)
                s = raw_s + max(0.0, float(trim_start_sec))
                e = raw_e - max(0.0, float(trim_end_sec))
                if e <= s + 1e-3:
                    # Fallback to raw span if trim is too aggressive for very short runs.
                    s, e = raw_s, raw_e
                t = " ".join(w["raw"] for w in zip_words[i1:i2]).strip()
                if t:
                    runs.append({"start_time": s, "end_time": e, "duration": round(e - s, 3), "text": t, "start_word_idx": i1, "end_word_idx": i2 - 1})
            if collect_events:
                events.append({"type": "equal", "zip_word_span": [i1, i2], "gip_word_span": [j1, j2]})
        else:
            mismatch += max(i2 - i1, j2 - j1)
            if collect_events:
                events.append({"type": "break", "tag": tag, "zip_word_span": [i1, i2], "gip_word_span": [j1, j2]})
    return runs, matched, mismatch, matched + mismatch, events


def merge_runs(audio: np.ndarray, sr: int, runs, gap_sec: float):
    segs = []
    for r in runs:
        s = max(0, int(r["start_time"] * sr))
        e = min(len(audio), int(r["end_time"] * sr))
        if e > s:
            segs.append((audio[s:e], r["text"], r))
    if not segs:
        return None
    gap = np.zeros(int(gap_sec * sr), dtype=np.float32) if gap_sec > 0 else None
    parts, texts, used = [], [], []
    for i, (clip, txt, r) in enumerate(segs):
        parts.append(clip.astype(np.float32, copy=False))
        texts.append(txt)
        used.append(r)
        if gap is not None and i < len(segs) - 1:
            parts.append(gap)
    return np.concatenate(parts), " ".join([t for t in texts if t.strip()]).strip(), used


def run_exact(args, p, zip_rec, gip_rec):
    df = pd.read_csv(p["denoise"])
    if "denoised_path" not in df.columns:
        raise ValueError("denoised_segments.csv missing denoised_path")
    for c in ["src", "duration_sec", "start_sec", "end_sec"]:
        if c not in df.columns:
            df[c] = np.nan

    if args.force_reprocess and p["all"].exists():
        p["all"].unlink()
    done = set()
    if args.resume and p["all"].exists() and not args.force_reprocess:
        try:
            done = set(pd.read_csv(p["all"], usecols=["denoised_path"])["denoised_path"].astype(str))
        except Exception:
            done = set()
    rows = [r for r in df.to_dict("records") if str(r["denoised_path"]) not in done]
    if args.max_items:
        rows = rows[: args.max_items]
    print(f"[exact] total={len(df)} done={len(done)} pending={len(rows)}")

    ok = fail = processed = 0
    buf = []
    zip_bs = gip_bs = int(args.batch_size)
    t0 = time.time()
    step = int(args.batch_size)
    pbar = tqdm(total=len(rows), desc="Exact Voting")
    for st in range(0, len(rows), step):
        if args.max_run_minutes and (time.time() - t0) / 60.0 >= args.max_run_minutes:
            print("[exact] stop by max_run_minutes for resume")
            break
        chunk = rows[st : st + step]
        audios, map_idx = [], []
        if int(args.load_workers) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.load_workers)) as ex:
                futs = [ex.submit(load_audio_16k, str(r["denoised_path"]), args.sample_rate) for r in chunk]
                for i, (r, fut) in enumerate(zip(chunk, futs)):
                    try:
                        audios.append(fut.result())
                        map_idx.append(i)
                    except Exception as e:
                        x = dict(r)
                        x.update({"zipformer_text": "", "gipformer_text": "", "zipformer_norm": "", "gipformer_norm": "", "decision_reason": f"audio_read_error:{e}", "accepted": False, "final_wav": "", "final_text": ""})
                        buf.append(x)
                        fail += 1
                        processed += 1
                        pbar.update(1)
        else:
            for i, r in enumerate(chunk):
                try:
                    audios.append(load_audio_16k(str(r["denoised_path"]), args.sample_rate))
                    map_idx.append(i)
                except Exception as e:
                    x = dict(r)
                    x.update({"zipformer_text": "", "gipformer_text": "", "zipformer_norm": "", "gipformer_norm": "", "decision_reason": f"audio_read_error:{e}", "accepted": False, "final_wav": "", "final_text": ""})
                    buf.append(x)
                    fail += 1
                    processed += 1
                    pbar.update(1)
        if map_idx:
            if args.parallel_dual_decode:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fut_zip = ex.submit(decode_zip_meta_batch, zip_rec, audios, args.sample_rate, zip_bs)
                    fut_gip = ex.submit(decode_texts_batch, gip_rec, audios, args.sample_rate, gip_bs)
                    z_meta, zip_bs = fut_zip.result()
                    gt, gip_bs = fut_gip.result()
            else:
                z_meta, zip_bs = decode_zip_meta_batch(zip_rec, audios, args.sample_rate, zip_bs)
                gt, gip_bs = decode_texts_batch(gip_rec, audios, args.sample_rate, gip_bs)
            for li, ri in enumerate(map_idx):
                r = dict(chunk[ri])
                z_item = z_meta[li] if li < len(z_meta) else {"text": "", "tokens": [], "timestamps": []}
                z = str(z_item.get("text", "") or "")
                z_tokens = list(z_item.get("tokens", []) or [])
                z_timestamps = list(z_item.get("timestamps", []) or [])
                g = str(gt[li] or "")
                zn = normalize_text(z)
                gn = normalize_text(g)
                acc = bool(zn) and zn == gn
                if not zn and not gn:
                    reason = "both_empty"
                elif not zn:
                    reason = "zip_empty"
                elif not gn:
                    reason = "gip_empty"
                elif acc:
                    reason = "exact"
                else:
                    reason = "mismatch"
                final_wav = str(r["denoised_path"]) if acc else ""
                r.update(
                    {
                        "zipformer_text": z,
                        "zipformer_tokens_json": json.dumps(z_tokens, ensure_ascii=False),
                        "zipformer_timestamps_json": json.dumps(z_timestamps, ensure_ascii=False),
                        "gipformer_text": g,
                        "zipformer_norm": zn,
                        "gipformer_norm": gn,
                        "decision_reason": reason,
                        "accepted": bool(acc),
                        "final_wav": final_wav,
                        "final_text": zn if acc else "",
                    }
                )
                buf.append(r)
                ok += int(acc)
                fail += int(not acc)
                processed += 1
                pbar.update(1)
                if args.save_every > 0 and processed % args.save_every == 0:
                    pd.DataFrame(buf).to_csv(p["all"], mode="a", header=(not p["all"].exists() or p["all"].stat().st_size == 0), index=False)
                    buf = []
                    print(f"[exact] checkpoint processed={processed} zip_bs={zip_bs} gip_bs={gip_bs}")
    pbar.close()
    if buf:
        pd.DataFrame(buf).to_csv(p["all"], mode="a", header=(not p["all"].exists() or p["all"].stat().st_size == 0), index=False)

    all_df = pd.read_csv(p["all"]) if p["all"].exists() else pd.DataFrame()
    if not all_df.empty and "denoised_path" in all_df.columns:
        all_df = all_df.drop_duplicates(subset=["denoised_path"], keep="last")
        all_df.to_csv(p["all"], index=False)
    acc_df = all_df[as_bool_series(all_df.get("accepted", False), len(all_df), all_df.index)].copy() if not all_df.empty else pd.DataFrame()
    keep = ["final_wav", "final_text", "duration_sec", "src", "start_sec", "end_sec"]
    for c in keep:
        if c not in acc_df.columns:
            acc_df[c] = np.nan
    acc_df = acc_df[keep] if not acc_df.empty else pd.DataFrame(columns=keep)
    acc_df.to_csv(p["accepted"], index=False)
    with open(p["accepted_jsonl"], "w", encoding="utf-8") as f:
        for r in acc_df.to_dict("records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[exact] ok={ok} fail={fail} all_rows={len(all_df)} accepted={len(acc_df)}")


def run_salvage(args, p, zip_rec, gip_rec):
    if not p["all"].exists():
        raise FileNotFoundError(p["all"])
    df = pd.read_csv(p["all"])
    rej = df[~as_bool_series(df.get("accepted", False), len(df), df.index)].copy()
    if args.debug_target_file:
        rej = rej[rej["denoised_path"].astype(str).str.contains(args.debug_target_file, regex=False, na=False)].copy()
        if args.debug_first and len(rej) > 1:
            rej = rej.head(1)
    print(f"[salvage] selected_rejected={len(rej)}")
    rows, run_rows, dbg = [], [], []
    cached_used = 0
    model_infer_used = 0
    skipped_no_cache = 0
    collect_debug = bool(args.save_salvage_debug or args.debug_verbose)
    zip_bs = int(args.batch_size)
    gip_bs = int(args.batch_size)
    step = max(1, int(args.salvage_batch_size))
    processed = 0
    rej_rows = rej.to_dict("records")
    pbar = tqdm(total=len(rej_rows), desc="Salvage Voting")

    def flush_buffers():
        nonlocal rows, run_rows, dbg
        if collect_debug and dbg and not args.debug_no_write:
            pd.DataFrame(dbg).to_csv(
                p["salvage_debug"],
                mode="a",
                header=(not p["salvage_debug"].exists() or p["salvage_debug"].stat().st_size == 0),
                index=False,
            )
        if rows and not args.debug_no_write:
            pd.DataFrame(rows).to_csv(
                p["salvage"],
                mode="a",
                header=(not p["salvage"].exists() or p["salvage"].stat().st_size == 0),
                index=False,
            )
        if run_rows and not args.debug_no_write:
            pd.DataFrame(run_rows).to_csv(
                p["salvage_runs"],
                mode="a",
                header=(not p["salvage_runs"].exists() or p["salvage_runs"].stat().st_size == 0),
                index=False,
            )
        rows, run_rows, dbg = [], [], []

    for st in range(0, len(rej_rows), step):
        chunk = rej_rows[st : st + step]
        audios, map_idx = [], []
        if int(args.load_workers) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.load_workers)) as ex:
                futs = [ex.submit(load_audio_16k, str(r.get("denoised_path", "") or ""), args.sample_rate) for r in chunk]
                for i, (r, fut) in enumerate(zip(chunk, futs)):
                    wav = str(r.get("denoised_path", "") or "")
                    if not wav or not Path(wav).exists():
                        processed += 1
                        pbar.update(1)
                        continue
                    try:
                        audio = fut.result()
                        if audio.size == 0:
                            processed += 1
                            pbar.update(1)
                            continue
                        audios.append(audio)
                        map_idx.append(i)
                    except Exception as e:
                        print(f"[salvage] audio error {Path(wav).name}: {e}")
                        processed += 1
                        pbar.update(1)
        else:
            for i, r in enumerate(chunk):
                wav = str(r.get("denoised_path", "") or "")
                if not wav or not Path(wav).exists():
                    processed += 1
                    pbar.update(1)
                    continue
                try:
                    audio = load_audio_16k(wav, args.sample_rate)
                    if audio.size == 0:
                        processed += 1
                        pbar.update(1)
                        continue
                    audios.append(audio)
                    map_idx.append(i)
                except Exception as e:
                    print(f"[salvage] audio error {Path(wav).name}: {e}")
                    processed += 1
                    pbar.update(1)

        if not map_idx:
            continue

        cached_payload = {}
        need_decode_local = []
        need_decode_audios = []
        for li, ri in enumerate(map_idx):
            r = chunk[ri]
            if args.salvage_reuse_phase1:
                z_tokens = parse_json_list(r.get("zipformer_tokens_json", ""))
                z_timestamps = parse_json_list(r.get("zipformer_timestamps_json", ""))
                g_text = str(r.get("gipformer_text", "") or "").strip()
                if z_tokens and z_timestamps and g_text:
                    cached_payload[li] = {
                        "zip_result": {
                            "text": str(r.get("zipformer_text", "") or ""),
                            "tokens": z_tokens,
                            "timestamps": z_timestamps,
                        },
                        "gip_text": g_text,
                    }
                    cached_used += 1
                    continue

            if args.salvage_allow_model_infer:
                need_decode_local.append(li)
                need_decode_audios.append(audios[li])
            else:
                skipped_no_cache += 1

        decoded_payload = {}
        if need_decode_local:
            if zip_rec is None or gip_rec is None:
                raise RuntimeError(
                    "Salvage needs ASR inference for some rows but recognizers are not initialized. "
                    "Enable --salvage-reuse-phase1 cache, or pass --salvage-allow-model-infer with model paths."
                )
            if args.parallel_dual_decode:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fut_zip = ex.submit(decode_zip_meta_batch, zip_rec, need_decode_audios, args.sample_rate, zip_bs)
                    fut_gip = ex.submit(decode_texts_batch, gip_rec, need_decode_audios, args.sample_rate, gip_bs)
                    zrs, zip_bs = fut_zip.result()
                    gts, gip_bs = fut_gip.result()
            else:
                zrs, zip_bs = decode_zip_meta_batch(zip_rec, need_decode_audios, args.sample_rate, zip_bs)
                gts, gip_bs = decode_texts_batch(gip_rec, need_decode_audios, args.sample_rate, gip_bs)

            for idx, li in enumerate(need_decode_local):
                decoded_payload[li] = {
                    "zip_result": zrs[idx] if idx < len(zrs) else {"text": "", "tokens": [], "timestamps": []},
                    "gip_text": str(gts[idx] or "") if idx < len(gts) else "",
                }
                model_infer_used += 1

        processable = set(cached_payload.keys()) | set(decoded_payload.keys())
        for li in range(len(map_idx)):
            if li not in processable:
                processed += 1
                pbar.update(1)

        for li, ri in enumerate(map_idx):
            if li not in processable:
                continue
            r = chunk[ri]
            wav = str(r.get("denoised_path", "") or "")
            audio = audios[li]
            payload = cached_payload.get(li) or decoded_payload.get(li)
            zr = payload["zip_result"]
            gr_text = payload["gip_text"]
            try:
                runs, matched, mismatch, aligned, events = find_consensus_runs(
                    zr["tokens"],
                    zr["timestamps"],
                    gr_text,
                    args.tail_pad_sec,
                    args.trim_start_sec,
                    args.trim_end_sec,
                    collect_events=collect_debug,
                )
                if collect_debug:
                    dbg.append(
                        {
                            "wav": wav,
                            "aligned_words": aligned,
                            "matched_words": matched,
                            "mismatch_words": mismatch,
                            "num_runs": len(runs),
                            "events_json": json.dumps(events, ensure_ascii=False),
                        }
                    )
                if args.debug_verbose:
                    print("=" * 100)
                    print("FILE:", wav)
                    print("ZIP_TEXT_FULL:\n", zr.get("text", ""))
                    print("GIP_TEXT_FULL:\n", gr_text)
                if runs:
                    merged = merge_runs(audio, args.sample_rate, runs, args.merge_gap_sec)
                    if merged:
                        m_audio, m_text, used = merged
                        if m_text.strip():
                            out = p["accepted_dir"] / f"salvaged_vote_{Path(wav).name}"
                            if not args.debug_no_write:
                                sf.write(str(out), m_audio, args.sample_rate)
                            nr = dict(r)
                            nr["final_wav"] = str(out)
                            nr["final_text"] = m_text
                            nr["duration_sec"] = round(len(m_audio) / args.sample_rate, 3)
                            nr["accepted"] = True
                            nr["is_salvaged_vote"] = True
                            nr["num_merged_runs"] = len(used)
                            nr["matched_words"] = matched
                            nr["mismatch_words"] = mismatch
                            nr["merged_runs"] = json.dumps(used, ensure_ascii=False)
                            rows.append(nr)
                            for i, u in enumerate(used):
                                run_rows.append(
                                    {
                                        "src_wav": wav,
                                        "out_wav": str(out),
                                        "run_idx": i,
                                        "start_word_idx": u["start_word_idx"],
                                        "end_word_idx": u["end_word_idx"],
                                        "start_time": u["start_time"],
                                        "end_time": u["end_time"],
                                        "duration": u["duration"],
                                        "text": u["text"],
                                    }
                                )
            except Exception as e:
                print(f"[salvage] error {Path(wav).name}: {e}")
            processed += 1
            pbar.update(1)
            if args.save_every > 0 and processed % args.save_every == 0:
                flush_buffers()
                print(f"[salvage] checkpoint processed={processed} zip_bs={zip_bs} gip_bs={gip_bs}")

    pbar.close()
    flush_buffers()
    if p["salvage"].exists():
        try:
            sdf = pd.read_csv(p["salvage"])
            print(f"[salvage] salvaged={len(sdf)}")
        except Exception:
            print("[salvage] output written")
    else:
        print("[salvage] no output")
    print(
        "[salvage] cache_used=",
        cached_used,
        "| model_infer_used=",
        model_infer_used,
        "| skipped_no_cache=",
        skipped_no_cache,
    )


def safe_sum_hours(df: pd.DataFrame, col="duration_sec"):
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum() / 3600.0)


def run_combine(p):
    all_df = pd.read_csv(p["all"])
    keep = ["final_wav", "final_text", "duration_sec", "src", "start_sec", "end_sec"]
    pre = all_df[as_bool_series(all_df.get("accepted", False), len(all_df), all_df.index)].copy()
    for c in keep:
        if c not in pre.columns:
            pre[c] = None
    pre = pre[keep]
    pre.to_csv(p["accepted"], index=False)
    sal = pd.read_csv(p["salvage"]) if p["salvage"].exists() else pd.DataFrame(columns=keep)
    for c in keep:
        if c not in sal.columns:
            sal[c] = None
    sal = sal[keep]
    comb = pd.concat([pre, sal], ignore_index=True)
    if not comb.empty:
        comb = comb.drop_duplicates(subset=["final_wav"], keep="last")
    comb.to_csv(p["accepted_combined"], index=False)
    vad = pd.read_csv(p["vad"]) if p["vad"].exists() else pd.DataFrame()
    den = pd.read_csv(p["denoise"]) if p["denoise"].exists() else pd.DataFrame()
    raw_h = 0.0
    if p["raw_dir"].exists():
        wavs = sorted(p["raw_dir"].glob("*_16k.wav")) or sorted(p["raw_dir"].glob("*.wav"))
        for w in wavs:
            try:
                raw_h += float(sf.info(str(w)).duration) / 3600.0
            except Exception:
                pass
    print("\n=== Pipeline Hours Breakdown ===")
    print("Raw input hours        :", round(raw_h, 3))
    print("VAD segment hours      :", round(safe_sum_hours(vad), 3))
    print("Denoised segment hours :", round(safe_sum_hours(den), 3))
    print("Accepted (pre-salvage) :", round(safe_sum_hours(pre), 3))
    print("Salvaged voting hours  :", round(safe_sum_hours(sal), 3))
    print("Accepted (combined)    :", round(safe_sum_hours(comb), 3))
    print("\n=== Acceptance Stats ===")
    print("Total segments :", len(all_df))
    print("Accepted       :", len(comb))
    print("Accept rate    :", round(100 * len(comb) / max(1, len(all_df)), 2), "%")
    print("\nSaved manifests:")
    print("-", p["accepted"])
    print("-", p["accepted_combined"])


def build_paths(base_dir: Path):
    m = base_dir / "manifests"
    p = {
        "base_dir": base_dir,
        "manifests": m,
        "raw_dir": base_dir / "raw",
        "accepted_dir": base_dir / "accepted",
        "rejected_dir": base_dir / "rejected",
        "denoise": m / "denoised_segments.csv",
        "vad": m / "vad_segments.csv",
        "all": m / "all_results.csv",
        "accepted": m / "accepted_manifest.csv",
        "accepted_jsonl": m / "accepted_manifest.jsonl",
        "salvage": m / "salvaged_vote_manifest.csv",
        "salvage_runs": m / "salvaged_vote_runs_manifest.csv",
        "salvage_debug": m / "salvaged_vote_debug.csv",
        "accepted_combined": m / "accepted_manifest_combined.csv",
    }
    for d in [p["manifests"], p["accepted_dir"], p["rejected_dir"]]:
        d.mkdir(parents=True, exist_ok=True)
    return p


def parse_args():
    ap = argparse.ArgumentParser("Voting pipeline local (exact + salvage + combine)")
    ap.add_argument("--base-dir", default="data_label_auto")
    ap.add_argument("--zip-model-dir", required=True)
    ap.add_argument("--gipformer-repo", required=True)
    ap.add_argument("--hf-cache-dir", default="/home/vietnam/voice_to_text/.hf-cache-local")
    ap.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--zip-provider", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--gip-provider", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--ort-probe", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--auto-install-ort", action="store_true")
    ap.add_argument("--steps", default="exact,salvage,combine")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--batch-size-cuda", type=int, default=96)
    ap.add_argument("--batch-size-cpu", type=int, default=6)
    ap.add_argument("--salvage-batch-size", type=int, default=192)
    ap.add_argument("--load-workers", type=int, default=32)
    ap.add_argument("--zip-num-threads", type=int, default=None)
    ap.add_argument("--gip-num-threads", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--parallel-dual-decode", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force-reprocess", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-run-minutes", type=float, default=None)
    ap.add_argument("--tail-pad-sec", type=float, default=0.0)
    ap.add_argument("--trim-start-sec", type=float, default=0.03)
    ap.add_argument("--trim-end-sec", type=float, default=0.05)
    ap.add_argument("--merge-gap-sec", type=float, default=0.0)
    ap.add_argument("--save-salvage-debug", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--salvage-reuse-phase1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Phase-2 salvage reuses phase-1 cached zip tokens/timestamps + gip text "
            "from all_results.csv (timestamp voting only, no ASR rerun)."
        ),
    )
    ap.add_argument(
        "--salvage-allow-model-infer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If cache is missing in phase-2 salvage, allow running ASR models as fallback.",
    )
    ap.add_argument("--debug-target-file", default=None)
    ap.add_argument("--debug-first", action="store_true")
    ap.add_argument("--debug-no-write", action="store_true")
    ap.add_argument("--debug-verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    steps = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
    base_dir = Path(args.base_dir).expanduser().resolve()
    p = build_paths(base_dir)
    if args.ort_probe:
        provider, providers = ensure_onnxruntime(args.provider, args.auto_install_ort)
    else:
        providers = ["probe_skipped"]
        if args.provider == "auto":
            provider = "cuda" if shutil.which("nvidia-smi") else "cpu"
        else:
            provider = args.provider
        if provider == "cuda" and not shutil.which("nvidia-smi"):
            raise RuntimeError("Requested cuda but nvidia-smi not found. Use --provider cpu")
    zip_provider = provider if args.zip_provider == "auto" else args.zip_provider
    gip_provider = provider if args.gip_provider == "auto" else args.gip_provider
    if args.ort_probe:
        if zip_provider == "cuda" and "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"zip-provider=cuda requested but available providers={providers}")
        if gip_provider == "cuda" and "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"gip-provider=cuda requested but available providers={providers}")
    gpu_name, gpu_mem = get_gpu_info()
    cpu_count = os.cpu_count() or 8
    args.batch_size = args.batch_size_cuda if zip_provider == "cuda" else args.batch_size_cpu
    zip_threads = int(args.zip_num_threads) if args.zip_num_threads else (2 if zip_provider == "cuda" else min(16, max(4, cpu_count // 8)))
    gip_threads = int(args.gip_num_threads) if args.gip_num_threads else (2 if gip_provider == "cuda" else min(96, max(8, cpu_count // 3)))

    print("=== Runtime ===")
    print("Providers:", providers)
    print("Chosen provider:", provider)
    print("Zip provider:", zip_provider)
    print("Gip provider:", gip_provider)
    if provider == "cuda":
        print("GPU:", gpu_name, f"({gpu_mem:.1f} GB)")
    print("Batch size:", args.batch_size)
    print("Salvage batch size:", args.salvage_batch_size)
    print("Load workers:", args.load_workers)
    print("Zip threads:", zip_threads)
    print("Gip threads:", gip_threads)
    print("Base dir:", base_dir)
    print(
        "Salvage trim:",
        f"tail_pad={args.tail_pad_sec}s",
        f"trim_start={args.trim_start_sec}s",
        f"trim_end={args.trim_end_sec}s",
    )

    zip_dir = Path(args.zip_model_dir).expanduser().resolve()
    gip_repo = Path(args.gipformer_repo).expanduser().resolve()
    hf_cache_dir = Path(args.hf_cache_dir).expanduser().resolve()
    if not zip_dir.exists():
        raise FileNotFoundError(f"zip model dir missing: {zip_dir}")
    if not gip_repo.exists():
        raise FileNotFoundError(f"gipformer repo missing: {gip_repo}")

    need_asr = ("exact" in steps) or ("salvage" in steps and args.salvage_allow_model_infer)
    zip_rec = gip_rec = None
    if need_asr:
        zip_rec = init_zip(zip_dir, zip_provider, args.sample_rate, zip_threads)
        gip_rec = init_gip(gip_repo, gip_provider, gip_threads, hf_cache_dir)

    if "exact" in steps:
        run_exact(args, p, zip_rec, gip_rec)
    if "salvage" in steps:
        run_salvage(args, p, zip_rec, gip_rec)
    if "combine" in steps:
        run_combine(p)
    print("\nDone.")


if __name__ == "__main__":
    main()
