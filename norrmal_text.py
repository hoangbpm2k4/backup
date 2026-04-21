#!/usr/bin/env python3
"""Vietnamese ASR text normalizer using Qwen3.5-9B GGUF (local inference via llama-cpp-python).

Model: unsloth/Qwen3.5-9B-GGUF  ->  Qwen3.5-9B-UD-Q4_K_XL.gguf
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing as mp
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_OLLAMA_MODEL = "qwen3:14b-q4_K_M"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_BACKEND = "ollama"

_LLAMA_MODEL = None
_LLAMA_MODEL_KEY: Optional[Tuple[str, int, int, int, int]] = None
_WORKER_NORM_KWARGS: Optional[Dict[str, Any]] = None

SYSTEM_PROMPT = """\
Bạn là bộ hậu xử lý văn bản ASR tiếng Việt.
Nhiệm vụ: chuẩn hoá chính tả hiển thị (viết hoa, viết thường, dấu câu) cho MỘT đoạn transcript.

Ràng buộc bắt buộc:
1) GIỮ NGUYÊN nội dung gốc, không thêm ý mới, không bớt ý, không diễn đạt lại.
2) Không đổi thứ tự từ nếu không thật sự cần cho dấu câu.
3) Chỉ thêm/chỉnh chữ hoa-thường và dấu câu (. , ? ! : ;) để dễ đọc.
4) Đoạn có thể bị cắt giữa câu; KHÔNG bắt buộc chữ đầu phải viết hoa.
5) Chỉ viết hoa tên riêng khi đủ chắc chắn theo ngữ cảnh; nếu không chắc thì giữ thường.
6) Chuẩn hoá số: ưu tiên chuyển số viết bằng chữ sang chữ số khi đủ chắc chắn.
7) Giữ đơn vị đo ngay sau số (ví dụ micromét, kg, %...).
8) Không chắc về giá trị số thì giữ nguyên cách viết gốc.
9) Không trả lời giải thích.

Đầu ra:
- Chỉ trả về đúng 1 dòng văn bản đã chuẩn hoá.
"""

PROPER_NAME_EXAMPLES = """\
VÍ DỤ THAM CHIẾU CHUẨN HÓA TÊN RIÊNG (chỉ khi chắc chắn):
- "hennike", "henike" -> "Heineken"
- "sauinbet", "inbet" -> "InBev"
- "samiler" -> "SABMiller"
- "an hiu sơ búto" -> "Anheuser-Busch"
- "milơ lí" -> "Miller Lite"
Nếu gặp tên quốc tế quen thuộc khác, hãy chuẩn hóa tương tự khi đủ chắc chắn.
"""

NUMBER_WORDS = {
    "không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm",
    "sáu", "bảy", "tám", "chín", "mười", "mươi", "chục", "trăm",
    "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "linh", "lẻ", "phẩy", "âm", "trừ",
}

RELATION_TITLES = {
    "ông", "bà", "anh", "chị", "em", "cô", "chú", "bác", "cậu", "mợ", "dì", "thím", "thầy", "u", "mụ",
}

NON_NAME_AFTER_TITLE = {
    "cả", "hai", "ba", "bốn", "tư", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "ta", "tôi", "mình", "nó", "họ", "người", "nhà", "vợ", "chồng", "con", "cháu",
    "anh", "chị", "em", "ông", "bà", "cô", "chú", "bác", "cậu", "mợ", "dì", "thím", "thầy", "u", "mụ",
    "lớn", "nhỏ", "trẻ", "già",
}

NAME_FOLLOW_VERBS = {
    "theo", "đi", "về", "lên", "xuống", "nói", "bảo", "kêu", "gọi",
    "đứng", "ngồi", "chạy", "đến", "rời", "vào", "ra", "qua", "đưa", "lấy", "mang", "xem",
}

KNOWN_FOREIGN_NAME_PATTERNS = (
    (re.compile(r"(?i)\b(?:hennike|henike)\b"), "Heineken"),
    (re.compile(r"(?i)\b(?:sau\s*inbet|sauinbet|inbet)\b"), "InBev"),
    (re.compile(r"(?i)\bsamiler\b"), "SABMiller"),
    (re.compile(r"(?i)\ban\s*hiu\s*sơ\s*búto\b"), "Anheuser-Busch"),
    (re.compile(r"(?i)\bmilơ\s*lí\b"), "Miller Lite"),
)

PUNCT_SPACE_RE = re.compile(r"[^\wÀ-ỹ]+", flags=re.UNICODE)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Normalize Vietnamese ASR text using Ollama or direct llama-cpp."
    )
    parser.add_argument(
        "--backend", choices=("ollama", "llama_cpp"), default=DEFAULT_BACKEND,
        help="Inference backend. ollama uses HTTP server; llama_cpp runs local GGUF directly.",
    )
    parser.add_argument(
        "--mode", choices=("sample", "full"), default="sample",
        help="sample: run random sample check. full: process full input with resume.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_OLLAMA_MODEL,
        help="Model name for Ollama backend.",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to GGUF model file for llama_cpp backend.",
    )
    parser.add_argument(
        "--ollama-url", type=str, default=DEFAULT_OLLAMA_URL,
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--input", type=Path, default=root / "selected_transcripts_full.txt",
        help="Input file in key<TAB>text or JSONL format.",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "selected_transcripts_qwen_norm.txt",
        help="Output file in key<TAB>normalized_text or JSONL format.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "selected_transcripts_qwen_norm.checkpoint.json",
        help="Checkpoint JSON path for resume.",
    )
    parser.add_argument(
        "--sample-output", type=Path, default=root / "qwen_norm_sample_5.csv",
        help="CSV output path for sample mode.",
    )
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-min-tokens", type=int, default=5)
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Generation temperature. Keep low for deterministic normalization.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--num-ctx", type=int, default=1024,
        help="Context window for each request.",
    )
    parser.add_argument(
        "--n-gpu-layers", type=int, default=0,
        help="llama_cpp only: number of layers to offload to GPU. 0 = CPU only.",
    )
    parser.add_argument(
        "--n-threads", type=int, default=0,
        help="llama_cpp only: CPU threads. 0 = llama_cpp default.",
    )
    parser.add_argument(
        "--llama-n-batch", type=int, default=512,
        help="llama_cpp only: prompt processing batch size (n_batch).",
    )
    parser.add_argument(
        "--keep-alive", type=str, default="30m",
        help=(
            "Ollama keep_alive value to keep model loaded between requests "
            "(e.g. 30m, 2h, -1 to keep forever)."
        ),
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Force exactly 1 model call per segment for maximum speed.",
    )
    parser.set_defaults(warmup_model=True)
    parser.add_argument(
        "--no-warmup-model",
        dest="warmup_model",
        action="store_false",
        help="Skip one-time warmup request before processing.",
    )
    parser.add_argument(
        "--save-every", type=int, default=20,
        help="Checkpoint frequency in full mode.",
    )
    parser.add_argument(
        "--flush-every", type=int, default=50,
        help="Flush output file every N rows in full mode.",
    )
    parser.add_argument("--start-line", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument(
        "--parallel-lines", type=int, default=1,
        help="Process N lines in parallel in full mode (llama_cpp backend only).",
    )
    parser.add_argument(
        "--input-format", choices=("auto", "tsv", "jsonl"), default="auto",
        help="How to parse input. auto uses .jsonl suffix, otherwise key<TAB>text.",
    )
    parser.add_argument(
        "--jsonl-text-field", type=str, default="text",
        help="JSONL field that contains the transcript text.",
    )
    parser.add_argument(
        "--jsonl-id-field", type=str, default="utt_id",
        help="JSONL field used as row identifier for logs/sample output.",
    )
    parser.add_argument(
        "--allow-word-edits", action="store_true",
        help="Allow model to modify words (disabled by default).",
    )
    parser.set_defaults(normalize_numbers=True, normalize_known_names=True)
    parser.add_argument(
        "--no-normalize-numbers", dest="normalize_numbers", action="store_false",
    )
    parser.add_argument("--normalize-known-names", action="store_true")
    parser.add_argument(
        "--no-normalize-known-names", dest="normalize_known_names", action="store_false",
    )
    parser.add_argument("--max-non-number-edits-for-numbers", type=int, default=2)
    parser.add_argument("--max-non-number-edits-for-names", type=int, default=20)
    parser.add_argument(
        "--enable-refine-passes",
        action="store_true",
        help="Enable extra retry/name/number refinement passes (slower).",
    )
    parser.set_defaults(force_punctuate=True)
    parser.add_argument(
        "--no-force-punctuate",
        dest="force_punctuate",
        action="store_false",
        help="Disable punctuation-focused retry when output is unchanged.",
    )
    return parser.parse_args()


def check_ollama_ready(base_url: str, model: str) -> None:
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    req = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Start server first, for example:\n"
            "OLLAMA_HOST=127.0.0.1:11434 /path/to/ollama serve"
        ) from exc

    models = payload.get("models", []) if isinstance(payload, dict) else []
    installed = {str(m.get("name", "")) for m in models if isinstance(m, dict)}
    candidates = {model}
    if ":" not in model:
        candidates.add(f"{model}:latest")

    if not (installed & candidates):
        raise RuntimeError(
            f"Ollama model '{model}' not found. Create it first with `ollama create` or `ollama pull`."
        )


def normalize_keep_alive(value: str) -> str:
    v = value.strip()
    if re.fullmatch(r"-?\d+", v):
        if v == "-1":
            # Ollama expects Go duration format; map "-1" to a very long duration.
            return "720h"
        return f"{v}s"
    return v


def warmup_ollama_model(
    *,
    base_url: str,
    model: str,
    num_ctx: int,
    keep_alive: str,
) -> None:
    started = time.time()
    req_body = {
        "model": model,
        "prompt": "warmup",
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_predict": 1,
            "temperature": 0.0,
            "num_ctx": min(num_ctx, 512),
        },
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0):
            pass
    except Exception as exc:
        print(f"[init][warn] warmup failed, continue anyway: {exc}")
        return
    elapsed = time.time() - started
    print(f"[init] model warmup done in {elapsed:.2f}s (keep_alive={keep_alive})")


def _load_llama_model(
    *,
    model_path: Path,
    num_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    llama_n_batch: int,
):
    global _LLAMA_MODEL, _LLAMA_MODEL_KEY
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    key = (str(model_path), num_ctx, n_gpu_layers, n_threads, llama_n_batch)
    if _LLAMA_MODEL is not None and _LLAMA_MODEL_KEY == key:
        return _LLAMA_MODEL

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install with: pip3 install --user llama-cpp-python"
        ) from exc

    started = time.time()
    kwargs: Dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": num_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": max(1, llama_n_batch),
        "verbose": False,
    }
    if n_threads > 0:
        kwargs["n_threads"] = n_threads

    _LLAMA_MODEL = Llama(**kwargs)
    _LLAMA_MODEL_KEY = key
    elapsed = time.time() - started
    print(
        f"[init] loaded llama_cpp model in {elapsed:.2f}s "
        f"(ctx={num_ctx}, gpu_layers={n_gpu_layers}, threads={n_threads or 'auto'}, "
        f"n_batch={max(1, llama_n_batch)})"
    )
    return _LLAMA_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────
def make_user_prompt(
    raw_text: str,
    *,
    strict_preserve_words: bool = False,
    normalize_numbers: bool = True,
    include_name_examples: bool = False,
    force_number_conversion: bool = False,
    force_name_normalization: bool = False,
) -> str:
    extra = ""
    if strict_preserve_words:
        extra = (
            "\nRÀNG BUỘC SIÊU NGHIÊM: giữ nguyên từng từ gốc theo đúng thứ tự; "
            "không đổi từ đồng âm, không sửa chính tả theo suy đoán, "
            "không chuẩn hoá tên riêng theo kiến thức ngoài."
        )

    name_rule = ""
    if include_name_examples:
        name_rule = (
            "\nCHUẨN HOÁ TÊN RIÊNG: được phép sửa cụm phiên âm tên riêng sang dạng chuẩn khi đủ chắc chắn.\n"
            f"{PROPER_NAME_EXAMPLES}"
        )
    if force_name_normalization:
        name_rule += (
            "\nƯU TIÊN CAO: thử chuẩn hóa tên riêng quốc tế bị phiên âm sai, "
            "nhưng chỉ được thay đổi tối thiểu ở các từ không liên quan tên riêng."
        )

    number_rule = ""
    if normalize_numbers:
        number_rule = (
            "\nCHUẨN HOÁ SỐ: đổi số chữ sang chữ số khi chắc chắn. "
            "Ví dụ: 'một trăm hai mươi ba micromét' -> '123 micromét'. "
            "Ví dụ: 'khoảng một trăm hai mươi đến một trăm ba mươi micromét' -> 'khoảng 120-130 micromét'. "
            "Ví dụ lỗi ASR: 'khoảng một trăm hai mươi một trăm ba mươi micomét' -> 'khoảng 120-130 micomét'."
        )
    if force_number_conversion:
        number_rule += (
            "\nƯU TIÊN CAO: cố gắng chuyển cụm số chữ thành chữ số. "
            "Nếu có hai mốc số liền nhau theo ngữ cảnh khoảng/dao động, ưu tiên chuẩn hóa dạng a-b."
        )

    return (
        "Hãy chuẩn hoá đoạn ASR sau theo đúng yêu cầu hệ thống."
        f"{extra}{name_rule}{number_rule}\n"
        f"ĐOẠN_GỐC: {raw_text}\n"
        "Trả về duy nhất 1 dòng kết quả."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────────────────────────────────────
def sanitize_model_output(text: str, fallback: str) -> str:
    text = text.strip().replace("\r", "\n")
    text = text.replace("```", "").strip()
    # Strip think-block patterns (Qwen3 thinking mode) — closed and unclosed
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    if not text:
        return fallback
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return fallback

    prompt_echo_patterns = (
        "hãy chuẩn hoá đoạn asr sau theo đúng yêu cầu hệ thống",
        "đoạn_gốc:",
        "trả về duy nhất 1 dòng kết quả",
    )
    first_line = lines[0]
    for ln in lines:
        lowered = ln.lower()
        if any(pat in lowered for pat in prompt_echo_patterns):
            continue
        first_line = ln
        break

    first_line = re.sub(r"^(normalized\s*[:：]\s*)", "", first_line, flags=re.IGNORECASE)
    first_line = re.sub(r"^(đoáng\s*[:：]\s*)", "", first_line, flags=re.IGNORECASE)
    first_line = first_line.strip(" \"'`")
    return first_line or fallback


def canonical_tokens_for_compare(text: str) -> List[str]:
    lowered = text.lower()
    lowered = PUNCT_SPACE_RE.sub(" ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return []
    return lowered.split(" ")


def is_digit_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)?", token))


def collapse_number_like_tokens(tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in NUMBER_WORDS or is_digit_token(token):
            j = i + 1
            while j < len(tokens) and (tokens[j] in NUMBER_WORDS or is_digit_token(tokens[j])):
                j += 1
            out.append("<num>")
            i = j
            continue
        out.append(token)
        i += 1
    return out


def contains_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def likely_has_number_words(text: str) -> bool:
    tokens = canonical_tokens_for_compare(text)
    count = sum(1 for tok in tokens if tok in NUMBER_WORDS)
    return count >= 2


def strip_number_like_tokens(tokens: List[str]) -> List[str]:
    return [tok for tok in tokens if tok not in NUMBER_WORDS and not is_digit_token(tok)]


def token_edit_distance(a: List[str], b: List[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, tb in enumerate(b, 1):
            cost = 0 if ta == tb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def non_number_edit_distance(raw_text: str, normalized_text: str) -> int:
    raw_tokens = strip_number_like_tokens(canonical_tokens_for_compare(raw_text))
    norm_tokens = strip_number_like_tokens(canonical_tokens_for_compare(normalized_text))
    return token_edit_distance(raw_tokens, norm_tokens)


def has_capitalized_latin_word(text: str) -> bool:
    return bool(re.search(r"\b[A-Z][A-Za-z][A-Za-z-]*\b", text))


def likely_has_foreign_name_noise(text: str) -> bool:
    tokens = canonical_tokens_for_compare(text)
    count = 0
    for tok in tokens:
        if tok in NUMBER_WORDS:
            continue
        if re.fullmatch(r"[a-z][a-z-]{4,}", tok):
            count += 1
            if count >= 2:
                return True
    return False


def words_preserved(raw_text: str, normalized_text: str, *, allow_number_transforms: bool) -> bool:
    raw_tokens = canonical_tokens_for_compare(raw_text)
    norm_tokens = canonical_tokens_for_compare(normalized_text)
    if allow_number_transforms:
        raw_tokens = collapse_number_like_tokens(raw_tokens)
        norm_tokens = collapse_number_like_tokens(norm_tokens)
    return raw_tokens == norm_tokens


def capitalize_sentence_starts(text: str) -> str:
    chars = list(text)
    should_cap = True
    for idx, ch in enumerate(chars):
        if should_cap and ch.isalpha():
            chars[idx] = ch.upper()
            should_cap = False
            continue
        if ch in ".?!":
            should_cap = True
    return "".join(chars)


def capitalize_word_everywhere(text: str, word: str) -> str:
    if not word:
        return text
    cap = word[0].upper() + word[1:]
    pattern = re.compile(rf"(?i)\b{re.escape(word)}\b")
    return pattern.sub(cap, text)


def find_name_candidates(text: str) -> List[str]:
    lower_text = text.lower()
    candidates: set[str] = set()

    title_pattern = re.compile(
        rf"\b(?:{'|'.join(re.escape(t) for t in sorted(RELATION_TITLES))})\s+([a-zà-ỹ][a-zà-ỹ'-]{{1,}})\b"
    )
    for m in title_pattern.finditer(lower_text):
        token = m.group(1)
        if token not in NON_NAME_AFTER_TITLE and token not in NAME_FOLLOW_VERBS:
            candidates.add(token)

    va_pattern = re.compile(
        rf"\bvà\s+([a-zà-ỹ][a-zà-ỹ'-]{{1,}})\s+(?:{'|'.join(re.escape(v) for v in sorted(NAME_FOLLOW_VERBS))})\b"
    )
    token_counts: Dict[str, int] = {}
    for tok in canonical_tokens_for_compare(lower_text):
        token_counts[tok] = token_counts.get(tok, 0) + 1
    for m in va_pattern.finditer(lower_text):
        token = m.group(1)
        if token in NON_NAME_AFTER_TITLE:
            continue
        if token_counts.get(token, 0) >= 2:
            candidates.add(token)

    return sorted(candidates, key=len, reverse=True)


def apply_known_foreign_name_normalization(text: str) -> str:
    out = text
    for pattern, replacement in KNOWN_FOREIGN_NAME_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def postprocess_capitalization(text: str) -> str:
    out = capitalize_sentence_starts(text)
    for name in find_name_candidates(out):
        out = capitalize_word_everywhere(out, name)
    return out


def finalize_output(text: str, *, normalize_known_names: bool) -> str:
    out = text
    if normalize_known_names:
        out = apply_known_foreign_name_normalization(out)
    return postprocess_capitalization(out)


# ──────────────────────────────────────────────────────────────────────────────
# Local LLM inference
# ──────────────────────────────────────────────────────────────────────────────
def call_llm_once(
    *,
    backend: str,
    base_url: str,
    model: str,
    model_path: Optional[Path],
    raw_text: str,
    prompt_user: str,
    temperature: float,
    max_output_tokens: int,
    num_ctx: int,
    keep_alive: str,
    n_gpu_layers: int,
    n_threads: int,
    llama_n_batch: int,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_user},
    ]
    if backend == "ollama":
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.95,
                "num_predict": max_output_tokens,
                "num_ctx": num_ctx,
            },
        }
        body["keep_alive"] = keep_alive
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body_text[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama connection error: {exc}") from exc

        raw_output = (
            response.get("message", {}).get("content", "")
            if isinstance(response, dict)
            else ""
        )
        return sanitize_model_output(raw_output, fallback=raw_text)

    if model_path is None:
        raise ValueError("--model-path is required when --backend llama_cpp")
    llm = _load_llama_model(
        model_path=model_path,
        num_ctx=num_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        llama_n_batch=llama_n_batch,
    )
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_output_tokens,
    )
    raw_output = ""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            if isinstance(first, dict):
                msg = first.get("message", {})
                if isinstance(msg, dict):
                    raw_output = str(msg.get("content", "") or "")
                if not raw_output:
                    raw_output = str(first.get("text", "") or "")
    return sanitize_model_output(raw_output, fallback=raw_text)


def call_qwen_normalize(
    *,
    backend: str,
    base_url: str,
    model: str,
    model_path: Optional[Path],
    raw_text: str,
    temperature: float,
    max_output_tokens: int,
    num_ctx: int,
    allow_word_edits: bool,
    normalize_numbers: bool,
    normalize_known_names: bool,
    max_non_number_edits_for_numbers: int,
    max_non_number_edits_for_names: int,
    enable_refine_passes: bool,
    force_punctuate: bool,
    keep_alive: str,
    single_pass: bool,
    n_gpu_layers: int,
    n_threads: int,
    llama_n_batch: int,
) -> str:
    def call_once(prompt: str) -> str:
        return call_llm_once(
            backend=backend,
            base_url=base_url,
            model=model,
            model_path=model_path,
            raw_text=raw_text,
            prompt_user=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            llama_n_batch=llama_n_batch,
        )

    safe_candidate = raw_text
    prompt_main = make_user_prompt(
        raw_text,
        strict_preserve_words=False,
        normalize_numbers=normalize_numbers,
        include_name_examples=normalize_known_names,
    )
    result = call_once(prompt_main)

    if single_pass:
        return finalize_output(result, normalize_known_names=normalize_known_names)

    # Fast path: keep only one call by default for speed.
    if not enable_refine_passes:
        candidate = result

        if (
            not allow_word_edits
            and not words_preserved(raw_text, candidate, allow_number_transforms=normalize_numbers)
        ):
            strict_prompt = make_user_prompt(
                raw_text,
                strict_preserve_words=True,
                normalize_numbers=normalize_numbers,
                include_name_examples=False,
            )
            strict_result = call_once(strict_prompt)
            if words_preserved(raw_text, strict_result, allow_number_transforms=normalize_numbers):
                candidate = strict_result

        if (
            normalize_numbers
            and not contains_digit(candidate)
            and likely_has_number_words(raw_text)
        ):
            prompt_numbers = make_user_prompt(
                raw_text,
                strict_preserve_words=True,
                normalize_numbers=True,
                include_name_examples=False,
                force_number_conversion=True,
            )
            number_result = call_once(prompt_numbers)
            if contains_digit(number_result):
                dist = non_number_edit_distance(raw_text, number_result)
                if dist <= max_non_number_edits_for_numbers:
                    candidate = number_result

        if candidate == raw_text and force_punctuate and len(raw_text.split()) >= 14:
            prompt_punct = (
                "Chỉ thêm/chỉnh dấu câu và chữ hoa-thường, không đổi bất kỳ từ nào. "
                "Ưu tiên thêm dấu phẩy/chấm tối thiểu để câu dễ đọc hơn.\n"
                f"ĐOẠN_GỐC: {raw_text}\n"
                "Trả về duy nhất 1 dòng kết quả."
            )
            punct_result = call_once(prompt_punct)
            if allow_word_edits or words_preserved(
                raw_text, punct_result, allow_number_transforms=normalize_numbers
            ):
                candidate = punct_result
        return finalize_output(candidate, normalize_known_names=normalize_known_names)

    should_try_name_pass = normalize_known_names and likely_has_foreign_name_noise(raw_text)

    if allow_word_edits or words_preserved(raw_text, result, allow_number_transforms=normalize_numbers):
        safe_candidate = result
        if not should_try_name_pass:
            return finalize_output(result, normalize_known_names=normalize_known_names)

    if should_try_name_pass and has_capitalized_latin_word(result):
        name_dist = non_number_edit_distance(raw_text, result)
        if name_dist <= max_non_number_edits_for_names:
            return finalize_output(result, normalize_known_names=normalize_known_names)

    # Retry with strict no-rewrite instructions
    prompt_strict = make_user_prompt(
        raw_text,
        strict_preserve_words=True,
        normalize_numbers=normalize_numbers,
        include_name_examples=False,
    )
    strict_result = call_once(prompt_strict)
    if words_preserved(raw_text, strict_result, allow_number_transforms=normalize_numbers):
        safe_candidate = strict_result
        if not should_try_name_pass:
            return finalize_output(strict_result, normalize_known_names=normalize_known_names)

    if should_try_name_pass:
        prompt_names = make_user_prompt(
            raw_text,
            strict_preserve_words=False,
            normalize_numbers=normalize_numbers,
            include_name_examples=True,
            force_name_normalization=True,
        )
        name_result = call_once(prompt_names)
        if has_capitalized_latin_word(name_result):
            name_dist = non_number_edit_distance(raw_text, name_result)
            if name_dist <= max_non_number_edits_for_names:
                return finalize_output(name_result, normalize_known_names=normalize_known_names)

    if normalize_numbers:
        prompt_numbers = make_user_prompt(
            raw_text,
            strict_preserve_words=True,
            normalize_numbers=True,
            include_name_examples=False,
            force_number_conversion=True,
        )
        number_result = call_once(prompt_numbers)
        if contains_digit(number_result):
            dist = non_number_edit_distance(raw_text, number_result)
            if dist <= max_non_number_edits_for_numbers:
                return finalize_output(number_result, normalize_known_names=normalize_known_names)

    return finalize_output(safe_candidate, normalize_known_names=normalize_known_names)


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def detect_input_format(path: Path, configured: str) -> str:
    if configured != "auto":
        return configured
    return "jsonl" if path.suffix.lower() == ".jsonl" else "tsv"


def split_record(
    line: str,
    line_no: int,
    *,
    input_format: str,
    jsonl_text_field: str,
    jsonl_id_field: str,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    line = line.rstrip("\n")
    if input_format == "jsonl":
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Line {line_no}: expected JSON object.")
        if jsonl_text_field not in obj:
            raise KeyError(f"Line {line_no}: missing JSONL text field '{jsonl_text_field}'.")
        key = str(obj.get(jsonl_id_field) or f"line_{line_no}")
        raw_text = obj[jsonl_text_field]
        if not isinstance(raw_text, str):
            raise TypeError(f"Line {line_no}: field '{jsonl_text_field}' must be a string.")
        return key, raw_text, obj

    if "\t" not in line:
        return f"line_{line_no}", line, None
    key, raw_text = line.split("\t", 1)
    return key, raw_text, None


def format_output_line(
    *,
    key: str,
    normalized_text: str,
    record: Optional[Dict[str, Any]],
    output_format: str,
    jsonl_text_field: str,
) -> str:
    if output_format == "jsonl":
        if record is None:
            raise ValueError("JSONL output requires the original JSON object.")
        out_record = dict(record)
        out_record[jsonl_text_field] = normalized_text
        return json.dumps(out_record, ensure_ascii=False) + "\n"
    return f"{key}\t{normalized_text}\n"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def atomic_write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def punctuation_count(text: str) -> int:
    return sum(ch in ".,?!:;" for ch in text)


def evaluate_pair(raw_text: str, normalized_text: str, *, allow_number_transforms: bool) -> Dict[str, object]:
    return {
        "changed": raw_text != normalized_text,
        "word_preserved": words_preserved(raw_text, normalized_text, allow_number_transforms=allow_number_transforms),
        "raw_tokens": len(raw_text.split()),
        "norm_tokens": len(normalized_text.split()),
        "raw_punct": punctuation_count(raw_text),
        "norm_punct": punctuation_count(normalized_text),
    }


def reservoir_sample_records(
    input_path: Path,
    sample_size: int,
    seed: int,
    min_tokens: int,
    *,
    input_format: str,
    jsonl_text_field: str,
    jsonl_id_field: str,
) -> List[Tuple[int, str, str]]:
    rng = random.Random(seed)
    sample: List[Tuple[int, str, str]] = []
    seen = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            key, text, _ = split_record(
                line,
                line_no,
                input_format=input_format,
                jsonl_text_field=jsonl_text_field,
                jsonl_id_field=jsonl_id_field,
            )
            if len(text.split()) < min_tokens:
                continue
            seen += 1
            item = (line_no, key, text)
            if len(sample) < sample_size:
                sample.append(item)
            else:
                j = rng.randint(1, seen)
                if j <= sample_size:
                    sample[j - 1] = item
    return sample


# ──────────────────────────────────────────────────────────────────────────────
# Run modes
# ──────────────────────────────────────────────────────────────────────────────
def _normalize(args: argparse.Namespace, raw_text: str) -> str:
    return call_qwen_normalize(
        backend=args.backend,
        base_url=args.ollama_url,
        model=args.model,
        model_path=args.model_path,
        raw_text=raw_text,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        num_ctx=args.num_ctx,
        allow_word_edits=args.allow_word_edits,
        normalize_numbers=args.normalize_numbers,
        normalize_known_names=args.normalize_known_names,
        max_non_number_edits_for_numbers=args.max_non_number_edits_for_numbers,
        max_non_number_edits_for_names=args.max_non_number_edits_for_names,
        enable_refine_passes=args.enable_refine_passes,
        force_punctuate=args.force_punctuate,
        keep_alive=args.keep_alive,
        single_pass=args.single_pass,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        llama_n_batch=args.llama_n_batch,
    )


def _build_norm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "backend": args.backend,
        "base_url": args.ollama_url,
        "model": args.model,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "num_ctx": args.num_ctx,
        "allow_word_edits": args.allow_word_edits,
        "normalize_numbers": args.normalize_numbers,
        "normalize_known_names": args.normalize_known_names,
        "max_non_number_edits_for_numbers": args.max_non_number_edits_for_numbers,
        "max_non_number_edits_for_names": args.max_non_number_edits_for_names,
        "enable_refine_passes": args.enable_refine_passes,
        "force_punctuate": args.force_punctuate,
        "keep_alive": args.keep_alive,
        "single_pass": args.single_pass,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "llama_n_batch": args.llama_n_batch,
    }


def _worker_init(norm_kwargs: Dict[str, Any]) -> None:
    global _WORKER_NORM_KWARGS
    _WORKER_NORM_KWARGS = norm_kwargs
    if norm_kwargs["backend"] == "llama_cpp":
        model_path = norm_kwargs.get("model_path")
        if model_path is None:
            raise ValueError("--model-path is required when --backend llama_cpp")
        _load_llama_model(
            model_path=Path(str(model_path)),
            num_ctx=int(norm_kwargs["num_ctx"]),
            n_gpu_layers=int(norm_kwargs["n_gpu_layers"]),
            n_threads=int(norm_kwargs["n_threads"]),
            llama_n_batch=int(norm_kwargs["llama_n_batch"]),
        )


def _worker_normalize(raw_text: str) -> Tuple[str, Optional[str]]:
    if _WORKER_NORM_KWARGS is None:
        raise RuntimeError("Worker not initialized.")
    kwargs = dict(_WORKER_NORM_KWARGS)
    try:
        normalized = call_qwen_normalize(raw_text=raw_text, **kwargs)
        return normalized, None
    except Exception as exc:
        return raw_text, str(exc)


def run_sample(args: argparse.Namespace) -> None:
    input_format = detect_input_format(args.input.resolve(), args.input_format)
    sample = reservoir_sample_records(
        args.input.resolve(),
        sample_size=args.sample_size,
        seed=args.sample_seed,
        min_tokens=args.sample_min_tokens,
        input_format=input_format,
        jsonl_text_field=args.jsonl_text_field,
        jsonl_id_field=args.jsonl_id_field,
    )
    if not sample:
        raise RuntimeError("No eligible sample rows found. Reduce --sample-min-tokens.")

    rows: List[Dict[str, object]] = []
    changed_count = 0
    punct_added_count = 0
    error_count = 0

    for idx, (line_no, key, raw_text) in enumerate(sample, start=1):
        try:
            normalized_text = _normalize(args, raw_text)
        except Exception as exc:
            error_count += 1
            normalized_text = raw_text
            print(f"[sample][warn] line={line_no} fallback raw. error: {exc}")

        metrics = evaluate_pair(raw_text, normalized_text, allow_number_transforms=args.normalize_numbers)
        changed_count += int(bool(metrics["changed"]))
        punct_added_count += int(metrics["norm_punct"] > metrics["raw_punct"])
        rows.append({
            "sample_idx": idx,
            "line_no": line_no,
            "key": key,
            "input_text": raw_text,
            "normalized_text": normalized_text,
            "changed": metrics["changed"],
            "word_preserved": metrics["word_preserved"],
            "raw_tokens": metrics["raw_tokens"],
            "norm_tokens": metrics["norm_tokens"],
            "raw_punct": metrics["raw_punct"],
            "norm_punct": metrics["norm_punct"],
        })
        print(f"[sample] {idx}/{len(sample)} done: line={line_no}")
        print(f"  IN : {raw_text}")
        print(f"  OUT: {normalized_text}")

    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sample_output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "line_no", "key", "input_text", "normalized_text",
                        "changed", "word_preserved", "raw_tokens", "norm_tokens",
                        "raw_punct", "norm_punct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[sample] wrote CSV: {args.sample_output}")
    print(
        f"[sample] changed={changed_count}/{len(rows)}, "
        f"punct_added={punct_added_count}/{len(rows)}, errors={error_count}"
    )


def run_full(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    checkpoint_path = args.checkpoint.resolve()
    input_format = detect_input_format(input_path, args.input_format)
    output_format = "jsonl" if input_format == "jsonl" else "tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    existing_output_lines = count_lines(output_path)
    if args.start_line is not None:
        resume_after_line = max(0, args.start_line - 1)
    else:
        resume_after_line = existing_output_lines

    processed_now = 0
    submitted_now = 0
    submitted_now = 0
    error_count = 0
    started_at = time.time()

    with input_path.open("r", encoding="utf-8") as input_f, \
         output_path.open("a", encoding="utf-8") as output_f:

        for line_no, line in enumerate(input_f, start=1):
            if line_no <= resume_after_line:
                continue
            if args.limit is not None and submitted_now >= args.limit:
                break

            key, raw_text, record = split_record(
                line,
                line_no,
                input_format=input_format,
                jsonl_text_field=args.jsonl_text_field,
                jsonl_id_field=args.jsonl_id_field,
            )
            try:
                normalized_text = _normalize(args, raw_text)
            except Exception as exc:
                error_count += 1
                normalized_text = raw_text
                print(f"[full][warn] line={line_no} fallback raw. error: {exc}")

            output_f.write(format_output_line(
                key=key,
                normalized_text=normalized_text,
                record=record,
                output_format=output_format,
                jsonl_text_field=args.jsonl_text_field,
            ))
            processed_now += 1
            if processed_now % args.flush_every == 0:
                output_f.flush()

            if processed_now % args.save_every == 0:
                atomic_write_json(checkpoint_path, {
                    "status": "running",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "model": args.model,
                    "last_input_line": line_no,
                    "processed_now": processed_now,
                    "error_count": error_count,
                    "output_lines": resume_after_line + processed_now,
                    "updated_at_unix": time.time(),
                })

            if processed_now % args.print_every == 0:
                elapsed = time.time() - started_at
                rate = processed_now / elapsed if elapsed > 0 else 0.0
                print(
                    f"[full] processed_now={processed_now}, "
                    f"last_input_line={line_no}, rate={rate:.2f} rows/sec, "
                    f"errors={error_count}"
                )

    final_output_lines = existing_output_lines + processed_now
    atomic_write_json(checkpoint_path, {
        "status": "completed",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model": args.model,
        "last_input_line": resume_after_line + processed_now,
        "processed_now": processed_now,
        "error_count": error_count,
        "output_lines": final_output_lines,
        "updated_at_unix": time.time(),
    })
    print(
        f"[full] done. processed_now={processed_now}, "
        f"output_lines={final_output_lines}, errors={error_count}, output={output_path}"
    )


def run_full_parallel(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    checkpoint_path = args.checkpoint.resolve()
    input_format = detect_input_format(input_path, args.input_format)
    output_format = "jsonl" if input_format == "jsonl" else "tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    existing_output_lines = count_lines(output_path)
    if args.start_line is not None:
        resume_after_line = max(0, args.start_line - 1)
    else:
        resume_after_line = existing_output_lines

    processed_now = 0
    submitted_now = 0
    error_count = 0
    started_at = time.time()
    next_line_to_write = resume_after_line + 1
    max_inflight = max(args.parallel_lines * 6, args.parallel_lines)

    norm_kwargs = _build_norm_kwargs(args)
    pending: Dict[int, Tuple[concurrent.futures.Future, str, Optional[Dict[str, Any]]]] = {}

    with input_path.open("r", encoding="utf-8") as input_f, \
         output_path.open("a", encoding="utf-8") as output_f, \
         concurrent.futures.ProcessPoolExecutor(
             max_workers=args.parallel_lines,
             initializer=_worker_init,
             initargs=(norm_kwargs,),
             mp_context=mp.get_context("spawn"),
         ) as pool:

        for line_no, line in enumerate(input_f, start=1):
            if line_no <= resume_after_line:
                continue
            if args.limit is not None and submitted_now >= args.limit:
                break

            key, raw_text, record = split_record(
                line,
                line_no,
                input_format=input_format,
                jsonl_text_field=args.jsonl_text_field,
                jsonl_id_field=args.jsonl_id_field,
            )
            pending[line_no] = (pool.submit(_worker_normalize, raw_text), key, record)
            submitted_now += 1

            while len(pending) >= max_inflight and next_line_to_write in pending:
                fut, key_w, record_w = pending.pop(next_line_to_write)
                normalized_text, err = fut.result()
                if err:
                    error_count += 1
                    print(f"[full][warn] line={next_line_to_write} fallback raw. error: {err}")
                output_f.write(format_output_line(
                    key=key_w,
                    normalized_text=normalized_text,
                    record=record_w,
                    output_format=output_format,
                    jsonl_text_field=args.jsonl_text_field,
                ))
                processed_now += 1
                next_line_to_write += 1

                if processed_now % args.flush_every == 0:
                    output_f.flush()
                if processed_now % args.save_every == 0:
                    atomic_write_json(checkpoint_path, {
                        "status": "running",
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "model": args.model,
                        "last_input_line": next_line_to_write - 1,
                        "processed_now": processed_now,
                        "error_count": error_count,
                        "output_lines": resume_after_line + processed_now,
                        "updated_at_unix": time.time(),
                    })
                if processed_now % args.print_every == 0:
                    elapsed = time.time() - started_at
                    rate = processed_now / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[full] processed_now={processed_now}, "
                        f"last_input_line={next_line_to_write - 1}, rate={rate:.2f} rows/sec, "
                        f"errors={error_count}, parallel_lines={args.parallel_lines}"
                    )

        while next_line_to_write in pending:
            fut, key_w, record_w = pending.pop(next_line_to_write)
            normalized_text, err = fut.result()
            if err:
                error_count += 1
                print(f"[full][warn] line={next_line_to_write} fallback raw. error: {err}")
            output_f.write(format_output_line(
                key=key_w,
                normalized_text=normalized_text,
                record=record_w,
                output_format=output_format,
                jsonl_text_field=args.jsonl_text_field,
            ))
            processed_now += 1
            next_line_to_write += 1

            if processed_now % args.flush_every == 0:
                output_f.flush()
            if processed_now % args.save_every == 0:
                atomic_write_json(checkpoint_path, {
                    "status": "running",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "model": args.model,
                    "last_input_line": next_line_to_write - 1,
                    "processed_now": processed_now,
                    "error_count": error_count,
                    "output_lines": resume_after_line + processed_now,
                    "updated_at_unix": time.time(),
                })
            if processed_now % args.print_every == 0:
                elapsed = time.time() - started_at
                rate = processed_now / elapsed if elapsed > 0 else 0.0
                print(
                    f"[full] processed_now={processed_now}, "
                    f"last_input_line={next_line_to_write - 1}, rate={rate:.2f} rows/sec, "
                    f"errors={error_count}, parallel_lines={args.parallel_lines}"
                )

    final_output_lines = existing_output_lines + processed_now
    atomic_write_json(checkpoint_path, {
        "status": "completed",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model": args.model,
        "last_input_line": resume_after_line + processed_now,
        "processed_now": processed_now,
        "error_count": error_count,
        "output_lines": final_output_lines,
        "updated_at_unix": time.time(),
    })
    print(
        f"[full] done. processed_now={processed_now}, "
        f"output_lines={final_output_lines}, errors={error_count}, output={output_path}, "
        f"parallel_lines={args.parallel_lines}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    if args.backend == "ollama":
        args.keep_alive = normalize_keep_alive(args.keep_alive)
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be > 0.")
    if args.parallel_lines <= 0:
        raise ValueError("--parallel-lines must be > 0.")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.backend == "ollama":
        check_ollama_ready(args.ollama_url, args.model)
        if args.warmup_model:
            warmup_ollama_model(
                base_url=args.ollama_url,
                model=args.model,
                num_ctx=args.num_ctx,
                keep_alive=args.keep_alive,
            )
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required when --backend llama_cpp")
        _load_llama_model(
            model_path=args.model_path,
            num_ctx=args.num_ctx,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.n_threads,
            llama_n_batch=args.llama_n_batch,
        )

    if args.mode == "sample":
        run_sample(args)
    else:
        if args.backend == "llama_cpp" and args.parallel_lines > 1:
            run_full_parallel(args)
        else:
            run_full(args)


if __name__ == "__main__":
    main()
