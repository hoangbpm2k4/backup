#!/usr/bin/env python3
import argparse
import csv
import math
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tqdm import tqdm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download YouTube URLs from CSV and normalize to 16kHz mono WAV."
        )
    )
    parser.add_argument(
        "--csv",
        default="link_ytb.csv",
        help="CSV file path that contains video URLs.",
    )
    parser.add_argument(
        "--url-column",
        default="video_url",
        help="Column name in CSV that contains YouTube URLs.",
    )
    parser.add_argument(
        "--cookies",
        default="youtube_auth_cookies.txt",
        help="Path to Netscape cookies.txt for YouTube auth.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/youtube_wav_16k",
        help="Output directory for normalized WAV files.",
    )
    parser.add_argument(
        "--archive-file",
        default="data/youtube_wav_16k.archive.txt",
        help="yt-dlp archive file to support resume and skip already downloaded URLs.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Target channels. Use 1 for mono.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers.",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Only process first N URLs (for testing).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=15,
        help="Retries per URL.",
    )
    return parser


def ensure_dependencies() -> None:
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency yt-dlp. Install with: pip install yt-dlp"
        ) from exc

    if shutil.which("ffmpeg") is None:
        raise SystemExit("Missing ffmpeg in PATH. Install ffmpeg first.")


def load_urls_from_csv(csv_path: Path, url_column: str) -> List[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    urls: List[str] = []
    seen = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or url_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{url_column}' not found in CSV. Columns: {reader.fieldnames}"
            )
        for row in reader:
            url = (row.get(url_column) or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return urls


def select_urls(urls: Iterable[str], max_urls: Optional[int]) -> List[str]:
    out = list(urls)
    if max_urls is None:
        return out
    if max_urls <= 0:
        raise ValueError(f"max_urls must be > 0, got {max_urls}")
    return out[:max_urls]


def split_urls(urls: Sequence[str], num_parts: int) -> List[List[str]]:
    if num_parts <= 1:
        return [list(urls)]
    chunk_size = max(1, math.ceil(len(urls) / num_parts))
    return [list(urls[i : i + chunk_size]) for i in range(0, len(urls), chunk_size)]


def archive_entries(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def merge_archives(target_archive: Path, archive_files: Sequence[Path]) -> None:
    all_entries = []
    seen = set()
    for p in archive_files:
        for line in archive_entries(p):
            if line in seen:
                continue
            seen.add(line)
            all_entries.append(line)
    target_archive.parent.mkdir(parents=True, exist_ok=True)
    if all_entries:
        target_archive.write_text("\n".join(all_entries) + "\n", encoding="utf-8")
    elif not target_archive.exists():
        target_archive.touch()


def build_ydl_options(
    *,
    output_dir: Path,
    archive_file: Path,
    cookies_file: Optional[Path],
    sample_rate: int,
    channels: int,
    retries: int,
    progress_hook=None,
):
    # Force audio-only download and normalize output to WAV 16k mono.
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "noplaylist": True,
        "ignoreerrors": False,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "outtmpl": str(output_dir / "%(uploader)s" / "%(id)s_%(title).180B.%(ext)s"),
        "download_archive": str(archive_file),
        "retries": retries,
        "fragment_retries": retries,
        "extractor_retries": retries,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": [
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
        ],
        "extractor_args": {
            "youtube": {
                "player_client": ["default"],
            }
        },
        "geo_bypass": True,
        "geo_bypass_country": "VN",
    }
    if cookies_file:
        tmp_cookie = tempfile.NamedTemporaryFile(
            prefix="yt_cookie_", suffix=".txt", delete=False
        )
        shutil.copy2(cookies_file.resolve(), tmp_cookie.name)
        opts["cookiefile"] = tmp_cookie.name
    if progress_hook is not None:
        opts["progress_hooks"] = [progress_hook]
    return opts


def make_progress_hook(progress_bar, seen_items, lock):
    def _hook(update):
        if update.get("status") != "finished":
            return
        info = update.get("info_dict") or {}
        key = info.get("webpage_url") or info.get("id") or update.get("filename")
        if key is None:
            return
        with lock:
            if key in seen_items:
                return
            seen_items.add(key)
            if progress_bar.n < progress_bar.total:
                progress_bar.update(1)

    return _hook


def should_skip(error: Exception) -> bool:
    text = str(error).lower()
    markers = (
        "video unavailable",
        "private video",
        "this video is unavailable",
        "members-only",
        "not made this video available in your country",
        "requested format is not available",
        "only images are available",
        "sign in to confirm you're not a bot",
        "premieres in",
        "this live event will begin in",
    )
    return any(x in text for x in markers)


def download_chunk(
    *,
    worker_id: int,
    urls: Sequence[str],
    output_dir: Path,
    archive_file: Path,
    cookies_file: Optional[Path],
    sample_rate: int,
    channels: int,
    retries: int,
    progress_hook=None,
) -> None:
    import yt_dlp

    ydl_opts = build_ydl_options(
        output_dir=output_dir,
        archive_file=archive_file,
        cookies_file=cookies_file,
        sample_rate=sample_rate,
        channels=channels,
        retries=retries,
        progress_hook=progress_hook,
    )
    print(f"Worker {worker_id}: {len(urls)} urls")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                rc = ydl.download([url])
            except Exception as exc:
                if should_skip(exc):
                    print(f"Worker {worker_id}: skip {url} ({exc})")
                    continue
                raise
            if rc not in (None, 0):
                raise RuntimeError(f"Worker {worker_id}: yt-dlp exit code {rc} for {url}")


def remove_non_wav_files(output_dir: Path, preserve: Sequence[Path]) -> int:
    preserve_set = {p.resolve() for p in preserve}
    removed = 0
    for p in output_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.resolve() in preserve_set:
            continue
        if p.suffix.lower() == ".wav":
            continue
        # Keep only final wav files as requested.
        p.unlink(missing_ok=True)
        removed += 1
    return removed


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dependencies()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    archive_file = Path(args.archive_file)
    cookies_file = Path(args.cookies) if args.cookies else None

    urls = load_urls_from_csv(csv_path, args.url_column)
    urls = select_urls(urls, args.max_urls)
    if not urls:
        raise SystemExit("No URLs found to download.")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    if not archive_file.exists():
        archive_file.touch()

    print(f"Total URLs: {len(urls)}")
    print(f"Output dir: {output_dir.resolve()}")
    print(
        f"Normalize: {args.sample_rate} Hz, {args.channels} channel(s), WAV only"
    )

    initial_done = min(len(archive_entries(archive_file)), len(urls))
    progress_lock = threading.Lock()
    seen_items = set()

    worker_count = min(args.workers, len(urls))
    chunks = split_urls(urls, worker_count)

    temp_dir = archive_file.parent / ".tmp_archives"
    temp_dir.mkdir(parents=True, exist_ok=True)

    worker_archives: List[Path] = []
    seed_entries = archive_entries(archive_file)
    for i in range(len(chunks)):
        ap = temp_dir / f"download_archive.worker{i}.txt"
        if seed_entries:
            ap.write_text("\n".join(seed_entries) + "\n", encoding="utf-8")
        else:
            ap.touch()
        worker_archives.append(ap)

    completed = False
    try:
        with tqdm(total=len(urls), initial=initial_done, desc="Download wav", unit="url") as pbar:
            with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
                futures = [
                    ex.submit(
                        download_chunk,
                        worker_id=i,
                        urls=chunk,
                        output_dir=output_dir,
                        archive_file=worker_archives[i],
                        cookies_file=cookies_file,
                        sample_rate=args.sample_rate,
                        channels=args.channels,
                        retries=args.retries,
                        progress_hook=make_progress_hook(pbar, seen_items, progress_lock),
                    )
                    for i, chunk in enumerate(chunks)
                ]
                for fut in as_completed(futures):
                    fut.result()
                    merge_archives(archive_file, worker_archives)
        completed = True
    finally:
        merge_archives(archive_file, worker_archives)
        if completed:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"Interrupted. Resume state kept in {temp_dir}")

    removed = remove_non_wav_files(output_dir, preserve=[archive_file])
    print(f"Done. Removed {removed} non-wav file(s).")


if __name__ == "__main__":
    main()
