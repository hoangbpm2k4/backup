#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuously delete source files listed in used_source_files.*.txt logs.")
    p.add_argument("--run-dir", type=Path, required=True, help="e.g. /home/vietnam/voice_to_text/data/overlap_runs")
    p.add_argument("--poll-sec", type=float, default=3.0)
    p.add_argument("--state-file", type=Path, default=None)
    p.add_argument("--activity-log", type=Path, default=None)
    p.add_argument("--stop-file", type=Path, default=None, help="If this file exists, daemon exits cleanly.")
    return p.parse_args()


def list_logs(run_dir: Path) -> List[Tuple[Path, Path]]:
    out: List[Tuple[Path, Path]] = []
    for sub in ["single", "overlap2", "overlap3"]:
        root = run_dir / sub
        if not root.exists():
            continue
        for done_marker in sorted(root.rglob("used_source_files.*.txt.done.json")):
            log_path = Path(str(done_marker).replace(".done.json", ""))
            if log_path.exists():
                out.append((log_path, done_marker))
    return out


def load_state(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, st: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip("\n") + "\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    state_file = args.state_file or (run_dir / "logs" / "deleter_state.json")
    activity_log = args.activity_log or (run_dir / "logs" / "deleter_activity.log")
    stop_file = args.stop_file or (run_dir / "logs" / "deleter.stop")

    state = load_state(state_file)
    deleted = 0
    missing = 0
    failed = 0
    cycles = 0

    log_line(activity_log, f"[START] run_dir={run_dir.as_posix()} poll_sec={args.poll_sec}")
    while True:
        cycles += 1
        if stop_file.exists():
            log_line(activity_log, "[STOP] stop-file detected, exiting")
            break

        logs = list_logs(run_dir)
        for lp, done_marker in logs:
            # Process only finalized logs; use marker mtime in state key to avoid stale offsets.
            mtime_ns = done_marker.stat().st_mtime_ns
            key = f"{lp.as_posix()}::{mtime_ns}"
            last = int(state.get(key, 0))
            try:
                with lp.open("rb") as f:
                    f.seek(last)
                    chunk = f.read()
                    state[key] = f.tell()
            except Exception as e:
                log_line(activity_log, f"[WARN] cannot read {key}: {e}")
                continue

            if not chunk:
                continue
            for line in chunk.decode("utf-8", errors="ignore").splitlines():
                p = line.strip()
                if not p:
                    continue
                try:
                    os.unlink(p)
                    deleted += 1
                except FileNotFoundError:
                    missing += 1
                except Exception as e:
                    failed += 1
                    log_line(activity_log, f"[ERR] delete failed {p}: {e}")

        if cycles % 20 == 0:
            log_line(activity_log, f"[STAT] deleted={deleted} missing={missing} failed={failed} tracked_logs={len(logs)}")
            save_state(state_file, state)

        time.sleep(args.poll_sec)

    save_state(state_file, state)
    log_line(activity_log, f"[DONE] deleted={deleted} missing={missing} failed={failed}")


if __name__ == "__main__":
    main()
