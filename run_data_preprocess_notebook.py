#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Execute data_preprocess.ipynb cells in production order: CELL6(VAD) -> CELL5(Denoise)."
    )
    p.add_argument(
        "--notebook",
        default="/home/vietnam/voice_to_text/data_preprocess.ipynb",
        help="Path to notebook file.",
    )
    p.add_argument(
        "--cell-order",
        default="1,0",
        help="Comma-separated cell indexes to execute in order. Default: 1,0",
    )
    return p.parse_args()


def display(x):
    print(x)


def main():
    args = parse_args()
    nb_path = Path(args.notebook)
    if not nb_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    order = []
    for x in args.cell_order.split(","):
        x = x.strip()
        if not x:
            continue
        order.append(int(x))

    # Execute notebook cells inside the real __main__ module namespace.
    # This keeps dynamically defined functions pickle-resolvable for
    # ProcessPoolExecutor (needed by VAD process mode).
    g = sys.modules["__main__"].__dict__
    g["display"] = display
    print("RUN_START", datetime.now().isoformat())
    print("NOTEBOOK", str(nb_path))
    print("CELL_ORDER", order)

    for idx in order:
        t0 = time.time()
        print("")
        print("===== START CELL", idx, "=====")
        src = "".join(nb["cells"][idx]["source"])
        exec(src, g, g)
        print("===== END CELL", idx, "seconds=", round(time.time() - t0, 3), "=====")

    print("RUN_END", datetime.now().isoformat())


if __name__ == "__main__":
    main()
