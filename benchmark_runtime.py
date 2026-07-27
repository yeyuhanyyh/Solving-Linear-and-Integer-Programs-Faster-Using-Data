#!/usr/bin/env python3
"""Rerun Figure 3-4 timings with the packaged algorithms."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNERS = ROOT / "code" / "runners"
TIMING = ROOT / "results" / "timing"


def run(name: str, *arguments: str) -> None:
    command = [sys.executable, str(RUNNERS / name), *arguments]
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("all", "figure3", "figure4"),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--legacy-ours",
        action="store_true",
        help="Use the archived ellipsoid implementation for Figure 4 timing.",
    )
    args = parser.parse_args()

    common = (
        "--repeats",
        str(args.repeats),
        "--out-dir",
        str(TIMING),
    )
    if args.force:
        common = (*common, "--force")


    if args.scope in ("all", "figure3"):
        run("run_figure3_ours_timing.py", *common)
        run("plot_figure3_ours_runtime.py")
    if args.scope in ("all", "figure4"):
        if args.legacy_ours:
            run(
                "run_figure4_matched_k_timing.py",
                *common,
                "--n-list",
                "20",
                "--device",
                str(args.device),
            )
        else:
            arguments = ("--repeats", str(args.repeats))
            if args.force:
                arguments = (*arguments, "--force")
            run("run_figure4_ours_timing.py", *arguments)

    if args.scope in ("all", "figure4") and int(args.repeats) != 3:
        print(
            "Figure 4 timing CSV generated. Its runtime plot requires the "
            "documented three-repeat protocol.",
            flush=True,
        )
        return
    if args.scope in ("all", "figure4") and args.legacy_ours:
        run("plot_runtime.py")
    if args.scope in ("all", "figure4") and not args.legacy_ours:
        run("plot_figure4_ours_runtime.py")


if __name__ == "__main__":
    main()

