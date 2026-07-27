#!/usr/bin/env python3
"""Re-evaluate bundled checkpoints, plot, or rerun Figure 2-4 experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNERS = ROOT / "code" / "runners"
RESULTS = ROOT / "results" / "fresh"


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Retrain every plotted model from the legacy source code.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Render bundled raw CSV files without re-evaluating checkpoints.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "figure2", "figure3", "figure4"),
        default="all",
        help="Limit a de-novo rerun to one figure.",
    )
    parser.add_argument(
        "--original-netlib-k50",
        action="store_true",
        help=(
            "Explicit alias for the default GROW7/SC205 K=50 checkpoint "
            "re-evaluation."
        ),
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.plot_only and args.rerun:
        parser.error("--plot-only cannot be combined with --rerun")
    if args.plot_only and args.original_netlib_k50:
        parser.error("--plot-only cannot be combined with --original-netlib-k50")
    if args.original_netlib_k50 and args.rerun:
        parser.error("--original-netlib-k50 cannot be combined with --rerun")

    if not args.rerun and not args.plot_only:
        run(
            [
                sys.executable,
                str(RUNNERS / "reevaluate_original_netlib_k50.py"),
                "--update-main-results",
            ]
        )
    if args.rerun and args.only in ("all", "figure2"):
        run(
            [
                sys.executable,
                str(RUNNERS / "run_figure2.py"),
                "--out-dir",
                str(RESULTS),
                "--force",
            ]
        )
    if args.rerun and args.only in ("all", "figure3"):
        run(
            [
                sys.executable,
                str(RUNNERS / "run_figure3_timing.py"),
                "--repeats",
                "1",
                "--out-dir",
                str(RESULTS),
                "--force",
            ]
        )
    if args.rerun and args.only in ("all", "figure4"):
        run(
            [
                sys.executable,
                str(RUNNERS / "run_figure4_timing.py"),
                "--repeats",
                "1",
                "--full-grid",
                "--out-dir",
                str(RESULTS),
                "--force",
            ]
        )

    run([sys.executable, str(ROOT / "plot_fresh_results.py")])


if __name__ == "__main__":
    main()
