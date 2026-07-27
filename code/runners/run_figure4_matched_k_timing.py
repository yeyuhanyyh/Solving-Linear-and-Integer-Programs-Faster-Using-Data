#!/usr/bin/env python3
"""Run matched-K timing with the documented half-size MinCostFlow case."""

from __future__ import annotations

import copy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _run_figure4_matched_k_timing_impl as implementation  # noqa: E402


ORIGINAL_GET_CASES = implementation.figure4.get_cases


def half_size_cases() -> dict[str, object]:
    cases = ORIGINAL_GET_CASES()
    mincostflow = copy.deepcopy(cases["mincostflow"])
    mincostflow.args_updates = dict(mincostflow.args_updates)
    mincostflow.args_updates["n_edges"] = 180
    cases["mincostflow"] = mincostflow
    return cases


def main() -> None:
    implementation.figure4.get_cases = half_size_cases
    implementation.main()


if __name__ == "__main__":
    main()
