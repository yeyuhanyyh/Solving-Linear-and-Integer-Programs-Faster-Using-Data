#!/usr/bin/env python3
"""Time Ours and combine it with the frozen matched-K FCNN-c rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CORE_CODE = ROOT / "code" / "core"
EXPERIMENT_CODE = ROOT / "code" / "experiments"
SCRIPT_DIR = Path(__file__).resolve().parent
for code_path in (CORE_CODE, EXPERIMENT_CODE, SCRIPT_DIR):
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

import _run_figure4_matched_k_timing_impl as legacy  # noqa: E402
import compare_fixedX_family_suite as cff  # noqa: E402
import ours  # noqa: E402


CASES = tuple(legacy.CASE_ORDER)
OURS = "Ours"
FCNN = "FCNNc"
CACHE = ROOT / "results" / "timing" / "cache"
CACHE_CSV = CACHE / "figure4_fcnn_baseline_legacy.csv"
CACHE_MANIFEST = CACHE / "figure4_fcnn_baseline_legacy_manifest.json"
OUTPUT = ROOT / "results" / "timing" / "figure4_ours_matched_k_timing_raw.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stats(values: list[float]) -> tuple[float, float, int]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan"), 0
    sd = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    return float(np.mean(clean)), sd, int(clean.size)


def load_cache() -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    actual_hash = sha256_file(CACHE_CSV)
    expected_hash = str(manifest["cache_sha256"]).upper()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"FCNN-c cache hash mismatch: {actual_hash} != {expected_hash}"
        )
    baseline = pd.read_csv(CACHE_CSV)
    if len(baseline) != 3 * len(CASES):
        raise RuntimeError("FCNN-c cache does not contain 3 x 8 rows")
    if set(baseline["method"].astype(str)) != {FCNN}:
        raise RuntimeError("Unexpected method in FCNN-c cache")
    return baseline, manifest


def half_size_cases() -> dict[str, object]:
    cases = legacy.figure4.get_cases()
    mincostflow = copy.deepcopy(cases["mincostflow"])
    mincostflow.args_updates = dict(mincostflow.args_updates)
    mincostflow.args_updates["n_edges"] = 180
    cases["mincostflow"] = mincostflow
    return cases


def cached_row(
    baseline: pd.DataFrame,
    repeat: int,
    case_name: str,
) -> pd.Series:
    selected = baseline[
        pd.to_numeric(baseline["repeat"]).eq(int(repeat))
        & baseline["case"].astype(str).eq(case_name)
        & pd.to_numeric(baseline["n_samples"]).eq(20)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Missing cached row for repeat={repeat}, case={case_name}"
        )
    return selected.iloc[0]


def run_ours(
    case_spec: object,
    case_name: str,
    repeat: int,
    cached: pd.Series,
) -> dict[str, object]:
    args = legacy.experiment_args("cpu")
    preparation_start = time.perf_counter()
    data = legacy.figure4.prepare_fixed_pool_case(case_spec, args, 20)
    preparation_time = time.perf_counter() - preparation_start
    train = [
        cff.convert_to_general_projection_instance(instance)
        for instance in data.bundle.train[:20]
    ]
    test = [
        cff.convert_to_general_projection_instance(instance)
        for instance in data.bundle.test
    ]
    d_full = int(train[0].n_vars)
    target_k = int(cached["K"])
    if d_full != int(cached["full_dimension"]):
        raise RuntimeError(
            f"{case_name}: prepared d={d_full}, "
            f"cached d={cached['full_dimension']}"
        )
    learned = ours.fit(
        train,
        target_dimension=target_k,
        seed=17 + 1009 * int(repeat) + int(data.case.seed_offset),
    )
    evaluated = ours.evaluate(test, learned)
    ratio_mean, ratio_sd, ratio_n = stats(
        [float(row["objective_ratio"]) for row in evaluated]
    )
    solve_mean, solve_sd, _ = stats(
        [float(row["solver_time_s"]) for row in evaluated]
    )
    prep_mean, prep_sd, _ = stats(
        [float(row["preparation_time_s"]) for row in evaluated]
    )
    online_mean, online_sd, _ = stats(
        [float(row["online_time_s"]) for row in evaluated]
    )
    print(
        f"[Ours] repeat={repeat}, case={case_name}, d={d_full}, "
        f"raw_rank={learned.raw_rank}, K={target_k}, "
        f"learning={learned.learning_time_s:.4f}s",
        flush=True,
    )
    return {
        "figure": "Figure 4",
        "protocol": "ours_one_full_lp_per_sample_then_svd",
        "repeat": int(repeat),
        "case": case_name,
        "method": OURS,
        "source_method": OURS,
        "rho": 0.1,
        "n_samples": 20,
        "K": target_k,
        "matched_K": target_k,
        "full_dimension": d_full,
        "learning_time_s": float(learned.learning_time_s),
        "data_preparation_time_s": float(preparation_time),
        "learning_plus_preparation_time_s": float(
            learned.learning_time_s + preparation_time
        ),
        "solve_time_mean_s": solve_mean,
        "solve_time_se_s": solve_sd,
        "projection_time_mean_s": prep_mean,
        "projection_time_se_s": prep_sd,
        "online_total_time_mean_s": online_mean,
        "online_total_time_se_s": online_sd,
        "full_lp_solve_time_mean_s": float(
            cached["full_lp_solve_time_mean_s"]
        ),
        "full_lp_solve_time_se_s": float(
            cached["full_lp_solve_time_se_s"]
        ),
        "full_lp_solve_observations": int(
            cached["full_lp_solve_observations"]
        ),
        "objective_ratio_mean": ratio_mean,
        "objective_ratio_se": ratio_sd,
        "solve_observations": ratio_n,
        "lp_backend": "scipy-highs-cpu",
        "raw_learned_rank": int(learned.raw_rank),
        "basis_padding_columns": target_k - int(learned.raw_rank),
        "training_full_lp_solves": int(learned.training_lp_solves),
        "training_full_lp_solver_s": float(
            learned.training_solver_time_s
        ),
        "training_schedule": (
            "one compact full LP per training sample, then one "
            "optimizer-difference SVD"
        ),
        "implementation": "optimizer_difference_span",
        "unique_optimizer_assumption": True,
        "timing_cached": False,
        "timing_provenance_id": "fresh_ours_2026_07_27",
    }


def validate(
    frame: pd.DataFrame,
    original_baseline: pd.DataFrame,
) -> None:
    if len(frame) != 48:
        raise RuntimeError(f"Expected 48 rows, found {len(frame)}")
    for _, group in frame.groupby(["repeat", "case", "n_samples"]):
        if set(group["method"].astype(str)) != {OURS, FCNN}:
            raise RuntimeError("A timing group is incomplete")
        if group["K"].astype(int).nunique() != 1:
            raise RuntimeError("Ours and FCNN-c use different K")
        if group["full_dimension"].astype(int).nunique() != 1:
            raise RuntimeError("Ours and FCNN-c use different full d")
    ours_rows = frame[frame["method"].astype(str).eq(OURS)]
    if (ours_rows["raw_learned_rank"].astype(int) > ours_rows["K"].astype(int)).any():
        raise RuntimeError("A learned rank exceeds the matched K")

    original_columns = list(original_baseline.columns)
    merged_baseline = frame[
        frame["method"].astype(str).eq(FCNN)
    ][original_columns]
    keys = ["repeat", "case", "n_samples", "K"]
    pd.testing.assert_frame_equal(
        original_baseline.sort_values(keys).reset_index(drop=True),
        merged_baseline.sort_values(keys).reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if int(args.repeats) != 3:
        raise ValueError("The committed baseline cache contains 3 repeats")
    if OUTPUT.exists() and not args.force:
        print(f"Reusing existing Ours timing: {OUTPUT}")
        return

    baseline, manifest = load_cache()
    cases = half_size_cases()
    ours_rows: list[dict[str, object]] = []
    for repeat in range(1, 4):
        for case_name in CASES:
            ours_rows.append(
                run_ours(
                    cases[case_name],
                    case_name,
                    repeat,
                    cached_row(baseline, repeat, case_name),
                )
            )

    cached = baseline.copy()
    cached["timing_cached"] = True
    cached["timing_provenance_id"] = str(manifest["provenance_id"])
    cached["baseline_source_sha256"] = str(manifest["source_sha256"])
    fresh = pd.DataFrame(ours_rows)
    fresh["baseline_source_sha256"] = str(manifest["source_sha256"])
    combined = pd.concat([fresh, cached], ignore_index=True, sort=False)
    validate(combined, baseline)
    combined = combined.sort_values(
        ["repeat", "case", "method"], kind="stable"
    ).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"Saved Figure 4 Ours timing to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
