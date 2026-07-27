#!/usr/bin/env python3
"""Time the optimizer-span Ours implementation and reuse Figure 3 baselines."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CORE_CODE = ROOT / "code" / "core"
EXPERIMENT_CODE = ROOT / "code" / "experiments"
for code_path in (CORE_CODE, EXPERIMENT_CODE):
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

import compare_fixedX_family_suite as cff  # noqa: E402
import ours  # noqa: E402
import run_dynamic_k_baseline_compare_four_lp as figure3  # noqa: E402


CASE_ORDER = ("packing", "sc205", "random_lp_A", "random_lp_C")
RHO_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
OURS = "OursEstC"
LEGACY = ROOT / "results" / "timing" / "figure3_timing_raw.csv"
OUTPUT_NAME = "figure3_ours_timing_raw.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def mean_se(values: list[float]) -> tuple[float, float, int]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return math.nan, math.nan, 0
    if clean.size == 1:
        return float(clean[0]), 0.0, 1
    return (
        float(np.mean(clean)),
        float(np.std(clean, ddof=1) / np.sqrt(clean.size)),
        int(clean.size),
    )


def load_legacy_baselines() -> tuple[pd.DataFrame, str]:
    legacy = pd.read_csv(LEGACY)
    required = {
        "repeat",
        "case",
        "method",
        "source_method",
        "rho",
        "n_samples",
        "K",
        "learning_time_s",
        "solve_time_mean_s",
        "online_total_time_mean_s",
    }
    if not required.issubset(legacy.columns):
        raise RuntimeError("Legacy Figure 3 timing CSV is incomplete")
    if len(legacy) != 240:
        raise RuntimeError(f"Expected 240 legacy rows, found {len(legacy)}")
    baselines = legacy[~legacy["method"].astype(str).eq(OURS)].copy()
    if len(baselines) != 180:
        raise RuntimeError(
            f"Expected 180 frozen baseline rows, found {len(baselines)}"
        )
    if set(baselines["method"].astype(str)) != {
        "DataDriven",
        "Rand",
        "FCNNc",
    }:
        raise RuntimeError("Unexpected Figure 3 baseline method")
    for _, group in baselines.groupby(["case", "rho"], sort=False):
        if group["K"].astype(int).nunique() != 1:
            raise RuntimeError("Legacy Figure 3 baselines do not share K")
        if group["repeat"].astype(int).nunique() != 3:
            raise RuntimeError("Legacy Figure 3 baseline is missing repeats")
    return baselines, sha256_file(LEGACY)


def archived_k(
    baselines: pd.DataFrame,
    case_name: str,
    rho: float,
) -> int:
    rows = baselines[
        baselines["case"].astype(str).eq(case_name)
        & np.isclose(pd.to_numeric(baselines["rho"]), float(rho))
    ]
    values = rows["K"].astype(int).unique()
    if len(values) != 1:
        raise RuntimeError(f"Missing archived K for {case_name}, rho={rho}")
    return int(values[0])


def retained_indices(
    data: object,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray(data.bundle.Ctrain[:40], dtype=float)
    prior, _metadata, scores = (
        figure3.olddyn.ridge_ours.estimate_ridge_calibrated_prior(
            costs,
            costs,
            rho=float(rho),
            ridge_scale=1e-3,
            ridge_abs=1e-10,
            delta0=0.20,
            rule="conformal",
            support_mode="empirical_span",
            svd_tol=1e-10,
        )
    )
    retained = np.flatnonzero(
        scores <= float(prior.radius) ** 2 + 1e-10
    )
    if retained.size == 0 and scores.size:
        retained = np.asarray([int(np.argmin(scores))], dtype=int)
    return retained.astype(int), np.asarray(prior.mu, dtype=float)


def pad_to_at_least(
    representation: ours.Representation,
    target_dimension: int,
    seed: int,
) -> ours.Representation:
    if representation.raw_rank >= int(target_dimension):
        return representation
    start = time.perf_counter()
    basis = ours.pad_basis(
        representation.basis,
        int(target_dimension),
        int(seed),
    )
    return ours.Representation(
        anchor=representation.anchor,
        basis=basis,
        raw_rank=representation.raw_rank,
        training_lp_solves=representation.training_lp_solves,
        training_solver_time_s=representation.training_solver_time_s,
        learning_time_s=(
            representation.learning_time_s + time.perf_counter() - start
        ),
    )


def run_point(
    data: object,
    test: list[object],
    case_name: str,
    rho: float,
    repeat: int,
    target_k: int,
    preparation_time: float,
    baseline_hash: str,
) -> dict[str, object]:
    retained, prior_center = retained_indices(data, rho)
    retained_train = [
        cff.convert_to_general_projection_instance(
            data.bundle.train[int(index)]
        )
        for index in retained
    ]
    anchor_template = retained_train[0]
    if prior_center.shape == anchor_template.c.shape:
        anchor_cost = -prior_center
    else:
        n_slack = int(data.bundle.stdlp.n_slack)
        anchor_cost = -prior_center[n_slack:]
    if anchor_cost.shape != anchor_template.c.shape:
        raise RuntimeError("Prior-center cost has the wrong dimension")
    anchor = dataclasses.replace(
        anchor_template,
        c=anchor_cost,
        name=f"{case_name}_rho_{rho:.1f}_prior_center",
    )
    train = [anchor, *retained_train]
    seed = (
        int(figure3.OLD_SUITE_DYNAMIC_CONFIGS[case_name].seed)
        + int(data.case.seed_offset)
        + 1009 * int(repeat)
        + int(round(1000.0 * rho))
    )
    learned = ours.fit(train)
    learned = pad_to_at_least(learned, target_k, seed)
    evaluated = ours.evaluate(test, learned)

    ratio_mean, ratio_se, observations = mean_se(
        [float(row["objective_ratio"]) for row in evaluated]
    )
    solve_mean, solve_se, _ = mean_se(
        [float(row["solver_time_s"]) for row in evaluated]
    )
    projection_mean, projection_se, _ = mean_se(
        [float(row["preparation_time_s"]) for row in evaluated]
    )
    online_mean, online_se, _ = mean_se(
        [float(row["online_time_s"]) for row in evaluated]
    )
    actual_k = int(learned.basis.shape[1])
    print(
        f"[Figure 3 new Ours] repeat={repeat}, case={case_name}, "
        f"rho={rho:.1f}, retained={len(retained)}, "
        f"raw_rank={learned.raw_rank}, K={actual_k}, "
        f"training_LPs={learned.training_lp_solves}, "
        f"learning={learned.learning_time_s:.4f}s",
        flush=True,
    )
    return {
        "figure": "Figure 3",
        "protocol": "ours_anchor_plus_one_lp_per_retained_sample_then_svd",
        "repeat": int(repeat),
        "case": case_name,
        "method": OURS,
        "source_method": "Ours (optimizer span)",
        "rho": float(rho),
        "n_samples": 40,
        "K": actual_k,
        "archived_baseline_K": int(target_k),
        "dimension_matches_archived_baselines": actual_k == int(target_k),
        "learning_time_s": float(learned.learning_time_s),
        "data_preparation_time_s": float(preparation_time),
        "solve_time_mean_s": solve_mean,
        "solve_time_se_s": solve_se,
        "projection_time_mean_s": projection_mean,
        "projection_time_se_s": projection_se,
        "online_total_time_mean_s": online_mean,
        "online_total_time_se_s": online_se,
        "objective_ratio_mean": ratio_mean,
        "objective_ratio_se": ratio_se,
        "solve_observations": observations,
        "retained_training_samples": int(len(retained)),
        "raw_learned_rank": int(learned.raw_rank),
        "basis_padding_columns": actual_k - int(learned.raw_rank),
        "training_full_lp_solves": int(learned.training_lp_solves),
        "training_full_lp_solver_s": float(
            learned.training_solver_time_s
        ),
        "training_schedule": (
            "one prior-center anchor LP, one compact full LP per "
            "retained training sample, then one optimizer-difference SVD"
        ),
        "implementation": "optimizer_difference_span",
        "lp_backend": "scipy-highs-cpu",
        "timing_cached": False,
        "baseline_source_sha256": baseline_hash,
    }


def validate(
    ours_rows: pd.DataFrame,
    repeats: int,
) -> None:
    if len(ours_rows) != int(repeats) * len(CASE_ORDER) * len(RHO_VALUES):
        raise RuntimeError("Fresh Ours timing grid is incomplete")
    if not ours_rows["training_full_lp_solves"].astype(int).equals(
        ours_rows["retained_training_samples"].astype(int) + 1
    ):
        raise RuntimeError("Ours LP count is not one anchor plus one per sample")
    if (
        ours_rows["raw_learned_rank"].astype(int)
        > ours_rows["K"].astype(int)
    ).any():
        raise RuntimeError("An exact learned span was truncated")

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "timing",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if int(args.repeats) < 1:
        raise ValueError("--repeats must be positive")
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / OUTPUT_NAME
    if output.exists() and not args.force:
        print(f"Reusing existing Figure 3 Ours timing: {output}")
        return

    baselines, baseline_hash = load_legacy_baselines()
    fresh_rows: list[dict[str, object]] = []
    for repeat in range(1, int(args.repeats) + 1):
        for case_name in CASE_ORDER:
            config = figure3.OLD_SUITE_DYNAMIC_CONFIGS[case_name]
            preparation_start = time.perf_counter()
            data = figure3.olddyn.prepare_case(case_name, config, 40)
            preparation_time = time.perf_counter() - preparation_start
            test = [
                cff.convert_to_general_projection_instance(instance)
                for instance in data.bundle.test
            ]
            for rho in RHO_VALUES:
                fresh_rows.append(
                    run_point(
                        data,
                        test,
                        case_name,
                        rho,
                        repeat,
                        archived_k(baselines, case_name, rho),
                        preparation_time,
                        baseline_hash,
                    )
                )

    fresh = pd.DataFrame(fresh_rows)
    validate(fresh, int(args.repeats))
    fresh = fresh.sort_values(
        ["case", "rho", "repeat"],
        kind="stable",
    ).reset_index(drop=True)
    fresh.to_csv(output, index=False)
    print(f"Saved Figure 3 Ours timing to {output}", flush=True)


if __name__ == "__main__":
    main()

