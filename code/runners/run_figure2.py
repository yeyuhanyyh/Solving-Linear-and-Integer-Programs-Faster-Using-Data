#!/usr/bin/env python3
"""Reproduce the Figure 2 optimizer-difference rank-growth curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CODE = ROOT / "code" / "experiments"
CORE_CODE = ROOT / "code" / "core"
for code_path in (EXPERIMENT_CODE, CORE_CODE):
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

import run_latest_alg1_tuned_beyond_prior_rank_growth as figure2  # noqa: E402
import run_ridge_calibrated_beyond_prior_ours as ridge_ours  # noqa: E402


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def run_case(case: object, case_name: str) -> list[dict[str, object]]:
    config = figure2.TUNED_CONFIG[case_name]
    args = figure2.base_user_args(seed=int(config.seed), n_train=25)
    args.n_test = 20
    data = figure2.prepare_case(case, config, args, n_max=25)
    costs = np.asarray(data.bundle.Ctrain[:25], dtype=float)
    prior, _prior_meta, train_scores = (
        ridge_ours.estimate_ridge_calibrated_prior(
            costs,
            costs,
            rho=float(args.rho),
            ridge_scale=float(args.ridge_scale),
            ridge_abs=float(args.ridge_abs),
            delta0=float(args.delta0),
            rule=str(args.calibration_rule),
            support_mode=str(args.support_mode),
            svd_tol=float(args.svd_tol),
        )
    )
    retained = np.flatnonzero(
        train_scores <= float(prior.radius) ** 2 + 1e-10
    )
    if retained.size == 0 and train_scores.size > 0:
        retained = np.asarray([int(np.argmin(train_scores))], dtype=int)
    stage = figure2.latest_alg1_with_trace(
        data.bundle.stdlp,
        costs[retained],
        tol=float(config.indep_tol),
        anchor_cost=np.asarray(prior.mu, dtype=float),
    )
    trace = np.concatenate(
        [[0], np.asarray(stage.rank_after_sample, dtype=int)]
    )
    if trace.size != 26:
        raise RuntimeError(
            f"{case_name}: expected 25 retained samples, got {trace.size - 1}"
        )
    return [
        {
            "figure": "Figure 2",
            "case": case_name,
            "method": "OursEstC",
            "source_method": "OursEstC",
            "rho": 0.1,
            "n_samples": sample_count,
            "K": int(rank),
            "rank": int(rank),
            "retained_samples": int(retained.size),
            "seed": int(config.seed),
        }
        for sample_count, rank in enumerate(trace)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases", default=",".join(figure2.CASE_ORDER))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "fresh",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cases = split_csv(args.cases)
    unknown = sorted(set(cases) - set(figure2.CASE_ORDER))
    if unknown:
        raise ValueError(f"Unknown Figure 2 cases: {unknown}")
    output_path = args.out_dir.resolve() / "figure2_rank_growth_raw.csv"
    if output_path.exists() and not args.force:
        print(f"Reuse {output_path}", flush=True)
        return

    case_specs = figure2.nested.get_cases()
    rows: list[dict[str, object]] = []
    for case_name in cases:
        print(f"[Figure 2] case={case_name}", flush=True)
        rows.extend(run_case(case_specs[case_name], case_name))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved Figure 2 rank growth to {output_path}", flush=True)


if __name__ == "__main__":
    main()
