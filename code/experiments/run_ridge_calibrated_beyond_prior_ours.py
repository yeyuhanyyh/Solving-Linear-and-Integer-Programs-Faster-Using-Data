#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OursEstC with ridge-Mahalanobis calibration for the beyond-prior figures.

This script keeps the existing LP instances/cost distributions, but replaces the
old estimated-C construction by the calibrated score set described in the
current manuscript:

    s(c) = (c - mu)^T (Sigma + lambda I)^{-1} (c - mu),
    C_hat = {c : s(c) <= S_(k)}.

By default we use the largest finite calibration order statistic, S_(m), because
the numerical section uses the same historical batch for fitting, calibration,
and representation learning.  The saved manifest records the exact rule.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
CORE_CODE = ROOT.parent / "core"
sys.path.insert(0, str(CORE_CODE))

import compare_fixedX_family_suite as cff  # noqa: E402
import run_beyond_prior_sample_sweeps_paper8 as nested  # noqa: E402
import run_subgaussian_beyond_prior_suite as bp  # noqa: E402
from iwata_sakaue_pelp_projection_compare import objective_ratio, summarize_ratios_and_times  # noqa: E402


CASE_ORDER = ["packing", "mincostflow", "random_lp_A", "random_lp_B", "random_lp_C", "random_lp_D", "grow7", "sc205"]
K_LIST = [10, 20, 30, 50]


def parse_int_list(spec: str) -> List[int]:
    return [int(s.strip()) for s in str(spec).split(",") if s.strip()]


def calibration_index(m: int, rho: float, delta0: float, rule: str) -> int:
    """Return a 1-based order statistic index."""
    if m <= 0:
        return 1
    rule = str(rule).lower()
    if rule == "max":
        return m
    if rule == "conformal":
        return min(m, int(math.ceil((m + 1) * (1.0 - float(rho)))))
    if rule == "tolerance":
        # Small m and small delta0 may request k=m+1, i.e. all of R^d.
        # For numerical ellipsoid experiments we clip to the largest finite
        # score and record this in the manifest.
        from math import comb

        p = 1.0 - float(rho)
        for k in range(1, m + 2):
            tail = 0.0
            for j in range(k, m + 1):
                tail += comb(m, j) * (p**j) * ((1.0 - p) ** (m - j))
            if tail <= float(delta0):
                return min(k, m)
        return m
    raise ValueError(f"unknown calibration rule: {rule}")


def estimate_ridge_calibrated_prior(
    fit_costs: np.ndarray,
    cal_costs: np.ndarray,
    rho: float,
    ridge_scale: float,
    ridge_abs: float,
    delta0: float,
    rule: str,
    support_mode: str,
    svd_tol: float,
) -> tuple[bp.EllipsoidPrior, Dict[str, object], np.ndarray]:
    fit = np.asarray(fit_costs, dtype=float)
    cal = np.asarray(cal_costs, dtype=float)
    if fit.ndim != 2 or fit.shape[0] == 0:
        raise ValueError("fit_costs must be a nonempty 2D array")
    d = int(fit.shape[1])
    mu = fit.mean(axis=0)
    centered = fit - mu.reshape(1, -1)
    if fit.shape[0] <= 1:
        cov = np.zeros((d, d), dtype=float)
    else:
        cov = centered.T @ centered / float(max(1, fit.shape[0] - 1))
    avg_var = float(np.trace(cov) / max(d, 1))
    lam = max(float(ridge_abs), float(ridge_scale) * max(avg_var, 1e-18))
    support_mode = str(support_mode).lower()
    if support_mode == "full":
        evals, V = np.linalg.eigh(cov + lam * np.eye(d))
        evals = np.maximum(evals, max(lam, 1e-18))
    elif support_mode == "empirical_span":
        if np.linalg.norm(centered) <= 1e-14:
            V = np.zeros((d, 0), dtype=float)
            evals = np.zeros(0, dtype=float)
        else:
            _u, s, vt = np.linalg.svd(centered / math.sqrt(max(1, fit.shape[0] - 1)), full_matrices=False)
            keep = s > float(svd_tol) * max(float(s[0]) if s.size else 1.0, 1.0)
            V = vt[keep, :].T.copy()
            evals = np.maximum(s[keep] ** 2 + lam, max(lam, 1e-18))
    else:
        raise ValueError(f"unknown support_mode: {support_mode}")
    E = V * np.sqrt(evals).reshape(1, -1) if V.shape[1] else np.zeros((d, 0), dtype=float)

    cal_centered = cal - mu.reshape(1, -1)
    if V.shape[1] == 0:
        scores = np.zeros(cal.shape[0], dtype=float)
    else:
        z = cal_centered @ V / np.sqrt(evals).reshape(1, -1)
        scores = np.sum(z * z, axis=1)
    if scores.size == 0:
        radius2 = 0.0
        k = 1
    else:
        k = calibration_index(scores.size, rho=rho, delta0=delta0, rule=rule)
        radius2 = float(np.sort(scores)[max(0, min(k, scores.size) - 1)])
    radius = math.sqrt(max(radius2, 0.0))
    prior = bp.EllipsoidPrior(
        mu=mu,
        E=E,
        radius=radius,
        rho=float(rho),
        pilot_n=int(cal.shape[0]),
        effective_dim=int(E.shape[1]),
        radius_scale=float("nan"),
        retained_train=0,
        pilot_maha_q=radius,
    )
    if fit.shape == cal.shape and np.allclose(fit, cal):
        train_scores = scores.copy()
    elif V.shape[1] == 0:
        train_scores = np.zeros(fit.shape[0], dtype=float)
    else:
        train_scores = np.sum(((fit - mu.reshape(1, -1)) @ V / np.sqrt(evals).reshape(1, -1)) ** 2, axis=1)
    meta = {
        "score": "ridge_mahalanobis",
        "ambient_dim": d,
        "support_mode": support_mode,
        "effective_dim": int(E.shape[1]),
        "ridge_lambda": lam,
        "ridge_scale": float(ridge_scale),
        "ridge_abs": float(ridge_abs),
        "calibration_rule": str(rule),
        "calibration_index_1based": int(k),
        "calibration_m": int(scores.size),
        "rho": float(rho),
        "delta0": float(delta0),
        "radius_squared": float(radius2),
        "radius": float(radius),
        "fit_n": int(fit.shape[0]),
        "cal_n": int(cal.shape[0]),
    }
    return prior, meta, train_scores


def evaluate_ours(data: object, stage1: object) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for inst in data.bundle.test:
        c_std = np.asarray(inst.c_std, dtype=float)
        x_std, _obj_min, runtime, success, msg = cff.solve_ours_exact_on_instance(
            data.bundle.stdlp,
            c_std,
            stage1.U,
            stage1.x_anchor,
        )
        obj = float("nan")
        if success and x_std is not None:
            obj = float(-c_std @ x_std) if np.asarray(inst.c).shape[0] == np.asarray(c_std).shape[0] else float(inst.c @ x_std[data.bundle.stdlp.n_slack :])
        rows.append(
            {
                "method": "OursEstC",
                "K": int(stage1.U.shape[1]),
                "intrinsic_K": int(stage1.U.shape[1]),
                "instance": inst.name,
                "objective": obj,
                "full_objective": inst.full_obj,
                "objective_ratio": objective_ratio(obj, inst.full_obj) if success else np.nan,
                "time": runtime,
                "success": float(success),
                "message": msg,
            }
        )
    return pd.DataFrame(rows)


def alg1_latest_optimizer_difference(
    lp: object,
    Ctrain: np.ndarray,
    indep_tol: float = 1e-8,
    verbose: bool = False,
) -> object:
    """Current manuscript Algorithm 1: append optimizer-difference directions.

    This is intentionally simpler than the older estimated-prior facet-hit
    learner.  Given an anchor optimizer x0, each observed optimizer contributes
    x_i - x0 only if it is linearly independent of the current learned span.
    """
    costs = np.asarray(Ctrain, dtype=float)
    if costs.ndim != 2 or costs.shape[0] == 0:
        raise ValueError("latest Algorithm 1 needs at least one retained training cost")
    x_anchor, _ = bp.solve_standard_form_min(lp, costs[0])
    D = np.zeros((lp.dim, 0), dtype=float)
    hard: List[int] = []
    rank_after_sample: List[int] = []
    messages: List[str] = []
    for i, c in enumerate(costs):
        x_i, _ = bp.solve_standard_form_min(lp, c)
        q_i = np.asarray(x_i, dtype=float) - np.asarray(x_anchor, dtype=float)
        Dnew, added, _ = bp.append_direction(D, q_i, tol=float(indep_tol))
        if added:
            D = Dnew
            hard.append(i)
            msg = "added optimizer-difference direction"
        else:
            msg = "optimizer already in current affine span"
        rank_after_sample.append(int(D.shape[1]))
        messages.append(msg)
        if verbose:
            print(
                f"Latest Alg1 sample {i+1:3d}/{len(costs):3d}: "
                f"rank {D.shape[1]:3d}, {msg}",
                flush=True,
            )
    U = np.zeros((lp.dim, 0), dtype=float) if D.size == 0 else np.linalg.qr(D, mode="reduced")[0]
    return bp.OurStageIResult(
        D=D,
        U=U,
        x_anchor=x_anchor,
        hard_indices=hard,
        rank_after_sample=rank_after_sample,
        rank_after_query=[],
        messages=messages,
    )


def run_case_prefix(data: object, args: argparse.Namespace, n_samples: int, case_out: Path, force: bool = False) -> pd.DataFrame:
    summary_path = case_out / "ours_ridge_summary.csv"
    if summary_path.exists() and not force:
        return pd.read_csv(summary_path)
    case_out.mkdir(parents=True, exist_ok=True)
    train_costs = np.asarray(data.bundle.Ctrain[: int(n_samples)], dtype=float)
    prior, prior_meta, train_scores = estimate_ridge_calibrated_prior(
        train_costs,
        train_costs,
        rho=float(args.rho),
        ridge_scale=float(args.ridge_scale),
        ridge_abs=float(args.ridge_abs),
        delta0=float(args.delta0),
        rule=str(args.calibration_rule),
        support_mode=str(args.support_mode),
        svd_tol=float(args.svd_tol),
    )
    retained = np.flatnonzero(train_scores <= float(prior.radius) ** 2 + 1e-10)
    if retained.size == 0 and train_scores.size > 0:
        retained = np.asarray([int(np.argmin(train_scores))], dtype=int)
    prior.retained_train = int(retained.size)

    t0 = time.perf_counter()
    retained_costs = train_costs[retained]
    if str(args.learner) == "latest_alg1":
        stage1 = alg1_latest_optimizer_difference(
            data.bundle.stdlp,
            retained_costs,
            indep_tol=float(args.indep_tol),
            verbose=bool(args.verbose),
        )
    else:
        stage1 = bp.alg2_cumulative_ellipsoid(data.bundle.stdlp, retained_costs, prior, verbose=bool(args.verbose))
    fit_time = time.perf_counter() - t0
    raw = evaluate_ours(data, stage1)
    raw["case"] = data.case.slug
    raw["title"] = data.case.title
    raw["family"] = data.case.family
    raw["n_samples"] = int(n_samples)
    raw.to_csv(case_out / "ours_ridge_raw.csv", index=False)
    summary = summarize_ratios_and_times(raw.to_dict("records"))
    summary["case"] = data.case.slug
    summary["title"] = data.case.title
    summary["family"] = data.case.family
    summary["n_samples"] = int(n_samples)
    summary["intrinsic_K"] = int(stage1.U.shape[1])
    summary["fit_time"] = float(fit_time)
    summary["retained_samples"] = int(retained.size)
    summary.to_csv(summary_path, index=False)
    ranks = np.concatenate([[0], np.asarray(stage1.rank_after_sample, dtype=int)])
    pd.DataFrame(
        {
            "sample": np.arange(len(ranks)),
            "rank_after_sample": ranks,
            "case": data.case.slug,
            "title": data.case.title,
            "n_samples": int(n_samples),
        }
    ).to_csv(case_out / "ours_ridge_rank_after_sample.csv", index=False)
    np.savez(
        case_out / "ours_ridge_stageI_and_prior.npz",
        U=stage1.U,
        D=stage1.D,
        x_anchor=stage1.x_anchor,
        hard_indices=np.asarray(stage1.hard_indices, dtype=int),
        train_scores=train_scores,
        retained_indices=retained,
        prior_mu=prior.mu,
        prior_E=prior.E,
        prior_radius=np.asarray([prior.radius], dtype=float),
    )
    meta = {
        "case": data.case.slug,
        "title": data.case.title,
        "n_samples": int(n_samples),
        "retained_samples": int(retained.size),
        "learned_dimension": int(stage1.U.shape[1]),
        "objective_ratio_mean": float(summary["objective_ratio_mean"].iloc[0]),
        "objective_ratio_se": float(summary["objective_ratio_se"].iloc[0]),
        **prior_meta,
    }
    (case_out / "ours_ridge_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def expand_for_k_sweep(df: pd.DataFrame, k_list: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        for k in k_list:
            rec = row.to_dict()
            rec["K"] = int(k)
            rec["method"] = "OursEstC"
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default="results_ridge_calibrated_beyond_prior_ours_20260506_probe")
    p.add_argument("--case_order", default=",".join(CASE_ORDER))
    p.add_argument("--sample_list", default="20")
    p.add_argument("--n_val", type=int, default=6)
    p.add_argument("--n_test", type=int, default=12)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--pilot_n", type=int, default=96)
    p.add_argument("--rho", type=float, default=0.10)
    p.add_argument("--delta0", type=float, default=0.20)
    p.add_argument("--calibration_rule", choices=["max", "conformal", "tolerance"], default="max")
    p.add_argument("--support_mode", choices=["full", "empirical_span"], default="empirical_span")
    p.add_argument("--ridge_scale", type=float, default=1e-3)
    p.add_argument("--ridge_abs", type=float, default=1e-10)
    p.add_argument("--radius_scale", type=float, default=1.0)
    p.add_argument("--gaussian_std_scale", type=float, default=0.35)
    p.add_argument("--case_gaussian_std_scales", default="random_lp_A=20,random_lp_B=20,random_lp_C=20,random_lp_D=20")
    p.add_argument("--cost_space", choices=["reward_subspace", "full_standard"], default="reward_subspace")
    p.add_argument("--svd_tol", type=float, default=1e-10)
    p.add_argument("--k_list", default="10,20,30,50")
    p.add_argument("--rand_trials", type=int, default=3)
    p.add_argument("--learner", choices=["latest_alg1", "old_ellipsoid_facet"], default="latest_alg1")
    p.add_argument("--indep_tol", type=float, default=1e-8)
    # Needed because nested.case_with_sample_size sets these fields.
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=120)
    args = p.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_list = parse_int_list(args.sample_list)
    k_list = parse_int_list(args.k_list)
    case_order = [s.strip() for s in str(args.case_order).split(",") if s.strip()]
    cases = nested.get_cases()
    n_max = max(sample_list)

    frames = []
    for case_name in case_order:
        case = cases[case_name]
        print(f"[ridge calibrated OursEstC] prepare {case.slug}, n_max={n_max}", flush=True)
        data = nested.prepare_fixed_pool_case(case, args, n_max)
        for n in sample_list:
            print(f"[ridge calibrated OursEstC] run {case.slug}, n={n}", flush=True)
            frames.append(run_case_prefix(data, args, n, out_dir / "ours_ridge_sample_sweep" / f"{case.slug}_n{n:03d}", force=bool(args.force)))
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(out_dir / "ours_ridge_accuracy_vs_samples_summary.csv", index=False)
    expand_for_k_sweep(summary[summary["n_samples"] == min(sample_list)].copy(), k_list).to_csv(out_dir / "ours_ridge_k_sweep_expanded_summary.csv", index=False)
    learned = summary.sort_values("n_samples").groupby("case", as_index=False).tail(1)[["case", "title", "intrinsic_K", "n_samples", "retained_samples", "objective_ratio_mean", "objective_ratio_se"]].copy()
    learned = learned.rename(columns={"intrinsic_K": "learned_dimension"})
    learned.to_csv(out_dir / "ours_ridge_learned_dimensions.csv", index=False)
    manifest = {
        "experiment": "ridge_calibrated_beyond_prior_ours",
        "case_order": case_order,
        "sample_list": sample_list,
        "k_list": k_list,
        "rho": float(args.rho),
        "delta0": float(args.delta0),
        "calibration_rule": str(args.calibration_rule),
        "ridge_scale": float(args.ridge_scale),
        "ridge_abs": float(args.ridge_abs),
        "cost_space": str(args.cost_space),
        "same_batch_for_fit_calibration_training": True,
        "old_code_reused_for_lp_generation_and_exact_lp_solver": str(CORE_CODE),
        "representation_learner": str(args.learner),
    }
    (out_dir / "ridge_calibrated_ours_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved ridge-calibrated OursEstC results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
