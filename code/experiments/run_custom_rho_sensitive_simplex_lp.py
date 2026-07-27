#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A deliberately rho-sensitive beyond-prior LP diagnostic.

LP instance:

    maximize c^T x   subject to x in Delta_m.

The feasible region is a simplex, so each cost vector selects one vertex.  The
latest exact-compression learner is equivalent here to keeping the optimizer
vertices observed among the retained training samples.  A large rho produces a
small calibrated prior, filters out high Mahalanobis-score training costs, and
therefore misses several rare/high-value optimizer vertices.

This script is intentionally diagnostic rather than part of the main paper
benchmark suite.  It uses the same ridge-Mahalanobis score and conformal order
statistic as the current beyond-prior code path.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results_custom_rho_sensitive_simplex_lp_20260507"
RHO_LIST = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]


def calibration_index(m: int, rho: float, rule: str = "conformal") -> int:
    if m <= 0:
        return 1
    if rule == "max":
        return m
    if rule == "conformal":
        return min(m, int(math.ceil((m + 1) * (1.0 - float(rho)))))
    raise ValueError(f"unsupported calibration rule: {rule}")


def ridge_mahalanobis_scores(
    costs: np.ndarray,
    ridge_scale: float = 1e-2,
    ridge_abs: float = 1e-10,
    support_mode: str = "full",
    svd_tol: float = 1e-10,
) -> Tuple[np.ndarray, Dict[str, float]]:
    costs = np.asarray(costs, dtype=float)
    n, d = costs.shape
    mu = costs.mean(axis=0)
    centered = costs - mu.reshape(1, -1)
    cov = centered.T @ centered / float(max(1, n - 1))
    avg_var = float(np.trace(cov) / max(d, 1))
    lam = max(float(ridge_abs), float(ridge_scale) * max(avg_var, 1e-18))
    if support_mode == "full":
        evals, V = np.linalg.eigh(cov + lam * np.eye(d))
        evals = np.maximum(evals, max(lam, 1e-18))
    elif support_mode == "empirical_span":
        if np.linalg.norm(centered) <= 1e-14:
            V = np.zeros((d, 0), dtype=float)
            evals = np.zeros(0, dtype=float)
        else:
            _u, s, vt = np.linalg.svd(centered / math.sqrt(max(1, n - 1)), full_matrices=False)
            keep = s > float(svd_tol) * max(float(s[0]) if s.size else 1.0, 1.0)
            V = vt[keep, :].T.copy()
            evals = np.maximum(s[keep] ** 2 + lam, max(lam, 1e-18))
    else:
        raise ValueError(f"unknown support_mode: {support_mode}")
    if V.shape[1] == 0:
        scores = np.zeros(n, dtype=float)
    else:
        z = centered @ V / np.sqrt(evals).reshape(1, -1)
        scores = np.sum(z * z, axis=1)
    meta = {
        "ridge_lambda": float(lam),
        "avg_variance": float(avg_var),
        "effective_dim": int(V.shape[1]),
    }
    return scores, meta


def make_mode_cost(mode: int, dim: int, reward: float, baseline: float, nuisance: float = 0.0) -> np.ndarray:
    c = np.full(dim, float(baseline), dtype=float)
    c[int(mode)] = float(reward)
    if nuisance:
        # A deterministic low-amplitude oscillation prevents exact duplicate
        # costs from making all common modes numerically indistinguishable.
        grid = np.arange(dim, dtype=float)
        c += float(nuisance) * np.sin(0.7 * grid + 0.31 * float(mode))
        c[int(mode)] = float(reward)
    return c


def build_costs(dim: int, baseline: float, nuisance: float) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Return training and test costs for the diagnostic simplex LP."""
    # Common modes are frequent and moderate.  Tail modes are rare in the
    # retained center of the distribution but important at test time.
    common_rewards = {0: 1.00, 1: 1.05, 2: 1.10, 3: 1.15}
    tail_rewards = {4: 3.0, 5: 4.0, 6: 5.0, 7: 6.0, 8: 7.5, 9: 9.0, 10: 11.0, 11: 13.0}

    train_modes: List[int] = []
    for mode in common_rewards:
        train_modes += [mode] * 5
    for mode in tail_rewards:
        train_modes += [mode] * 2
    # Four extra middle/tail observations make n=40 and smooth the transition.
    train_modes += [4, 5, 6, 7]
    assert len(train_modes) == 40

    rows: List[Dict[str, object]] = []
    costs: List[np.ndarray] = []
    for i, mode in enumerate(train_modes):
        rewards = common_rewards | tail_rewards
        c = make_mode_cost(mode, dim, rewards[mode], baseline=baseline, nuisance=nuisance)
        rows.append(
            {
                "sample": i + 1,
                "mode": mode,
                "reward": rewards[mode],
                "family": "common" if mode in common_rewards else "tail",
            }
        )
        costs.append(c)

    test_modes: List[int] = []
    # Weight the test set toward the tail so missing tail vertices visibly hurts.
    for mode in common_rewards:
        test_modes += [mode] * 2
    for mode in tail_rewards:
        test_modes += [mode] * 6
    test_rows: List[Dict[str, object]] = []
    test_costs: List[np.ndarray] = []
    for i, mode in enumerate(test_modes):
        rewards = common_rewards | tail_rewards
        c = make_mode_cost(mode, dim, rewards[mode], baseline=baseline, nuisance=nuisance)
        test_rows.append(
            {
                "test_id": i,
                "mode": mode,
                "reward": rewards[mode],
                "family": "common" if mode in common_rewards else "tail",
            }
        )
        test_costs.append(c)
    return pd.DataFrame(rows), np.vstack(costs), pd.DataFrame(test_rows), np.vstack(test_costs)


def evaluate_simplex_exact(
    train_costs: np.ndarray,
    train_scores: np.ndarray,
    test_costs: np.ndarray,
    rho: float,
    rule: str,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    k = calibration_index(train_scores.size, rho=float(rho), rule=rule)
    radius2 = float(np.sort(train_scores)[max(0, k - 1)])
    retained = np.flatnonzero(train_scores <= radius2 + 1e-12)
    if retained.size == 0:
        retained = np.asarray([int(np.argmin(train_scores))], dtype=int)
    retained_modes = np.argmax(train_costs[retained], axis=1)
    # Latest exact learner: first retained optimizer is the anchor; every new
    # retained optimizer adds one independent vertex-difference direction.
    learned_modes = sorted(set(int(v) for v in retained_modes))
    d_star = max(0, len(learned_modes) - 1)

    rows: List[Dict[str, object]] = []
    for i, c in enumerate(test_costs):
        full_mode = int(np.argmax(c))
        full_obj = float(c[full_mode])
        reduced_values = c[np.asarray(learned_modes, dtype=int)]
        reduced_pos = int(np.argmax(reduced_values))
        reduced_mode = int(learned_modes[reduced_pos])
        reduced_obj = float(reduced_values[reduced_pos])
        rows.append(
            {
                "test_id": i,
                "full_mode": full_mode,
                "reduced_mode": reduced_mode,
                "full_objective": full_obj,
                "reduced_objective": reduced_obj,
                "objective_ratio": reduced_obj / full_obj if full_obj > 0 else np.nan,
                "hit": float(reduced_mode == full_mode),
            }
        )
    raw = pd.DataFrame(rows)
    summary = {
        "rho": float(rho),
        "calibration_index_1based": int(k),
        "radius_squared": radius2,
        "retained_samples": int(retained.size),
        "retained_modes": ",".join(str(int(v)) for v in sorted(set(retained_modes))),
        "learned_modes": ",".join(str(v) for v in learned_modes),
        "d_star": int(d_star),
        "objective_ratio_mean": float(raw["objective_ratio"].mean()),
        "objective_ratio_se": float(raw["objective_ratio"].std(ddof=1) / math.sqrt(len(raw))) if len(raw) > 1 else 0.0,
        "hit_rate": float(raw["hit"].mean()),
    }
    return summary, raw


def plot_results(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )
    rhos = list(summary["rho"].astype(float))
    x = np.arange(len(rhos))
    labels = [f"{r:.2f}".rstrip("0").rstrip(".") for r in rhos]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.45))
    color = "#d62728"
    axes[0].plot(x, summary["retained_samples"], marker="o", color="#555555", linewidth=2, linestyle=(0, (5, 2)))
    axes[0].set_ylabel("Retained samples")
    axes[1].plot(x, summary["d_star"], marker="D", color=color, linewidth=2, linestyle=(0, (5, 2)))
    axes[1].set_ylabel("Final learned dimension d*")
    axes[2].errorbar(
        x,
        summary["objective_ratio_mean"],
        yerr=summary["objective_ratio_se"],
        marker="D",
        color=color,
        linewidth=2,
        linestyle=(0, (5, 2)),
        capsize=2,
        label="OursEstC",
    )
    axes[2].set_ylabel("Average test objective ratio")
    axes[2].set_ylim(max(0.0, float(summary["objective_ratio_mean"].min()) - 0.06), 1.02)
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    for ax in axes:
        ax.set_xlabel("Target outside mass rho")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_custom_rho_sensitive_simplex_lp.png", bbox_inches="tight")
    fig.savefig(out_dir / "figure_custom_rho_sensitive_simplex_lp.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--rho_list", default=",".join(str(v) for v in RHO_LIST))
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--baseline", type=float, default=0.05)
    p.add_argument("--nuisance", type=float, default=0.002)
    p.add_argument("--ridge_scale", type=float, default=1e-2)
    p.add_argument("--ridge_abs", type=float, default=1e-10)
    p.add_argument("--support_mode", choices=["full", "empirical_span"], default="full")
    p.add_argument("--calibration_rule", choices=["conformal", "max"], default="conformal")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rho_list = [float(s.strip()) for s in str(args.rho_list).split(",") if s.strip()]

    train_meta, train_costs, test_meta, test_costs = build_costs(
        dim=int(args.dim),
        baseline=float(args.baseline),
        nuisance=float(args.nuisance),
    )
    scores, score_meta = ridge_mahalanobis_scores(
        train_costs,
        ridge_scale=float(args.ridge_scale),
        ridge_abs=float(args.ridge_abs),
        support_mode=str(args.support_mode),
    )
    train_meta["ridge_mahalanobis_score"] = scores
    train_meta["score_rank_low_to_high"] = pd.Series(scores).rank(method="first").astype(int).to_numpy()

    summaries: List[Dict[str, object]] = []
    raws: List[pd.DataFrame] = []
    for rho in rho_list:
        summary, raw = evaluate_simplex_exact(
            train_costs=train_costs,
            train_scores=scores,
            test_costs=test_costs,
            rho=float(rho),
            rule=str(args.calibration_rule),
        )
        summaries.append(summary)
        raw["rho"] = float(rho)
        raws.append(raw)

    summary_df = pd.DataFrame(summaries)
    raw_df = pd.concat(raws, ignore_index=True)
    summary_df.to_csv(out_dir / "custom_rho_sensitive_simplex_summary.csv", index=False)
    raw_df.to_csv(out_dir / "custom_rho_sensitive_simplex_raw.csv", index=False)
    train_meta.to_csv(out_dir / "custom_rho_sensitive_simplex_train_costs.csv", index=False)
    test_meta.to_csv(out_dir / "custom_rho_sensitive_simplex_test_costs.csv", index=False)
    manifest = {
        "experiment": "custom_rho_sensitive_simplex_lp",
        "lp": "maximize c^T x over the probability simplex Delta_m",
        "dim": int(args.dim),
        "n_train": int(train_costs.shape[0]),
        "n_test": int(test_costs.shape[0]),
        "rho_list": rho_list,
        "calibration_rule": str(args.calibration_rule),
        "ridge_score": score_meta,
        "baseline": float(args.baseline),
        "nuisance": float(args.nuisance),
        "mechanism": "large rho keeps only low-score/common optimizer modes, dropping high-value tail vertices",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_results(summary_df, out_dir)
    print(summary_df.to_string(index=False))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
