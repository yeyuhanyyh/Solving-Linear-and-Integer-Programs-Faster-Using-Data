#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A more natural rho-sensitive beyond-prior diagnostic.

LP: long-only portfolio allocation with asset and sector concentration caps,

    maximize r^T x
    s.t.     sum_i x_i = 1,
             0 <= x_i <= asset_cap,
             sum_{i in sector s} x_i <= sector_cap.

Return distribution: a factor/regime model with common market/sector noise and
rare sector-boom regimes.  Large rho filters high Mahalanobis-score boom samples
from the calibrated working prior, so the exact-compression learner misses the
corresponding optimal allocation directions.  The resulting ratio drop is the
intended diagnostic signal.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results_natural_rho_sensitive_portfolio_lp_20260507"
RHO_LIST = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]


def linprog_options() -> Dict[str, object]:
    return {"presolve": True, "dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9}


def calibration_index(m: int, rho: float, rule: str = "conformal") -> int:
    if rule == "max":
        return m
    if rule == "conformal":
        return min(m, int(math.ceil((m + 1) * (1.0 - float(rho)))))
    raise ValueError(f"unknown calibration rule: {rule}")


def sector_matrix(n_sectors: int, assets_per_sector: int) -> np.ndarray:
    n_assets = n_sectors * assets_per_sector
    A = np.zeros((n_sectors, n_assets), dtype=float)
    for s in range(n_sectors):
        A[s, s * assets_per_sector : (s + 1) * assets_per_sector] = 1.0
    return A


def make_portfolio_lp(n_sectors: int, assets_per_sector: int, asset_cap: float, sector_cap: float) -> Dict[str, object]:
    n_assets = n_sectors * assets_per_sector
    A_sector = sector_matrix(n_sectors, assets_per_sector)
    return {
        "n_assets": n_assets,
        "n_sectors": n_sectors,
        "assets_per_sector": assets_per_sector,
        "asset_cap": float(asset_cap),
        "sector_cap": float(sector_cap),
        "A_ub": A_sector,
        "b_ub": np.full(n_sectors, float(sector_cap), dtype=float),
        "A_eq": np.ones((1, n_assets), dtype=float),
        "b_eq": np.asarray([1.0], dtype=float),
        "bounds": [(0.0, float(asset_cap)) for _ in range(n_assets)],
    }


def solve_full(lp: Dict[str, object], r: np.ndarray) -> Tuple[np.ndarray, float, bool, float, str]:
    t0 = time.perf_counter()
    res = linprog(
        c=-np.asarray(r, dtype=float),
        A_ub=lp["A_ub"],
        b_ub=lp["b_ub"],
        A_eq=lp["A_eq"],
        b_eq=lp["b_eq"],
        bounds=lp["bounds"],
        method="highs",
        options=linprog_options(),
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        return np.zeros(int(lp["n_assets"]), dtype=float), float("nan"), False, runtime, str(res.message)
    x = np.asarray(res.x, dtype=float)
    return x, float(r @ x), True, runtime, "ok"


def solve_reduced(lp: Dict[str, object], r: np.ndarray, U: np.ndarray, x_anchor: np.ndarray) -> Tuple[np.ndarray, float, bool, float, str]:
    t0 = time.perf_counter()
    U = np.asarray(U, dtype=float)
    x_anchor = np.asarray(x_anchor, dtype=float)
    if U.shape[1] == 0:
        return x_anchor.copy(), float(r @ x_anchor), True, time.perf_counter() - t0, "anchor only"

    A_ub_parts = [
        np.asarray(lp["A_ub"], dtype=float) @ U,
        U,
        -U,
    ]
    b_ub_parts = [
        np.asarray(lp["b_ub"], dtype=float) - np.asarray(lp["A_ub"], dtype=float) @ x_anchor,
        np.full(int(lp["n_assets"]), float(lp["asset_cap"]), dtype=float) - x_anchor,
        x_anchor,
    ]
    A_ub = np.vstack(A_ub_parts)
    b_ub = np.concatenate(b_ub_parts)
    A_eq = np.asarray(lp["A_eq"], dtype=float) @ U
    b_eq = np.asarray(lp["b_eq"], dtype=float) - np.asarray(lp["A_eq"], dtype=float) @ x_anchor
    res = linprog(
        c=-(U.T @ np.asarray(r, dtype=float)),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(None, None) for _ in range(U.shape[1])],
        method="highs",
        options=linprog_options(),
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        return x_anchor.copy(), float("nan"), False, runtime, str(res.message)
    x = x_anchor + U @ np.asarray(res.x, dtype=float)
    x[np.abs(x) < 1e-10] = 0.0
    return x, float(r @ x), True, runtime, "ok"


def append_direction(D: np.ndarray, q: np.ndarray, tol: float) -> Tuple[np.ndarray, bool]:
    q = np.asarray(q, dtype=float)
    if D.size == 0:
        norm = float(np.linalg.norm(q))
        if norm > tol:
            return q.reshape(-1, 1), True
        return D, False
    Q, _r = np.linalg.qr(D, mode="reduced")
    resid = q - Q @ (Q.T @ q)
    norm = float(np.linalg.norm(resid))
    if norm > tol:
        return np.column_stack([D, resid]), True
    return D, False


def learn_exact_space(lp: Dict[str, object], train_returns: np.ndarray, retained: np.ndarray, tol: float) -> Dict[str, object]:
    retained_returns = train_returns[np.asarray(retained, dtype=int)]
    x_anchor, _obj, success, _time, msg = solve_full(lp, retained_returns[0])
    if not success:
        raise RuntimeError(f"anchor solve failed: {msg}")
    D = np.zeros((int(lp["n_assets"]), 0), dtype=float)
    ranks: List[int] = []
    for r in retained_returns:
        x_i, _obj_i, success_i, _time_i, msg_i = solve_full(lp, r)
        if not success_i:
            raise RuntimeError(f"training solve failed: {msg_i}")
        D, _added = append_direction(D, x_i - x_anchor, tol=float(tol))
        ranks.append(int(D.shape[1]))
    U = np.zeros_like(D) if D.size == 0 else np.linalg.qr(D, mode="reduced")[0]
    return {"x_anchor": x_anchor, "D": D, "U": U, "rank_after_sample": ranks}


def ridge_mahalanobis_scores(costs: np.ndarray, ridge_scale: float, ridge_abs: float) -> Tuple[np.ndarray, Dict[str, object]]:
    costs = np.asarray(costs, dtype=float)
    n, d = costs.shape
    mu = costs.mean(axis=0)
    centered = costs - mu.reshape(1, -1)
    cov = centered.T @ centered / float(max(1, n - 1))
    avg_var = float(np.trace(cov) / max(d, 1))
    lam = max(float(ridge_abs), float(ridge_scale) * max(avg_var, 1e-18))
    evals, V = np.linalg.eigh(cov + lam * np.eye(d))
    evals = np.maximum(evals, max(lam, 1e-18))
    z = centered @ V / np.sqrt(evals).reshape(1, -1)
    scores = np.sum(z * z, axis=1)
    return scores, {"ridge_lambda": float(lam), "effective_dim": int(V.shape[1]), "avg_variance": avg_var}


def generate_returns(
    n_common: int,
    boom_counts: Sequence[int],
    seed: int,
    n_sectors: int,
    assets_per_sector: int,
    shock_size: float,
    noise_scale: float,
    market_scale: float,
    sector_scale: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n_assets = n_sectors * assets_per_sector
    asset_alpha = np.tile(np.linspace(0.006, -0.006, assets_per_sector), n_sectors)
    sector_base = np.asarray([0.080, 0.075, 0.070, 0.020, 0.018, 0.016], dtype=float)
    beta = np.tile(np.linspace(0.85, 1.15, assets_per_sector), n_sectors)
    rows: List[Dict[str, object]] = []
    returns: List[np.ndarray] = []
    modes: List[int] = [-1] * int(n_common)
    for sector, count in zip([3, 4, 5], boom_counts):
        modes += [int(sector)] * int(count)
    for idx, mode in enumerate(modes):
        market = rng.normal(0.0, float(market_scale))
        sector_noise = rng.normal(0.0, float(sector_scale), size=n_sectors)
        idio = rng.normal(0.0, float(noise_scale), size=n_assets)
        r = np.repeat(sector_base, assets_per_sector) + asset_alpha + beta * market + np.repeat(sector_noise, assets_per_sector) + idio
        if mode >= 0:
            sl = slice(mode * assets_per_sector, (mode + 1) * assets_per_sector)
            r[sl] += float(shock_size)
            # A mild drag on the previously dominant sectors makes the regime
            # switch economically meaningful, not just a small perturbation.
            r[: 3 * assets_per_sector] -= 0.025
        rows.append({"sample": idx + 1, "regime": "common" if mode < 0 else f"sector_{mode}_boom", "boom_sector": mode})
        returns.append(r)
    return pd.DataFrame(rows), np.vstack(returns)


def evaluate(lp: Dict[str, object], test_returns: np.ndarray, stage: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for i, r in enumerate(test_returns):
        x_full, obj_full, ok_full, _t_full, msg_full = solve_full(lp, r)
        x_red, obj_red, ok_red, t_red, msg_red = solve_reduced(lp, r, stage["U"], stage["x_anchor"])
        rows.append(
            {
                "test_id": i,
                "full_objective": obj_full,
                "reduced_objective": obj_red,
                "objective_ratio": obj_red / obj_full if ok_full and ok_red and obj_full > 1e-12 else np.nan,
                "success": float(ok_full and ok_red),
                "time": float(t_red),
                "full_top_sector": int(np.argmax(x_full.reshape(int(lp["n_sectors"]), int(lp["assets_per_sector"])).sum(axis=1))) if ok_full else -1,
                "reduced_top_sector": int(np.argmax(x_red.reshape(int(lp["n_sectors"]), int(lp["assets_per_sector"])).sum(axis=1))) if ok_red else -1,
                "message": msg_red if ok_full else msg_full,
            }
        )
    return pd.DataFrame(rows)


def summarize_raw(raw: pd.DataFrame) -> Dict[str, object]:
    ratio = raw["objective_ratio"].dropna()
    return {
        "objective_ratio_mean": float(ratio.mean()),
        "objective_ratio_se": float(ratio.std(ddof=1) / math.sqrt(len(ratio))) if len(ratio) > 1 else 0.0,
        "success_rate": float(raw["success"].mean()),
    }


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9, "figure.dpi": 160, "savefig.dpi": 220})
    rhos = list(summary["rho"].astype(float))
    x = np.arange(len(rhos))
    labels = [f"{r:.2f}".rstrip("0").rstrip(".") for r in rhos]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    axes[0].plot(x, summary["retained_samples"], marker="o", color="#555555", linewidth=2, linestyle=(0, (5, 2)))
    axes[1].plot(x, summary["intrinsic_K"], marker="D", color="#d62728", linewidth=2, linestyle=(0, (5, 2)))
    axes[2].errorbar(
        x,
        summary["objective_ratio_mean"],
        yerr=summary["objective_ratio_se"],
        marker="D",
        color="#d62728",
        linewidth=2,
        linestyle=(0, (5, 2)),
        capsize=2,
    )
    axes[0].set_ylabel("Retained samples")
    axes[1].set_ylabel("Final learned dimension d*")
    axes[2].set_ylabel("Average test objective ratio")
    axes[2].set_ylim(max(0.0, float(summary["objective_ratio_mean"].min()) - 0.04), 1.01)
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    for ax in axes:
        ax.set_xlabel("Target outside mass rho")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_natural_rho_sensitive_portfolio_lp.png", bbox_inches="tight")
    fig.savefig(out_dir / "figure_natural_rho_sensitive_portfolio_lp.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--rho_list", default=",".join(str(v) for v in RHO_LIST))
    p.add_argument("--seed_train", type=int, default=123)
    p.add_argument("--seed_test", type=int, default=991)
    p.add_argument("--n_common_train", type=int, default=28)
    p.add_argument("--boom_counts_train", default="4,4,4")
    p.add_argument("--n_common_test", type=int, default=36)
    p.add_argument("--boom_counts_test", default="16,16,16")
    p.add_argument("--n_sectors", type=int, default=6)
    p.add_argument("--assets_per_sector", type=int, default=4)
    p.add_argument("--asset_cap", type=float, default=0.18)
    p.add_argument("--sector_cap", type=float, default=0.45)
    p.add_argument("--shock_size", type=float, default=0.62)
    p.add_argument("--noise_scale", type=float, default=0.006)
    p.add_argument("--market_scale", type=float, default=0.020)
    p.add_argument("--sector_scale", type=float, default=0.012)
    p.add_argument("--ridge_scale", type=float, default=1e-2)
    p.add_argument("--ridge_abs", type=float, default=1e-10)
    p.add_argument("--calibration_rule", choices=["conformal", "max"], default="conformal")
    p.add_argument("--indep_tol", type=float, default=1e-8)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rho_list = [float(s.strip()) for s in str(args.rho_list).split(",") if s.strip()]
    boom_train = [int(s.strip()) for s in str(args.boom_counts_train).split(",") if s.strip()]
    boom_test = [int(s.strip()) for s in str(args.boom_counts_test).split(",") if s.strip()]

    lp = make_portfolio_lp(args.n_sectors, args.assets_per_sector, args.asset_cap, args.sector_cap)
    train_meta, train_returns = generate_returns(
        int(args.n_common_train),
        boom_train,
        seed=int(args.seed_train),
        n_sectors=int(args.n_sectors),
        assets_per_sector=int(args.assets_per_sector),
        shock_size=float(args.shock_size),
        noise_scale=float(args.noise_scale),
        market_scale=float(args.market_scale),
        sector_scale=float(args.sector_scale),
    )
    test_meta, test_returns = generate_returns(
        int(args.n_common_test),
        boom_test,
        seed=int(args.seed_test),
        n_sectors=int(args.n_sectors),
        assets_per_sector=int(args.assets_per_sector),
        shock_size=float(args.shock_size),
        noise_scale=float(args.noise_scale),
        market_scale=float(args.market_scale),
        sector_scale=float(args.sector_scale),
    )
    scores, score_meta = ridge_mahalanobis_scores(train_returns, ridge_scale=float(args.ridge_scale), ridge_abs=float(args.ridge_abs))
    train_meta["ridge_mahalanobis_score"] = scores
    train_meta.to_csv(out_dir / "portfolio_train_returns_metadata.csv", index=False)
    test_meta.to_csv(out_dir / "portfolio_test_returns_metadata.csv", index=False)

    summary_rows: List[Dict[str, object]] = []
    raw_frames: List[pd.DataFrame] = []
    for rho in rho_list:
        k = calibration_index(scores.size, rho=float(rho), rule=str(args.calibration_rule))
        radius2 = float(np.sort(scores)[max(0, k - 1)])
        retained = np.flatnonzero(scores <= radius2 + 1e-12)
        if retained.size == 0:
            retained = np.asarray([int(np.argmin(scores))], dtype=int)
        stage = learn_exact_space(lp, train_returns, retained, tol=float(args.indep_tol))
        raw = evaluate(lp, test_returns, stage)
        raw["rho"] = float(rho)
        raw_frames.append(raw)
        rec = {
            "rho": float(rho),
            "calibration_index_1based": int(k),
            "radius_squared": radius2,
            "retained_samples": int(retained.size),
            "intrinsic_K": int(stage["U"].shape[1]),
            "n_train": int(train_returns.shape[0]),
            "n_test": int(test_returns.shape[0]),
            **summarize_raw(raw),
        }
        summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows)
    raw_all = pd.concat(raw_frames, ignore_index=True)
    summary.to_csv(out_dir / "natural_rho_sensitive_portfolio_summary.csv", index=False)
    raw_all.to_csv(out_dir / "natural_rho_sensitive_portfolio_raw.csv", index=False)
    manifest = {
        "experiment": "natural_rho_sensitive_portfolio_lp",
        "lp": "long-only portfolio with asset and sector caps",
        "n_assets": int(lp["n_assets"]),
        "n_sectors": int(args.n_sectors),
        "assets_per_sector": int(args.assets_per_sector),
        "asset_cap": float(args.asset_cap),
        "sector_cap": float(args.sector_cap),
        "n_train": int(train_returns.shape[0]),
        "n_test": int(test_returns.shape[0]),
        "train_regimes": train_meta["regime"].value_counts().to_dict(),
        "test_regimes": test_meta["regime"].value_counts().to_dict(),
        "return_model": {
            "common": "base sector returns + market factor + sector factors + idiosyncratic noise",
            "tail": "rare sector-boom regimes for sectors 3,4,5 with positive shock and mild drag on common sectors",
            "shock_size": float(args.shock_size),
            "noise_scale": float(args.noise_scale),
            "market_scale": float(args.market_scale),
            "sector_scale": float(args.sector_scale),
        },
        "ridge_score": score_meta,
        "calibration_rule": str(args.calibration_rule),
        "rho_list": rho_list,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot(summary, out_dir)
    print(summary.to_string(index=False))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
