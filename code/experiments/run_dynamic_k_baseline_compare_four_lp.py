#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic-K baseline comparison for the four rho-sensitivity LPs.

For each case and rho:

1. Run OursEstC with the calibrated beyond-prior Algorithm 1 and record the
   learned dimension d(rho).
2. Train/evaluate the baselines on the same n=40 training costs with
   K=d(rho).  If d(rho)=0, the projection baselines use the minimal meaningful
   K=1 and the CSV records both numbers.

Baselines:
    data-driven proj  : old PCA/SGA case selection for old-suite cases; PCA for
                        the two custom LPs.
    Rand              : random coordinate/asset projection, averaged over trials.
    FCNN-c            : objective-only neural projection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
CORE_CODE = ROOT.parent / "core"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE_CODE))

import final_projection_figure_suite as fps  # noqa: E402
import run_custom_rho_sensitive_simplex_lp as simplex  # noqa: E402
import run_dynamic_k_baseline_compare_rho as olddyn  # noqa: E402
import run_natural_rho_sensitive_portfolio_lp as port  # noqa: E402
from iwata_sakaue_pelp_projection_compare import (  # noqa: E402
    CostOnlyProjectionNet,
    LPInstance,
    append_fixed_projection_rows,
    append_learned_projector_rows,
    ensure_full_solutions,
    pca_projection_from_training,
    random_column_projection,
    summarize_ratios_and_times,
    train_implicit_projection_model,
)


RHO_VALUES_DEFAULT = [0.10, 0.30, 0.50, 0.70, 0.90]

TITLE = {
    "packing": "Packing",
    "sc205": "SC205",
    "mode_switch_simplex": "Mode-Switch Simplex",
    "regime_portfolio": "Regime Portfolio",
}

CASE_ORDER = ["packing", "sc205", "mode_switch_simplex", "regime_portfolio"]

OLD_SUITE_SELECTED_PROJECTION = {
    "packing": "SGA",
    "maxflow": "SGA",
    "random_lp_A": "SGA",
    "random_lp_B": "SGA",
    "random_lp_C": "SGA",
    "random_lp_D": "SGA",
    "grow7": "SGA",
    "mincostflow": "PCA",
    "shortest_path": "PCA",
    "sc205": "PCA",
}

OLD_SUITE_DYNAMIC_CONFIGS = {
    "packing": olddyn.CASE_CONFIGS["packing"],
    "sc205": olddyn.CASE_CONFIGS["sc205"],
    "random_lp_A": olddyn.CaseConfig("reward_subspace", 25.0, 1e-8, 21, "SGA", "RandomLP A"),
    "random_lp_B": olddyn.CaseConfig("reward_subspace", 26.0, 1e-8, 22, "SGA", "RandomLP B"),
    "random_lp_C": olddyn.CaseConfig("reward_subspace", 25.0, 1e-8, 23, "SGA", "RandomLP C"),
    "random_lp_D": olddyn.CaseConfig("reward_subspace", 22.0, 1e-8, 24, "SGA", "RandomLP D"),
    "grow7": olddyn.CaseConfig("reward_subspace", 0.35, 0.08, 7, "SGA", "GROW7"),
    "mincostflow": olddyn.CaseConfig("reward_subspace", 10.0, 1e-8, 12, "PCA", "MinCostFlow"),
}

TITLE.update(
    {
        "random_lp_A": "RandomLP A",
        "random_lp_B": "RandomLP B",
        "random_lp_C": "RandomLP C",
        "random_lp_D": "RandomLP D",
        "grow7": "GROW7",
        "mincostflow": "MinCostFlow",
    }
)


def mean_se(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def high_rho_stress_value(rho: float, margin: float, start: float, cap: float) -> float:
    rho = float(rho)
    margin = float(max(0.0, margin))
    start = float(start)
    cap = float(cap)
    if margin <= 0.0 or rho <= start:
        return rho
    denom = max(1e-12, 0.90 - start)
    weight = min(1.0, max(0.0, (rho - start) / denom))
    return float(min(cap, rho + margin * weight))


def normalize_projection_raw(raw: pd.DataFrame, selected_projection: str) -> pd.DataFrame:
    out = raw.copy()
    out["source_method"] = out["method"]
    out.loc[out["method"].eq(selected_projection), "method"] = "DataDriven"
    out.loc[out["method"].eq("CostOnly"), "method"] = "FCNNc"
    out.loc[out["method"].str.startswith("Rand"), "method"] = "Rand"
    return out


def evaluate_old_suite_baselines(
    data: object,
    selected_projection: str,
    rho: float,
    k_budget: int,
    n_samples: int,
    out_dir: Path,
    include_fcnn: bool,
    neural_args: argparse.Namespace | None = None,
) -> pd.DataFrame:
    k_projection = max(1, int(k_budget))
    args = copy.copy(data.args)
    args.k_list = str(k_projection)
    # Keep the neural budget aligned with the current paper experiments.
    args.costonly_init = str(getattr(neural_args, "costonly_init", "plain_torch_random"))
    args.costonly_solver_space = str(getattr(neural_args, "costonly_solver_space", getattr(args, "costonly_solver_space", "auto")))
    args.costonly_warm_cols = int(getattr(neural_args, "costonly_warm_cols", getattr(args, "costonly_warm_cols", 2)))
    args.costonly_hidden_dim = int(getattr(neural_args, "hidden_dim", getattr(args, "costonly_hidden_dim", 32)))
    args.costonly_epochs = int(getattr(neural_args, "epochs", getattr(args, "costonly_epochs", 500)))
    args.costonly_batch_size = int(getattr(neural_args, "batch_size", getattr(args, "costonly_batch_size", 8)))
    args.costonly_lr = float(getattr(neural_args, "lr", getattr(args, "costonly_lr", 1e-3)))
    args.costonly_patience = int(getattr(neural_args, "patience", getattr(args, "costonly_patience", 120)))
    args.rand_trials = int(getattr(neural_args, "rand_trials", getattr(args, "rand_trials", 3)))
    fps_data = fps.PreparedCaseData(
        case=data.case,
        args=args,
        problem=data.problem,
        bundle=data.bundle,
        proj_train=data.proj_train,
        proj_val=data.proj_val,
        proj_test=data.proj_test,
    )
    methods = ["Rand", selected_projection]
    if include_fcnn:
        methods.append("CostOnly")
    result = fps.run_case_from_prepared(
        fps_data,
        out_dir,
        methods=methods,
        train_limit=int(n_samples),
        force=True,
        write_rank=False,
    )
    raw_path = out_dir / "raw_results.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else result["summary"].copy()
    keep = raw[raw["method"].isin(["Rand", selected_projection, "CostOnly"])].copy()
    keep = normalize_projection_raw(keep, selected_projection)
    keep["rho"] = float(rho)
    keep["K_budget"] = int(k_budget)
    keep["K_projection"] = int(k_projection)
    keep["retained_samples"] = np.nan
    return keep


def lp_instances_from_costs(costs: np.ndarray, prefix: str, A: np.ndarray, b: np.ndarray, Aeq: np.ndarray, beq: np.ndarray) -> List[LPInstance]:
    out: List[LPInstance] = []
    for i, c in enumerate(np.asarray(costs, dtype=float)):
        out.append(
            LPInstance(
                c=np.asarray(c, dtype=float).reshape(-1),
                A=np.asarray(A, dtype=float),
                b=np.asarray(b, dtype=float),
                A_eq=np.asarray(Aeq, dtype=float),
                b_eq=np.asarray(beq, dtype=float),
                name=f"{prefix}_{i:05d}",
            )
        )
    ensure_full_solutions(out)
    return out


def make_simplex_special() -> Dict[str, object]:
    train_meta, train_costs, test_meta, test_costs = simplex.build_costs(dim=12, baseline=0.50, nuisance=0.002)
    # Independent validation set with the same mode mix, but a tiny different
    # nuisance phase via rolling the costs.
    _val_meta, val_costs, _tmp_meta, _tmp_costs = simplex.build_costs(dim=12, baseline=0.50, nuisance=0.002)
    val_costs = np.roll(val_costs[:16], shift=1, axis=1)
    n = train_costs.shape[1]
    A = np.zeros((0, n), dtype=float)
    b = np.zeros(0, dtype=float)
    Aeq = np.ones((1, n), dtype=float)
    beq = np.asarray([1.0], dtype=float)
    train = lp_instances_from_costs(train_costs, "simplex_train", A, b, Aeq, beq)
    val = lp_instances_from_costs(val_costs, "simplex_val", A, b, Aeq, beq)
    test = lp_instances_from_costs(test_costs, "simplex_test", A, b, Aeq, beq)
    scores, _meta = simplex.ridge_mahalanobis_scores(
        train_costs,
        ridge_scale=0.01,
        ridge_abs=1e-10,
        support_mode="full",
    )
    return {
        "case": "mode_switch_simplex",
        "title": TITLE["mode_switch_simplex"],
        "train_costs": train_costs,
        "scores": scores,
        "train": train,
        "val": val,
        "test": test,
    }


def simplex_ours_raw(special: Dict[str, object], rho: float) -> Tuple[pd.DataFrame, int, int]:
    summary, raw0 = simplex.evaluate_simplex_exact(
        train_costs=np.asarray(special["train_costs"], dtype=float),
        train_scores=np.asarray(special["scores"], dtype=float),
        test_costs=np.vstack([inst.c for inst in special["test"]]),
        rho=float(rho),
        rule="conformal",
    )
    raw = pd.DataFrame(
        {
            "method": "OursEstC",
            "source_method": "OursEstC",
            "K": int(summary["d_star"]),
            "instance": [f"simplex_test_{int(i):05d}" for i in raw0["test_id"]],
            "objective": raw0["reduced_objective"].astype(float),
            "full_objective": raw0["full_objective"].astype(float),
            "objective_ratio": raw0["objective_ratio"].astype(float),
            "time": 0.0,
            "success": 1.0,
            "rho": float(rho),
            "K_budget": int(summary["d_star"]),
            "K_projection": int(max(1, summary["d_star"])),
            "retained_samples": int(summary["retained_samples"]),
        }
    )
    return raw, int(summary["d_star"]), int(summary["retained_samples"])


def make_portfolio_special() -> Dict[str, object]:
    lp = port.make_portfolio_lp(n_sectors=6, assets_per_sector=4, asset_cap=0.18, sector_cap=0.45)
    _train_meta, train_returns = port.generate_returns(
        28,
        [4, 4, 4],
        seed=123,
        n_sectors=6,
        assets_per_sector=4,
        shock_size=0.70,
        noise_scale=0.006,
        market_scale=0.020,
        sector_scale=0.012,
    )
    _val_meta, val_returns = port.generate_returns(
        12,
        [2, 2, 2],
        seed=555,
        n_sectors=6,
        assets_per_sector=4,
        shock_size=0.70,
        noise_scale=0.006,
        market_scale=0.020,
        sector_scale=0.012,
    )
    _test_meta, test_returns = port.generate_returns(
        36,
        [16, 16, 16],
        seed=991,
        n_sectors=6,
        assets_per_sector=4,
        shock_size=0.70,
        noise_scale=0.006,
        market_scale=0.020,
        sector_scale=0.012,
    )
    A = np.vstack([np.asarray(lp["A_ub"], dtype=float), np.eye(int(lp["n_assets"]), dtype=float)])
    b = np.concatenate([np.asarray(lp["b_ub"], dtype=float), np.full(int(lp["n_assets"]), float(lp["asset_cap"]))])
    Aeq = np.asarray(lp["A_eq"], dtype=float)
    beq = np.asarray(lp["b_eq"], dtype=float)
    train = lp_instances_from_costs(train_returns, "portfolio_train", A, b, Aeq, beq)
    val = lp_instances_from_costs(val_returns, "portfolio_val", A, b, Aeq, beq)
    test = lp_instances_from_costs(test_returns, "portfolio_test", A, b, Aeq, beq)
    scores, _meta = port.ridge_mahalanobis_scores(train_returns, ridge_scale=0.01, ridge_abs=1e-10)
    return {
        "case": "regime_portfolio",
        "title": TITLE["regime_portfolio"],
        "lp": lp,
        "train_returns": train_returns,
        "scores": scores,
        "train": train,
        "val": val,
        "test": test,
    }


def portfolio_ours_raw(special: Dict[str, object], rho: float) -> Tuple[pd.DataFrame, int, int]:
    scores = np.asarray(special["scores"], dtype=float)
    k_idx = port.calibration_index(scores.size, float(rho), rule="conformal")
    radius2 = float(np.sort(scores)[max(0, k_idx - 1)])
    retained = np.flatnonzero(scores <= radius2 + 1e-12)
    if retained.size == 0:
        retained = np.asarray([int(np.argmin(scores))], dtype=int)
    stage = port.learn_exact_space(
        special["lp"],
        np.asarray(special["train_returns"], dtype=float),
        retained=retained,
        tol=1e-8,
    )
    raw0 = port.evaluate(
        special["lp"],
        np.vstack([inst.c for inst in special["test"]]),
        stage,
    )
    k_budget = int(stage["U"].shape[1])
    raw = pd.DataFrame(
        {
            "method": "OursEstC",
            "source_method": "OursEstC",
            "K": k_budget,
            "instance": [f"portfolio_test_{int(i):05d}" for i in raw0["test_id"]],
            "objective": raw0["reduced_objective"].astype(float),
            "full_objective": raw0["full_objective"].astype(float),
            "objective_ratio": raw0["objective_ratio"].astype(float),
            "time": raw0["time"].astype(float),
            "success": raw0["success"].astype(float),
            "rho": float(rho),
            "K_budget": k_budget,
            "K_projection": int(max(1, k_budget)),
            "retained_samples": int(retained.size),
        }
    )
    return raw, k_budget, int(retained.size)


def evaluate_special_baselines(
    special: Dict[str, object],
    rho: float,
    k_budget: int,
    out_dir: Path,
    include_fcnn: bool,
    rand_trials: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    hidden_dim: int,
    seed: int,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = special["train"]
    val = special["val"]
    test = special["test"]
    k_projection = max(1, int(k_budget))
    rows: List[Dict[str, object]] = []
    fit_rows: List[Dict[str, object]] = []
    n_vars = train[0].n_vars

    for tr in range(int(rand_trials)):
        rng = np.random.default_rng(int(seed) + 10000 + 97 * tr + k_projection)
        P_rand = random_column_projection(n_vars, k_projection, rng)
        append_fixed_projection_rows(rows, f"Rand#{tr + 1}", P_rand, test, k_projection)
    fit_rows.append({"method": "Rand", "K": k_projection, "fit_time": 0.0, "epochs": 0})

    t0 = time.perf_counter()
    P_pca = pca_projection_from_training(train, k_projection)
    fit_rows.append({"method": "DataDriven", "K": k_projection, "fit_time": time.perf_counter() - t0, "epochs": 1})
    append_fixed_projection_rows(rows, "DataDriven", P_pca, test, k_projection)
    np.save(out_dir / f"DataDriven_PCA_K{k_projection}.npy", P_pca)

    if include_fcnn:
        import torch

        device = torch.device("cpu")
        t0 = time.perf_counter()
        model = CostOnlyProjectionNet(n_vars, k_projection, hidden_dim=int(hidden_dim))
        model, hist = train_implicit_projection_model(
            model,
            train,
            val,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            device=device,
            seed=int(seed) + 3000 + k_projection,
            patience=int(patience),
            verbose=False,
        )
        fit_rows.append(
            {
                "method": "FCNNc",
                "K": k_projection,
                "fit_time": time.perf_counter() - t0,
                "epochs": int(len(hist.get("val_ratio", []))),
            }
        )
        append_learned_projector_rows(rows, "FCNNc", model, test, k_projection, device)
        pd.DataFrame(hist).to_csv(out_dir / f"history_FCNNc_K{k_projection}.csv", index=False)
        torch.save(model.state_dict(), out_dir / f"FCNNc_K{k_projection}.pt")

    raw = pd.DataFrame(rows)
    raw["method"] = raw["method"].str.replace(r"Rand#\d+", "Rand", regex=True)
    raw["source_method"] = raw["method"]
    raw["rho"] = float(rho)
    raw["K_budget"] = int(k_budget)
    raw["K_projection"] = int(k_projection)
    raw["retained_samples"] = np.nan
    raw.to_csv(out_dir / "raw_projection_baselines.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out_dir / "fit_times_projection_baselines.csv", index=False)
    return raw


def summarize_case_raw(raw: pd.DataFrame, case: str, title: str, n_samples: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for (rho, method), sub in raw.groupby(["rho", "method"], sort=True):
        summary = summarize_ratios_and_times(sub.to_dict("records"))
        summary["case"] = case
        summary["title"] = title
        summary["rho"] = float(rho)
        summary["n_samples"] = int(n_samples)
        summary["K_budget"] = int(sub["K_budget"].iloc[0])
        summary["K_projection"] = int(sub["K_projection"].iloc[0])
        summary["retained_samples"] = (
            int(sub["retained_samples"].dropna().iloc[0]) if sub["retained_samples"].notna().any() else np.nan
        )
        source = sub["source_method"].dropna().astype(str).unique()
        summary["source_method"] = source[0] if len(source) else str(method)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_old_case(case_name: str, rhos: Sequence[float], n_samples: int, out_dir: Path, include_fcnn: bool, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = OLD_SUITE_DYNAMIC_CONFIGS[case_name]
    data = olddyn.prepare_case(case_name, config, int(n_samples))
    raw_frames: List[pd.DataFrame] = []
    for rho in rhos:
        rho_eff = high_rho_stress_value(
            float(rho),
            margin=float(getattr(args, "ours_high_rho_margin", 0.0)),
            start=float(getattr(args, "ours_high_rho_start", 0.50)),
            cap=float(getattr(args, "ours_high_rho_cap", 0.98)),
        )
        if abs(rho_eff - float(rho)) > 1e-12:
            print(f"[dynamic-K four LP] {case_name}, rho={rho} (ours calibration rho_eff={rho_eff:.4f})", flush=True)
        else:
            print(f"[dynamic-K four LP] {case_name}, rho={rho}", flush=True)
        ours_raw, d, _retained = olddyn.evaluate_ours_at_rho(data, config, float(rho_eff), int(n_samples))
        ours_raw = ours_raw.copy()
        ours_raw["rho"] = float(rho)
        ours_raw["rho_effective"] = float(rho_eff)
        ours_raw["K_projection"] = np.maximum(1, ours_raw["K_budget"].astype(int))
        base_raw = evaluate_old_suite_baselines(
            data,
            selected_projection=config.selected_projection,
            rho=float(rho),
            k_budget=int(d),
            n_samples=int(n_samples),
            out_dir=out_dir / "baseline_runs" / f"{case_name}_rho{str(rho).replace('.', 'p')}_K{max(1, int(d))}",
            include_fcnn=include_fcnn,
            neural_args=args,
        )
        base_raw["rho_effective"] = float(rho_eff)
        raw_frames.extend([ours_raw, base_raw])
    raw = pd.concat(raw_frames, ignore_index=True)
    raw["case"] = case_name
    raw["title"] = config.title
    summary = summarize_case_raw(raw, case_name, config.title, int(n_samples))
    return raw, summary


def run_special_case(case_name: str, rhos: Sequence[float], n_samples: int, out_dir: Path, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    special = make_simplex_special() if case_name == "mode_switch_simplex" else make_portfolio_special()
    raw_frames: List[pd.DataFrame] = []
    for rho in rhos:
        print(f"[dynamic-K four LP] {case_name}, rho={rho}", flush=True)
        if case_name == "mode_switch_simplex":
            ours_raw, d, _retained = simplex_ours_raw(special, float(rho))
        else:
            ours_raw, d, _retained = portfolio_ours_raw(special, float(rho))
        base_raw = evaluate_special_baselines(
            special,
            rho=float(rho),
            k_budget=int(d),
            out_dir=out_dir / "baseline_runs" / f"{case_name}_rho{str(rho).replace('.', 'p')}_K{max(1, int(d))}",
            include_fcnn=bool(args.include_fcnn),
            rand_trials=int(args.rand_trials),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            patience=int(args.patience),
            hidden_dim=int(args.hidden_dim),
            seed=int(args.seed) + (5000 if case_name == "regime_portfolio" else 4000),
        )
        raw_frames.extend([ours_raw, base_raw])
    raw = pd.concat(raw_frames, ignore_index=True)
    raw["case"] = case_name
    raw["title"] = TITLE[case_name]
    summary = summarize_case_raw(raw, case_name, TITLE[case_name], int(n_samples))
    return raw, summary


def plot_summary(summary: pd.DataFrame, out_path: Path, rhos: Sequence[float], case_order: Sequence[str]) -> None:
    methods = ["OursEstC", "DataDriven", "Rand", "FCNNc"]
    labels = {
        "OursEstC": "OursEstC",
        "DataDriven": "data-driven proj",
        "Rand": "Rand",
        "FCNNc": "FCNN-c",
    }
    styles = {
        "OursEstC": ("#d62728", "D", (0, (6, 2))),
        "DataDriven": ("#1f77b4", "s", (0, (1, 2))),
        "Rand": ("#6f7d55", "x", (0, (2, 2))),
        "FCNNc": ("#8c564b", "v", (0, (4, 1, 1, 1))),
    }
    rho_labels = [f"{r:.2f}" for r in rhos]
    x = np.arange(len(rhos), dtype=float)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )
    ncols = len(case_order)
    fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 6.4), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for j, case in enumerate(case_order):
        sub_case = summary[summary["case"] == case].copy()
        ax = axes[0, j]
        for method in methods:
            sub = sub_case[sub_case["method"] == method].sort_values("rho")
            if sub.empty:
                continue
            color, marker, linestyle = styles[method]
            y = sub["objective_ratio_mean"].to_numpy(float)
            se = sub["objective_ratio_se"].fillna(0.0).to_numpy(float)
            ax.plot(x[: len(y)], y, color=color, marker=marker, linestyle=linestyle, linewidth=2.0, label=labels[method])
            ax.fill_between(x[: len(y)], y - se, y + se, color=color, alpha=0.09, linewidth=0)
        title = str(sub_case["title"].dropna().iloc[0]) if "title" in sub_case and not sub_case.empty else TITLE.get(case, case)
        ax.set_title(f"({chr(97 + j)}) {title}")
        ax.set_xticks(x)
        ax.set_xticklabels(rho_labels, rotation=35)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.set_ylabel("Objective ratio")

        ax2 = axes[1, j]
        ours = sub_case[sub_case["method"] == "OursEstC"].sort_values("rho")
        d = ours["K_budget"].to_numpy(float)
        ax2.plot(x[: len(d)], d, color="#d62728", marker="D", linestyle="--", linewidth=2.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(rho_labels, rotation=35)
        ax2.set_xlabel("Target outside mass rho")
        ax2.grid(True, alpha=0.25)
        if j == 0:
            ax2.set_ylabel("K = learned dimension d(rho)")
        if d.size:
            ax2.set_ylim(-0.5, float(np.nanmax(d)) + max(1.0, 0.08 * float(np.nanmax(d))))
    handles: List[object] = []
    labs: List[str] = []
    for ax in axes.reshape(-1):
        h_ax, l_ax = ax.get_legend_handles_labels()
        for h, lab in zip(h_ax, l_ax):
            if lab not in labs:
                handles.append(h)
                labs.append(lab)
    if handles:
        fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(handles), frameon=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="results_dynamic_k_baseline_compare_four_lp_n40_20260507")
    parser.add_argument("--cases", default=",".join(CASE_ORDER))
    parser.add_argument("--rho_list", default=",".join(str(v) for v in RHO_VALUES_DEFAULT))
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--include_fcnn", action="store_true", default=True)
    parser.add_argument("--rand_trials", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--costonly_init", default="plain_torch_random")
    parser.add_argument("--costonly_solver_space", default="auto")
    parser.add_argument("--costonly_warm_cols", type=int, default=2)
    parser.add_argument("--ours_high_rho_margin", type=float, default=0.0)
    parser.add_argument("--ours_high_rho_start", type=float, default=0.50)
    parser.add_argument("--ours_high_rho_cap", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rhos = [float(v.strip()) for v in str(args.rho_list).split(",") if v.strip()]
    cases = [v.strip() for v in str(args.cases).split(",") if v.strip()]

    raw_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    for case in cases:
        if case in OLD_SUITE_DYNAMIC_CONFIGS:
            raw, summ = run_old_case(case, rhos, int(args.n_samples), out_dir / case, include_fcnn=bool(args.include_fcnn), args=args)
        elif case in {"mode_switch_simplex", "regime_portfolio"}:
            raw, summ = run_special_case(case, rhos, int(args.n_samples), out_dir / case, args)
        else:
            raise ValueError(f"Unknown case: {case}")
        raw_frames.append(raw)
        summary_frames.append(summ)

    raw_all = pd.concat(raw_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    raw_all.to_csv(out_dir / "dynamic_k_baseline_compare_four_lp_raw.csv", index=False)
    summary_all.to_csv(out_dir / "dynamic_k_baseline_compare_four_lp_summary.csv", index=False)
    plot_summary(summary_all, out_dir / "figure_dynamic_k_baseline_compare_four_lp.png", rhos, cases)
    manifest = {
        "experiment": "dynamic_k_baseline_compare_four_lp",
        "n_samples": int(args.n_samples),
        "rho_values": rhos,
        "cases": cases,
        "baselines": ["data-driven proj", "Rand", "FCNN-c"],
        "rule": "For each rho, baselines use K=d(rho), where d(rho) is learned by OursEstC; if d=0, projection baselines use K=1 and this is recorded separately.",
        "special_lp_parameters": {
            "simplex": {"dim": 12, "baseline": 0.50, "nuisance": 0.002},
            "portfolio": {"n_assets": 24, "asset_cap": 0.18, "sector_cap": 0.45, "shock_size": 0.70},
        },
        "old_case_configs": {k: asdict(v) for k, v in OLD_SUITE_DYNAMIC_CONFIGS.items() if k in cases},
        "neural": {
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "patience": int(args.patience),
        },
        "ours_high_rho_stress": {
            "margin": float(args.ours_high_rho_margin),
            "start": float(args.ours_high_rho_start),
            "cap": float(args.ours_high_rho_cap),
            "description": "If margin>0, OursEstC uses rho_eff=rho+margin*clip((rho-start)/(0.90-start),0,1), capped by cap; the plotted rho remains the nominal target.",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary_all[["case", "method", "rho", "K_budget", "K_projection", "objective_ratio_mean", "objective_ratio_se"]].to_string(index=False))
    print(f"Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
