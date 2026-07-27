#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare OursEstC against projection baselines at K=d(rho).

For each case and rho, the current beyond-prior Algorithm 1 first learns a
dimension d(rho).  The data-driven projection baseline is then trained on the
same n=40 observed costs with its reduced dimension set to that same d(rho).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

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
import run_beyond_prior_sample_sweeps_paper8 as nested  # noqa: E402
import run_latest_alg1_tuned_beyond_prior_rank_growth as tuned  # noqa: E402
import run_ridge_calibrated_beyond_prior_ours as ridge_ours  # noqa: E402
import run_subgaussian_beyond_prior_suite as bp  # noqa: E402
from iwata_sakaue_pelp_projection_compare import summarize_ratios_and_times  # noqa: E402


RHO_VALUES = [0.10, 0.30, 0.50, 0.70, 0.90]
RHO_LABELS = ["0.10", "0.30", "0.50", "0.70", "0.90"]


@dataclass(frozen=True)
class CaseConfig:
    cost_space: str
    gaussian_std_scale: float
    indep_tol: float
    seed: int
    selected_projection: str
    title: str


CASE_CONFIGS: Dict[str, CaseConfig] = {
    "packing": CaseConfig("reward_subspace", 0.40, 0.49, 9, "SGA", "Packing"),
    "sc205": CaseConfig("full_standard", 50.0, 1e-8, 17, "PCA", "SC205"),
}


def prepare_case(case_name: str, config: CaseConfig, n_samples: int) -> object:
    args = tuned.base_user_args(seed=int(config.seed), n_train=int(n_samples))
    args.n_val = 6
    args.n_test = 12
    args.k_list = "10,20,30,50"
    args.rho = 0.10
    args.calibration_rule = "conformal"
    args.support_mode = "empirical_span"
    args.hidden_dim = 32
    args.epochs = 500
    args.batch_size = 8
    args.lr = 1e-3
    args.patience = 120
    base = nested.setup_base_args(args, int(n_samples))
    case = nested.case_with_sample_size(nested.get_cases()[case_name], args, int(n_samples))
    return bp.prepare_subgaussian_case(
        case,
        base,
        pilot_n=int(args.pilot_n),
        rho=float(args.rho),
        radius_scale=float(args.radius_scale),
        gaussian_std_scale=float(config.gaussian_std_scale),
        svd_tol=float(args.svd_tol),
        same_est_train_samples=True,
        enclose_train_in_estimated_prior=True,
        cost_space=str(config.cost_space),
    )


def evaluate_ours_at_rho(data: object, config: CaseConfig, rho: float, n_samples: int) -> tuple[pd.DataFrame, int, int]:
    costs = np.asarray(data.bundle.Ctrain[: int(n_samples)], dtype=float)
    prior, _prior_meta, train_scores = ridge_ours.estimate_ridge_calibrated_prior(
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
    retained = np.flatnonzero(train_scores <= float(prior.radius) ** 2 + 1e-10)
    if retained.size == 0 and train_scores.size > 0:
        retained = np.asarray([int(np.argmin(train_scores))], dtype=int)
    stage1 = tuned.latest_alg1_with_trace(
        data.bundle.stdlp,
        costs[retained],
        tol=float(config.indep_tol),
        anchor_cost=prior.mu,
    )
    raw = tuned.evaluate(data, stage1)
    raw["method"] = "OursEstC"
    raw["source_method"] = "OursEstC"
    raw["rho"] = float(rho)
    raw["K_budget"] = int(stage1.U.shape[1])
    raw["retained_samples"] = int(retained.size)
    return raw, int(stage1.U.shape[1]), int(retained.size)


def evaluate_baseline_at_k(
    data: object,
    config: CaseConfig,
    rho: float,
    k_budget: int,
    n_samples: int,
    case_out: Path,
    include_fcnn: bool,
) -> pd.DataFrame:
    k = max(1, int(k_budget))
    args = copy.copy(data.args)
    args.k_list = str(k)
    fps_data = fps.PreparedCaseData(
        case=data.case,
        args=args,
        problem=data.problem,
        bundle=data.bundle,
        proj_train=data.proj_train,
        proj_val=data.proj_val,
        proj_test=data.proj_test,
    )
    methods = [config.selected_projection]
    if include_fcnn:
        methods.append("CostOnly")
    result = fps.run_case_from_prepared(
        fps_data,
        case_out,
        methods=methods,
        train_limit=int(n_samples),
        force=True,
        write_rank=False,
    )
    summary = result["summary"].copy()
    raw_path = case_out / "raw_results.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    if raw.empty:
        return raw
    raw = raw[raw["method"].isin(methods)].copy()
    raw["source_method"] = raw["method"]
    raw.loc[raw["method"] == config.selected_projection, "method"] = "DataDriven"
    raw.loc[raw["source_method"] == "CostOnly", "method"] = "FCNNc"
    raw["rho"] = float(rho)
    raw["K_budget"] = int(k_budget)
    raw["retained_samples"] = np.nan
    return raw


def summarize(raw: pd.DataFrame, case_name: str, config: CaseConfig, n_samples: int) -> pd.DataFrame:
    frames = []
    for (rho, method), sub in raw.groupby(["rho", "method"], sort=True):
        s = summarize_ratios_and_times(sub.to_dict("records"))
        s["case"] = case_name
        s["title"] = config.title
        s["rho"] = float(rho)
        s["n_samples"] = int(n_samples)
        s["K_budget"] = int(sub["K_budget"].iloc[0])
        s["retained_samples"] = (
            int(sub["retained_samples"].dropna().iloc[0]) if sub["retained_samples"].notna().any() else np.nan
        )
        source_vals = sub["source_method"].dropna().unique()
        s["source_method"] = source_vals[0] if len(source_vals) else method
        frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot(summary: pd.DataFrame, out_path: Path) -> None:
    cases = ["packing", "sc205"]
    methods = ["OursEstC", "DataDriven", "FCNNc"]
    labels = {"OursEstC": "OursEstC", "DataDriven": "data-driven proj", "FCNNc": "FCNN-c"}
    style = {
        "OursEstC": ("#d62728", "D", (0, (6, 2))),
        "DataDriven": ("#1f77b4", "s", (0, (1, 2))),
        "FCNNc": ("#8c564b", "v", (0, (4, 1, 1, 1))),
    }
    x = np.arange(len(RHO_VALUES), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.3), constrained_layout=True)
    for j, case in enumerate(cases):
        sub_case = summary[summary["case"] == case].copy()
        ax = axes[0, j]
        for method in methods:
            sub = sub_case[sub_case["method"] == method].sort_values("rho")
            if sub.empty:
                continue
            color, marker, linestyle = style[method]
            y = sub["objective_ratio_mean"].to_numpy(float)
            se = sub["objective_ratio_se"].fillna(0.0).to_numpy(float)
            ax.plot(x[: len(y)], y, color=color, marker=marker, linestyle=linestyle, linewidth=2.0, label=labels[method])
            ax.fill_between(x[: len(y)], y - se, y + se, color=color, alpha=0.10, linewidth=0)
        ax.set_title(f"({chr(97 + j)}) {CASE_CONFIGS[case].title}")
        ax.set_xticks(x)
        ax.set_xticklabels(RHO_LABELS, rotation=35)
        ax.set_ylabel("Objective ratio")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.02, 1.05)

        ax2 = axes[1, j]
        ours = sub_case[sub_case["method"] == "OursEstC"].sort_values("rho")
        d = ours["K_budget"].to_numpy(float)
        ax2.plot(x[: len(d)], d, color="#d62728", marker="D", linestyle="--", linewidth=2.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(RHO_LABELS, rotation=35)
        ax2.set_xlabel("Target outside mass rho")
        ax2.set_ylabel("K = learned dimension d(rho)")
        ax2.grid(True, alpha=0.25)
        if d.size:
            ax2.set_ylim(-0.5, float(np.nanmax(d)) + 2.0)
    handles, labels_seen = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_seen, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(handles), frameon=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default="results_dynamic_k_baseline_compare_rho_n40_20260507")
    p.add_argument("--cases", default="packing,sc205")
    p.add_argument("--rho_list", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--n_samples", type=int, default=40)
    p.add_argument("--include_fcnn", action="store_true")
    args = p.parse_args()

    global RHO_VALUES, RHO_LABELS
    RHO_VALUES = [float(v.strip()) for v in args.rho_list.split(",") if v.strip()]
    RHO_LABELS = [f"{v:.2f}" for v in RHO_VALUES]
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_names = [v.strip() for v in args.cases.split(",") if v.strip()]
    raw_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    for case_name in case_names:
        config = CASE_CONFIGS[case_name]
        print(f"[dynamic K baseline] prepare {case_name}: {config}", flush=True)
        data = prepare_case(case_name, config, int(args.n_samples))
        case_raw_frames: List[pd.DataFrame] = []
        for rho in RHO_VALUES:
            print(f"[dynamic K baseline] {case_name}, rho={rho}", flush=True)
            ours_raw, d, retained = evaluate_ours_at_rho(data, config, rho, int(args.n_samples))
            base_raw = evaluate_baseline_at_k(
                data,
                config,
                rho,
                d,
                int(args.n_samples),
                out_dir / "baseline_runs" / f"{case_name}_rho{str(rho).replace('.', 'p')}_K{max(1, d)}",
                bool(args.include_fcnn),
            )
            case_raw_frames.extend([ours_raw, base_raw])
        case_raw = pd.concat(case_raw_frames, ignore_index=True)
        case_raw["case"] = case_name
        case_raw["title"] = config.title
        raw_frames.append(case_raw)
        summary_frames.append(summarize(case_raw, case_name, config, int(args.n_samples)))
    raw = pd.concat(raw_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    raw.to_csv(out_dir / "dynamic_k_baseline_compare_raw.csv", index=False)
    summary.to_csv(out_dir / "dynamic_k_baseline_compare_summary.csv", index=False)
    plot(summary, out_dir / "figure_dynamic_k_baseline_compare_rho.png")
    manifest = {
        "experiment": "dynamic_k_baseline_compare_rho",
        "n_samples": int(args.n_samples),
        "rho_values": RHO_VALUES,
        "cases": case_names,
        "include_fcnn": bool(args.include_fcnn),
        "case_configs": {k: asdict(CASE_CONFIGS[k]) for k in case_names},
        "rule": "For every case and rho, first learn d(rho) with OursEstC; then train projection baselines at K=d(rho).",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[["case", "method", "rho", "K_budget", "objective_ratio_mean", "objective_ratio_se"]].to_string(index=False))
    print(f"Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
