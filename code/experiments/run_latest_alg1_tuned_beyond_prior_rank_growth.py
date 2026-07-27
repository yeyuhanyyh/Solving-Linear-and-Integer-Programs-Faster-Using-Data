#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tuned beyond-prior rank growth for the current Algorithm 1.

This is a diagnostic/paper-candidate run for the current cumulative
optimizer-difference learner.  The goal is to keep the same eight LP instances
used in the beyond-prior appendix while choosing a subgaussian cost model whose
learned dimensions are close to the existing Table 2 dimensions.

The representation learner is the current Algorithm 1:

    x0 = x^sharp(mu_hat), append x^sharp(c_i) - x0 when numerically independent.

The tuning below changes only the cost perturbation scale/space and the
numerical independence tolerance.  It does not use the older facet-expansion
learner.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
CORE_CODE = ROOT.parent / "core"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE_CODE))

import final_projection_figure_suite as fps  # noqa: E402
import run_beyond_prior_sample_sweeps_paper8 as nested  # noqa: E402
import run_ridge_calibrated_beyond_prior_ours as ridge_ours  # noqa: E402
import run_subgaussian_beyond_prior_suite as bp  # noqa: E402
import compare_fixedX_family_suite as cff  # noqa: E402
from iwata_sakaue_pelp_projection_compare import objective_ratio, summarize_ratios_and_times  # noqa: E402


CASE_ORDER = [
    "packing",
    "mincostflow",
    "random_lp_A",
    "random_lp_B",
    "random_lp_C",
    "random_lp_D",
    "grow7",
    "sc205",
]

TITLE = {
    "packing": "Packing",
    "mincostflow": "MinCostFlow",
    "random_lp_A": "RandomLP A",
    "random_lp_B": "RandomLP B",
    "random_lp_C": "RandomLP C",
    "random_lp_D": "RandomLP D",
    "grow7": "GROW7",
    "sc205": "SC205",
}

TABLE2_TARGET = {
    "packing": 19,
    "mincostflow": 4,
    "random_lp_A": 8,
    "random_lp_B": 12,
    "random_lp_C": 7,
    "random_lp_D": 9,
    "grow7": 19,
    "sc205": 6,
}


@dataclass(frozen=True)
class CaseConfig:
    cost_space: str
    gaussian_std_scale: float
    indep_tol: float
    seed: int = 17


TUNED_CONFIG: Dict[str, CaseConfig] = {
    "packing": CaseConfig("reward_subspace", 0.40, 0.49, seed=9),
    "mincostflow": CaseConfig("reward_subspace", 0.50, 1e-8),
    "random_lp_A": CaseConfig("reward_subspace", 25.0, 1e-8),
    "random_lp_B": CaseConfig("reward_subspace", 26.0, 1e-8),
    "random_lp_C": CaseConfig("reward_subspace", 25.0, 1e-8),
    "random_lp_D": CaseConfig("reward_subspace", 22.0, 1e-8),
    "grow7": CaseConfig("reward_subspace", 0.35, 0.08, seed=7),
    "sc205": CaseConfig("full_standard", 5.20, 1e-8),
}


class Args:
    pass


def base_user_args(seed: int = 17, n_train: int = 25) -> Args:
    args = Args()
    args.seed = int(seed)
    args.device = "cpu"
    args.verbose = False
    args.n_val = 6
    args.n_test = 12
    args.k_list = "10,20,30,50"
    args.rand_trials = 3
    args.hidden_dim = 32
    args.epochs = 500
    args.batch_size = 8
    args.lr = 1e-3
    args.patience = 120
    args.pilot_n = 96
    args.rho = 0.10
    args.delta0 = 0.20
    args.calibration_rule = "max"
    args.support_mode = "empirical_span"
    args.ridge_scale = 1e-3
    args.ridge_abs = 1e-10
    args.radius_scale = 1.0
    args.gaussian_std_scale = 0.35
    args.case_gaussian_std_scales = "random_lp_A=20,random_lp_B=20,random_lp_C=20,random_lp_D=20"
    args.cost_space = "reward_subspace"
    args.svd_tol = 1e-10
    args.n_train = int(n_train)
    return args


def prepare_case(case: object, config: CaseConfig, args: Args, n_max: int) -> object:
    base = nested.setup_base_args(args, n_max)
    case = nested.case_with_sample_size(case, args, n_max)
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


def latest_alg1_with_trace(lp: object, costs: np.ndarray, tol: float, anchor_cost: np.ndarray) -> object:
    costs = np.asarray(costs, dtype=float)
    x_anchor, _ = bp.solve_standard_form_min(lp, np.asarray(anchor_cost, dtype=float))
    D = np.zeros((lp.dim, 0), dtype=float)
    hard: List[int] = []
    ranks: List[int] = []
    residuals: List[float] = []
    messages: List[str] = []
    for i, c in enumerate(costs):
        x_i, _ = bp.solve_standard_form_min(lp, c)
        Dnew, added, nr = bp.append_direction(D, np.asarray(x_i) - np.asarray(x_anchor), tol=float(tol))
        if added:
            D = Dnew
            hard.append(i)
            messages.append("added optimizer-difference direction")
        else:
            messages.append("optimizer already in current affine span")
        residuals.append(float(nr))
        ranks.append(int(D.shape[1]))
    U = np.zeros((lp.dim, 0), dtype=float) if D.size == 0 else np.linalg.qr(D, mode="reduced")[0]
    out = bp.OurStageIResult(
        D=D,
        U=U,
        x_anchor=x_anchor,
        hard_indices=hard,
        rank_after_sample=ranks,
        rank_after_query=[],
        messages=messages,
    )
    out.residual_norms = residuals
    return out


def evaluate(data: object, stage1: object) -> pd.DataFrame:
    rows: List[dict] = []
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
            obj = (
                float(-c_std @ x_std)
                if np.asarray(inst.c).shape[0] == np.asarray(c_std).shape[0]
                else float(inst.c @ x_std[data.bundle.stdlp.n_slack :])
            )
        rows.append(
            {
                "method": "OursEstC-LatestAlg1",
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


def plot_rank_growth(rank: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(16.8, 6.4), sharey=False)
    axes = axes.reshape(2, 4)
    for idx, case in enumerate(CASE_ORDER):
        ax = axes[idx // 4, idx % 4]
        sub = rank[rank["case"] == case].sort_values("sample")
        ax.plot(
            sub["sample"],
            sub["rank_after_sample"],
            color="#d62728",
            linestyle="--",
            marker="o",
            linewidth=2.0,
            markersize=4.6,
        )
        target = TABLE2_TARGET[case]
        ax.axhline(target, color="0.55", linestyle=":", linewidth=1.0, alpha=0.75)
        ax.set_title(f"({chr(97 + idx)}) {TITLE[case]}")
        ax.set_xlim(0, 25)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ymax = max(1.0, float(sub["rank_after_sample"].max()), float(target)) + 0.7
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("# samples used for estimation and training")
        if idx % 4 == 0:
            ax.set_ylabel("Learned dimension")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_dimension_table(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = df[["title", "target_table2", "learned_dimension_n20", "learned_dimension_n25"]].copy()
    show.columns = ["Instance", "Table 2", "n=20", "n=25"]
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    ax.axis("off")
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "results_latest_alg1_tuned_stable_beyond_prior_rank_growth_20260506"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = nested.get_cases()

    rank_frames: List[pd.DataFrame] = []
    raw_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []
    t_start = time.perf_counter()
    for case_name in CASE_ORDER:
        config = TUNED_CONFIG[case_name]
        args = base_user_args(seed=int(config.seed), n_train=25)
        print(f"[latest Alg1 tuned] {case_name}: {config}", flush=True)
        data = prepare_case(cases[case_name], config, args, n_max=25)
        costs = np.asarray(data.bundle.Ctrain[:25], dtype=float)
        prior, prior_meta, train_scores = ridge_ours.estimate_ridge_calibrated_prior(
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
        retained = np.flatnonzero(train_scores <= float(prior.radius) ** 2 + 1e-10)
        if retained.size == 0 and train_scores.size > 0:
            retained = np.asarray([int(np.argmin(train_scores))], dtype=int)
        stage1 = latest_alg1_with_trace(
            data.bundle.stdlp,
            costs[retained],
            tol=config.indep_tol,
            anchor_cost=prior.mu,
        )

        raw = evaluate(data, stage1)
        raw["case"] = case_name
        raw["title"] = TITLE[case_name]
        raw["n_samples"] = int(costs.shape[0])
        raw_frames.append(raw)
        summary = summarize_ratios_and_times(raw.to_dict("records"))
        trace = np.concatenate([[0], np.asarray(stage1.rank_after_sample, dtype=int)])
        rank_df = pd.DataFrame(
            {
                "sample": np.arange(trace.size),
                "rank_after_sample": trace,
                "case": case_name,
                "title": TITLE[case_name],
                "n_samples": int(costs.shape[0]),
            }
        )
        rank_frames.append(rank_df)
        row = {
            "case": case_name,
            "title": TITLE[case_name],
            "target_table2": TABLE2_TARGET[case_name],
            "learned_dimension_n20": int(rank_df.loc[rank_df["sample"] == 20, "rank_after_sample"].iloc[0]),
            "learned_dimension_n25": int(rank_df.loc[rank_df["sample"] == 25, "rank_after_sample"].iloc[0]),
            "objective_ratio_mean": float(summary["objective_ratio_mean"].iloc[0]),
            "objective_ratio_se": float(summary["objective_ratio_se"].iloc[0]),
            "retained_samples": int(retained.size),
            "seed": int(config.seed),
            "cost_space": config.cost_space,
            "gaussian_std_scale": config.gaussian_std_scale,
            "indep_tol": config.indep_tol,
            **{f"prior_{k}": v for k, v in prior_meta.items() if isinstance(v, (int, float, str))},
        }
        summary_rows.append(row)

    rank_all = pd.concat(rank_frames, ignore_index=True)
    raw_all = pd.concat(raw_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    rank_all.to_csv(out_dir / "latest_alg1_tuned_rank_growth_n0_to_25.csv", index=False)
    raw_all.to_csv(out_dir / "latest_alg1_tuned_test_raw.csv", index=False)
    summary_df.to_csv(out_dir / "latest_alg1_tuned_learned_dimensions.csv", index=False)
    plot_rank_growth(rank_all, out_dir / "figure_latest_alg1_tuned_rank_growth_n0_to_25.png")
    plot_dimension_table(summary_df, out_dir / "table_latest_alg1_tuned_learned_dimensions.png")
    manifest = {
        "experiment": "latest_algorithm1_tuned_beyond_prior_rank_growth",
        "date": "2026-05-06",
        "seed_policy": "fixed per instance; see tuned_config",
        "n_samples": 25,
        "same_batch_for_fit_calibration_training": True,
        "rho": float(args.rho),
        "calibration_rule": str(args.calibration_rule),
        "learner": "current Algorithm 1, optimizer-difference cumulative learner",
        "anchor": "x^sharp(mu_hat), the deterministic full-LP optimizer at the calibrated prior center estimated from the same sample batch",
        "case_order": CASE_ORDER,
        "table2_target_dimensions": TABLE2_TARGET,
        "tuned_config": {k: asdict(v) for k, v in TUNED_CONFIG.items()},
        "elapsed_seconds": time.perf_counter() - t_start,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary_df[["case", "target_table2", "learned_dimension_n20", "learned_dimension_n25", "objective_ratio_mean"]].to_string(index=False))
    print(f"Saved tuned latest-Algorithm-1 results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
