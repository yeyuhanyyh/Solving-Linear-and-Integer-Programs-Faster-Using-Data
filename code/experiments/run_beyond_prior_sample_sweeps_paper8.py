#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample-size diagnostics for the paper-eight beyond-prior experiment.

The sample size is shared by C-hat estimation and training.  For the
projection-learning comparison we keep the same PCA/SGA case selection used in
the merged paper-eight figures.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
CORE_CODE = ROOT.parent / "core"
sys.path.insert(0, str(CORE_CODE))

import final_projection_figure_suite as fps  # noqa: E402
import compare_fixedX_family_suite as cff  # noqa: E402
from iwata_sakaue_pelp_projection_compare import objective_ratio, summarize_ratios_and_times  # noqa: E402

import run_subgaussian_beyond_prior_suite as bp  # noqa: E402
import fcnn_training_protocols as fcnn_protocols  # noqa: E402


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
KEEP_SGA_CASES = {"packing", "maxflow", "random_lp_A", "random_lp_B", "random_lp_C", "random_lp_D", "grow7"}
KEEP_PCA_CASES = {"mincostflow", "shortest_path", "sc205"}
TITLE_OVERRIDES = {
    "packing": "Packing",
    "maxflow": "MaxFlow",
    "mincostflow": "MinCostFlow",
    "shortest_path": "ShortestPath",
    "random_lp_A": "RandomLP A",
    "random_lp_B": "RandomLP B",
    "random_lp_C": "RandomLP C",
    "random_lp_D": "RandomLP D",
    "grow7": "GROW7",
    "sc205": "SC205",
}


def selected_projection(case: str) -> str:
    if case in KEEP_SGA_CASES:
        return "SGA"
    if case in KEEP_PCA_CASES:
        return "PCA"
    return "PCA"


def case_with_sample_size(case: object, args: argparse.Namespace, n_train: int) -> object:
    """Override per-case defaults that otherwise cap netlib cases at n_train=24."""
    out = copy.deepcopy(case)
    out.args_updates["n_train"] = int(n_train)
    out.args_updates["n_val"] = int(args.n_val)
    out.args_updates["n_test"] = int(args.n_test)
    out.args_updates["k_list"] = str(args.k_list)
    out.args_updates["costonly_hidden_dim"] = int(args.hidden_dim)
    out.args_updates["costonly_epochs"] = int(args.epochs)
    out.args_updates["costonly_patience"] = int(args.patience)
    out.args_updates["costonly_batch_size"] = int(args.batch_size)
    out.args_updates["costonly_lr"] = float(args.lr)
    out.args_updates["costonly_init"] = "plain_torch_random"
    return out


def setup_base_args(args: argparse.Namespace, n_train: int) -> argparse.Namespace:
    base = fps.default_namespace()
    base.seed = int(args.seed)
    base.device = str(args.device)
    base.verbose = bool(args.verbose)
    base.n_train = int(n_train)
    base.n_val = int(args.n_val)
    base.n_test = int(args.n_test)
    base.k_list = str(args.k_list)
    base.rand_trials = int(args.rand_trials)
    base.make_quality_figures = False
    return base


def get_cases() -> Dict[str, object]:
    all_cases = [*fps.synthetic_cases(PACKAGE_ROOT), *fps.netlib_cases(PACKAGE_ROOT)]
    return {c.slug: c for c in all_cases}


def patch_styles() -> None:
    fps.METHOD_STYLE["OursEstC"] = {"color": "#d62728", "marker": "D", "linestyle": (0, (6, 2))}
    fps.METHOD_DISPLAY["OursEstC"] = "Ours (est. C)"
    fps.METHOD_STYLE["DataDriven"] = {"color": "#1f77b4", "marker": "s", "linestyle": (0, (1, 2))}
    fps.METHOD_DISPLAY["DataDriven"] = "data-driven proj"
    fps.METHOD_STYLE["FCNNc"] = {"color": "#8c564b", "marker": "v", "linestyle": (0, (4, 1, 1, 1))}
    fps.METHOD_DISPLAY["FCNNc"] = "FCNN-c"


def summarize_ours(
    case: object,
    args: argparse.Namespace,
    n_samples: int,
    out_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    case_out = out_dir / "ours_sample_sweep" / f"{case.slug}_n{n_samples:03d}"
    summary_path = case_out / "ours_summary.csv"
    if summary_path.exists() and not force:
        return pd.read_csv(summary_path)
    case_out.mkdir(parents=True, exist_ok=True)

    base = setup_base_args(args, n_samples)
    case = case_with_sample_size(case, args, n_samples)
    data = bp.prepare_subgaussian_case(
        case,
        base,
        pilot_n=int(args.pilot_n),
        rho=float(args.rho),
        radius_scale=float(args.radius_scale),
        gaussian_std_scale=float(case_gaussian_scale(case.slug, args)),
        svd_tol=float(args.svd_tol),
        same_est_train_samples=True,
        enclose_train_in_estimated_prior=True,
        cost_space=str(args.cost_space),
    )
    t0 = time.perf_counter()
    stage1 = bp.alg2_cumulative_ellipsoid(data.bundle.stdlp, data.bundle.Ctrain, data.estimated_prior, verbose=bool(args.verbose))
    fit_time = time.perf_counter() - t0

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
    raw = pd.DataFrame(rows)
    raw["case"] = case.slug
    raw["title"] = case.title
    raw["family"] = case.family
    raw["n_samples"] = int(n_samples)
    raw.to_csv(case_out / "ours_raw.csv", index=False)

    summary = summarize_ratios_and_times(raw.to_dict("records"))
    summary["case"] = case.slug
    summary["title"] = case.title
    summary["family"] = case.family
    summary["n_samples"] = int(n_samples)
    summary["intrinsic_K"] = int(stage1.U.shape[1])
    summary["fit_time"] = float(fit_time)
    summary["estimated_prior_effective_dim"] = int(data.estimated_prior.effective_dim)
    summary["estimated_prior_radius"] = float(data.estimated_prior.radius)
    summary["n_train_retained_for_ours"] = int(data.estimated_prior.retained_train)
    summary.to_csv(summary_path, index=False)

    pd.DataFrame(
        {
            "sample": np.arange(0, len(stage1.rank_after_sample) + 1),
            "rank_after_sample": np.concatenate([[0], np.asarray(stage1.rank_after_sample, dtype=int)]),
            "case": case.slug,
            "title": case.title,
            "n_samples": int(n_samples),
        }
    ).to_csv(case_out / "ours_rank_after_sample.csv", index=False)
    return summary


def summarize_projection(
    case: object,
    args: argparse.Namespace,
    n_samples: int,
    out_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    case_out = out_dir / "projection_sample_k_sweep" / f"{case.slug}_n{n_samples:03d}"
    selected_path = case_out / "data_driven_summary.csv"
    if selected_path.exists() and not force:
        return pd.read_csv(selected_path)
    case_out.mkdir(parents=True, exist_ok=True)

    base = setup_base_args(args, n_samples)
    case = case_with_sample_size(case, args, n_samples)
    data = bp.prepare_subgaussian_case(
        case,
        base,
        pilot_n=int(args.pilot_n),
        rho=float(args.rho),
        radius_scale=float(args.radius_scale),
        gaussian_std_scale=float(case_gaussian_scale(case.slug, args)),
        svd_tol=float(args.svd_tol),
        same_est_train_samples=True,
        enclose_train_in_estimated_prior=True,
        cost_space=str(args.cost_space),
    )
    fps_data = fps.PreparedCaseData(
        case=data.case,
        args=data.args,
        problem=data.problem,
        bundle=data.bundle,
        proj_train=data.proj_train,
        proj_val=data.proj_val,
        proj_test=data.proj_test,
    )
    method = selected_projection(case.slug)
    result = fps.run_case_from_prepared(
        fps_data,
        case_out,
        methods=[method],
        train_limit=n_samples,
        force=True,
        write_rank=False,
    )
    summary = result["summary"].copy()
    summary = summary[summary["method"] == method].copy()
    summary["source_method"] = method
    summary["method"] = "DataDriven"
    summary["case"] = case.slug
    summary["title"] = case.title
    summary["family"] = case.family
    summary["n_samples"] = int(n_samples)
    summary.to_csv(selected_path, index=False)
    return summary


def parse_case_float_map(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        out[key.strip()] = float(value.strip())
    return out


def case_gaussian_scale(case: str, args: argparse.Namespace) -> float:
    return float(parse_case_float_map(args.case_gaussian_std_scales).get(case, float(args.gaussian_std_scale)))


def prepare_fixed_pool_case(case: object, args: argparse.Namespace, n_max: int) -> bp.SubgaussianPreparedCase:
    """Prepare one fixed train/val/test split; smaller n use prefixes of Ctrain."""
    base = setup_base_args(args, n_max)
    case = case_with_sample_size(case, args, n_max)
    return bp.prepare_subgaussian_case(
        case,
        base,
        pilot_n=int(args.pilot_n),
        rho=float(args.rho),
        radius_scale=float(args.radius_scale),
        gaussian_std_scale=float(case_gaussian_scale(case.slug, args)),
        svd_tol=float(args.svd_tol),
        same_est_train_samples=True,
        enclose_train_in_estimated_prior=True,
        cost_space=str(args.cost_space),
    )


def estimate_prior_from_prefix(data: bp.SubgaussianPreparedCase, args: argparse.Namespace, n_samples: int) -> bp.EllipsoidPrior:
    train_costs = np.asarray(data.bundle.Ctrain[: int(n_samples)], dtype=float)
    prior, _maha = bp._estimate_ellipsoid_prior(  # noqa: SLF001
        train_costs,
        rho=float(args.rho),
        radius_scale=float(args.radius_scale),
        svd_tol=float(args.svd_tol),
    )
    train_maha = bp._mahalanobis_in_estimated_prior(train_costs, prior)  # noqa: SLF001
    if train_maha.size > 0:
        prior.radius = max(float(prior.radius), float(np.max(train_maha)) * (1.0 + 1e-8))
        train_maha = bp._mahalanobis_in_estimated_prior(train_costs, prior)  # noqa: SLF001
    prior.retained_train = int(np.sum(train_maha <= prior.radius + 1e-10))
    return prior


def summarize_ours_from_pool(
    data: bp.SubgaussianPreparedCase,
    args: argparse.Namespace,
    n_samples: int,
    out_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    case_out = out_dir / "ours_sample_sweep" / f"{data.case.slug}_n{n_samples:03d}"
    summary_path = case_out / "ours_summary.csv"
    if summary_path.exists() and not force:
        return pd.read_csv(summary_path)
    case_out.mkdir(parents=True, exist_ok=True)

    prior = estimate_prior_from_prefix(data, args, n_samples)
    train_costs = np.asarray(data.bundle.Ctrain[: int(n_samples)], dtype=float)
    t0 = time.perf_counter()
    stage1 = bp.alg2_cumulative_ellipsoid(data.bundle.stdlp, train_costs, prior, verbose=bool(args.verbose))
    fit_time = time.perf_counter() - t0

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
    raw = pd.DataFrame(rows)
    raw["case"] = data.case.slug
    raw["title"] = data.case.title
    raw["family"] = data.case.family
    raw["n_samples"] = int(n_samples)
    raw.to_csv(case_out / "ours_raw.csv", index=False)

    summary = summarize_ratios_and_times(raw.to_dict("records"))
    summary["case"] = data.case.slug
    summary["title"] = data.case.title
    summary["family"] = data.case.family
    summary["n_samples"] = int(n_samples)
    summary["intrinsic_K"] = int(stage1.U.shape[1])
    summary["fit_time"] = float(fit_time)
    summary["estimated_prior_effective_dim"] = int(prior.effective_dim)
    summary["estimated_prior_radius"] = float(prior.radius)
    summary["n_train_retained_for_ours"] = int(prior.retained_train)
    summary.to_csv(summary_path, index=False)

    pd.DataFrame(
        {
            "sample": np.arange(0, len(stage1.rank_after_sample) + 1),
            "rank_after_sample": np.concatenate([[0], np.asarray(stage1.rank_after_sample, dtype=int)]),
            "case": data.case.slug,
            "title": data.case.title,
            "n_samples": int(n_samples),
        }
    ).to_csv(case_out / "ours_rank_after_sample.csv", index=False)
    return summary


def summarize_projection_and_neural_from_pool(
    data: bp.SubgaussianPreparedCase,
    args: argparse.Namespace,
    n_samples: int,
    out_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    case_out = out_dir / "projection_neural_sample_k_sweep" / f"{data.case.slug}_n{n_samples:03d}"
    selected_path = case_out / "data_driven_and_neural_summary.csv"
    if selected_path.exists() and not force:
        return pd.read_csv(selected_path)
    case_out.mkdir(parents=True, exist_ok=True)

    fps_data = fps.PreparedCaseData(
        case=data.case,
        args=data.args,
        problem=data.problem,
        bundle=data.bundle,
        proj_train=data.proj_train,
        proj_val=data.proj_val,
        proj_test=data.proj_test,
    )
    method = selected_projection(data.case.slug)
    result = fps.run_case_from_prepared(
        fps_data,
        case_out,
        methods=[method, "CostOnly"],
        train_limit=int(n_samples),
        force=True,
        write_rank=False,
    )
    summary = result["summary"].copy()
    keep = summary[summary["method"].isin([method, "CostOnly"])].copy()
    keep["source_method"] = keep["method"]
    keep.loc[keep["method"] == method, "method"] = "DataDriven"
    keep.loc[keep["source_method"] == "CostOnly", "method"] = "FCNNc"
    keep["case"] = data.case.slug
    keep["title"] = data.case.title
    keep["family"] = data.case.family
    keep["n_samples"] = int(n_samples)
    keep.to_csv(selected_path, index=False)
    return keep


def learned_dimension_table(source_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for case in CASE_ORDER:
        meta_path = source_dir / "k_sweep" / case / "case_manifest.json"
        rank_csv = source_dir / "k_sweep" / case / "ours_rank_after_sample.csv"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        final_rank = int(meta.get("intrinsic_rank_ours_estC", -1))
        if rank_csv.exists():
            rdf = pd.read_csv(rank_csv)
            if not rdf.empty:
                final_rank = int(rdf["rank_after_sample"].iloc[-1])
        rows.append(
            {
                "case": case,
                "title": TITLE_OVERRIDES.get(case, meta.get("title", case)),
                "learned_dimension": final_rank,
                "n_samples": int(meta.get("n_train", -1)),
                "retained_samples": int(meta.get("n_train_retained_for_ours", -1)),
                "rho": float(meta.get("rho_outside_mass", math.nan)),
                "estimated_prior_effective_dim": int(meta.get("estimated_prior_effective_dim", -1)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "learned_dimensions_paper8.csv", index=False)
    plot_dimension_table(df, out_dir / "table_learned_dimensions_paper8.png")
    return df


def plot_dimension_table(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    ax.axis("off")
    show = df[["title", "learned_dimension", "n_samples", "retained_samples"]].copy()
    show.columns = ["Instance", "learned dim.", "samples", "retained"]
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


def title_map() -> Dict[str, str]:
    return {c: TITLE_OVERRIDES.get(c, c) for c in CASE_ORDER}


def parse_case_order(spec: str) -> List[str]:
    return [s.strip() for s in str(spec).split(",") if s.strip()]


def plot_ours_sample_efficiency(df: pd.DataFrame, out_path: Path) -> None:
    plot_grid_lines(
        df,
        out_path,
        x_col="n_samples",
        y_col="objective_ratio_mean",
        yerr_col="objective_ratio_se",
        xlabel="# estimation/training samples",
        ylabel="Average test objective ratio",
        group_col=None,
        methods=["OursEstC"],
        legend_labels={"OursEstC": "Ours (est. C)"},
    )


def plot_projection_sample_k(df: pd.DataFrame, ours_n20: pd.DataFrame, out_path: Path) -> None:
    proj = df.copy()
    label_map = {"DataDriven": "data-driven proj", "FCNNc": "FCNN-c"}
    proj["curve"] = proj.apply(lambda r: f"{label_map.get(str(r['method']), str(r['method']))}, n={int(r['n_samples'])}", axis=1)
    ours = ours_n20.copy()
    k_list = sorted(proj["K"].dropna().astype(int).unique())
    expanded = []
    for _, row in ours.iterrows():
        for k in k_list:
            rec = row.to_dict()
            rec["K"] = k
            rec["curve"] = "Ours (est. C), n=20"
            expanded.append(rec)
    both = pd.concat([proj, pd.DataFrame(expanded)], ignore_index=True)
    plot_grid_lines(
        both,
        out_path,
        x_col="K",
        y_col="objective_ratio_mean",
        yerr_col="objective_ratio_se",
        xlabel="Reduced dimension K",
        ylabel="Average test objective ratio",
        group_col="curve",
        methods=None,
        legend_labels=None,
    )


def plot_grid_lines(
    df: pd.DataFrame,
    out_path: Path,
    x_col: str,
    y_col: str,
    yerr_col: str,
    xlabel: str,
    ylabel: str,
    group_col: str | None,
    methods: Sequence[str] | None,
    legend_labels: Dict[str, str] | None,
) -> None:
    cases = CASE_ORDER
    titles = title_map()
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.6), sharey=False)
    axes = axes.reshape(-1)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
    for idx, case in enumerate(cases):
        ax = axes[idx]
        sub = df[df["case"] == case].copy()
        if group_col is None:
            groups = [(m, sub[sub["method"] == m]) for m in (methods or sorted(sub["method"].unique()))]
        else:
            groups = [(g, gdf) for g, gdf in sub.groupby(group_col, sort=True)]
        for gi, (name, gdf) in enumerate(groups):
            if gdf.empty:
                continue
            gdf = gdf.sort_values(x_col)
            name_s = str(name)
            if name_s.startswith("Ours"):
                color, marker, linestyle = "#d62728", "D", (0, (6, 2))
            elif name_s.startswith("FCNN-c"):
                color, marker, linestyle = "#8c564b", "v", (0, (4, 1, 1, 1))
            else:
                color, marker, linestyle = palette[gi % len(palette)], "o", "-"
            label = legend_labels.get(name, name) if legend_labels else name
            x = gdf[x_col].astype(float).to_numpy()
            y = gdf[y_col].astype(float).to_numpy()
            se = gdf[yerr_col].fillna(0.0).astype(float).to_numpy() if yerr_col in gdf else np.zeros_like(y)
            ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=2.0, markersize=4.5, label=label)
            ax.fill_between(x, y - se, y + se, color=color, alpha=0.10, linewidth=0)
        ax.set_title(f"({chr(ord('a') + idx)}) {titles[case]}", fontsize=12)
        ax.set_xlabel(xlabel)
        if idx % 4 == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=-0.02, top=max(1.05, float(sub[y_col].max()) + 0.08 if not sub.empty else 1.05))
    handles, labels = axes[0].get_legend_handles_labels()
    if group_col is not None:
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global CASE_ORDER
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", default="results_beyond_prior_nested_samples_paper8_k5_30_neural_20260504")
    p.add_argument("--source_dir", default="results_subgaussian_beyond_prior_rho010_oldcode_same_samples_fair_normal_rank_v1")
    p.add_argument("--case_order", default=",".join(CASE_ORDER))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--sample_list", default="20,40,60,80")
    p.add_argument("--projection_sample_list", default="20,40,60,80")
    p.add_argument("--n_val", type=int, default=6)
    p.add_argument("--n_test", type=int, default=12)
    p.add_argument("--pilot_n", type=int, default=96)
    p.add_argument("--rho", type=float, default=0.10)
    p.add_argument("--radius_scale", type=float, default=1.10)
    p.add_argument("--gaussian_std_scale", type=float, default=0.35)
    p.add_argument("--case_gaussian_std_scales", default="random_lp_A=20,random_lp_B=20,random_lp_C=20,random_lp_D=20")
    p.add_argument("--cost_space", choices=["reward_subspace", "full_standard"], default="reward_subspace")
    p.add_argument("--svd_tol", type=float, default=1e-10)
    p.add_argument("--k_list", default="5,10,20,30")
    p.add_argument("--rand_trials", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=120)
    args = p.parse_args()

    CASE_ORDER = parse_case_order(args.case_order)
    patch_styles()
    fcnn_protocols.use_plain_random_protocol()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = get_cases()
    selected_cases = [cases[c] for c in CASE_ORDER]

    source_dir = (ROOT / args.source_dir).resolve()
    learned = learned_dimension_table(source_dir, out_dir)

    sample_list = [int(x) for x in str(args.sample_list).split(",") if x.strip()]
    proj_sample_list = [int(x) for x in str(args.projection_sample_list).split(",") if x.strip()]
    n_max = int(max([*sample_list, *proj_sample_list]))

    prepared = {}
    for case in selected_cases:
        print(f"[prepare fixed pool] {case.slug}, n_max={n_max}", flush=True)
        prepared[case.slug] = prepare_fixed_pool_case(case, args, n_max)

    ours_frames = []
    for case in selected_cases:
        data = prepared[case.slug]
        for n in sample_list:
            print(f"[ours sample sweep] {case.slug}, n={n}", flush=True)
            ours_frames.append(summarize_ours_from_pool(data, args, n, out_dir, force=bool(args.force)))
    ours_df = pd.concat(ours_frames, ignore_index=True)
    ours_df.to_csv(out_dir / "ours_accuracy_vs_samples_summary.csv", index=False)
    plot_ours_sample_efficiency(ours_df, out_dir / "figure_ours_accuracy_vs_samples_paper8.png")

    proj_frames = []
    for case in selected_cases:
        data = prepared[case.slug]
        for n in proj_sample_list:
            print(f"[projection/neural sample K-sweep] {case.slug}, n={n}, projection={selected_projection(case.slug)}, neural=FCNN-c", flush=True)
            proj_frames.append(summarize_projection_and_neural_from_pool(data, args, n, out_dir, force=bool(args.force)))
    proj_df = pd.concat(proj_frames, ignore_index=True)
    proj_df.to_csv(out_dir / "data_driven_and_neural_k_sweep_by_samples_summary.csv", index=False)

    ours_n20 = ours_df[ours_df["n_samples"] == 20].copy()
    plot_projection_sample_k(proj_df, ours_n20, out_dir / "figure_projection_neural_k_sweep_by_samples_vs_ours_n20_paper8.png")

    manifest = {
        "experiment": "paper8_beyond_prior_sample_sweeps",
        "source_dir_for_learned_dimension_table": str(source_dir),
        "old_code_path": str(CORE_CODE),
        "case_order": CASE_ORDER,
        "sample_list": sample_list,
        "projection_sample_list": proj_sample_list,
        "fixed_sample_pool": True,
        "nested_training_prefixes": True,
        "fixed_val_test_for_all_sample_sizes": True,
        "same_est_train_samples": True,
        "enclose_train_in_estimated_prior": True,
        "rho": float(args.rho),
        "radius_scale": float(args.radius_scale),
        "gaussian_std_scale": float(args.gaussian_std_scale),
        "case_gaussian_std_scales": str(args.case_gaussian_std_scales),
        "cost_space": str(args.cost_space),
        "k_list": str(args.k_list),
        "neural_method": "FCNN-c / CostOnly",
        "neural_plain_random_init": True,
        "neural_training": {
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "patience": int(args.patience),
        },
        "pca_sga_selection": {"SGA": sorted(KEEP_SGA_CASES), "PCA": sorted(KEEP_PCA_CASES), "merged_label": "data-driven proj"},
        "learned_dimensions": learned.to_dict("records"),
    }
    (out_dir / "sample_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved sample-sweep diagnostics to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
