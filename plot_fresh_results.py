#!/usr/bin/env python3
"""Validate and plot only de-novo legacy-code results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "fresh"
OUT = ROOT / "figures"
FIGURE3_CASES = ["packing", "sc205", "random_lp_A", "random_lp_C"]
FIGURE4_CASES = [
    "packing",
    "mincostflow",
    "random_lp_A",
    "random_lp_B",
    "random_lp_C",
    "random_lp_D",
    "grow7",
    "sc205",
]
FIGURE2_CASES = FIGURE4_CASES
FIGURE2_TARGET = {
    "packing": 19,
    "mincostflow": 4,
    "random_lp_A": 8,
    "random_lp_B": 12,
    "random_lp_C": 7,
    "random_lp_D": 9,
    "grow7": 19,
    "sc205": 6,
}
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
METHOD_STYLE = {
    "OursEstC": ("#c43c35", "D", "-"),
    "DataDriven": ("#2673b8", "s", "--"),
    "FCNNc": ("#76538b", "v", "-."),
    "Rand": ("#66705a", "x", ":"),
}
K_STYLE = {
    10: ("#2673b8", "o", (0, (6, 2))),
    20: ("#d0782c", "s", (0, (3, 2))),
    30: ("#418a5b", "^", (0, (1, 2))),
    50: ("#76538b", "v", (0, (5, 1, 1, 1))),
}


def band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    spread: np.ndarray,
    color: str,
) -> None:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(spread)
    if np.any(valid):
        ax.fill_between(
            x[valid],
            y[valid] - spread[valid],
            y[valid] + spread[valid],
            color=color,
            alpha=0.10,
            linewidth=0,
        )


def collect_legend(fig: plt.Figure, axes: np.ndarray, ncol: int) -> None:
    handles: list[object] = []
    labels: list[str] = []
    for ax in axes.reshape(-1):
        local_handles, local_labels = ax.get_legend_handles_labels()
        for handle, label in zip(local_handles, local_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(ncol, len(labels)),
        frameon=True,
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def validate_figure2(data: pd.DataFrame) -> None:
    required = {"case", "n_samples", "rank", "method"}
    if not required.issubset(data.columns):
        raise RuntimeError("Figure 2 raw CSV is missing required columns")
    ours = data[data["method"].astype(str).eq("OursEstC")]
    expected = {
        (case, n_samples)
        for case in FIGURE2_CASES
        for n_samples in range(26)
    }
    actual = {
        (str(row.case), int(row.n_samples))
        for row in ours.itertuples()
    }
    if len(ours) != 208 or actual != expected:
        raise RuntimeError(
            f"Figure 2 grid is incomplete: rows={len(ours)}, "
            f"missing={sorted(expected - actual)}"
        )


def plot_figure2(data: pd.DataFrame) -> None:
    validate_figure2(data)
    data = data[data["method"].astype(str).eq("OursEstC")]
    fig, axes = plt.subplots(2, 4, figsize=(16.2, 6.4), sharex=True)
    for index, case in enumerate(FIGURE2_CASES):
        ax = axes.reshape(-1)[index]
        group = data[data["case"].astype(str).eq(case)].sort_values(
            "n_samples"
        )
        x = group["n_samples"].to_numpy(dtype=float)
        y = group["rank"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color="#c43c35",
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label="OursEstC",
        )
        ax.axhline(
            FIGURE2_TARGET[case],
            color="#666666",
            linestyle=":",
            linewidth=1.3,
            label="Reported target",
        )
        ax.set_title(f"({chr(97 + index)}) {TITLE[case]}")
        ax.set_xlabel("Samples processed")
        if index % 4 == 0:
            ax.set_ylabel("Learned dimension")
        ax.set_xlim(0, 25)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    collect_legend(fig, axes, ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "figure2_fresh_rank_growth")


def figure3_label(method: str, source_method: str) -> str:
    if method == "DataDriven":
        return f"DataDrivenProj ({source_method})"
    if method == "FCNNc":
        return "FCNN-c"
    return method


def validate_figure3(data: pd.DataFrame) -> None:
    expected = {
        (case, rho, method)
        for case in FIGURE3_CASES
        for rho in (0.1, 0.3, 0.5, 0.7, 0.9)
        for method in ("OursEstC", "DataDriven", "Rand", "FCNNc")
    }
    actual = {
        (str(row.case), round(float(row.rho), 1), str(row.method))
        for row in data.itertuples()
    }
    if len(data) != 80 or actual != expected:
        raise RuntimeError(
            f"Figure 3 grid is incomplete: rows={len(data)}, "
            f"missing={sorted(expected - actual)}"
        )


def plot_figure3(data: pd.DataFrame) -> None:
    validate_figure3(data)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True)
    for index, case in enumerate(FIGURE3_CASES):
        ax = axes.reshape(-1)[index]
        subset = data[data["case"].astype(str).eq(case)]
        for method in ("OursEstC", "DataDriven", "Rand", "FCNNc"):
            group = subset[subset["method"].astype(str).eq(method)].sort_values(
                "rho"
            )
            source_method = str(group["source_method"].iloc[0])
            color, marker, linestyle = METHOD_STYLE[method]
            x = group["rho"].to_numpy(dtype=float)
            y = group["objective_ratio_mean"].to_numpy(dtype=float)
            se = group["objective_ratio_se"].fillna(0).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=4.6,
                label=figure3_label(method, source_method),
            )
            band(ax, x, y, se, color)
        ax.set_title(f"({chr(97 + index)}) {TITLE[case]}")
        ax.set_xlabel("Target outside mass rho")
        if index % 2 == 0:
            ax.set_ylabel("Average objective ratio")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_ylim(-0.12, 1.05)
        ax.grid(alpha=0.25)
    collect_legend(fig, axes, ncol=4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "figure3_fresh_legacy_execution")


def validate_figure4(data: pd.DataFrame) -> None:
    expected = {
        (case, n_samples, method, k)
        for case in FIGURE4_CASES
        for n_samples in (20, 40, 60, 80)
        for method, k_values in (
            ("OursEstC", (None,)),
            ("FCNNc", (10, 20, 30, 50)),
        )
        for k in k_values
    }
    actual = {
        (
            str(row.case),
            int(row.n_samples),
            str(row.method),
            None if str(row.method) == "OursEstC" else int(row.K),
        )
        for row in data.itertuples()
    }
    if len(data) != 160 or actual != expected:
        raise RuntimeError(
            f"Figure 4 grid is incomplete: rows={len(data)}, "
            f"missing={sorted(expected - actual)}"
        )


def sample_sd(group: pd.DataFrame) -> np.ndarray:
    count = (
        pd.to_numeric(group["solve_observations"], errors="coerce")
        .fillna(1)
        .clip(lower=1)
        .to_numpy(dtype=float)
    )
    se = group["objective_ratio_se"].fillna(0).to_numpy(dtype=float)
    return se * np.sqrt(count)


def plot_figure4(data: pd.DataFrame) -> None:
    validate_figure4(data)
    fig, axes = plt.subplots(2, 4, figsize=(16.4, 6.8), sharey=True)
    for index, case in enumerate(FIGURE4_CASES):
        ax = axes.reshape(-1)[index]
        subset = data[data["case"].astype(str).eq(case)]
        for k in (10, 20, 30, 50):
            group = subset[
                subset["method"].astype(str).eq("FCNNc")
                & subset["K"].astype(int).eq(k)
            ].sort_values("n_samples")
            color, marker, linestyle = K_STYLE[k]
            x = group["n_samples"].to_numpy(dtype=float)
            y = group["objective_ratio_mean"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4.0,
                label=f"FCNN-c, K={k}",
            )
            band(ax, x, y, sample_sd(group), color)
        ours = subset[subset["method"].astype(str).eq("OursEstC")].sort_values(
            "n_samples"
        )
        x = ours["n_samples"].to_numpy(dtype=float)
        y = ours["objective_ratio_mean"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color="#c43c35",
            marker="D",
            linestyle="-",
            linewidth=2.2,
            markersize=4.2,
            label="OursEstC",
        )
        band(ax, x, y, sample_sd(ours), "#c43c35")
        ax.set_title(f"({chr(97 + index)}) {TITLE[case]}")
        ax.set_xlabel("Training samples")
        if index % 4 == 0:
            ax.set_ylabel("Average objective ratio")
        ax.set_xticks([20, 40, 60, 80])
        ax.set_ylim(-0.12, 1.05)
        ax.grid(alpha=0.25)
    collect_legend(fig, axes, ncol=5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "figure4_fresh_legacy_execution")


def main() -> None:
    figure2_path = RESULTS / "figure2_rank_growth_raw.csv"
    figure3_path = RESULTS / "figure3_timing_raw.csv"
    figure4_path = RESULTS / "figure4_timing_raw.csv"
    if not any(path.exists() for path in (figure2_path, figure3_path, figure4_path)):
        raise FileNotFoundError(
            "No fresh result CSV is present. Run reproduce_figures.py --rerun."
        )
    if figure2_path.exists():
        plot_figure2(pd.read_csv(figure2_path))
    if figure3_path.exists():
        plot_figure3(pd.read_csv(figure3_path))
    if figure4_path.exists():
        plot_figure4(pd.read_csv(figure4_path))
    print(f"Validated fresh-only figures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
