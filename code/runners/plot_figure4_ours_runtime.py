#!/usr/bin/env python3
"""Plot default Ours timing with the cached matched-K FCNN and Full LP rows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "timing" / "figure4_ours_matched_k_timing_raw.csv"
OUT = ROOT / "figures"
OURS = "Ours"
FCNN = "FCNNc"
CASES = (
    "packing",
    "mincostflow",
    "random_lp_A",
    "random_lp_B",
    "random_lp_C",
    "random_lp_D",
    "grow7",
    "sc205",
)
TITLES = {
    "packing": "Packing",
    "mincostflow": "MinCostFlow",
    "random_lp_A": "RandomLP A",
    "random_lp_B": "RandomLP B",
    "random_lp_C": "RandomLP C",
    "random_lp_D": "RandomLP D",
    "grow7": "GROW7",
    "sc205": "SC205",
}


def mean_sd(group: pd.DataFrame, column: str) -> tuple[float, float]:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), sd


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "learning_time_s",
        "solve_time_mean_s",
        "online_total_time_mean_s",
        "full_lp_solve_time_mean_s",
        "raw_learned_rank",
        "objective_ratio_mean",
    )
    rows: list[dict[str, object]] = []
    for keys, group in raw.groupby(["case", "method", "K"], sort=True):
        case_name, method, k = keys
        row: dict[str, object] = {
            "case": case_name,
            "method": method,
            "K": int(k),
            "full_dimension": int(group["full_dimension"].iloc[0]),
            "timing_repeats": int(group["repeat"].nunique()),
        }
        for metric in metrics:
            if metric not in group:
                continue
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(
                group, metric
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate(raw: pd.DataFrame) -> None:
    if len(raw) != 48:
        raise RuntimeError(f"Expected 48 timing rows, found {len(raw)}")
    for _, group in raw.groupby(["repeat", "case", "n_samples"]):
        if set(group["method"].astype(str)) != {OURS, FCNN}:
            raise RuntimeError("A matched-K timing group is incomplete")
        if group["K"].astype(int).nunique() != 1:
            raise RuntimeError("Ours and FCNN-c use different K")
        if group["full_dimension"].astype(int).nunique() != 1:
            raise RuntimeError("Ours and FCNN-c use different full d")


def method_frame(summary: pd.DataFrame, method: str) -> pd.DataFrame:
    return (
        summary[summary["method"].astype(str).eq(method)]
        .set_index("case")
        .reindex(CASES)
    )


def main() -> None:
    raw = pd.read_csv(RAW)
    validate(raw)
    summary = aggregate(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "figure4_runtime_summary.csv", index=False)

    ours_frame = method_frame(summary, OURS)
    fcnn_frame = method_frame(summary, FCNN)
    compact = pd.DataFrame(
        {
            "Instance": [TITLES[case] for case in CASES],
            "Full LP dimension d": ours_frame["full_dimension"].to_numpy(int),
            "Matched reduced dimension K": ours_frame["K"].to_numpy(int),
            "Raw optimizer-span rank": np.rint(
                ours_frame["raw_learned_rank_mean"].to_numpy(float)
            ).astype(int),
            "Ours learning mean (s)": ours_frame[
                "learning_time_s_mean"
            ].to_numpy(float),
            "Ours learning SD (s)": ours_frame[
                "learning_time_s_sd"
            ].to_numpy(float),
            "FCNN-c learning mean (s), cached": fcnn_frame[
                "learning_time_s_mean"
            ].to_numpy(float),
            "FCNN-c learning SD (s), cached": fcnn_frame[
                "learning_time_s_sd"
            ].to_numpy(float),
            "Learning speedup": (
                fcnn_frame["learning_time_s_mean"].to_numpy(float)
                / ours_frame["learning_time_s_mean"].to_numpy(float)
            ),
            "Ours objective ratio mean": ours_frame[
                "objective_ratio_mean_mean"
            ].to_numpy(float),
        }
    )
    compact.to_csv(OUT / "figure4_runtime_table.csv", index=False)

    labels = [
        f"{TITLES[case]}\nK={int(ours_frame.loc[case, 'K'])}, "
        f"d={int(ours_frame.loc[case, 'full_dimension'])}"
        for case in CASES
    ]
    x = np.arange(len(CASES), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.0))
    methods = (
        (OURS, "Ours", "#c43c35"),
        (FCNN, "FCNN-c (cached)", "#2673b8"),
    )
    width = 0.30
    for offset, (method, label, color) in zip(
        (-width / 2, width / 2), methods
    ):
        frame = method_frame(summary, method)
        axes[0].bar(
            x + offset,
            frame["learning_time_s_mean"].to_numpy(float),
            width,
            yerr=frame["learning_time_s_sd"].to_numpy(float),
            color=color,
            alpha=0.88,
            capsize=2.5,
            label=label,
        )

    bar_width = 0.24
    for axis, metric, title in (
        (axes[1], "solve_time_mean_s", "Reduced-LP solve call"),
        (axes[2], "online_total_time_mean_s", "End-to-end online time"),
    ):
        for offset, (method, label, color) in zip(
            (-bar_width, 0.0), methods
        ):
            frame = method_frame(summary, method)
            axis.bar(
                x + offset,
                frame[f"{metric}_mean"].to_numpy(float),
                bar_width,
                yerr=frame[f"{metric}_sd"].to_numpy(float),
                color=color,
                alpha=0.88,
                capsize=2.5,
                label=label,
            )
        axis.bar(
            x + bar_width,
            ours_frame["full_lp_solve_time_mean_s_mean"].to_numpy(float),
            bar_width,
            yerr=ours_frame[
                "full_lp_solve_time_mean_s_sd"
            ].to_numpy(float),
            color="#222222",
            alpha=0.88,
            capsize=2.5,
            label="Full LP (cached)",
        )
        axis.set_title(title)
        axis.set_ylabel("Time per held-out LP (s)")

    axes[0].set_title("Offline representation learning")
    axes[0].set_ylabel("Learning time (s)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle(
        "Figure 4 runtime: matched K, mean +/- 1 SD across "
        "three wall-clock repeats"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        OUT / "figure4_runtime_matched_k.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "figure4_runtime_matched_k.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
