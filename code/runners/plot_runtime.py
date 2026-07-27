#!/usr/bin/env python3
"""Plot timing means and repeat standard deviations on linear axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TIMING = ROOT / "results" / "timing"
OUT = ROOT / "figures"
F3_CASES = ("packing", "sc205", "random_lp_A", "random_lp_C")
F4_CASES = (
    "packing",
    "mincostflow",
    "random_lp_A",
    "random_lp_B",
    "random_lp_C",
    "random_lp_D",
    "grow7",
    "sc205",
)
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
STYLE = {
    "OursEstC": ("#c43c35", "D", "-"),
    "DataDriven": ("#2a8a6e", "P", "-."),
    "Rand": ("#66705a", "x", ":"),
    "FCNNc": ("#2673b8", "s", "--"),
}


def sample_sd(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.std(ddof=1)) if len(clean) > 1 else 0.0


def aggregate(
    data: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for group_key, group in data.groupby(keys, dropna=False, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(keys, values))
        row["timing_repeats"] = int(group["repeat"].nunique())
        for metric in metrics:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(numeric.mean())
            row[f"{metric}_sd"] = sample_sd(numeric)
        rows.append(row)
    return pd.DataFrame(rows)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def f3_label(method: str, source: str) -> str:
    if method == "DataDriven":
        return source
    return {
        "OursEstC": "OursEstC",
        "Rand": "Random projection",
        "FCNNc": "FCNN-c",
    }[method]


def plot_figure3() -> None:
    raw = pd.read_csv(TIMING / "figure3_timing_raw.csv")
    summary = aggregate(
        raw,
        ["case", "method", "source_method", "rho"],
        ["learning_time_s", "online_total_time_mean_s"],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "figure3_runtime_summary.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.8), sharex=True)
    for column, case_name in enumerate(F3_CASES):
        case_data = summary[summary["case"].astype(str).eq(case_name)]
        for row, (metric, ylabel) in enumerate(
            (
                ("learning_time_s", "Learning time (s)"),
                ("online_total_time_mean_s", "Online time per LP (s)"),
            )
        ):
            axis = axes[row, column]
            for method in ("OursEstC", "DataDriven", "Rand", "FCNNc"):
                group = (
                    case_data[case_data["method"].astype(str).eq(method)]
                    .sort_values("rho")
                )
                if group.empty:
                    continue
                color, marker, linestyle = STYLE[method]
                x = pd.to_numeric(group["rho"]).to_numpy(float)
                mean = pd.to_numeric(
                    group[f"{metric}_mean"]
                ).to_numpy(float)
                sd = pd.to_numeric(
                    group[f"{metric}_sd"]
                ).fillna(0).to_numpy(float)
                axis.plot(
                    x,
                    mean,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.7,
                    markersize=4,
                    label=f3_label(
                        method,
                        str(group["source_method"].iloc[0]),
                    ),
                )
                axis.fill_between(
                    x,
                    np.maximum(0, mean - sd),
                    mean + sd,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            if row == 0:
                axis.set_title(TITLE[case_name])
            axis.set_xlabel("Target outside mass rho")
            if column == 0:
                axis.set_ylabel(ylabel)
            axis.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("Mean +/- 1 SD across three wall-clock repeats")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save(fig, "figure3_runtime_legacy_algorithms")


def matched_timing_path() -> tuple[Path, bool]:
    matched = TIMING / "figure4_matched_k_timing_raw.csv"
    if matched.exists():
        return matched, True
    return TIMING / "figure4_timing_raw.csv", False


def validate_matched(raw: pd.DataFrame) -> None:
    for _, group in raw.groupby(["repeat", "case", "n_samples"]):
        if set(group["method"]) != {"OursEstC", "FCNNc"}:
            raise RuntimeError("Incomplete matched-K timing group")
        if group["K"].astype(int).nunique() != 1:
            raise RuntimeError("OursEstC and FCNN-c use different K")
        if group["full_dimension"].astype(int).nunique() != 1:
            raise RuntimeError("Methods use different full LP dimensions")


def dimensions_for(raw: pd.DataFrame) -> pd.DataFrame:
    ours = raw[raw["method"].astype(str).eq("OursEstC")]
    dimensions = (
        ours.groupby("case", as_index=False)
        .agg(learned_K=("K", "first"))
    )
    if "full_dimension" in raw:
        full = (
            ours.groupby("case", as_index=False)
            .agg(full_dimension=("full_dimension", "first"))
        )
        return dimensions.merge(full, on="case", validate="one_to_one")
    sizes = pd.read_csv(
        ROOT / "results" / "accuracy" / "instance_sizes.csv"
    )
    return dimensions.merge(
        sizes[["case_id", "standard_form_variables"]].rename(
            columns={
                "case_id": "case",
                "standard_form_variables": "full_dimension",
            }
        ),
        on="case",
        how="left",
        validate="one_to_one",
    )


def plot_figure4() -> None:
    timing_path, matched = matched_timing_path()
    raw = pd.read_csv(timing_path)
    if matched:
        validate_matched(raw)
    metrics = [
        "learning_time_s",
        "solve_time_mean_s",
        "online_total_time_mean_s",
    ]
    has_full = "full_lp_solve_time_mean_s" in raw
    if has_full:
        metrics.append("full_lp_solve_time_mean_s")
    summary = aggregate(
        raw,
        ["case", "method", "K", "n_samples"],
        metrics,
    )
    dimensions = dimensions_for(raw)
    summary = summary.merge(
        dimensions,
        on="case",
        how="left",
        validate="many_to_one",
    )
    if has_full:
        summary = summary.rename(
            columns={
                "full_lp_solve_time_mean_s_mean": (
                    "full_lp_solve_time_s"
                ),
                "full_lp_solve_time_mean_s_sd": (
                    "full_lp_solve_time_sd_s"
                ),
            }
        )
    else:
        full = pd.read_csv(TIMING / "full_lp_timing_summary.csv")
        summary = summary.merge(
            full[
                [
                    "case",
                    "full_lp_solve_time_mean_s",
                    "full_lp_solve_time_repeat_sd_s",
                ]
            ].rename(
                columns={
                    "full_lp_solve_time_mean_s": "full_lp_solve_time_s",
                    "full_lp_solve_time_repeat_sd_s": (
                        "full_lp_solve_time_sd_s"
                    ),
                }
            ),
            on="case",
            how="left",
            validate="many_to_one",
        )
    summary["matched_k_protocol"] = matched
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "figure4_runtime_summary.csv", index=False)

    x = np.arange(len(F4_CASES), dtype=float)
    labels = []
    for case_name in F4_CASES:
        row = dimensions[
            dimensions["case"].astype(str).eq(case_name)
        ].iloc[0]
        labels.append(
            f"{TITLE[case_name]}\nK={int(row['learned_K'])}, "
            f"d={int(row['full_dimension'])}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.0))
    methods = (
        ("OursEstC", "OursEstC", "#c43c35"),
        ("FCNNc", "FCNN-c", "#2673b8"),
    )
    width = 0.30
    for offset, (method, label, color) in zip(
        (-width / 2, width / 2),
        methods,
    ):
        group = (
            summary[summary["method"].astype(str).eq(method)]
            .set_index("case")
            .reindex(F4_CASES)
        )
        axes[0].bar(
            x + offset,
            group["learning_time_s_mean"].to_numpy(float),
            width,
            yerr=group["learning_time_s_sd"].to_numpy(float),
            color=color,
            alpha=0.88,
            capsize=2.5,
            label=label,
        )

    bar_width = 0.24
    for axis, metric, title in (
        (axes[1], "solve_time_mean_s", "HiGHS solver time only"),
        (axes[2], "online_total_time_mean_s", "End-to-end online time"),
    ):
        for offset, (method, label, color) in zip(
            (-bar_width, 0.0),
            methods,
        ):
            group = (
                summary[summary["method"].astype(str).eq(method)]
                .set_index("case")
                .reindex(F4_CASES)
            )
            axis.bar(
                x + offset,
                group[f"{metric}_mean"].to_numpy(float),
                bar_width,
                yerr=group[f"{metric}_sd"].to_numpy(float),
                color=color,
                alpha=0.88,
                capsize=2.5,
                label=label,
            )
        full = (
            summary[summary["method"].astype(str).eq("OursEstC")]
            .set_index("case")
            .reindex(F4_CASES)
        )
        axis.bar(
            x + bar_width,
            full["full_lp_solve_time_s"].to_numpy(float),
            bar_width,
            yerr=full["full_lp_solve_time_sd_s"].to_numpy(float),
            color="#222222",
            alpha=0.88,
            capsize=2.5,
            label="Full LP",
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
    protocol = "matched K" if matched else "historical fixed K"
    fig.suptitle(
        f"Figure 4 runtime ({protocol}): mean +/- 1 SD across "
        "three wall-clock repeats"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "figure4_runtime_legacy_algorithms_matched_k")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    plot_figure3()
    plot_figure4()


if __name__ == "__main__":
    main()
