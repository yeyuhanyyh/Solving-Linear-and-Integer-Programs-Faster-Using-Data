#!/usr/bin/env python3
"""Plot fresh optimizer-span Ours timing with frozen Figure 3 baselines."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TIMING = ROOT / "results" / "timing"
OUT = ROOT / "figures"
RAW = TIMING / "figure3_ours_timing_raw.csv"
LEGACY = TIMING / "figure3_timing_raw.csv"
CASES = ("packing", "sc205", "random_lp_A", "random_lp_C")
TITLE = {
    "packing": "Packing",
    "sc205": "SC205",
    "random_lp_A": "RandomLP A",
    "random_lp_C": "RandomLP C",
}
STYLE = {
    "OursEstC": ("#c43c35", "D", "-"),
    "DataDriven": ("#2a8a6e", "P", "-."),
    "Rand": ("#66705a", "x", ":"),
    "FCNNc": ("#2673b8", "s", "--"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sample_sd(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.std(ddof=1)) if len(clean) > 1 else 0.0


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["case", "method", "source_method", "rho"]
    for group_key, group in raw.groupby(keys, dropna=False, sort=True):
        case_name, method, source_method, rho = group_key
        row: dict[str, object] = {
            "case": case_name,
            "method": method,
            "source_method": source_method,
            "rho": float(rho),
            "timing_repeats": int(group["repeat"].nunique()),
            "K_min": int(pd.to_numeric(group["K"]).min()),
            "K_max": int(pd.to_numeric(group["K"]).max()),
        }
        for metric in ("learning_time_s", "solve_time_mean_s"):
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(numeric.mean())
            row[f"{metric}_sd"] = sample_sd(numeric)
        rows.append(row)
    return pd.DataFrame(rows)


def method_label(method: str, source: str) -> str:
    if method == "DataDriven":
        return source
    return {
        "OursEstC": "Ours (optimizer span)",
        "Rand": "Random projection",
        "FCNNc": "FCNN-c",
    }[method]


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        OUT / f"{stem}.svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def validate(ours: pd.DataFrame, baselines: pd.DataFrame) -> None:
    if len(ours) == 0 or len(baselines) != 180:
        raise RuntimeError("Figure 3 Ours/baseline timing grid is incomplete")
    hashes = set(ours["baseline_source_sha256"].astype(str))
    if hashes != {sha256_file(LEGACY)}:
        raise RuntimeError("Figure 3 baseline source hash mismatch")
    if set(baselines["method"].astype(str)) != {
        "DataDriven", "Rand", "FCNNc"
    }:
        raise RuntimeError("Unexpected Figure 3 baseline method")
    expected = {
        (case_name, rho)
        for case_name in CASES
        for rho in (0.1, 0.3, 0.5, 0.7, 0.9)
    }
    actual = {
        (str(row.case), round(float(row.rho), 1))
        for row in ours.itertuples()
    }
    if actual != expected:
        raise RuntimeError("Fresh Figure 3 Ours grid is incomplete")


def main() -> None:
    ours = pd.read_csv(RAW)
    legacy = pd.read_csv(LEGACY)
    baselines = legacy[
        ~legacy["method"].astype(str).eq("OursEstC")
    ].copy()
    validate(ours, baselines)
    raw = pd.concat([ours, baselines], ignore_index=True, sort=False)
    summary = aggregate(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        OUT / "figure3_runtime_new_ours_summary.csv",
        index=False,
    )

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.8), sharex=True)
    for column, case_name in enumerate(CASES):
        case_data = summary[
            summary["case"].astype(str).eq(case_name)
        ]
        for row, (metric, ylabel) in enumerate(
            (
                ("learning_time_s", "Learning time (s)"),
                ("solve_time_mean_s", "Reduced LP solver time (s)"),
            )
        ):
            axis = axes[row, column]
            for method in ("OursEstC", "DataDriven", "Rand", "FCNNc"):
                group = case_data[
                    case_data["method"].astype(str).eq(method)
                ].sort_values("rho")
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
                    label=method_label(
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
    ours_repeats = int(
        summary[
            summary["method"].astype(str).eq("OursEstC")
        ]["timing_repeats"].max()
    )
    baseline_repeats = int(
        summary[
            ~summary["method"].astype(str).eq("OursEstC")
        ]["timing_repeats"].max()
    )
    fig.suptitle(
        f"Fresh Ours: {ours_repeats} wall-clock run; "
        f"archived baselines: {baseline_repeats} repeats"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save(fig, "figure3_runtime_new_ours")
    print(
        f"Saved Figure 3 runtime figure in {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()

