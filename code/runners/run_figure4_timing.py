#!/usr/bin/env python3
"""Time the clean Figure 4 OursEstC/FCNN-c workload without envelopes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CODE = ROOT / "code" / "experiments"
CORE_CODE = ROOT / "code" / "core"
SCRIPT_DIR = Path(__file__).resolve().parent
for code_path in (SCRIPT_DIR, EXPERIMENT_CODE, CORE_CODE):
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

import run_beyond_prior_sample_sweeps_paper8 as figure4  # noqa: E402
import fcnn_training_protocols as fcnn_protocols  # noqa: E402
from run_figure3_timing import (  # noqa: E402
    append_fixed_with_timing,
    append_learned_with_timing,
    append_signed_with_timing,
    mean_se,
)


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


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def split_int_csv(value: str) -> list[int]:
    return [int(part) for part in split_csv(value)]


def experiment_args(k_list: list[int]) -> argparse.Namespace:
    return argparse.Namespace(
        seed=17,
        device="cpu",
        verbose=False,
        n_val=6,
        n_test=12,
        k_list=",".join(str(value) for value in k_list),
        rand_trials=3,
        hidden_dim=32,
        epochs=500,
        batch_size=8,
        lr=1e-3,
        patience=120,
        pilot_n=96,
        rho=0.10,
        radius_scale=1.10,
        gaussian_std_scale=0.35,
        case_gaussian_std_scales=(
            "random_lp_A=20,random_lp_B=20,"
            "random_lp_C=20,random_lp_D=20"
        ),
        cost_space="reward_subspace",
        svd_tol=1e-10,
    )


def install_timing_hooks() -> None:
    figure4.fps.append_learned_projector_rows = append_learned_with_timing
    figure4.fps.append_fixed_projection_rows = append_fixed_with_timing
    figure4.fps.append_signed_costonly_rows = append_signed_with_timing


def ours_record(
    summary: pd.DataFrame,
    repeat: int,
    case_name: str,
    n_samples: int,
    preparation_time: float,
) -> dict[str, object]:
    row = summary.iloc[0]
    return {
        "figure": "Figure 4",
        "repeat": int(repeat),
        "case": case_name,
        "method": "OursEstC",
        "source_method": "OursEstC",
        "rho": 0.1,
        "n_samples": int(n_samples),
        "K": int(row["K"]),
        "learning_time_s": float(row["fit_time"]),
        "data_preparation_time_s": float(preparation_time),
        "solve_time_mean_s": float(row["time_mean"]),
        "solve_time_se_s": float(row["time_se"]),
        "projection_time_mean_s": 0.0,
        "projection_time_se_s": 0.0,
        "online_total_time_mean_s": float(row["time_mean"]),
        "online_total_time_se_s": float(row["time_se"]),
        "objective_ratio_mean": float(row["objective_ratio_mean"]),
        "objective_ratio_se": float(row["objective_ratio_se"]),
        "solve_observations": int(row["n"]),
    }


def fcnn_records(
    raw: pd.DataFrame,
    fit: pd.DataFrame,
    repeat: int,
    case_name: str,
    n_samples: int,
    preparation_time: float,
    initialization_seed: int,
    initialization_protocol: str,
    training_args: argparse.Namespace,
) -> list[dict[str, object]]:
    fit_map = {
        int(row["K"]): float(row["fit_time"])
        for _, row in fit[fit["method"].astype(str).eq("CostOnly")].iterrows()
    }
    records: list[dict[str, object]] = []
    for k, group in raw[raw["method"].astype(str).eq("CostOnly")].groupby("K"):
        k_int = int(k)
        ratio_mean, ratio_se, observations = mean_se(group["objective_ratio"])
        solve_mean, solve_se, _ = mean_se(group["lp_solve_time_s"])
        projection_mean, projection_se, _ = mean_se(group["projection_time_s"])
        online_mean, online_se, _ = mean_se(group["online_total_time_s"])
        records.append(
            {
                "figure": "Figure 4",
                "repeat": int(repeat),
                "case": case_name,
                "method": "FCNNc",
                "source_method": "FCNN-c",
                "rho": 0.1,
                "n_samples": int(n_samples),
                "K": k_int,
                "learning_time_s": fit_map[k_int],
                "data_preparation_time_s": float(preparation_time),
                "solve_time_mean_s": solve_mean,
                "solve_time_se_s": solve_se,
                "projection_time_mean_s": projection_mean,
                "projection_time_se_s": projection_se,
                "online_total_time_mean_s": online_mean,
                "online_total_time_se_s": online_se,
                "objective_ratio_mean": ratio_mean,
                "objective_ratio_se": ratio_se,
                "solve_observations": observations,
                "fcnn_initialization_seed": int(initialization_seed),
                "fcnn_initialization_protocol": initialization_protocol,
                "fcnn_hidden_dim": int(training_args.hidden_dim),
                "fcnn_epochs": int(training_args.epochs),
                "fcnn_batch_size": int(training_args.batch_size),
                "fcnn_patience": int(training_args.patience),
                "fcnn_lr": float(training_args.lr),
                "fcnn_validation_samples": int(training_args.n_val),
                "fcnn_test_samples": int(training_args.n_test),
            }
        )
    return records


def run_case(
    case: object,
    case_name: str,
    repeat: int,
    n_values: list[int],
    k_values: list[int],
    out_dir: Path,
) -> list[dict[str, object]]:
    args = experiment_args(k_values)
    if case_name == "packing":
        args.hidden_dim = 12
        args.epochs = 6
        args.batch_size = 4
        args.patience = 3
        fcnn_protocols.use_legacy_cost_pca_protocol()
        initialization_protocol = "legacy_cost_pca"
    else:
        fcnn_protocols.use_plain_random_protocol()
        initialization_protocol = "plain_torch_random"
    # The archived sweep did not store Torch RNG state; fix a case-level seed.
    initialization_seed = int(args.seed) + int(case.seed_offset) + 19911
    fcnn_protocols.seed_all(initialization_seed)
    preparation_start = time.perf_counter()
    data = figure4.prepare_fixed_pool_case(case, args, max(n_values))
    preparation_time = time.perf_counter() - preparation_start
    records: list[dict[str, object]] = []

    repeat_dir = out_dir / f"repeat_{repeat}" / case_name
    for n_samples in n_values:
        print(
            f"[Figure 4 timing] repeat={repeat}, case={case_name}, "
            f"n={n_samples}, K={k_values}",
            flush=True,
        )
        ours_summary = figure4.summarize_ours_from_pool(
            data,
            args,
            n_samples,
            repeat_dir,
            force=True,
        )
        records.append(
            ours_record(
                ours_summary,
                repeat,
                case_name,
                n_samples,
                preparation_time,
            )
        )

        case_out = repeat_dir / "fcnn" / f"n_{n_samples:03d}"
        prepared = figure4.fps.PreparedCaseData(
            case=data.case,
            args=data.args,
            problem=data.problem,
            bundle=data.bundle,
            proj_train=data.proj_train,
            proj_val=data.proj_val,
            proj_test=data.proj_test,
        )
        figure4.fps.run_case_from_prepared(
            prepared,
            case_out,
            methods=["CostOnly"],
            train_limit=int(n_samples),
            force=True,
            write_rank=False,
        )
        raw = pd.read_csv(case_out / "raw_results.csv")
        fit = pd.read_csv(case_out / "fit_times.csv")
        records.extend(
            fcnn_records(
                raw,
                fit,
                repeat,
                case_name,
                n_samples,
                preparation_time,
                initialization_seed,
                initialization_protocol,
                args,
            )
        )
    return records


def update_combined(output_dir: Path) -> None:
    files = sorted(output_dir.glob("figure*_timing_raw.csv"))
    frames = [pd.read_csv(path) for path in files if path.name != "timing_repeated.csv"]
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            output_dir / "timing_repeated.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", default=",".join(CASE_ORDER))
    parser.add_argument("--n-list", default="20")
    parser.add_argument("--k-list", default="20")
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "timing",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cases = split_csv(args.cases)
    unknown = sorted(set(cases) - set(CASE_ORDER))
    if unknown:
        raise ValueError(f"Unknown Figure 4 cases: {unknown}")
    n_values = [20, 40, 60, 80] if args.full_grid else split_int_csv(args.n_list)
    k_values = [10, 20, 30, 50] if args.full_grid else split_int_csv(args.k_list)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "figure4_timing_raw.csv"
    existing = (
        pd.read_csv(output_path)
        if output_path.exists() and not args.force
        else pd.DataFrame()
    )
    frames: list[pd.DataFrame] = []
    if not existing.empty:
        frames.append(existing)

    figure4.patch_styles()

    install_timing_hooks()
    case_specs = figure4.get_cases()
    for repeat in range(1, int(args.repeats) + 1):
        for case_name in cases:
            if not existing.empty:
                done_base = existing[
                    existing["repeat"].astype(int).eq(repeat)
                    & existing["case"].astype(str).eq(case_name)
                    & existing["n_samples"].astype(int).isin(n_values)
                ]
                done_ours = done_base[
                    done_base["method"].astype(str).eq("OursEstC")
                ]
                done_fcnn = done_base[
                    done_base["method"].astype(str).eq("FCNNc")
                    & done_base["K"].astype(int).isin(k_values)
                ]
                if (
                    len(done_ours) >= len(n_values)
                    and len(done_fcnn) >= len(n_values) * len(k_values)
                ):
                    print(
                        f"[Figure 4 timing] reuse repeat={repeat}, case={case_name}",
                        flush=True,
                    )
                    continue
            case_records = run_case(
                case_specs[case_name],
                case_name,
                repeat,
                n_values,
                k_values,
                out_dir / "figure4_runs",
            )
            frames.append(pd.DataFrame(case_records))
            pd.concat(frames, ignore_index=True, sort=False).to_csv(
                output_path,
                index=False,
            )
    update_combined(out_dir)
    print(f"Saved Figure 4 timing to {output_path}", flush=True)


if __name__ == "__main__":
    main()
