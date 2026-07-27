#!/usr/bin/env python3
"""Time legacy OursEstC and FCNN-c with the same learned dimension."""

from __future__ import annotations

import argparse
import copy
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


def experiment_args(device: str) -> argparse.Namespace:
    return argparse.Namespace(
        seed=17,
        device=str(device),
        verbose=False,
        n_val=6,
        n_test=12,
        k_list="20",
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


def full_lp_statistics(data: object) -> tuple[float, float, int, int]:
    times = pd.Series(
        [float(instance.full_time) for instance in data.proj_test],
        dtype=float,
    )
    mean, se, observations = mean_se(times)
    full_dimension = int(data.proj_test[0].n_vars)
    return mean, se, observations, full_dimension


def ours_record(
    summary: pd.DataFrame,
    repeat: int,
    case_name: str,
    n_samples: int,
    preparation_time: float,
    full_lp_mean: float,
    full_lp_se: float,
    full_lp_observations: int,
    full_dimension: int,
) -> dict[str, object]:
    row = summary.iloc[0]
    learning_core = float(row["fit_time"])
    matched_k = int(row["K"])
    return {
        "figure": "Figure 4",
        "protocol": "matched_k_legacy_algorithms",
        "repeat": int(repeat),
        "case": case_name,
        "method": "OursEstC",
        "source_method": "OursEstC",
        "rho": 0.1,
        "n_samples": int(n_samples),
        "K": matched_k,
        "matched_K": matched_k,
        "full_dimension": int(full_dimension),
        "learning_time_s": learning_core,
        "data_preparation_time_s": float(preparation_time),
        "learning_plus_preparation_time_s": (
            learning_core + float(preparation_time)
        ),
        "solve_time_mean_s": float(row["time_mean"]),
        "solve_time_se_s": float(row["time_se"]),
        "projection_time_mean_s": 0.0,
        "projection_time_se_s": 0.0,
        "online_total_time_mean_s": float(row["time_mean"]),
        "online_total_time_se_s": float(row["time_se"]),
        "full_lp_solve_time_mean_s": float(full_lp_mean),
        "full_lp_solve_time_se_s": float(full_lp_se),
        "full_lp_solve_observations": int(full_lp_observations),
        "objective_ratio_mean": float(row["objective_ratio_mean"]),
        "objective_ratio_se": float(row["objective_ratio_se"]),
        "solve_observations": int(row["n"]),
        "lp_backend": "scipy-highs-cpu",
    }


def fcnn_record(
    raw: pd.DataFrame,
    fit: pd.DataFrame,
    repeat: int,
    case_name: str,
    n_samples: int,
    preparation_time: float,
    initialization_seed: int,
    matched_k: int,
    full_lp_mean: float,
    full_lp_se: float,
    full_lp_observations: int,
    full_dimension: int,
    device: str,
) -> dict[str, object]:
    selected_raw = raw[
        raw["method"].astype(str).eq("CostOnly")
        & pd.to_numeric(raw["K"], errors="coerce").eq(int(matched_k))
    ]
    selected_fit = fit[
        fit["method"].astype(str).eq("CostOnly")
        & pd.to_numeric(fit["K"], errors="coerce").eq(int(matched_k))
    ]
    if selected_raw.empty or len(selected_fit) != 1:
        raise RuntimeError(
            f"Missing legacy FCNN-c output for {case_name}, "
            f"n={n_samples}, K={matched_k}"
        )

    ratio_mean, ratio_se, observations = mean_se(
        selected_raw["objective_ratio"]
    )
    solve_mean, solve_se, _ = mean_se(selected_raw["lp_solve_time_s"])
    projection_mean, projection_se, _ = mean_se(
        selected_raw["projection_time_s"]
    )
    online_mean, online_se, _ = mean_se(
        selected_raw["online_total_time_s"]
    )
    learning_core = float(selected_fit.iloc[0]["fit_time"])
    return {
        "figure": "Figure 4",
        "protocol": "matched_k_legacy_algorithms",
        "repeat": int(repeat),
        "case": case_name,
        "method": "FCNNc",
        "source_method": "FCNN-c legacy",
        "rho": 0.1,
        "n_samples": int(n_samples),
        "K": int(matched_k),
        "matched_K": int(matched_k),
        "full_dimension": int(full_dimension),
        "learning_time_s": learning_core,
        "data_preparation_time_s": float(preparation_time),
        "learning_plus_preparation_time_s": (
            learning_core + float(preparation_time)
        ),
        "solve_time_mean_s": solve_mean,
        "solve_time_se_s": solve_se,
        "projection_time_mean_s": projection_mean,
        "projection_time_se_s": projection_se,
        "online_total_time_mean_s": online_mean,
        "online_total_time_se_s": online_se,
        "full_lp_solve_time_mean_s": float(full_lp_mean),
        "full_lp_solve_time_se_s": float(full_lp_se),
        "full_lp_solve_observations": int(full_lp_observations),
        "objective_ratio_mean": ratio_mean,
        "objective_ratio_se": ratio_se,
        "solve_observations": observations,
        "fcnn_initialization_seed": int(initialization_seed),
        "torch_device": str(device),
        "lp_backend": "scipy-highs-cpu",
    }


def run_case(
    case: object,
    case_name: str,
    repeat: int,
    n_values: list[int],
    out_dir: Path,
    device: str,
) -> list[dict[str, object]]:
    args = experiment_args(device)
    preparation_start = time.perf_counter()
    data = figure4.prepare_fixed_pool_case(case, args, max(n_values))
    preparation_time = time.perf_counter() - preparation_start
    full_lp_mean, full_lp_se, full_lp_n, full_dimension = (
        full_lp_statistics(data)
    )
    records: list[dict[str, object]] = []

    repeat_dir = out_dir / f"repeat_{repeat}" / case_name
    for n_samples in n_values:
        ours_summary = figure4.summarize_ours_from_pool(
            data,
            args,
            n_samples,
            repeat_dir,
            force=True,
        )
        matched_k = int(ours_summary.iloc[0]["K"])
        if not 1 <= matched_k <= full_dimension:
            raise RuntimeError(
                f"Invalid learned K={matched_k} for d={full_dimension}"
            )
        print(
            f"[Figure 4 matched-K] repeat={repeat}, case={case_name}, "
            f"n={n_samples}, d={full_dimension}, K={matched_k}",
            flush=True,
        )
        records.append(
            ours_record(
                ours_summary,
                repeat,
                case_name,
                n_samples,
                preparation_time,
                full_lp_mean,
                full_lp_se,
                full_lp_n,
                full_dimension,
            )
        )

        prepared_args = copy.deepcopy(data.args)
        prepared_args.k_list = str(matched_k)
        prepared = figure4.fps.PreparedCaseData(
            case=data.case,
            args=prepared_args,
            problem=data.problem,
            bundle=data.bundle,
            proj_train=data.proj_train,
            proj_val=data.proj_val,
            proj_test=data.proj_test,
        )
        initialization_seed = (
            int(args.seed)
            + int(case.seed_offset)
            + 19911
            + 101 * int(n_samples)
            + int(matched_k)
        )
        fcnn_protocols.seed_all(initialization_seed)
        case_out = repeat_dir / "fcnn_matched" / f"n_{n_samples:03d}"
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
        records.append(
            fcnn_record(
                raw,
                fit,
                repeat,
                case_name,
                n_samples,
                preparation_time,
                initialization_seed,
                matched_k,
                full_lp_mean,
                full_lp_se,
                full_lp_n,
                full_dimension,
                device,
            )
        )
    return records


def validate_output(frame: pd.DataFrame, repeats: int, cases: list[str],
                    n_values: list[int]) -> None:
    expected_rows = int(repeats) * len(cases) * len(n_values) * 2
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, found {len(frame)}"
        )
    keys = ["repeat", "case", "n_samples"]
    for _, group in frame.groupby(keys, sort=False):
        if set(group["method"]) != {"OursEstC", "FCNNc"}:
            raise RuntimeError("A matched-K group is missing a method")
        if group["K"].astype(int).nunique() != 1:
            raise RuntimeError("OursEstC and FCNN-c use different K")
        if group["full_dimension"].astype(int).nunique() != 1:
            raise RuntimeError("Methods use different full LP dimensions")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", default=",".join(CASE_ORDER))
    parser.add_argument("--n-list", default="20")
    parser.add_argument("--device", default="cpu")
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
    n_values = split_int_csv(args.n_list)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "figure4_matched_k_timing_raw.csv"

    if output_path.exists() and not args.force:
        existing = pd.read_csv(output_path)
        validate_output(existing, int(args.repeats), cases, n_values)
        print(f"Validated existing matched-K timing: {output_path}")
        return

    figure4.patch_styles()
    fcnn_protocols.use_plain_random_protocol()
    install_timing_hooks()
    case_specs = figure4.get_cases()
    records: list[dict[str, object]] = []
    for repeat in range(1, int(args.repeats) + 1):
        for case_name in cases:
            records.extend(
                run_case(
                    case_specs[case_name],
                    case_name,
                    repeat,
                    n_values,
                    out_dir / "figure4_matched_k_runs",
                    str(args.device),
                )
            )
            pd.DataFrame(records).to_csv(output_path, index=False)

    frame = pd.DataFrame(records)
    validate_output(frame, int(args.repeats), cases, n_values)
    frame.to_csv(output_path, index=False)
    print(f"Saved matched-K Figure 4 timing to {output_path}", flush=True)


if __name__ == "__main__":
    main()
