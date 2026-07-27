#!/usr/bin/env python3
"""Repeat the exact Figure 3 method mapping and record fit/solve time."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CODE = ROOT / "code" / "experiments"
CORE_CODE = ROOT / "code" / "core"
for code_path in (EXPERIMENT_CODE, CORE_CODE):
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

import run_dynamic_k_baseline_compare_four_lp as figure3  # noqa: E402
import iwata_sakaue_pelp_projection_compare as iw  # noqa: E402
import fcnn_training_protocols as fcnn_protocols  # noqa: E402


CASE_ORDER = ["packing", "sc205", "random_lp_A", "random_lp_C"]
NEURAL_BUDGET = {
    "packing": {"epochs": 500, "patience": 120},
    "sc205": {"epochs": 500, "patience": 120},
    "random_lp_A": {"epochs": 10, "patience": 3},
    "random_lp_C": {"epochs": 10, "patience": 3},
}
RHO_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
METHOD_MAP = {
    "Rand": ("Rand", "Rand"),
    "PCA": ("DataDriven", "PCA"),
    "SGA": ("DataDriven", "SGA"),
    "CostOnly": ("FCNNc", "FCNN-c"),
}


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def mean_se(values: pd.Series) -> tuple[float, float, int]:
    data = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return math.nan, math.nan, 0
    if data.size == 1:
        return float(data[0]), 0.0, 1
    return (
        float(np.mean(data)),
        float(np.std(data, ddof=1) / np.sqrt(data.size)),
        int(data.size),
    )


@torch.no_grad()
def append_learned_with_timing(
    rows: list[dict[str, object]],
    method: str,
    model: torch.nn.Module,
    instances: list[object],
    k: int,
    device: torch.device,
) -> None:
    model.eval()
    for instance in instances:
        c_tensor, a_tensor, b_tensor = iw.to_torch_instance(instance, device)
        projection_start = time.perf_counter()
        projection = (
            model(c_tensor, a_tensor, b_tensor)
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
        projection_time = time.perf_counter() - projection_start
        solution = iw.solve_projected_lp_max(instance, projection)
        rows.append(
            {
                "method": method,
                "K": int(k),
                "instance": instance.name,
                "objective": solution.obj,
                "full_objective": instance.full_obj,
                "objective_ratio": (
                    iw.objective_ratio(solution.obj, instance.full_obj)
                    if solution.success
                    else np.nan
                ),
                "time": float(projection_time + solution.runtime),
                "projection_time_s": float(projection_time),
                "lp_solve_time_s": float(solution.runtime),
                "online_total_time_s": float(projection_time + solution.runtime),
                "success": float(solution.success),
            }
        )


def append_fixed_with_timing(
    rows: list[dict[str, object]],
    method: str,
    projection: np.ndarray,
    instances: list[object],
    k: int,
) -> None:
    for instance in instances:
        solution = iw.solve_projected_lp_max(instance, projection)
        rows.append(
            {
                "method": method,
                "K": int(k),
                "instance": instance.name,
                "objective": solution.obj,
                "full_objective": instance.full_obj,
                "objective_ratio": (
                    iw.objective_ratio(solution.obj, instance.full_obj)
                    if solution.success
                    else np.nan
                ),
                "time": float(solution.runtime),
                "projection_time_s": 0.0,
                "lp_solve_time_s": float(solution.runtime),
                "online_total_time_s": float(solution.runtime),
                "success": float(solution.success),
            }
        )


@torch.no_grad()
def append_signed_with_timing(
    rows: list[dict[str, object]],
    model: torch.nn.Module,
    instances: list[object],
    nullspace: np.ndarray,
    reference: np.ndarray,
    k: int,
    device: torch.device,
    n_slack: int,
) -> None:
    model.eval()
    for instance in instances:
        reward = torch.as_tensor(
            instance.g_reward,
            dtype=torch.float32,
            device=device,
        )
        projection_start = time.perf_counter()
        projection = model(reward).detach().cpu().numpy().astype(float)
        projection_time = time.perf_counter() - projection_start
        (
            success,
            objective,
            solve_time,
            _y,
            _dual,
            x_standard,
        ) = figure3.fps.solve_signed_nullspace_lp_max(
            instance,
            projection,
            nullspace,
            reference,
        )
        rows.append(
            {
                "method": "CostOnly",
                "K": int(k),
                "instance": instance.name,
                "objective": objective,
                "full_objective": instance.full_objective,
                "objective_ratio": (
                    iw.objective_ratio(objective, instance.full_objective)
                    if success
                    else np.nan
                ),
                "time": float(projection_time + solve_time),
                "projection_time_s": float(projection_time),
                "lp_solve_time_s": float(solve_time),
                "online_total_time_s": float(projection_time + solve_time),
                "success": float(success),
                "signed_nullspace_obj": objective,
                "x_std_norm": (
                    float(np.linalg.norm(x_standard))
                    if success and x_standard is not None
                    else float("nan")
                ),
                "y_support_dim": (
                    int(max(0, np.count_nonzero(x_standard[n_slack:] > 1e-8)))
                    if success and x_standard is not None
                    else 0
                ),
            }
        )


def install_timing_hooks() -> None:
    figure3.fps.append_learned_projector_rows = append_learned_with_timing
    figure3.fps.append_fixed_projection_rows = append_fixed_with_timing
    figure3.fps.append_signed_costonly_rows = append_signed_with_timing


def neural_args(case_name: str) -> argparse.Namespace:
    budget = NEURAL_BUDGET[case_name]
    return argparse.Namespace(
        costonly_init="plain_torch_random",
        costonly_solver_space="auto",
        costonly_warm_cols=2,
        hidden_dim=32,
        epochs=int(budget["epochs"]),
        batch_size=8,
        lr=1e-3,
        patience=int(budget["patience"]),
        rand_trials=3,
    )


def summarize_method(
    raw: pd.DataFrame,
    figure: str,
    repeat: int,
    case_name: str,
    method: str,
    source_method: str,
    rho: float,
    n_samples: int,
    k: int,
    learning_time: float,
    preparation_time: float,
) -> dict[str, object]:
    ratio_mean, ratio_se, observations = mean_se(raw["objective_ratio"])
    online_mean, online_se, _ = mean_se(raw["online_total_time_s"])
    solve_mean, solve_se, _ = mean_se(raw["lp_solve_time_s"])
    projection_mean, projection_se, _ = mean_se(raw["projection_time_s"])
    return {
        "figure": figure,
        "repeat": int(repeat),
        "case": case_name,
        "method": method,
        "source_method": source_method,
        "rho": float(rho),
        "n_samples": int(n_samples),
        "K": int(k),
        "learning_time_s": float(learning_time),
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
    }


def run_case(
    case_name: str,
    repeat: int,
    out_dir: Path,
) -> list[dict[str, object]]:
    config = figure3.OLD_SUITE_DYNAMIC_CONFIGS[case_name]
    preparation_start = time.perf_counter()
    data = figure3.olddyn.prepare_case(case_name, config, n_samples=40)
    preparation_time = time.perf_counter() - preparation_start
    args = neural_args(case_name)
    rows: list[dict[str, object]] = []

    for rho in RHO_VALUES:
        print(
            f"[Figure 3 timing] repeat={repeat}, case={case_name}, rho={rho:.1f}",
            flush=True,
        )
        ours_start = time.perf_counter()
        ours_raw, learned_dimension, _retained = (
            figure3.olddyn.evaluate_ours_at_rho(data, config, rho, 40)
        )
        ours_learning_time = time.perf_counter() - ours_start
        ours_raw = ours_raw.copy()
        ours_raw["lp_solve_time_s"] = pd.to_numeric(
            ours_raw["time"], errors="coerce"
        )
        ours_raw["projection_time_s"] = 0.0
        ours_raw["online_total_time_s"] = pd.to_numeric(
            ours_raw["time"], errors="coerce"
        )
        rows.append(
            summarize_method(
                ours_raw,
                "Figure 3",
                repeat,
                case_name,
                "OursEstC",
                "OursEstC",
                rho,
                40,
                int(learned_dimension),
                ours_learning_time,
                preparation_time,
            )
        )

        run_dir = (
            out_dir
            / f"repeat_{repeat}"
            / case_name
            / f"rho_{rho:.1f}_K{max(1, int(learned_dimension))}"
        )
        # The archived nested sweep omitted a Torch seed. Use the related
        # beyond-prior runner's seed family plus rho, and record it below.
        initialization_seed = (
            int(config.seed)
            + int(data.case.seed_offset)
            + 19911
            + int(round(1000.0 * rho))
        )
        fcnn_protocols.seed_all(initialization_seed)
        baseline_raw = figure3.evaluate_old_suite_baselines(
            data,
            selected_projection=config.selected_projection,
            rho=rho,
            k_budget=int(learned_dimension),
            n_samples=40,
            out_dir=run_dir,
            include_fcnn=True,
            neural_args=args,
        )
        fit = pd.read_csv(run_dir / "fit_times.csv")
        fit_map = {
            str(row["method"]): float(row["fit_time"])
            for _, row in fit.iterrows()
        }
        for raw_method, (method, source_method) in METHOD_MAP.items():
            normalized_method = (
                "DataDriven"
                if raw_method in {"PCA", "SGA"}
                else ("FCNNc" if raw_method == "CostOnly" else raw_method)
            )
            group = baseline_raw[
                baseline_raw["method"].astype(str).eq(normalized_method)
            ].copy()
            if group.empty or raw_method not in fit_map:
                continue
            historical_total = pd.to_numeric(group["time"], errors="coerce")
            missing_timing = pd.Series(
                np.nan,
                index=group.index,
                dtype=float,
            )
            solve_time = pd.to_numeric(
                (
                    group["lp_solve_time_s"]
                    if "lp_solve_time_s" in group
                    else missing_timing
                ),
                errors="coerce",
            )
            projection_time = pd.to_numeric(
                (
                    group["projection_time_s"]
                    if "projection_time_s" in group
                    else missing_timing
                ),
                errors="coerce",
            )
            online_time = pd.to_numeric(
                (
                    group["online_total_time_s"]
                    if "online_total_time_s" in group
                    else missing_timing
                ),
                errors="coerce",
            )
            # Historical fixed-projector rows store the LP runtime in `time`.
            # Mixed-method CSVs still contain the newer timing columns, but
            # those cells are NaN for PCA/SGA and must be filled row by row.
            group["lp_solve_time_s"] = solve_time.fillna(historical_total)
            group["projection_time_s"] = projection_time.fillna(0.0)
            group["online_total_time_s"] = online_time.fillna(historical_total)
            record = (
                summarize_method(
                    group,
                    "Figure 3",
                    repeat,
                    case_name,
                    method,
                    source_method,
                    rho,
                    40,
                    max(1, int(learned_dimension)),
                    fit_map[raw_method],
                    preparation_time,
                )
            )
            if method == "FCNNc":
                record["fcnn_initialization_seed"] = int(initialization_seed)
            rows.append(record)
    return rows


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
        raise ValueError(f"Unknown Figure 3 cases: {unknown}")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "figure3_timing_raw.csv"
    existing = (
        pd.read_csv(output_path)
        if output_path.exists() and not args.force
        else pd.DataFrame()
    )
    frames: list[pd.DataFrame] = []
    if not existing.empty:
        frames.append(existing)

    install_timing_hooks()
    for repeat in range(1, int(args.repeats) + 1):
        for case_name in cases:
            if not existing.empty:
                done = existing[
                    existing["repeat"].astype(int).eq(repeat)
                    & existing["case"].astype(str).eq(case_name)
                ]
                if len(done) == 20:
                    print(
                        f"[Figure 3 timing] reuse repeat={repeat}, case={case_name}",
                        flush=True,
                    )
                    continue
            case_rows = run_case(case_name, repeat, out_dir / "figure3_runs")
            frames.append(pd.DataFrame(case_rows))
            pd.concat(frames, ignore_index=True, sort=False).to_csv(
                output_path,
                index=False,
            )
    update_combined(out_dir)
    print(f"Saved Figure 3 timing to {output_path}", flush=True)


if __name__ == "__main__":
    main()
