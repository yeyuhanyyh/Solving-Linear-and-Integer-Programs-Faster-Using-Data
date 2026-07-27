#!/usr/bin/env python3
"""Re-evaluate the bundled GROW7/SC205 Figure 4 K=50 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CODE = ROOT / "code" / "experiments"
CORE_CODE = ROOT / "code" / "core"
SCRIPT_DIR = Path(__file__).resolve().parent
for code_path in (SCRIPT_DIR, EXPERIMENT_CODE, CORE_CODE):
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

import final_projection_figure_suite as fps  # noqa: E402
import run_beyond_prior_sample_sweeps_paper8 as figure4  # noqa: E402
from run_figure4_timing import experiment_args  # noqa: E402


CHECKPOINTS = ROOT / "data" / "checkpoints" / "figure4_netlib_k50"
FRESH = ROOT / "results" / "fresh"
CASES = ("grow7", "sc205")
N_VALUES = (20, 40, 60, 80)
RATIO_RTOL = 1e-6
RATIO_ATOL = 1e-8


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mean_se(values: pd.Series) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if array.size == 0:
        return math.nan, math.nan
    if array.size == 1:
        return float(array[0]), 0.0
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1) / math.sqrt(array.size)),
    )


def evaluate_checkpoints() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(CHECKPOINTS / "manifest.csv")
    case_map = figure4.get_cases()
    experiment = experiment_args([50])
    summaries: list[dict[str, object]] = []
    test_frames: list[pd.DataFrame] = []

    for case_name in CASES:
        data = figure4.prepare_fixed_pool_case(
            case_map[case_name],
            experiment,
            max(N_VALUES),
        )
        prepared = fps.PreparedCaseData(
            case=data.case,
            args=data.args,
            problem=data.problem,
            bundle=data.bundle,
            proj_train=data.proj_train,
            proj_val=data.proj_val,
            proj_test=data.proj_test,
        )
        signed = fps.prepare_signed_nullspace_data(prepared)
        reduced_dim = int(signed.N.shape[1])

        for n_samples in N_VALUES:
            expected = manifest[
                manifest["case"].astype(str).eq(case_name)
                & manifest["n_samples"].astype(int).eq(n_samples)
            ]
            if len(expected) != 1:
                raise RuntimeError(
                    f"Missing manifest row for {case_name}, n={n_samples}"
                )
            expected_row = expected.iloc[0]
            checkpoint_origin = str(
                expected_row.get(
                    "checkpoint_origin",
                    "original_archived_checkpoint",
                )
            )
            initialization_seed = expected_row.get(
                "initialization_seed", math.nan
            )
            trainer_seed = expected_row.get("trainer_seed", math.nan)
            history_file = expected_row.get("history_file", math.nan)
            if pd.notna(history_file) and str(history_file).strip():
                history = pd.read_csv(CHECKPOINTS / str(history_file))
                best_index = pd.to_numeric(
                    history["val_ratio"], errors="raise"
                ).idxmax()
                best_row = history.loc[best_index]
                expected_epoch = int(expected_row["best_validation_epoch"])
                expected_ratio = float(expected_row["best_validation_ratio"])
                if int(best_row["epoch"]) != expected_epoch or not math.isclose(
                    float(best_row["val_ratio"]),
                    expected_ratio,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"Validation-history mismatch for {case_name}, "
                        f"n={n_samples}"
                    )
            checkpoint = CHECKPOINTS / str(expected_row["checkpoint"])
            observed_hash = file_sha256(checkpoint)
            if observed_hash != str(expected_row["sha256"]):
                raise RuntimeError(f"SHA-256 mismatch: {checkpoint}")

            model = fps.SignedCostOnlyProjectionNet(
                reduced_dim,
                reduced_dim,
                50,
                hidden_dim=32,
            )
            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            validation_rows: list[dict[str, object]] = []
            fps.append_signed_costonly_rows(
                validation_rows,
                model,
                signed.val,
                signed.N,
                signed.x_ref,
                50,
                torch.device("cpu"),
                data.bundle.stdlp.n_slack,
            )
            validation = pd.DataFrame(validation_rows)
            validation_mean, validation_se = mean_se(
                validation["objective_ratio"]
            )
            expected_validation = expected_row.get(
                "best_validation_ratio", math.nan
            )
            if pd.notna(expected_validation) and not math.isclose(
                validation_mean,
                float(expected_validation),
                rel_tol=RATIO_RTOL,
                abs_tol=RATIO_ATOL,
            ):
                raise RuntimeError(
                    f"Checkpoint validation mismatch for {case_name}, "
                    f"n={n_samples}: {validation_mean} != "
                    f"{expected_validation}"
                )
            rows: list[dict[str, object]] = []
            fps.append_signed_costonly_rows(
                rows,
                model,
                signed.test,
                signed.N,
                signed.x_ref,
                50,
                torch.device("cpu"),
                data.bundle.stdlp.n_slack,
            )
            raw = pd.DataFrame(rows)
            raw["case"] = case_name
            raw["n_samples"] = n_samples
            raw["checkpoint_sha256"] = observed_hash
            test_frames.append(raw)

            ratio_mean, ratio_se = mean_se(raw["objective_ratio"])
            online_mean, online_se = mean_se(raw["time"])
            archived_mean = float(
                expected_row["archived_objective_ratio_mean"]
            )
            difference = abs(ratio_mean - archived_mean)
            if not math.isclose(
                ratio_mean,
                archived_mean,
                rel_tol=RATIO_RTOL,
                abs_tol=RATIO_ATOL,
            ):
                raise RuntimeError(
                    f"{case_name}, n={n_samples}: checkpoint mean "
                    f"{ratio_mean} != archived mean {archived_mean}"
                )
            summaries.append(
                {
                    "figure": "Figure 4",
                    "repeat": 1,
                    "case": case_name,
                    "method": "FCNNc",
                    "source_method": "FCNN-c",
                    "rho": 0.1,
                    "n_samples": n_samples,
                    "K": 50,
                    "learning_time_s": math.nan,
                    "solve_time_mean_s": math.nan,
                    "solve_time_se_s": math.nan,
                    "projection_time_mean_s": math.nan,
                    "projection_time_se_s": math.nan,
                    "online_total_time_mean_s": online_mean,
                    "online_total_time_se_s": online_se,
                    "objective_ratio_mean": ratio_mean,
                    "objective_ratio_se": ratio_se,
                    "solve_observations": int(len(raw)),
                    "fcnn_initialization_seed": (
                        int(initialization_seed)
                        if pd.notna(initialization_seed)
                        else math.nan
                    ),
                    "fcnn_initialization_protocol": "plain_torch_random_checkpoint",
                    "fcnn_hidden_dim": 32,
                    "fcnn_epochs": 500,
                    "fcnn_batch_size": 8,
                    "fcnn_patience": 120,
                    "fcnn_lr": 1e-3,
                    "validation_objective_ratio_mean": validation_mean,
                    "validation_objective_ratio_se": validation_se,
                    "fcnn_validation_samples": 6,
                    "fcnn_test_samples": 12,
                    "result_provenance": f"{checkpoint_origin}_reevaluation",
                    "checkpoint_origin": checkpoint_origin,
                    "trainer_seed": (
                        int(trainer_seed)
                        if pd.notna(trainer_seed)
                        else math.nan
                    ),
                    "selection_split": str(
                        expected_row.get("selection_split", "validation")
                    ),
                    "selection_rule": str(
                        expected_row.get("selection_rule", "")
                    ),
                    "best_validation_epoch": expected_row.get(
                        "best_validation_epoch", math.nan
                    ),
                    "best_validation_ratio": expected_row.get(
                        "best_validation_ratio", math.nan
                    ),
                    "checkpoint_epoch_selection_used_test": False,
                    "seed_search_performed": False,
                    "checkpoint_sha256": observed_hash,
                    "archived_objective_ratio_mean": archived_mean,
                    "checkpoint_reproduction_abs_error": difference,
                }
            )
            print(
                f"{case_name} n={n_samples}: test={ratio_mean:.12g}, "
                f"validation={validation_mean:.12g}, archived={archived_mean:.12g}",
                flush=True,
            )

    return (
        pd.DataFrame(summaries),
        pd.concat(test_frames, ignore_index=True),
    )


def update_main_results(summary: pd.DataFrame) -> None:
    output_path = FRESH / "figure4_timing_raw.csv"
    current = pd.read_csv(output_path)
    target = (
        current["case"].astype(str).isin(CASES)
        & current["method"].astype(str).eq("FCNNc")
        & current["K"].astype(int).eq(50)
        & current["n_samples"].astype(int).isin(N_VALUES)
    )
    if int(target.sum()) != 8:
        raise RuntimeError(
            f"Expected 8 replaceable Figure 4 rows, found {target.sum()}"
        )
    retained = current.loc[~target].copy()
    merged = pd.concat([retained, summary], ignore_index=True, sort=False)
    merged = merged.sort_values(
        ["case", "n_samples", "method", "K"],
        kind="stable",
    )
    merged.to_csv(output_path, index=False)

    figure3 = pd.read_csv(FRESH / "figure3_timing_raw.csv")
    pd.concat([figure3, merged], ignore_index=True, sort=False).to_csv(
        FRESH / "timing_repeated.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--update-main-results",
        action="store_true",
        help="Replace only the eight GROW7/SC205 K=50 rows in Figure 4.",
    )
    args = parser.parse_args()

    summary, test_rows = evaluate_checkpoints()
    FRESH.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        FRESH / "figure4_netlib_k50_checkpoint_summary.csv",
        index=False,
    )
    test_rows.to_csv(
        FRESH / "figure4_netlib_k50_checkpoint_test_rows.csv",
        index=False,
    )
    if args.update_main_results:
        update_main_results(summary)
        print("Updated the eight Figure 4 K=50 Netlib rows.", flush=True)


if __name__ == "__main__":
    main()
