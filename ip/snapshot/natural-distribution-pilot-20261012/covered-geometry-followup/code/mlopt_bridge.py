"""CPU bridge to the existing MLOPT implementation for new n-label cases.

Only Python source directories are added to sys.path.  In particular, the
Python 3.12 supplement dependency directory must not be added to the Python
3.14 experiment runtime.  This module neither saves checkpoints nor loads old
training data or fitted models.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
import sys

import numpy as np
from threadpoolctl import threadpool_limits
import mlopt_impl as mlopt_baseline
from strategy_match import propose_strategy_fixings


def _single_thread_torch() -> dict[str, Any]:
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            if torch.get_num_interop_threads() != 1:
                raise RuntimeError(
                    "Set Torch interop threads to one before training starts in this worker"
                ) from exc
    return dict(torch_version=torch.__version__, intraop_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(), device="cpu")


def _checked_case(case: Any) -> int:
    features = np.asarray(case.train_features, dtype=float)
    costs = np.asarray(case.train_costs, dtype=float)
    solutions = np.asarray(case.train_solutions, dtype=float)
    n = len(features)
    if n not in (48, 64, 96):
        raise ValueError("This natural-cost protocol trains MLOPT on n=48, n=64 or n=96")
    if (features.ndim != 2 or not features.shape[1] or not np.isfinite(features).all()
            or costs.shape != (n, int(case.model.A.shape[1]))
            or solutions.shape != costs.shape
            or not np.isfinite(costs).all() or not np.isfinite(solutions).all()):
        raise ValueError("Expected complete finite n-label features, original costs and solutions")
    return n


def fit_mlopt(case: Any, *, seed: int = 0) -> tuple[Any, dict[str, Any]]:
    """Run the original eight-config development search and all-n refit on CPU.

    The original routine creates an internal 80/20 split of these supplied
    labels and no external validation or test data.  The final strategy bank
    and standardization are built from all supplied n training examples.
    """
    n = _checked_case(case)
    runtime = _single_thread_torch()
    config = mlopt_baseline.Config(
        device="cpu", seed=int(seed), hidden_candidates=((32, 32, 32), (64, 64, 64)),
        learning_rates=(1e-3, 1e-2), epoch_candidates=(50, 150),
        batch_size=32, dropout=0.1, validation_fraction=0.2, n_best=10,
    )
    started = perf_counter()
    with threadpool_limits(limits=1):
        model = mlopt_baseline.fit(case, config=config)
    metadata = dict(model.metadata)
    metadata.update(
        method="MLOPT-native-small-sample-natural-cost adaptation",
        training_cost_budget=n, source_of_labels="only supplied new certified n-label optimizers",
        old_120_training_labels_used=False, old_fitted_predictor_used=False,
        external_test_information_used=False,
        neural_parameter_grid_count=8, final_refit_training_count=n,
        training_runtime=runtime, wrapper_fit_seconds=perf_counter()-started,
        implementation_source=str(Path(mlopt_baseline.__file__).resolve()),
        matched_prediction="separate declared strategy-guided coordinate-fixing variant",
    )
    if len(metadata["validation_trials"]) != 8:
        raise RuntimeError("Original MLOPT did not run the required eight development configurations")
    if int(metadata["final_bank"]["n_samples"]) != n:
        raise RuntimeError("Final MLOPT bank did not use exactly the supplied n labels")
    if metadata["trainable_parameter_count"] != sum(int(w.size+b.size) for w,b in model.weights):
        raise RuntimeError("Actual Torch parameter count differs from its exported tensors")
    model.metadata = metadata
    return model, metadata


def _check_prediction_case(model: Any, case: Any) -> None:
    if (str(model.bank.case.case_id) != str(case.case_id)
            or int(model.bank.n_variables) != int(case.model.A.shape[1])):
        raise ValueError("MLOPT model belongs to a different new case")


def predict_native(model: Any, case: Any, cost: Any, features: Any
                   ) -> tuple[dict[str, Any], np.ndarray | None]:
    """Return native cached-strategy decoding as the runner's (raw, x) pair.

    Native decoding has no original or restricted solver-optimality
    certificate.  A feasible decoded point is eligible for reference-based
    quality scoring, whose optimum is never supplied to this helper.
    """
    _check_prediction_case(model, case)
    result = dict(model.predict(np.asarray(features, dtype=float), np.asarray(cost, dtype=float)))
    value = result.pop("x", None)
    x = None if value is None else np.asarray(value, dtype=float)
    result["valid"] = bool(result.get("valid", False) and x is not None)
    result["certified"] = False
    result["original_global_solver_certified"] = False
    result["certificate_scope"] = "no solver optimum certificate; decoded candidate feasibility only"
    result["objective"] = None if x is None else float(np.asarray(cost, dtype=float)@x)
    result.setdefault("status", "no_feasible_predicted_strategy" if x is None else "feasible_strategy_decoded")
    result.setdefault("diagnostics", {})
    result["diagnostics"].update(
        training_samples=int(model.metadata["training_cost_budget"]),
        retained_strategy_count=len(model.selected),
        native_candidate_limit=int(model.config.n_best),
        dimension_matched=False, test_optimum_used=False,
    )
    return result, x


def propose_matched(model: Any, case: Any, features: Any
                    ) -> tuple[dict[int, int], dict[str, Any]]:
    """Use the unchanged temperature-one strategy-agreement matching rule."""
    _check_prediction_case(model, case)
    fixed, diagnostic = propose_strategy_fixings(model, case, np.asarray(features, dtype=float), "mlopt")
    diagnostic["training_samples"] = int(model.metadata["training_cost_budget"])
    diagnostic["target_common_dimension"] = int(case.affine["rank"])
    diagnostic["actual_common_dimension"] = len(case.commitment_indices)-len(fixed)
    return fixed, diagnostic
