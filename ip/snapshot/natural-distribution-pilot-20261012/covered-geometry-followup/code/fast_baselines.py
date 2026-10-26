"""Training-only fast controls for the separate small-sample experiment.

These controls consume only the arrays supplied in ``case``.  They never load
the original 120-label cohort, read test references, or change frozen sources.
The historical controls are declared new controls, not author implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from dimension_diagnostics import (
    assert_matched_fixings,
    common_coordinate_indices,
    target_common_dimension,
)
from svm_match import CoordinateSVMModel


def _checked_training(case: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(case.train_features, dtype=float)
    solutions = np.asarray(case.train_solutions, dtype=float)
    common = common_coordinate_indices(case)
    n = len(features)
    if (features.ndim != 2 or not n or not features.shape[1]
            or not np.isfinite(features).all()):
        raise ValueError("Expected nonempty finite training feature vectors")
    if (solutions.shape != (n, int(case.model.A.shape[1]))
            or not np.isfinite(solutions).all()):
        raise ValueError("Training solutions differ from the supplied original model")
    values = solutions[:, common]
    binary = np.rint(values).astype(np.int8)
    if (np.max(np.abs(values-binary), initial=0.) > 1e-6
            or not np.isin(binary, [0, 1]).all()):
        raise ValueError("Common training coordinates must be binary")
    return features.copy(), solutions.copy(), binary


def fit_coordinate_svm_small(case: Any) -> tuple[CoordinateSVMModel, dict[str, Any]]:
    """Use the fixed linear-SVC rule on exactly the supplied n training labels.

    ``probability=True`` runs libsvm's training-only sigmoid calibration.
    Calibration can be weak with tiny class counts.  It is not a validation
    gate or a claim of well-calibrated small-sample probabilities.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    started = perf_counter()
    features, solutions, binary = _checked_training(case)
    common = common_coordinate_indices(case)
    n, seed = len(features), int(case.seed)
    predictors: list[Any | None] = []
    probabilities = np.full(len(common), np.nan, dtype=float)
    constant_count = 0
    rare_class_count = 0
    class_counts = []
    for j in range(len(common)):
        labels = binary[:, j]
        positives = int(labels.sum())
        class_counts.append([n-positives, positives])
        if np.all(labels == labels[0]):
            probabilities[j] = float(labels[0])
            predictors.append(None)
            constant_count += 1
            continue
        rare_class_count += min(positives, n-positives) < 5
        predictor = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1.0, probability=True, random_state=seed),
        )
        predictor.fit(features, labels)
        if not np.array_equal(np.asarray(predictor.classes_), np.array([0, 1])):
            raise RuntimeError("SVC did not preserve the expected binary class mapping")
        predictors.append(predictor)
    model = CoordinateSVMModel(
        predictors=predictors, constant_probabilities=probabilities,
        common_indices=common.copy(), n_features=features.shape[1],
        random_state=seed, training_samples=n,
    )
    metadata = dict(
        method="Small-sample coordinate-SVM matched-K control",
        is_native_algorithm=False, fit_seconds=perf_counter()-started,
        training_samples=n, training_labels="supplied certified training optimizers",
        common_ambient_dimension=len(common), n_features=features.shape[1],
        constant_coordinate_count=constant_count,
        fitted_classifier_count=len(common)-constant_count,
        nonconstant_coordinates_with_class_count_below_five=int(rare_class_count),
        coordinate_class_counts=class_counts,
        estimator="StandardScaler followed by linear SVC(C=1, probability=True)",
        probability_calibration="libsvm training-only sigmoid calibration",
        scaling="StandardScaler fitted on the supplied training features",
        calibration_caveat="Tiny classes can yield weak confidence estimates",
        random_state=seed, original_Table1_gate_used=False,
        parameter_selection="fixed protocol; no test information used",
        prediction="class-1 probability MAP; p=0.5 selects 1",
        confidence="max(p,1-p)", tie_break="original common-column index",
        target_K="affine rank of the same supplied training optimizers",
    )
    return model, metadata


@dataclass
class HistoricalBank:
    """All supplied full points plus stable deduplication of integer strategies."""

    case_id: str
    family: str
    full_solutions: np.ndarray
    integer_indices: np.ndarray
    unique_integer_assignments: np.ndarray
    representative_solutions: np.ndarray
    first_training_indices: np.ndarray
    common_indices: np.ndarray
    stability: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def prepare_historical(case: Any) -> HistoricalBank:
    """Build a bank and coordinate stability using only supplied training data."""
    started = perf_counter()
    features, solutions, binary = _checked_training(case)
    if case.family not in ("uc", "knapsack"):
        raise ValueError("Historical controls support UC and knapsack only")
    model = case.model
    integer = np.flatnonzero(np.asarray(model.integrality) != 0)
    assignment_values = solutions[:, integer]
    assignments = np.rint(assignment_values).astype(np.int64)
    if np.max(np.abs(assignment_values-assignments), initial=0.) > 1e-6:
        raise ValueError("Historical complete integer assignments must be integral")
    activities = np.asarray(model.A @ solutions.T).T
    violation = max(
        float(np.max(model.row_lower[None, :]-activities, initial=0.)),
        float(np.max(activities-model.row_upper[None, :], initial=0.)),
        float(np.max(model.col_lower[None, :]-solutions, initial=0.)),
        float(np.max(solutions-model.col_upper[None, :], initial=0.)),
    )
    if violation > 1e-6:
        raise ValueError("Historical training solutions violate the original constraints")
    seen: set[tuple[int, ...]] = set()
    first = []
    for i, row in enumerate(assignments):
        key = tuple(map(int, row))
        if key not in seen:
            seen.add(key)
            first.append(i)
    first_array = np.asarray(first, dtype=np.int64)
    stability = np.abs(binary.mean(axis=0)-0.5)
    return HistoricalBank(
        case_id=str(case.case_id), family=str(case.family),
        full_solutions=solutions, integer_indices=integer.copy(),
        unique_integer_assignments=assignments[first_array].copy(),
        representative_solutions=solutions[first_array].copy(),
        first_training_indices=first_array,
        common_indices=common_coordinate_indices(case), stability=stability,
        metadata=dict(
            training_samples=len(features), unique_integer_strategies=len(first),
            integer_coordinate_count=len(integer),
            deduplication="complete integer assignment; retain first occurrence",
            stability="abs(training common-coordinate mean - 0.5)",
            preparation_seconds=perf_counter()-started,
            test_information_used=False,
        ),
    )


def _checked_bank_cost(bank: HistoricalBank, case: Any, cost: Any) -> np.ndarray:
    common = common_coordinate_indices(case)
    integer = np.flatnonzero(np.asarray(case.model.integrality) != 0)
    if (bank.case_id != str(case.case_id) or bank.family != str(case.family)
            or not np.array_equal(common, bank.common_indices)
            or not np.array_equal(integer, bank.integer_indices)):
        raise ValueError("Historical bank differs from the supplied case")
    value = np.asarray(cost, dtype=float)
    if value.shape != (int(case.model.A.shape[1]),) or not np.isfinite(value).all():
        raise ValueError("Expected one finite original objective vector")
    if not len(bank.full_solutions) or not len(bank.unique_integer_assignments):
        raise ValueError("Historical bank must not be empty")
    return value


def propose_historical_matched(bank: HistoricalBank, case: Any,
                               cost: Any) -> tuple[dict[int, int], dict[str, Any]]:
    """Choose a stored feasible anchor and fix D-K stable common coordinates."""
    started = perf_counter()
    cost = _checked_bank_cost(bank, case, cost)
    common = bank.common_indices
    K = target_common_dimension(case)
    stored_objectives = bank.full_solutions @ cost
    # np.argmin retains the first training index in an exact tie.
    anchor_index = int(np.argmin(stored_objectives))
    anchor = np.rint(bank.full_solutions[anchor_index, common]).astype(np.int64)
    order = np.lexsort((common, -bank.stability))
    chosen = order[:len(common)-K]
    fixed = {int(common[j]): int(anchor[j]) for j in chosen}
    dimensions = assert_matched_fixings(case, fixed, target_K=K)
    diagnostic = dict(
        method="Historical-strategy matched-K control", is_native_algorithm=False,
        training_samples=len(bank.full_solutions),
        anchor_training_index=anchor_index,
        anchor_stored_full_objective=float(stored_objectives[anchor_index]),
        anchor_rule="lowest current objective among stored full feasible training solutions",
        anchor_objective_caveat="UC stored dispatch used only to choose the anchor; the sub-IP reoptimizes dispatch",
        confidence="abs(training common-coordinate mean - 0.5)",
        common_stability=bank.stability.tolist(),
        selected_common_indices=common[chosen].tolist(),
        tie_break="first training anchor index; original common-column index",
        target_common_dimension=K, actual_common_dimension=len(common)-len(fixed),
        target_rank=len(common)-K, actual_rank=len(fixed), exact_match=True,
        dimension_diagnostics=dimensions, prediction_seconds=perf_counter()-started,
        test_optimum_used=False,
    )
    return fixed, diagnostic


def solve_best_observed(bank: HistoricalBank, case: Any, cost: Any,
                        adapter: Any, seed: int, time_limit: float
                        ) -> tuple[dict[str, Any], np.ndarray | None]:
    """Evaluate the supplied finite strategy bank under one shared wall budget.

    UC fixes every original integer variable and reoptimizes continuous
    dispatch in an LP for each unique strategy.  Knapsack directly evaluates
    the stored points.  ``certified`` refers only to completing this finite
    bank evaluation; it never certifies the original global optimum.
    """
    started = perf_counter()
    cost = _checked_bank_cost(bank, case, cost)
    time_limit = float(time_limit)
    if not np.isfinite(time_limit) or time_limit <= 0.:
        raise ValueError("Expected a positive finite shared time budget")
    total = len(bank.unique_integer_assignments)
    visited = 0
    all_certified = True
    best_x = None
    best_objective = None
    best_id = None
    best_candidate_certified = False
    best_solver_result = None
    candidates = []
    for i, assignment in enumerate(bank.unique_integer_assignments):
        remaining = time_limit-(perf_counter()-started)
        if remaining <= 0.:
            break
        if bank.family == "uc":
            fixed = {int(j): int(v) for j, v in zip(bank.integer_indices, assignment)}
            result = adapter.solve(
                case.model, cost, fixed=fixed, relax_integrality=True,
                seed=int(seed)+i, time_limit=remaining,
            )
            x = result.x
            raw = result.to_dict(include_solution=False)
            candidate_valid = bool(result.valid and x is not None)
            candidate_certified = bool(result.certified and candidate_valid)
        else:
            x = bank.representative_solutions[i].copy()
            original = adapter.validate_solution(case.model, x)
            candidate_valid = bool(original["valid"])
            candidate_certified = candidate_valid
            raw = dict(
                status="stored integer point evaluated", valid=candidate_valid,
                certified=candidate_certified, objective=float(cost@x),
                original_check=original, n_fixed=len(bank.integer_indices),
                solve_seconds=0., build_seconds=0.,
            )
        visited += 1
        all_certified = all_certified and candidate_certified
        # Cross-check the winning candidate's complete historical assignment.
        restriction_residual = (None if x is None else float(np.max(
            np.abs(np.asarray(x)[bank.integer_indices]-assignment), initial=0.)))
        candidate_valid = bool(candidate_valid and restriction_residual is not None
                               and restriction_residual <= 1e-6)
        candidate_certified = bool(candidate_certified and candidate_valid)
        all_certified = all_certified and candidate_certified
        objective = None if x is None else float(cost@x)
        candidates.append(dict(
            strategy_id=i, first_training_index=int(bank.first_training_indices[i]),
            valid=candidate_valid, certified=candidate_certified,
            objective=objective, restriction_residual=restriction_residual,
            solver_result=raw,
        ))
        if candidate_valid and (best_objective is None or objective < best_objective):
            best_x = np.asarray(x, dtype=float).copy()
            best_objective = objective
            best_id = i
            best_candidate_certified = candidate_certified
            best_solver_result = raw
    elapsed = perf_counter()-started
    complete = visited == total
    finite_bank_certified = bool(complete and all_certified and best_x is not None)
    raw = dict(
        method="Best observed complete integer strategy control",
        status=("finite strategy bank evaluated" if complete
                else "shared budget exhausted before completing strategy bank"),
        objective=best_objective, valid=best_x is not None,
        certified=finite_bank_certified, optimal=finite_bank_certified,
        certificate_scope="finite historical strategy bank only",
        original_global_solver_certified=False,
        winner_conditional_certified=best_candidate_certified,
        winning_candidate_certified=best_candidate_certified,
        strategy_id=best_id,
        first_training_index=(None if best_id is None
                              else int(bank.first_training_indices[best_id])),
        training_samples=len(bank.full_solutions), full_bank_count=total,
        visited_count=visited, candidate_count=visited,
        bank_complete=complete, all_visited_candidates_certified=all_certified,
        method_budget_seconds=time_limit, method_wall_budget_exceeded=elapsed > time_limit,
        online_seconds=elapsed, seconds=elapsed,
        candidates=candidates, winning_solver_result=best_solver_result,
        dimension_matched=False, test_optimum_used=False, full_fallback_used=False,
    )
    return raw, best_x
