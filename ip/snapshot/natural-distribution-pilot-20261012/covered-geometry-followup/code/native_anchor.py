"""Native-output-anchor matching for the fresh natural-distribution pilot.

Mechanically reuse the current native final candidate. No training, test
reference, fallback, or warm start is performed by this module.
"""
from __future__ import annotations
from time import perf_counter, process_time
from typing import Any
import numpy as np
import portable_adapter as adapter
from dimension_diagnostics import common_coordinate_indices, target_common_dimension, assert_matched_fixings
from strategy_match import _standardized_features, _native_scores
from mlopt_bridge import predict_native
METHOD = "mlopt_native_anchor_matched"

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def propose_native_anchor(model: Any, case: Any, cost: Any, features: Any,
                          *, native_predict_fn=None
                          ) -> tuple[dict[int, int] | None, dict[str, Any], np.ndarray | None]:
    """Use native top-ten objective selection, then all-class coordinate agreement.

    A missing native winner produces no fixings and no fallback.  The optional
    predictor argument supports a no-solver unit test of that failure branch.
    """
    native_predict_fn = predict_native if native_predict_fn is None else native_predict_fn
    native_raw, native_x = native_predict_fn(model, case, cost, features)
    diagnostic = dict(
        method=METHOD, original_algorithm=False,
        anchor_source="native MLOPT final feasible candidate: minimum objective among decoded top-ten",
        native_prediction_raw=native_raw, native_solution=native_x,
        native_objective=None if native_x is None else float(np.asarray(cost)@native_x),
        anchor_available=False,
        test_optimum_used=False, warm_start_used=False, full_fallback_used=False,
    )
    if not native_raw.get("valid", False) or native_x is None:
        diagnostic.update(status="no_valid_native_winner", native_anchor_strategy_id=None,
                          feasible_witness_valid=False)
        return None, diagnostic, None
    common = common_coordinate_indices(case)
    K = target_common_dimension(case)
    original_integer = np.flatnonzero(np.asarray(case.model.integrality))
    bank_integer = np.asarray(model.bank.integer_indices, dtype=np.int64)
    if not np.array_equal(bank_integer, original_integer):
        raise ValueError("Native bank integer columns differ from the original case")
    strategy_id = int(native_raw["strategy_id"])
    selected = np.asarray(model.selected, dtype=np.int64)
    if strategy_id not in set(map(int, selected)):
        raise ValueError("Native winner is not in the retained strategy bank")
    strategy = model.bank.strategies[strategy_id]
    native_x = np.asarray(native_x, dtype=float)
    original = adapter.validate_solution(case.model, native_x)
    intres = float(np.max(np.abs(native_x[bank_integer]-strategy.integer_values), initial=0.))
    equalityres = float(np.max(np.abs(model.bank.E@native_x-model.bank.beq), initial=0.))
    activeres = float(np.max(np.abs(model.bank.G[strategy.active_rows]@native_x
                                     -model.bank.h[strategy.active_rows]), initial=0.))
    native_strategy_valid = bool(original["valid"] and max(intres, equalityres, activeres) <= 1e-6)
    if not native_strategy_valid:
        raise RuntimeError("Native winner fails its original or complete-strategy witness check")
    standardized = _standardized_features(model, np.asarray(features, dtype=float))
    scores = _native_scores(model, standardized, "mlopt")
    if scores.shape != selected.shape:
        raise ValueError("Native output class mapping differs from retained strategies")
    integer_location = {int(j): i for i, j in enumerate(bank_integer)}
    locations = [integer_location[int(j)] for j in common]
    historical = np.asarray([model.bank.strategies[int(j)].integer_values[locations]
                             for j in selected], dtype=np.int64)
    anchor = np.asarray(strategy.integer_values[locations], dtype=np.int64)
    weights = np.exp(scores-np.max(scores))
    weights /= weights.sum()
    confidence = np.sum(weights[:, None]*(historical == anchor), axis=0)
    order = np.lexsort((common, -confidence))
    chosen = order[:len(common)-K]
    fixed = {int(common[j]): int(anchor[j]) for j in chosen}
    dimensions = assert_matched_fixings(case, fixed, target_K=K)
    witness_valid = all(abs(float(native_x[j])-v) <= 1e-6 for j, v in fixed.items())
    if not witness_valid:
        raise RuntimeError("Native feasible witness violates the proposed partial fixing")
    diagnostic.update(
        status="native_output_anchor_selected", native_anchor_strategy_id=strategy_id,
        anchor_available=True,
        native_anchor_retained_position=int(np.flatnonzero(selected == strategy_id)[0]),
        native_anchor_representative_training_index=int(strategy.representative_training_index),
        native_anchor_common_values=anchor.tolist(), common_original_columns=common.tolist(),
        retained_strategy_ids=selected.tolist(), all_retained_strategy_scores=scores.tolist(),
        strategy_confidence_weights=weights.tolist(), softmax_temperature=1.0,
        confidence_scope="temperature-one agreement surrogate, not calibrated probabilities",
        common_weighted_agreement=confidence.tolist(),
        confidence_sorted_common_columns=common[order].tolist(),
        fixed_original_columns=common[chosen].tolist(),
        native_strategy_check=dict(valid=native_strategy_valid, integer_residual=intres,
                                   permanent_equality_residual=equalityres, active_equality_residual=activeres),
        native_original_check=original, feasible_witness_valid=witness_valid,
        target_common_dimension=K, actual_common_dimension=len(common)-len(fixed),
        dimension_diagnostics=dimensions,
        native_candidate_count=native_raw["diagnostics"]["n_candidates"],
        coordinate_selection="agreement with native final output descending, original column ascending",
        native_score_top_one_strategy_id=int(selected[int(np.argmax(scores))]),
        training_samples=int(model.metadata["training_cost_budget"]),
    )
    return fixed, diagnostic, native_x

def execute_native_anchor(model, case, cost, features, *, solver_seed, budget=30.):
    started = perf_counter()
    cpu_started = process_time()
    fixed, diagnostic, native_x = propose_native_anchor(model, case, cost, features)
    prediction_seconds = perf_counter()-started
    remaining = budget-prediction_seconds
    attempted = bool(fixed is not None and remaining > 0)
    if fixed is None or remaining <= 0:
        raw, x = dict(valid=False, certified=False,
                      status="no_valid_native_winner" if fixed is None else "prediction_budget_exhausted"), None
    else:
        # Deliberately use the same cold solve as the original matched cohort.
        result = adapter.solve(case.model, cost, fixed=fixed, seed=solver_seed, time_limit=remaining)
        raw, x = result.to_dict(include_solution=False), result.x
    original = adapter.validate_solution(case.model, x) if x is not None else dict(valid=False)
    residual = None if x is None or fixed is None else max((abs(float(x[j])-v) for j, v in fixed.items()), default=0.)
    restriction_valid = bool(x is not None and fixed is not None and residual <= 1e-6)
    objective = None if x is None else float(np.asarray(cost)@x)
    elapsed = perf_counter()-started
    cpu_seconds = process_time()-cpu_started
    valid = bool(raw.get("valid", False) and original["valid"] and restriction_valid
                 and objective is not None and np.isfinite(objective) and elapsed <= budget)
    # These diagnostics do not affect the returned solution and stay outside
    # the timer, consistent with the main experiment's independent audits.
    physics = None
    if case.family == "uc" and x is not None:
        from uc_model import validate_uc_solution
        physics = validate_uc_solution(case.original_base, x)
        if bool(physics["valid"]) != bool(original["valid"]):
            raise RuntimeError("UC physics disagrees with the original constraints")
    locally_certified = bool(raw.get("certified", False) and restriction_valid)
    native_objective = diagnostic["native_objective"]
    dominance_eligible = bool(valid and locally_certified and native_x is not None)
    dominance_passed = (None if not dominance_eligible else
                        objective <= native_objective+adapter.recovery_tolerance(native_objective))
    return clean(dict(
        method=METHOD, seconds=elapsed, process_cpu_seconds=cpu_seconds,
        prediction_seconds=prediction_seconds, construction_seconds=float(raw.get("build_seconds", 0.)),
        solve_seconds=float(raw.get("solve_seconds", 0.)),
        validation_seconds=max(0., elapsed-prediction_seconds-float(raw.get("build_seconds",0.))-float(raw.get("solve_seconds",0.))), valid=valid, original_feasible=original["valid"],
        original_check=original, restriction_valid=restriction_valid, restriction_residual=residual,
        objective=objective, solution=x, fixed=fixed,
        actual_K=None if fixed is None else len(case.commitment_indices)-len(fixed),
        solved_model_certified=locally_certified, status=raw.get("status"),
        certificate_scope="restricted IP", original_global_solver_certified=False,
        method_wall_budget_exceeded=elapsed>budget, raw_method_result=raw,
        diagnostic=diagnostic, physics=physics,
        native_objective_dominance_eligible=dominance_eligible,
        native_objective_dominance_passed=dominance_passed,
        subip_attempted=attempted,
        full_fallback_used=False, warm_start_used=False,
    ))
