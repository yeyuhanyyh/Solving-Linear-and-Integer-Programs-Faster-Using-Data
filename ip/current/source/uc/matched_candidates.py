"""Candidate-level coordinate matching for frozen MLOPT and Li predictors.

This is an explicit inference adaptation. It does not call a native decoder,
change any learned parameter, or consult a reference optimum. Native methods
remain separate. Each originally predicted top-k strategy supplies one partial
integer assignment before a restricted IP is solved. All original constraints
and integer domains must remain in the caller's solve callback.

MLOPT uses its native best-valid-objective selection. Li defaults to its native
Eq. (21) minimum normalized original-constraint violation, with predicted-order
ties. ``best_valid_objective`` is a separately declared Li enhancement, never
an alias for Eq. (21). The complete candidate trace supports both prespecified
rules without changing the candidate budget or doing additional solves.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np


VERSION = "candidate-first-common-k-v1"
RULES = ("eq21", "best_valid_objective")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer_vector(values, name):
    raw = np.asarray(values)
    _require(raw.ndim == 1 and np.all(np.isfinite(raw)), name + " must be a finite vector")
    rounded = raw.astype(np.int64)
    _require(np.array_equal(raw, rounded), name + " must contain integers")
    return rounded


def _scores(model, features, kind):
    if kind == "mlopt":
        # Match MLOPTModel.predict without importing neural vendor modules or
        # performing any native decoding. This also works in MLOPT-only runs.
        mean, scale = np.asarray(model.mean), np.asarray(model.scale)
        features = np.asarray(features, dtype=float)
        _require(features.shape == mean.shape == scale.shape and features.ndim == 1 and
                 np.isfinite(features).all() and np.isfinite(mean).all() and
                 np.isfinite(scale).all() and np.all(scale > 0), "Invalid fitted feature scaling")
        values = ((features - mean) / scale).astype(np.float32)
        _require(bool(model.weights), "MLOPT has no fitted predictor layers")
        for i, (weight, bias) in enumerate(model.weights):
            values = weight @ values + bias
            if i != len(model.weights) - 1:
                values = np.maximum(values, 0)
        return values
    # Li's frozen CPU helper is loaded only after its verified runtime loader.
    # Deferred import lets this module and pure tests run without Torch.
    from strategy_match import _native_scores, _standardized_features
    return _native_scores(model, _standardized_features(model, features), kind)


def candidate_plan(model, case, features, kind, target_K, *, score_function=None):
    """Read a fitted predictor and construct every original top-k restriction.

    ``score_function`` is an injection point for pure mock tests. Production
    callers leave it unset and use the frozen native inference helper.
    """
    _require(kind in ("mlopt", "li"), "Unknown strategy learner")
    n = int(case.model.A.shape[1])
    common = _integer_vector(case.commitment_indices, "Common columns")
    _require(len(common) == len(set(common)) and np.all((0 <= common) & (common < n)),
             "Common columns must be distinct model columns")
    K = int(target_K)
    _require(K == target_K and 0 <= K <= len(common), "Invalid target K")
    original_integer = np.flatnonzero(np.asarray(case.model.integrality) != 0)
    bank_integer = _integer_vector(model.bank.integer_indices, "Strategy integer columns")
    _require(np.array_equal(np.sort(bank_integer), original_integer),
             "Strategy bank and model have different integer columns")
    _require(set(common).issubset(set(bank_integer)), "Common columns must be integer")
    selected = _integer_vector(model.selected, "Retained strategy IDs")
    _require(len(selected) > 0 and len(set(selected)) == len(selected) and
             np.all((selected >= 0) & (selected < len(model.bank.strategies))),
             "Retained strategy IDs must be distinct valid bank positions")
    score_function = _scores if score_function is None else score_function
    scores = np.asarray(score_function(model, features, kind), dtype=float)
    _require(scores.shape == selected.shape and np.isfinite(scores).all(),
             "Scores must be finite and aligned to retained strategies")
    native_limit = model.config.n_best if kind == "mlopt" else model.top_k
    limit = int(native_limit)
    _require(limit == native_limit and limit > 0, "Native candidate limit must be a positive integer")
    ranked = np.argsort(-scores, kind="stable")[:min(limit, len(selected))]
    location = {int(j): i for i, j in enumerate(bank_integer)}
    common_positions = np.asarray([location[int(j)] for j in common], dtype=int)
    patterns = []
    for sid in selected:
        values = np.asarray(model.bank.strategies[int(sid)].integer_values, dtype=float)
        _require(values.shape == bank_integer.shape and np.isfinite(values).all(),
                 "Malformed historical integer strategy")
        values = values[common_positions]
        rounded = np.rint(values)
        _require(np.all(np.abs(values - rounded) <= 1e-9) and
                 np.all((rounded == 0) | (rounded == 1)), "Common strategy values must be binary")
        _require(np.all(rounded >= np.asarray(case.model.col_lower)[common]) and
                 np.all(rounded <= np.asarray(case.model.col_upper)[common]),
                 "Historical common pattern violates model bounds")
        patterns.append(rounded.astype(np.int64))
    patterns = np.asarray(patterns)
    # Preserve each existing wrapper's confidence population, not just top-1.
    confidence_positions = ranked if kind == "mlopt" else np.arange(len(selected))
    logits = scores[confidence_positions]
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    candidates = []
    for rank, position in enumerate(ranked):
        pattern = patterns[position]
        agreement = np.sum(weights[:, None] * (patterns[confidence_positions] == pattern), axis=0)
        coordinate_order = np.lexsort((common, -agreement))
        chosen = coordinate_order[:len(common) - K]
        fixed = {int(common[j]): int(pattern[j]) for j in chosen}
        _require(len(fixed) == len(common) - K, "Incorrect number of fixed common coordinates")
        candidates.append(dict(rank=rank, strategy_id=int(selected[position]),
            retained_position=int(position), predicted_score=float(scores[position]),
            fixed=fixed, weighted_agreement=agreement.tolist(),
            confidence_sorted_common_columns=common[coordinate_order].tolist()))
    return dict(kind=kind, target_K=K, common_dimension=len(common),
        common_columns=common.tolist(), native_candidate_limit=limit,
        retained_strategy_ids=selected.tolist(), all_retained_scores=scores.tolist(),
        confidence_retained_positions=confidence_positions.tolist(),
        confidence_weights=weights.tolist(), confidence_temperature=1.0,
        confidence_population="native_top_k" if kind == "mlopt" else "all_retained_strategies",
        candidate_count=len(candidates), candidates=candidates)


def normalized_infeasibility(bank, x, floor):
    """Frozen Li Eq. (21) normalization, including all canonical constraints."""
    if x is None or not np.isfinite(x).all():
        return float("inf")
    rhs = np.concatenate((np.asarray(bank.h, dtype=float).ravel(),
                          np.asarray(bank.beq, dtype=float).ravel()))
    rhs = rhs[np.isfinite(rhs)]
    denominator = max(float(np.abs(rhs).max()) if rhs.size else 0.0, float(floor))
    _require(np.isfinite(denominator) and denominator > 0, "Invalid Eq. (21) denominator")
    violation = 0.0
    if bank.G.shape[0]:
        violation = max(violation, float(np.maximum(np.asarray(bank.G @ x).ravel() - bank.h, 0.0).max()))
    if bank.E.shape[0]:
        violation = max(violation, float(np.abs(np.asarray(bank.E @ x).ravel() - bank.beq).max()))
    return violation / denominator


def select_candidate(candidates, rule):
    """Return a trace index; stable iteration preserves predicted-order ties."""
    _require(rule in RULES, "Unknown candidate selection rule")
    available = [(i, row) for i, row in enumerate(candidates) if row.get("attempted")]
    if rule == "best_valid_objective":
        available = [(i, row) for i, row in available if row.get("valid")]
        return min(available, key=lambda p: p[1]["objective"])[0] if available else None
    return min(available, key=lambda p: p[1]["normalized_infeasibility"])[0] if available else None


def _result_dict(value):
    if isinstance(value, Mapping):
        return dict(value)
    return dict(vars(value))


def reselect_matched_result(result, selection_rule):
    """Score a prespecified second rule on the identical saved candidate run.

    This performs no prediction or solve. In particular, the Li supplementary
    best-objective row can reuse the primary Eq. (21) candidate completion run.
    The caller must retain the complete ``candidate_solutions`` collection.
    """
    previous = result["diagnostic"]
    _require(previous["adapter_version"] == VERSION, "Unknown candidate trace version")
    _require(selection_rule in RULES, "Unknown candidate selection rule")
    kind = previous["candidate_plan"]["kind"]
    _require(kind == "li" or selection_rule == "best_valid_objective", "MLOPT preserves its native selection")
    rows = previous["candidates"]
    index = select_candidate(rows, selection_rule)
    winner = None if index is None else rows[index]
    valid = bool(winner is not None and winner["valid"] and not previous["time_limit_exhausted"])
    points = {s["rank"]: s["x"] for s in result["candidate_solutions"]}
    _require(not valid or winner["rank"] in points, "Selected candidate point is missing")
    diagnostic = dict(previous, selection_rule=selection_rule, selected_candidate_trace_index=index,
        li_best_objective_is_separate_enhancement=(kind == "li" and selection_rule == "best_valid_objective"),
        identical_candidate_execution_reused=True)
    out = dict(result, diagnostic=diagnostic,
        x=points[winner["rank"]] if valid else None, valid=valid,
        objective=winner["objective"] if valid else None,
        fixed=dict(winner["fixed"]) if valid else None,
        strategy_id=winner["strategy_id"] if winner is not None else None,
        actual_K=result["target_K"] if valid else None,
        status=("total_wall_budget_exhausted" if previous["time_limit_exhausted"] else
                "matched_candidate_selected" if valid else "no_valid_selected_candidate"))
    return out


def solve_matched_candidates(learner, case, cost, features, method, *, target_K,
                             time_limit, seed, solve_restriction, validate_original,
                             selection_rule=None, score_function=None, clock=None):
    """Run original top-k candidates under one shared wall-clock budget.

    ``solve_restriction(cost, fixed=..., time_limit=..., seed=...)`` must solve
    the unchanged common IP with only those additional coordinate fixings.
    ``validate_original(x, cost)`` must verify the complete original model and
    return ``valid`` and ``objective_recomputed``. Neither callback receives a
    reference optimum. Candidate solutions are returned separately for audit.
    """
    _require(method in ("mlopt", "preference_learning"), "Unsupported matched method")
    kind = "mlopt" if method == "mlopt" else "li"
    rule = selection_rule or ("best_valid_objective" if kind == "mlopt" else "eq21")
    _require(rule in RULES and (kind == "li" or rule == "best_valid_objective"),
             "MLOPT preserves native best-valid-objective selection")
    _require(np.isfinite(time_limit) and time_limit > 0, "Positive finite total budget required")
    cost = np.asarray(cost, dtype=float)
    _require(cost.shape == (case.model.A.shape[1],) and np.isfinite(cost).all(),
             "Cost must be a finite full-model vector")
    clock = perf_counter if clock is None else clock
    started = clock()
    model = learner.model
    plan = candidate_plan(model, case, features, kind, target_K, score_function=score_function)
    prediction_seconds = clock() - started
    rows, solutions = [], []
    exhausted = False
    for candidate in plan["candidates"]:
        row = dict(candidate, attempted=False, valid=False, objective=None,
                   normalized_infeasibility=float("inf"), status="not_attempted_budget_exhausted")
        remaining = float(time_limit) - (clock() - started)
        row["remaining_budget_before_solve"] = max(0.0, remaining)
        if remaining <= 0:
            exhausted = True
            rows.append(row)
            continue
        row["attempted"] = True
        begin = clock()
        x = None
        try:
            raw = _result_dict(solve_restriction(cost, fixed=dict(candidate["fixed"]),
                time_limit=remaining, seed=int(seed)))
            x = raw.pop("x", None)
            row["solver"] = raw
            row["status"] = raw.get("status", "solver_returned")
            if x is not None:
                x = np.asarray(x, dtype=float).copy()
                _require(x.shape == cost.shape and np.isfinite(x).all(), "Malformed solver point")
                # Preserve a finite returned point even if subsequent original
                # model verification raises; it stays invalid but is replayable.
                solutions.append(dict(rank=candidate["rank"], strategy_id=candidate["strategy_id"], x=x))
                check = dict(validate_original(x, cost))
                check.pop("x_original", None)
                row["original_validation"] = check
                objective = float(cost @ x)
                row["objective"] = objective
                fixed_ok = all(abs(float(x[j]) - v) <= 1e-6 for j, v in candidate["fixed"].items())
                row["fixed_assignment_valid"] = fixed_ok
                objective_ok = abs(objective - float(check["objective_recomputed"])) <= 1e-6 + 1e-9 * abs(objective)
                row["objective_validation_passed"] = objective_ok
                row["valid"] = bool(raw.get("valid") and check.get("valid") and fixed_ok and objective_ok)
                row["normalized_infeasibility"] = normalized_infeasibility(
                    model.bank, x, getattr(model.config, "denominator_floor", 1e-12))
        except Exception as error:
            row.update(valid=False, status="candidate_exception", error=repr(error))
        row["candidate_seconds"] = clock() - begin
        row["elapsed_seconds_after_candidate"] = clock() - started
        if row["elapsed_seconds_after_candidate"] > time_limit:
            exhausted = True
            row["late_solver_status"] = row["status"]
            row.update(valid=False, status="candidate_wall_budget_exceeded",
                       normalized_infeasibility=float("inf"))
        rows.append(row)
    elapsed = clock() - started
    exhausted = exhausted or elapsed > time_limit
    indices = {name: select_candidate(rows, name) for name in RULES}
    winner_index = indices[rule]
    winner = rows[winner_index] if winner_index is not None else None
    valid = bool(winner is not None and winner["valid"] and not exhausted)
    selected_solution = next((s["x"] for s in solutions if winner is not None and s["rank"] == winner["rank"]), None)
    selected_status = "matched_candidate_selected" if valid else "no_valid_selected_candidate"
    if exhausted:
        selected_status = "total_wall_budget_exhausted"
    diagnostic = dict(adapter_version=VERSION, original_algorithm=False, native_method_unchanged=True,
        selection_rule=rule, li_best_objective_is_separate_enhancement=(kind == "li" and rule == "best_valid_objective"),
        restriction_before_decoding=True, native_decoder_called=False, full_solver_fallback=False,
        unrestricted_solve_used=False, test_optimum_used=False, duplicate_restrictions_deduplicated=False,
        candidate_plan=plan, candidates=rows, selected_candidate_trace_index=winner_index,
        prespecified_selection_trace_indices=indices, planned_candidate_count=len(rows),
        attempted_candidate_count=sum(row["attempted"] for row in rows),
        total_time_limit=float(time_limit), time_limit_exhausted=exhausted,
        budget_policy="One shared wall budget; late candidates and an over-budget final return are invalid.")
    return dict(x=selected_solution if valid else None, valid=valid,
        objective=winner["objective"] if valid else None,
        fixed=dict(winner["fixed"]) if valid else None,
        strategy_id=winner["strategy_id"] if winner is not None else None,
        status=selected_status, seconds=elapsed, prediction_seconds=prediction_seconds,
        within_budget=not exhausted, target_K=int(target_K),
        actual_K=int(target_K) if valid else None, diagnostic=diagnostic,
        candidate_solutions=solutions, certified=False, certificate_scope="restricted_model")
