"""MLOPT on fixed-feasible-set binary knapsack; native and matched-K queries.

The immutable training/pruning/decoding implementation is vendored unchanged.
Partial fixing is an adaptation, preserving the native top-ten winner as its
feasible anchor. It is not a native MLOPT compression-dimension parameter.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from . import mlopt_runtime as runtime

RUNTIME_SHA256 = "4886b0283fbec2e59d23f4901bfcd3c1a9d46fe2fb8a54e515e177e07665a322"
IMPLEMENTATION_VERSION = "asc-mlopt-native-output-anchor-v1-20261030"


def _verify_runtime():
    actual = sha256(Path(runtime.__file__).read_bytes()).hexdigest()
    if actual != RUNTIME_SHA256:
        raise RuntimeError("Vendored native MLOPT implementation differs from its frozen source")


def _query(model, profits):
    p = np.asarray(profits, dtype=float)
    if p.shape != (model.bank.n_variables,) or not np.isfinite(p).all():
        raise ValueError("profits must be a finite vector of original variable dimension")
    return p


def make_case(weights, capacity, train_profits, train_solutions, *, case_id):
    """Create a training-only minimization case from positive knapsack profits."""
    w = np.asarray(weights, dtype=float)
    p, x = np.asarray(train_profits, dtype=float), np.asarray(train_solutions, dtype=float)
    if (w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w <= 0)
            or not np.isfinite(capacity) or capacity < 0):
        raise ValueError("Invalid knapsack weights or capacity")
    if (p.ndim != 2 or p.shape[1] != len(w) or x.shape != p.shape
            or len(p) < 2 or not np.isfinite(p).all() or np.any(p <= 0)
            or not np.isfinite(x).all() or np.any(x != np.rint(x))
            or np.any(x < 0) or np.any(x > 1)):
        raise ValueError("Require finite positive training profits and complete binary labels")
    if np.any(x @ w > float(capacity) + 1e-7):
        raise ValueError("A training label violates original capacity")
    d = len(w)
    original = SimpleNamespace(A=sp.csr_matrix(w[None, :]),
        row_lower=np.array([-np.inf]), row_upper=np.array([float(capacity)]),
        col_lower=np.zeros(d), col_upper=np.ones(d), integrality=np.ones(d, dtype=np.int64))
    return SimpleNamespace(case_id=str(case_id), model=original,
        train_features=p.copy(), train_costs=-p.copy(), train_solutions=x.copy())


def fit(weights, capacity, train_profits, train_solutions, *, case_id,
        device="cpu", seed=0):
    """Fit the complete predetermined native grid; returns (model, metadata).

    The runtime's fit_seconds includes strategy construction, frequency pruning,
    all eight validation configurations, and the chosen all-training-data refit.
    It excludes shared optimizer-label generation. We retain that scope directly.
    """
    _verify_runtime()
    case = make_case(weights, capacity, train_profits, train_solutions, case_id=case_id)
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu or cuda training device is required")
    config = runtime.Config(device=device, seed=int(seed),
        hidden_candidates=((32, 32, 32), (64, 64, 64)), learning_rates=(1e-3, 1e-2),
        epoch_candidates=(50, 150), dropout=0.1, batch_size=32,
        validation_fraction=0.2, n_best=10)
    with threadpool_limits(limits=1):
        model = runtime.fit(case, config=config)
    if len(model.metadata["validation_trials"]) != 8:
        raise RuntimeError("Native MLOPT grid omitted a declared configuration")
    if model.metadata["final_bank"]["n_samples"] != len(case.train_features):
        raise RuntimeError("Native MLOPT final refit omitted training data")
    if not np.array_equal(model.bank.integer_indices, np.arange(len(weights))):
        raise RuntimeError("This adapter is only for all-binary knapsack")
    model.metadata.update(method="MLOPT", adapter_version=IMPLEMENTATION_VERSION,
        vendored_runtime_sha256=RUNTIME_SHA256,
        objective_conversion="max positive profit p*x becomes min -p*x",
        input_features="all original item profits, standardized within training folds",
        native_K=None, native_free_binary_variables_per_strategy=0,
        no_strategy_library_cap=True, native_top_candidates=10,
        matching_policy="partial coordinate-fixing adaptation anchored to native top-ten decoded best",
        matching_native_algorithm=False,
        learning_seconds=model.fit_seconds,
        label_generation_included=False)
    return model, dict(model.metadata)


def _predict_native(model, profits):
    p = _query(model, profits)
    decoded = model.predict(p, -p)
    result = dict(decoded)
    result["native_minimization_objective"] = result["objective"]
    result["objective"] = -float(result["objective"]) if result.get("valid") else None
    result["K"] = None
    result["native_free_binary_variables_per_strategy"] = 0
    result["solver_seconds"] = 0.0
    result["objective_sense"] = "maximize original positive profit"
    return result


def predict_native(model, profits):
    """Native top-ten feasible-strategy decoding, with no MIP fallback.

    Returns original positive-profit objective and original solution in x.
    seconds/total_seconds include prediction, decoding, and original checks.
    solver_seconds is zero because native decoding invokes no IP solver.
    """
    started = perf_counter()
    with threadpool_limits(limits=1):
        result = _predict_native(model, profits)
    result["total_seconds"] = perf_counter() - started
    return result


def propose_fixed(model, profits, K):
    """Return exactly d-K fixings around the native top-ten feasible winner.

    All retained logits determine temperature-one softmax confidence. Coordinates
    are ranked by agreement with the native winner, then original column index.
    The original winning assignment witnesses feasibility of every fixing. A
    caller still solves the original IP with these bounds, keeping all original
    constraints. No warm start or full-solver fallback is provided here.
    """
    started = perf_counter()
    p = _query(model, profits)
    d = len(p)
    if isinstance(K, bool) or int(K) != K or not 0 <= int(K) <= d:
        raise ValueError("K must be an integer between zero and original dimension")
    K = int(K)
    with threadpool_limits(limits=1):
        native = _predict_native(model, p)
        witness = native.pop("x", None)
        diagnostic = dict(method="MLOPT", adapter_version=IMPLEMENTATION_VERSION,
            original_algorithm=False, native_prediction=native, target_K=K,
            anchor_source="best feasible decoded candidate among native top ten",
            warm_start_used=False, full_solver_fallback=False,
            score_scope="all retained strategy logits; no candidate-library cap",
            softmax_temperature=1.0, actual_K=None)
        if not native.get("valid") or witness is None:
            diagnostic.update(status="no_valid_native_winner",
                prediction_seconds=perf_counter()-started)
            return None, diagnostic
        witness = np.asarray(witness, dtype=float)
        selected = np.asarray(model.selected, dtype=np.int64)
        winner = int(native["strategy_id"])
        if winner not in selected or not model.bank.check_original(witness)["valid"]:
            raise RuntimeError("Native anchor does not have its claimed feasible witness")
        anchor = np.asarray(model.bank.strategies[winner].integer_values, dtype=np.int64)
        if not np.array_equal(witness, anchor):
            raise RuntimeError("Native winner differs from its complete integer strategy")
        values = ((p-model.mean)/model.scale).astype(np.float32)
        for i, (weight, bias) in enumerate(model.weights):
            values = weight @ values + bias
            if i < len(model.weights)-1:
                values = np.maximum(values, 0)
        scores = np.asarray(values, dtype=float)
        if scores.shape != selected.shape or not np.isfinite(scores).all():
            raise ValueError("MLOPT scores and retained strategies disagree")
        confidence_weights = np.exp(scores - scores.max())
        confidence_weights /= confidence_weights.sum()
        historical = np.asarray([model.bank.strategies[int(j)].integer_values for j in selected])
        confidence = np.sum(confidence_weights[:, None] * (historical == anchor[None, :]), axis=0)
        columns = np.arange(d, dtype=np.int64)
        order = np.lexsort((columns, -confidence))
        chosen = order[:d-K]
        fixed = {int(j): int(anchor[j]) for j in chosen}
        if len(fixed) != d-K or any(witness[j] != v for j, v in fixed.items()):
            raise RuntimeError("Exact matched dimension or native witness check failed")
        diagnostic.update(status="native_output_anchor_fixings_proposed", actual_K=d-len(fixed),
            native_anchor_strategy_id=winner,
            raw_top_one_strategy_id=int(selected[int(np.argmax(scores))]),
            native_objective=native["objective"],
            selected_original_columns=chosen.tolist(), fixed_count=len(fixed),
            retained_strategy_ids=selected.tolist(), retained_scores=scores.tolist(),
            confidence_weights=confidence_weights.tolist(), coordinate_confidence=confidence.tolist(),
            native_anchor=anchor.tolist(), feasible_witness_valid=True,
            coordinate_selection="agreement with native winner descending, original column ascending",
            confidence_calibrated=False, prediction_seconds=perf_counter()-started)
    return fixed, diagnostic
