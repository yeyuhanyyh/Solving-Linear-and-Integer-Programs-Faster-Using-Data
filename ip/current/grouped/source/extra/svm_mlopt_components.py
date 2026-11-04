"""Training-only SVM and original-grid MLOPT controls for hard knapsack.

Both methods propose exactly D-K coordinate fixings; the caller solves the
restricted original IP. No development bank, reference optimum or old category
multiplier map is opened here. MLOPT anchors to its native top-ten feasible
objective winner, then ranks coordinates by all-retained-strategy agreement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import uuid
from time import perf_counter, process_time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "extra-runtime"
MLOPT_SOURCE = RUNTIME / "hard_knap_mlopt_runtime.py"
MLOPT_EXPECTED_SHA256 = "4886b0283fbec2e59d23f4901bfcd3c1a9d46fe2fb8a54e515e177e07665a322"
SCHEMA = "hard-knapsack-svm-mlopt-v1-20261025"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _mlopt_runtime():
    if file_hash(MLOPT_SOURCE) != MLOPT_EXPECTED_SHA256:
        raise ValueError("Immutable original-grid MLOPT source changed")
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))
    import hard_knap_mlopt_runtime
    if Path(hard_knap_mlopt_runtime.__file__).resolve() != MLOPT_SOURCE.resolve():
        raise ValueError("Wrong MLOPT runtime module was imported")
    return hard_knap_mlopt_runtime


def _checked_case(case):
    """Use only certified training arrays and the explicit signed affine map."""
    features = np.asarray(case.train_features, dtype=float)
    costs = np.asarray(case.train_costs, dtype=float)
    solutions = np.asarray(case.train_solutions, dtype=float)
    model = case.model
    D = int(model.A.shape[1])
    if features.ndim != 2 or len(features) < 2 or not features.shape[1]:
        raise ValueError("Need at least two complete training feature rows")
    n, q = features.shape
    if costs.shape != (n, D) or solutions.shape != costs.shape:
        raise ValueError("Original training costs and complete solutions have wrong shape")
    if not all(np.isfinite(x).all() for x in (features, costs, solutions)):
        raise ValueError("Training matrices must be finite")
    common = np.asarray(case.commitment_indices, dtype=np.int64)
    if not np.array_equal(common, np.arange(D)):
        raise ValueError("Hard-knapsack scope must be all original item columns")
    if not np.array_equal(np.asarray(model.integrality), np.ones(D, dtype=int)):
        raise ValueError("This component supports binary original variables only")
    if not (np.array_equal(model.col_lower, np.zeros(D))
            and np.array_equal(model.col_upper, np.ones(D))):
        raise ValueError("Expected original item bounds [0,1]")
    binary = np.rint(solutions).astype(np.int64)
    if not np.isin(binary, [0, 1]).all() or not np.allclose(binary, solutions, rtol=0, atol=1e-6):
        raise ValueError("Training optimizers must be binary")
    activity = np.asarray(model.A @ solutions.T).T
    violation = max(float(np.max(np.asarray(model.row_lower)[None, :]-activity, initial=0.)),
                    float(np.max(activity-np.asarray(model.row_upper)[None, :], initial=0.)))
    if violation > 1e-6:
        raise ValueError("Training solutions violate original constraints")
    offset = np.asarray(case.metadata["cost_offset"], dtype=float)
    coefficient = np.asarray(case.metadata["cost_feature_coefficients"], dtype=float)
    if offset.shape != (D,) or coefficient.shape != (D, q):
        raise ValueError("Explicit signed affine objective map has wrong shape")
    if not np.allclose(offset+features @ coefficient.T, costs, rtol=1e-14, atol=1e-12):
        raise ValueError("Signed affine training objective map changed")
    reference = np.asarray(case.train_objectives, dtype=float)
    if reference.shape != (n,) or not np.all(np.isfinite(reference)):
        raise ValueError("Need all supplied training objective values")
    if not np.allclose(np.einsum("ij,ij->i", costs, solutions), reference, rtol=1e-9, atol=1e-6):
        raise ValueError("Training solutions and objective values differ")
    K = int(case.affine["rank"])
    if not 0 <= K <= D:
        raise ValueError("Training-derived target K outside original dimension")
    return features, costs, binary, common, K


def source_binding(case):
    features, costs, binary, common, K = _checked_case(case)
    A = case.model.A.tocsr()
    return dict(schema=SCHEMA, case_id=str(case.case_id), n_train=len(features),
                D=len(common), K=K, source_hashes=dict(case.source_hashes),
                dataset_fingerprint=str(case.dataset_fingerprint),
                training_arrays_sha256=_array_hash(features, costs, binary, case.train_objectives),
                original_model_sha256=_array_hash(A.data, A.indices, A.indptr,
                    np.asarray(A.shape), case.model.row_lower, case.model.row_upper,
                    case.model.col_lower, case.model.col_upper, case.model.integrality),
                objective_map_sha256=_array_hash(case.metadata["cost_offset"],
                    case.metadata["cost_feature_coefficients"]),
                component_source_sha256=file_hash(__file__),
                immutable_mlopt_source_sha256=file_hash(MLOPT_SOURCE),
                objective_map="signed affine: offset + coefficients @ feature",
                development_used_in_training=False, confirmation_used=False)


def _seed(case):
    return int(case.metadata.get("baseline_seed", case.seed)) % (2**31-1)


def _check_model(model, case, method):
    if getattr(model, "component_method", None) != method:
        raise ValueError("Baseline method mismatch")
    if getattr(model, "component_binding", None) != source_binding(case):
        raise ValueError("Model source, training data, original matrix or objective map changed")


def _query(case, cost, features):
    value = np.asarray(features, dtype=float)
    c = np.asarray(cost, dtype=float)
    q = np.asarray(case.train_features).shape[1]
    D = int(case.model.A.shape[1])
    if value.shape != (q,) or c.shape != (D,) or not all(np.isfinite(x).all() for x in (value, c)):
        raise ValueError("Need finite query features and original objective vector")
    expected = np.asarray(case.metadata["cost_offset"]) + np.asarray(case.metadata["cost_feature_coefficients"]) @ value
    if not np.allclose(expected, c, rtol=1e-14, atol=1e-12):
        raise ValueError("Query does not obey explicit signed affine objective map")
    return c, value


@dataclass
class SVMModel:
    predictors: list[Any | None]
    constant_probabilities: np.ndarray
    common_indices: np.ndarray
    n_features: int
    fit_seconds: float = 0.
    metadata: dict = field(default_factory=dict)
    component_binding: dict = field(default_factory=dict)
    component_method: str = "svm"


def fit_svm(case):
    """Fixed standardization/linear C=1/probability=True coordinate model.

    libsvm uses its internal five-fold sigmoid probability calibration. As in
    the existing fixed-K control, scaling is fitted on supplied training rows
    before that internal calibration. No acceptance gates or tuning are added.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import sklearn

    started, cpu_started = perf_counter(), process_time()
    features, _, labels, common, K = _checked_case(case)
    predictors, probabilities, class_counts = [], np.full(len(common), np.nan), []
    seed = _seed(case)
    with threadpool_limits(limits=1):
        for j in common:
            y = labels[:, j]
            positives = int(y.sum())
            class_counts.append([len(y)-positives, positives])
            if np.all(y == y[0]):
                predictors.append(None)
                probabilities[j] = float(y[0])
                continue
            predictor = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.,
                probability=True, random_state=seed))
            predictor.fit(features, y)
            if not np.array_equal(predictor.classes_, [0, 1]):
                raise ValueError("Unexpected SVM class-probability order")
            predictors.append(predictor)
    seconds, cpu = perf_counter()-started, process_time()-cpu_started
    model = SVMModel(predictors, probabilities, common.copy(), features.shape[1], seconds)
    model.component_binding = source_binding(case)
    model.metadata = dict(method="SVM", protocol="coordinate-MAP fixed-K adaptation",
        training_samples=len(features), fitted_classifier_count=sum(p is not None for p in predictors),
        coordinate_class_counts=class_counts,
        nonconstant_coordinates_with_class_count_below_five=sum(0 < min(a,b) < 5 for a,b in class_counts),
        target_K=K, fit_seconds=seconds, fit_cpu_seconds=cpu, label_generation_included=False,
        estimator="StandardScaler + linear SVC(C=1, probability=True)",
        probability_calibration="libsvm internal training-only five-fold sigmoid calibration",
        standardization="supplied training features; before libsvm internal calibration",
        confidence="max(p, 1-p)", fixed_count=len(common)-K,
        random_state=seed, original_Xavier_acceptance_gates_used=False,
        parameter_search=False, test_information_used=False,
        sklearn_version=sklearn.__version__, python_version=platform.python_version(),
        source_binding=model.component_binding)
    return model, dict(model.metadata)


def propose_svm(model, case, cost, features):
    started = perf_counter()
    _check_model(model, case, "svm")
    _, feature = _query(case, cost, features)
    common = model.common_indices
    K = int(case.affine["rank"])
    probabilities = model.constant_probabilities.copy()
    for j, predictor in enumerate(model.predictors):
        if predictor is not None:
            index = np.flatnonzero(np.asarray(predictor.classes_) == 1)
            if len(index) != 1:
                raise ValueError("SVM class-1 probability has no unique column")
            probabilities[j] = predictor.predict_proba(feature[None, :])[0, index[0]]
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("SVM produced invalid confidence")
    confidence = np.maximum(probabilities, 1-probabilities)
    order = np.lexsort((common, -confidence))
    chosen = order[:len(common)-K]
    fixed = {int(common[j]): int(probabilities[j] >= .5) for j in chosen}
    if len(fixed) != len(common)-K:
        raise RuntimeError("SVM coordinate budget was not attained")
    return fixed, dict(method="svm", status="coordinate_fixings_proposed",
        probabilities=probabilities.tolist(), confidence=confidence.tolist(),
        selected_original_columns=common[chosen].tolist(), K=K, actual_K=len(common)-len(fixed),
        confidence_rule="max(p,1-p)", map_tie_break="p=0.5 selects one",
        coordinate_tie_break="original column ascending", full_fallback_used=False,
        prediction_seconds=perf_counter()-started)


def fit_mlopt(case, *, device="cpu", seed=None):
    """Unmodified eight configurations plus chosen all-n refit (no reductions)."""
    _checked_case(case)
    runtime = _mlopt_runtime()
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu or cuda device required")
    config = runtime.Config(device=device, seed=_seed(case) if seed is None else int(seed),
        hidden_candidates=((32,32,32),(64,64,64)), learning_rates=(1e-3,1e-2),
        epoch_candidates=(50,150), dropout=.1, batch_size=32,
        validation_fraction=.2, n_best=10)
    started, cpu_started = perf_counter(), process_time()
    with threadpool_limits(limits=1):
        model = runtime.fit(case, config=config)
    wrapper_wall, wrapper_cpu = perf_counter()-started, process_time()-cpu_started
    if len(model.metadata["validation_trials"]) != 8:
        raise RuntimeError("Required full eight-config MLOPT grid was not completed")
    if model.metadata["final_bank"]["n_samples"] != len(case.train_features):
        raise RuntimeError("MLOPT final refit omitted training examples")
    if model.metadata["trainable_parameter_count"] != sum(w.size+b.size for w,b in model.weights):
        raise RuntimeError("MLOPT exported tensors and parameter count differ")
    model.component_method = "mlopt"
    model.component_binding = source_binding(case)
    model.metadata.update(method="MLOPT", protocol="native-output-anchor fixed-K adaptation",
        wrapper_fit_seconds=wrapper_wall, wrapper_fit_cpu_seconds=wrapper_cpu,
        label_generation_included=False, neural_parameter_grid_count=8,
        matching_anchor="minimum objective among native decoded top-ten feasible candidates",
        matching_confidence="all retained strategy softmax-temperature-one coordinate agreement",
        matching_fixed_count=case.model.A.shape[1]-int(case.affine["rank"]),
        source_binding=model.component_binding, python_version=platform.python_version(),
        torch_version=torch.__version__)
    return model, dict(model.metadata)


def _mlopt_logits(model, feature):
    values = ((feature-model.mean)/model.scale).astype(np.float32)
    for i, (weight,bias) in enumerate(model.weights):
        values = weight @ values + bias
        if i < len(model.weights)-1:
            values = np.maximum(values, 0.)
    return np.asarray(values, dtype=float)


def propose_mlopt(model, case, cost, features):
    """Native top-ten winner, never raw top-one, anchors D-K fixings."""
    started = perf_counter()
    _check_model(model, case, "mlopt")
    c, feature = _query(case, cost, features)
    native = dict(model.predict(feature, c))
    witness = native.pop("x", None)
    diagnostic = dict(method="mlopt", native_prediction=native,
        anchor_source="minimum objective among decoded native top-ten feasible candidates",
        full_fallback_used=False, warm_start_used=False,
        actual_K=None, K=int(case.affine["rank"]))
    if not native.get("valid", False) or witness is None:
        diagnostic.update(status="no_valid_native_winner", prediction_seconds=perf_counter()-started)
        return None, diagnostic
    witness = np.asarray(witness, dtype=float)
    selected = np.asarray(model.selected, dtype=np.int64)
    winner = int(native["strategy_id"])
    if winner not in selected:
        raise RuntimeError("Native MLOPT winner missing from retained strategies")
    integer = np.asarray(model.bank.integer_indices, dtype=np.int64)
    common = np.asarray(case.commitment_indices, dtype=np.int64)
    if not np.array_equal(integer, common):
        raise ValueError("Bank integer-coordinate map differs from the original binary IP")
    anchor = np.asarray(model.bank.strategies[winner].integer_values, dtype=np.int64)
    if not model.bank.check_original(witness)["valid"] or np.max(np.abs(witness-anchor), initial=0.) > 1e-6:
        raise RuntimeError("Native anchor lacks its claimed original-feasible integer witness")
    scores = _mlopt_logits(model, feature)
    if scores.shape != selected.shape or not np.isfinite(scores).all():
        raise ValueError("MLOPT class scores and retained strategies disagree")
    weights = np.exp(scores-scores.max())
    weights /= weights.sum()
    historical = np.asarray([model.bank.strategies[int(j)].integer_values for j in selected])
    confidence = np.sum(weights[:, None]*(historical == anchor[None,:]), axis=0)
    order = np.lexsort((common, -confidence))
    K = int(case.affine["rank"])
    chosen = order[:len(common)-K]
    fixed = {int(common[j]): int(anchor[j]) for j in chosen}
    if len(fixed) != len(common)-K or any(abs(witness[j]-v) > 1e-6 for j,v in fixed.items()):
        raise RuntimeError("Matched fixing count or native witness failed")
    diagnostic.update(status="native_output_anchor_fixings_proposed",
        native_anchor_strategy_id=winner,
        native_score_top_one_strategy_id=int(selected[int(np.argmax(scores))]),
        native_anchor_representative_training_index=int(model.bank.strategies[winner].representative_training_index),
        native_anchor=anchor.tolist(), native_objective=float(c @ witness),
        native_candidate_count=int(native["diagnostics"]["n_candidates"]),
        retained_strategy_ids=selected.tolist(), all_retained_strategy_scores=scores.tolist(),
        strategy_weights=weights.tolist(), softmax_temperature=1.,
        confidence=confidence.tolist(), selected_original_columns=common[chosen].tolist(),
        coordinate_selection="agreement with native winner descending, original column ascending",
        feasible_witness_valid=True, actual_K=len(common)-len(fixed),
        prediction_seconds=perf_counter()-started)
    return fixed, diagnostic


def save_component(model, path):
    """Write a new checkpoint atomically; existing checkpoints are immutable."""
    path = Path(path)
    if getattr(model, "component_method", None) not in ("svm", "mlopt"):
        raise ValueError("Not a guarded SVM/MLOPT component")
    binding = model.component_binding
    if binding["component_source_sha256"] != file_hash(__file__):
        raise ValueError("Component source changed since fit")
    if binding["immutable_mlopt_source_sha256"] != file_hash(MLOPT_SOURCE):
        raise ValueError("MLOPT runtime changed since fit")
    envelope = dict(schema=SCHEMA, method=model.component_method,
                    source_binding=binding, model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            pickle.dump(envelope, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename is exclusive (an existing destination is an error).
        # POSIX rename overwrites, so there use an exclusive hard-link publish.
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path), sha256=file_hash(path), method=model.component_method,
                source_binding=binding, metadata=model.metadata)


def load_component(path, case, method):
    """Load only trusted locally generated files, then verify all bound inputs."""
    if method not in ("svm", "mlopt"):
        raise ValueError("Unknown component method")
    _mlopt_runtime()  # Stable import name is also required by the trusted pickle.
    with Path(path).open("rb") as stream:
        envelope = pickle.load(stream)
    if envelope.get("schema") != SCHEMA or envelope.get("method") != method:
        raise ValueError("Checkpoint schema or method changed")
    if envelope.get("source_binding") != source_binding(case):
        raise ValueError("Checkpoint source/data binding changed")
    model = envelope["model"]
    _check_model(model, case, method)
    return model
