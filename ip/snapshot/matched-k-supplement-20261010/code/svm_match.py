"""Two explicitly labelled SVM dimension-control variants.

The frozen original ``dimension_diagnostics`` module must be on sys.path.
The budget cap reuses the original Xavier-style accepted equations and does
not add predictions. The coordinate-MAP learner fits a predictor for every
nonconstant common coordinate and drops the original acceptance gates. It is
therefore a separate variant, not the native Xavier implementation.

Neither fitting routine reads confirmation/test data. K always comes from
the frozen training-derived affine rank, in the common u/x coordinate block.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from dimension_diagnostics import (
    assert_matched_fixings,
    common_coordinate_indices,
    select_equalities_by_rank,
    target_common_dimension,
)


CAPPED_METHOD = "SVM budget-capped variant"
COORDINATE_METHOD = "SVM coordinate-MAP matched-K variant"


def _equality_key(equality: Any) -> tuple[str, int, int | None, int]:
    return (str(equality.kind), int(equality.i),
            None if equality.j is None else int(equality.j), int(equality.rhs))


def _feature_vector(features: Any, n_features: int) -> np.ndarray:
    value = np.asarray(features, dtype=float)
    if value.shape != (n_features,) or not np.isfinite(value).all():
        raise ValueError("Expected one finite cost-feature vector with the training dimension")
    return value


def propose_capped_equalities(model: Any, case: Any,
                              features: Any) -> tuple[list[Any], dict[str, Any]]:
    """Keep up to D-K consistent independent native accepted equations.

    Confidence is the pooled training-only OOF positive precision for an
    accepted classifier, or the training positive frequency for an
    unconditional equation. Ties retain the native prediction order. If the
    original accepted predictions cannot reach the requested rank, every
    usable independent equation is kept and the mismatch is reported.
    """
    started = perf_counter()
    common = common_coordinate_indices(case)
    D, K = len(common), target_common_dimension(case)
    if int(model.n_binary_) != D:
        raise ValueError("Native SVM and case use different common-coordinate dimensions")
    feature = _feature_vector(features, int(model.n_features_in_))
    equalities, native_info = model.predict_equalities(feature)
    native_seconds = perf_counter() - started
    records = {_equality_key(record.equality): record for record in model.records_}
    scores = []
    for equality in equalities:
        record = records.get(_equality_key(equality))
        if record is None:
            raise ValueError("Native prediction lacks a corresponding training candidate record")
        if record.status == "unconditional":
            score = record.frequency
        elif record.status == "accepted_classifier":
            score = record.oof_precision
        else:
            raise ValueError("Native output contains an equation that did not pass the original gate")
        if score is None or not np.isfinite(score) or not 0. <= float(score) <= 1.:
            raise ValueError("Candidate confidence must be a finite training reliability in [0,1]")
        scores.append(float(score))
    selected = select_equalities_by_rank(equalities, D, D-K, scores)
    selection_seconds = perf_counter() - started - native_seconds
    equations = list(selected.pop("selected"))
    diagnostic = dict(
        selected, method=CAPPED_METHOD, is_native_algorithm=False,
        confidence="training OOF positive precision; unconditional training positive frequency",
        tie_break="original native prediction order",
        native_equalities_count=len(equalities), selected_equalities_count=len(equations),
        selected_confidence=[scores[i] for i in selected["selected_indices"]],
        native_prediction_diagnostic=native_info,
        native_prediction_seconds=native_seconds,
        selection_seconds=selection_seconds,
        variant_seconds=perf_counter()-started,
        added_synthetic_equalities=0,
        dimension_scope="common-coordinate affine space imposed by selected equations",
    )
    return equations, diagnostic


@dataclass
class CoordinateSVMModel:
    """Training-only predictors, with None estimators for constant columns."""

    predictors: list[Any | None]
    constant_probabilities: np.ndarray
    common_indices: np.ndarray
    n_features: int
    random_state: int
    training_samples: int


def fit_coordinate_svm(case: Any) -> tuple[CoordinateSVMModel, dict[str, Any]]:
    """Fit C=1 linear probabilistic SVCs using exactly the 120 train labels.

    libsvm's probability=True sigmoid calibration uses its internal five-fold
    procedure on the supplied training data. StandardScaler is fitted on the
    training features before that procedure; this routine does not estimate
    validation performance or tune hyperparameters. The seed is fixed by case.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    started = perf_counter()
    common = common_coordinate_indices(case)
    features = np.array(case.train_features, dtype=float, copy=True)
    solutions = np.asarray(case.train_solutions, dtype=float)
    if (features.ndim != 2 or features.shape[0] != 120 or not features.shape[1]
            or not np.isfinite(features).all()):
        raise ValueError("Coordinate SVM requires exactly 120 finite training feature vectors")
    if (solutions.shape != (120, int(case.model.A.shape[1]))
            or not np.isfinite(solutions[:, common]).all()):
        raise ValueError("Training solution shape differs from the original model")
    values = solutions[:, common]
    binary = np.rint(values).astype(np.int8)
    if np.max(np.abs(values-binary), initial=0.) > 1e-6 or not np.isin(binary, [0, 1]).all():
        raise ValueError("Common training coordinates must be binary")
    seed = int(case.seed)
    predictors, probabilities = [], np.full(len(common), np.nan, dtype=float)
    constant_count = 0
    for j in range(len(common)):
        labels = binary[:, j]
        if np.all(labels == labels[0]):
            probabilities[j] = float(labels[0])
            predictors.append(None)
            constant_count += 1
        else:
            predictor = make_pipeline(
                StandardScaler(),
                SVC(kernel="linear", C=1.0, probability=True, random_state=seed),
            )
            predictor.fit(features, labels)
            if not np.array_equal(np.asarray(predictor.classes_), np.array([0, 1])):
                raise RuntimeError("Binary SVC did not preserve the expected probability class mapping")
            predictors.append(predictor)
    model = CoordinateSVMModel(predictors=predictors,
        constant_probabilities=probabilities,
        common_indices=common.copy(), n_features=features.shape[1],
        random_state=seed, training_samples=120)
    metadata = dict(
        method=COORDINATE_METHOD, is_native_algorithm=False,
        fit_seconds=perf_counter()-started, training_samples=120,
        training_labels="original certified optimizer common u/x coordinates",
        common_ambient_dimension=len(common), n_features=features.shape[1],
        constant_coordinate_count=constant_count,
        fitted_classifier_count=len(common)-constant_count,
        estimator="StandardScaler followed by linear SVC(C=1, probability=True)",
        probability_calibration="libsvm internal training-only five-fold Platt sigmoid calibration",
        scaling="StandardScaler fitted on supplied training features before libsvm calibration",
        random_state=seed, original_Table1_gate_used=False,
        parameter_selection="fixed protocol; no confirmation/test data or labels accessed",
        prediction="class-1 probability MAP, tie p=0.5 selects 1",
        confidence="max(p,1-p)", tie_break="original common-column index",
        target_K="frozen training-derived Ours affine rank",
    )
    return model, metadata


def propose_coordinate_fixings(model: CoordinateSVMModel, case: Any,
                                features: Any) -> tuple[dict[int, int], dict[str, Any]]:
    """Fix exactly D-K distinct common coordinates using calibrated MAP."""
    started = perf_counter()
    common = common_coordinate_indices(case)
    if not np.array_equal(common, model.common_indices):
        raise ValueError("Coordinate SVM common-column mapping differs from the case")
    D, K = len(common), target_common_dimension(case)
    if len(model.predictors) != D or model.constant_probabilities.shape != (D,):
        raise ValueError("Coordinate SVM predictor arrays have invalid dimensions")
    feature = _feature_vector(features, model.n_features)
    p = model.constant_probabilities.copy()
    for j, predictor in enumerate(model.predictors):
        if predictor is not None:
            classes = np.asarray(predictor.classes_)
            location = np.flatnonzero(classes == 1)
            if len(location) != 1:
                raise ValueError("Class-1 probability has no unique classifier column")
            p[j] = float(predictor.predict_proba(feature.reshape(1, -1))[0, location[0]])
    if not np.isfinite(p).all() or np.any(p < 0.) or np.any(p > 1.):
        raise ValueError("Coordinate SVM produced an invalid probability")
    confidence = np.maximum(p, 1.-p)
    order = np.lexsort((common, -confidence))
    chosen = order[:D-K]
    fixed = {int(common[j]): int(p[j] >= .5) for j in chosen}
    metrics = assert_matched_fixings(case, fixed, target_K=K)
    diagnostic = dict(
        method=COORDINATE_METHOD, is_native_algorithm=False,
        probability_class="1", common_probabilities=p.tolist(),
        common_confidence=confidence.tolist(), selected_common_indices=common[chosen].tolist(),
        target_common_dimension=K, actual_common_dimension=D-len(fixed),
        target_rank=D-K, actual_rank=len(fixed), exact_match=True,
        status="budget_reached", original_Table1_gate_used=False,
        dimension_diagnostics=metrics, prediction_seconds=perf_counter()-started,
        confidence="max(p,1-p)", tie_break="original common-column index",
        map_tie_break="p=0.5 selects 1",
    )
    return fixed, diagnostic
