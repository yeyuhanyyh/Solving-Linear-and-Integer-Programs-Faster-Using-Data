"""Declared matched-K coordinate-fixing variants of frozen strategy learners.

These are new prediction-guided variants, not native MLOPT or Li decoding.
No decoder, solver, training procedure, test optimum, or active-set constraint
is used here. The caller solves the original formulation with only the returned
common-coordinate fixings added. UC startup/shutdown/dispatch stay unrestricted
by this helper. Existing complete-strategy native methods remain separate.

The caller places the original frozen code directory on sys.path and verifies
checkpoint/data provenance before constructing the fitted model. Its validated
historical strategy bank supplies the anchor. In the present cost-only families,
that historical feasible strategy also supplies a feasible witness for the
partial fixings; this helper does not re-solve or re-certify the witness.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from dimension_diagnostics import (
    assert_matched_fixings,
    common_coordinate_indices,
    target_common_dimension,
)


IMPLEMENTATION_VERSION = "strategy-guided-matched-k-v1-20261010"


def _standardized_features(model: Any, features: np.ndarray) -> np.ndarray:
    mean = np.asarray(model.mean, dtype=float)
    scale = np.asarray(model.scale, dtype=float)
    features = np.asarray(features, dtype=float)
    if mean.ndim != 1 or features.shape != mean.shape or scale.shape != mean.shape:
        raise ValueError("Features and fitted standardization must be equal-size vectors")
    if not (np.isfinite(features).all() and np.isfinite(mean).all()
            and np.isfinite(scale).all() and np.all(scale > 0)):
        raise ValueError("Features/standardization must be finite with positive scales")
    # Match the original CPU inference conversion exactly.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        standardized = ((features - mean) / scale).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized float32 features are not finite")
    return standardized


def _native_scores(model: Any, standardized: np.ndarray, kind: str) -> np.ndarray:
    if kind == "mlopt":
        values = standardized
        if not model.weights:
            raise ValueError("MLOPT checkpoint has no fitted network layers")
        for i, (weight, bias) in enumerate(model.weights):
            weight, bias = np.asarray(weight), np.asarray(bias)
            if (weight.ndim != 2 or weight.shape[1] != len(values)
                    or bias.shape != (weight.shape[0],)
                    or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                raise ValueError("Invalid MLOPT layer dimensions or nonfinite weights")
            values = weight @ values + bias
            if i != len(model.weights) - 1:
                values = np.maximum(values, 0)
        scores = values
    elif kind == "li":
        import torch

        network = model.network
        if network is None or network.training:
            raise ValueError("Li checkpoint must have a fitted network in eval mode")
        if any(tensor.device.type != "cpu" for tensor in
               list(network.parameters()) + list(network.buffers())):
            raise ValueError("Matched-K inference requires the frozen Li network on CPU")
        parameters = torch.as_tensor(standardized[None, :], dtype=torch.float32)
        with torch.inference_mode():
            output = network(parameters)
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError("Li score output must have shape (1, retained_strategy_count)")
        scores = output[0].detach().cpu().numpy()
    else:
        raise ValueError("kind must be 'mlopt' or 'li'")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Strategy scores must be a finite vector")
    return scores


def propose_strategy_fixings(model: Any, case: Any, features: np.ndarray,
                             kind: str) -> tuple[dict[int, int], dict[str, Any]]:
    """Fix exactly D-K common columns to one predicted historical anchor.

    All retained strategy scores are evaluated, independently of native n_best
    or top_k. Temperature-one softmax is a confidence surrogate, not a claim of
    calibration. Coordinate confidence is weighted agreement with the highest
    scoring strategy. Ties in strategy scores preserve the checkpoint's retained
    order; ties in coordinate confidence use the original variable column.
    """
    if kind not in ("mlopt", "li"):
        raise ValueError("kind must be 'mlopt' or 'li'")
    common = common_coordinate_indices(case)
    target_K = target_common_dimension(case)
    standardized = _standardized_features(model, features)
    scores = _native_scores(model, standardized, kind)

    selected_raw = np.asarray(model.selected)
    if selected_raw.ndim != 1 or not len(selected_raw):
        raise ValueError("Checkpoint must retain at least one strategy")
    selected = selected_raw.astype(np.int64)
    if (not np.array_equal(selected_raw, selected) or len(np.unique(selected)) != len(selected)
            or np.any(selected < 0) or np.any(selected >= len(model.bank.strategies))):
        raise ValueError("Retained strategies must be distinct valid bank indices")
    if scores.shape != selected.shape:
        raise ValueError("Scores must correspond to all retained strategies in checkpoint order")

    bank_integer_raw = np.asarray(model.bank.integer_indices)
    bank_integer = bank_integer_raw.astype(np.int64)
    original_integer = np.flatnonzero(np.asarray(case.model.integrality) != 0)
    if (bank_integer.ndim != 1 or not np.array_equal(bank_integer_raw, bank_integer)
            or len(np.unique(bank_integer)) != len(bank_integer)
            or not np.array_equal(np.sort(bank_integer), original_integer)):
        raise ValueError("Strategy-bank integer columns differ from the original model")
    integer_location = {int(column): i for i, column in enumerate(bank_integer)}
    common_locations = np.asarray([integer_location[int(column)] for column in common])
    common_strategies = []
    for index in selected:
        full_integer = np.asarray(model.bank.strategies[int(index)].integer_values, dtype=float)
        if full_integer.shape != bank_integer.shape or not np.isfinite(full_integer).all():
            raise ValueError("Historical strategy has invalid integer assignment shape/values")
        values = full_integer[common_locations]
        rounded = np.rint(values)
        if (np.any(np.abs(values - rounded) > 1e-9)
                or np.any(rounded < np.asarray(case.model.col_lower)[common])
                or np.any(rounded > np.asarray(case.model.col_upper)[common])):
            raise ValueError("Historical common values must be integers within original bounds")
        common_strategies.append(rounded.astype(np.int64))
    common_strategies = np.asarray(common_strategies, dtype=np.int64)

    # Stable argsort reproduces the native retained-order score tie rule.
    ranking = np.argsort(-scores, kind="stable")
    anchor_position = int(ranking[0])
    anchor = common_strategies[anchor_position]
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    weights /= weights.sum()
    confidence = np.sum(weights[:, None] * (common_strategies == anchor), axis=0)
    coordinate_order = np.lexsort((common, -confidence))
    chosen = coordinate_order[:len(common) - target_K]
    fixed = {int(common[position]): int(anchor[position]) for position in chosen}
    dimensions = assert_matched_fixings(case, fixed, target_K=target_K)
    native_limit = (getattr(model.config, "n_best", None) if kind == "mlopt"
                    else getattr(model, "top_k", None))
    anchor_strategy = model.bank.strategies[int(selected[anchor_position])]
    representative_index = getattr(anchor_strategy, "representative_training_index", None)
    diagnostics = dict(
        implementation_version=IMPLEMENTATION_VERSION,
        method=f"{kind}-guided matched-K coordinate-fixing variant",
        original_algorithm=False,
        native_method_unchanged=True,
        score_type="classifier logits" if kind == "mlopt" else "predicted strategy rewards",
        retained_strategy_ids=selected.tolist(),
        all_retained_strategy_scores=scores.tolist(),
        strategy_score_ranking_positions=ranking.tolist(),
        softmax_temperature=1.0,
        strategy_confidence_weights=weights.tolist(),
        confidence_scope="temperature-one confidence surrogate; not calibrated probabilities",
        anchor_retained_position=anchor_position,
        anchor_strategy_id=int(selected[anchor_position]),
        anchor_representative_training_index=(None if representative_index is None
                                              else int(representative_index)),
        common_original_columns=common.tolist(),
        anchor_common_values=anchor.tolist(),
        common_weighted_agreement=confidence.tolist(),
        confidence_sorted_common_columns=common[coordinate_order].tolist(),
        fixed_original_columns=common[chosen].tolist(),
        native_candidate_limit=(None if native_limit is None else int(native_limit)),
        native_candidate_limit_used=False,
        planned_submip_count=1,
        active_constraints_used=False,
        original_constraints_to_retain="all",
        additional_integer_fixings="common u/x only; no UC startup/shutdown/dispatch fixing",
        coordinate_selection="weighted agreement descending, original column ascending",
        strategy_score_ties="checkpoint retained order",
        test_optimum_used=False,
        training_performed=False,
        decoder_called=False,
        dimension_diagnostics=dimensions,
    )
    return fixed, diagnostics
