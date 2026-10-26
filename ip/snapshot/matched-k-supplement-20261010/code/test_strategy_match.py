"""Synthetic-only checks of the new matched-K strategy fixing policy.

No existing experiment, training, checkpoint, solver, or cloud API is used.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sparse

ORIGINAL_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(ORIGINAL_CODE))

from strategy_match import propose_strategy_fixings


class ForbiddenDecodeStrategy:
    def __init__(self, values, source_index):
        self.integer_values = np.asarray(values, dtype=np.int64)
        self.representative_training_index = source_index

    @property
    def active_rows(self):
        raise AssertionError("Matched variant must not read active rows")


class ArrayTensor:
    """Minimal CPU tensor facade; no real Torch network is run by these tests."""
    def __init__(self, values):
        self.values = np.asarray(values)
        self.device = SimpleNamespace(type="cpu")

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return ArrayTensor(self.values[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class ConstantScoreNetwork:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def parameters(self):
        return ()

    def buffers(self):
        return (ArrayTensor(self.scores),)

    def __call__(self, features):
        return ArrayTensor(np.repeat(self.scores[None, :], features.shape[0], axis=0))


def fixture(K=2, family="uc"):
    # Non-contiguous common columns expose accidental indexing by array position.
    common = np.asarray([1, 4, 6, 9], dtype=np.int64)
    integer = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    case = SimpleNamespace(
        family=family,
        commitment_indices=common,
        affine={"rank": K},
        model=SimpleNamespace(A=sparse.csr_matrix((0, 12)),
                              integrality=np.r_[np.ones(10), np.zeros(2)],
                              col_lower=np.zeros(12), col_upper=np.ones(12)),
    )
    common_values = [[1, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0]]
    strategies = []
    for index, values in enumerate(common_values):
        full = np.ones(10, dtype=np.int64)
        full[common] = values
        strategies.append(ForbiddenDecodeStrategy(full, index + 10))

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("No native decoding or solver is allowed")

    bank = SimpleNamespace(integer_indices=integer, strategies=strategies,
                           decode=forbidden_decode)
    model = SimpleNamespace(
        mean=np.asarray([1., 2.]), scale=np.asarray([2., 4.]),
        selected=np.asarray([2, 0, 1]), bank=bank,
        weights=[(np.asarray([[1., 0.], [0., -1.]], dtype=np.float32),
                  np.asarray([.5, -.5], dtype=np.float32)),
                 (np.asarray([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32),
                  np.zeros(3, dtype=np.float32))],
        config=SimpleNamespace(n_best=1), top_k=1,
        network=ConstantScoreNetwork([1000., 999., 998.]).eval(),
    )
    return case, model, np.asarray([5., -6.])


class StrategyMatchTests(unittest.TestCase):
    def setUp(self):
        # Verify the Li adapter interface with a fake CPU tensor API. Real
        # checkpoint/PyTorch integration is explicitly outside this local test.
        torch_stub = SimpleNamespace(
            float32=np.float32,
            as_tensor=lambda values, dtype: ArrayTensor(np.asarray(values, dtype=dtype)),
            inference_mode=nullcontext,
        )
        self.torch_patch = patch.dict(sys.modules, {"torch": torch_stub})
        self.torch_patch.start()

    def tearDown(self):
        self.torch_patch.stop()

    def test_mlopt_exact_count_values_and_original_column_tie(self):
        case, model, features = fixture()
        fixed, report = propose_strategy_fixings(model, case, features, "mlopt")
        self.assertEqual(fixed, {6: 1, 1: 1})
        self.assertEqual(report["all_retained_strategy_scores"], [2.5, 1.5, -2.5])
        self.assertEqual(report["anchor_strategy_id"], 2)
        self.assertEqual(report["anchor_representative_training_index"], 12)
        self.assertEqual(report["retained_strategy_ids"], [2, 0, 1])
        self.assertEqual(report["dimension_diagnostics"]["common_nominal_free_dimension"], 2)
        self.assertEqual(report["dimension_diagnostics"]["other_integer_fixed_count"], 0)
        self.assertTrue(report["dimension_diagnostics"]["matches_expected_common_dimension"])
        self.assertFalse(report["native_candidate_limit_used"])
        self.assertEqual(len(report["all_retained_strategy_scores"]), 3)
        json.dumps(report, allow_nan=False)

    def test_li_stable_softmax_and_anchor_agreement(self):
        case, model, features = fixture()
        before = model.network.scores.copy()
        fixed, report = propose_strategy_fixings(model, case, features, "li")
        self.assertEqual(fixed, {6: 1, 1: 1})
        weights = np.exp(np.asarray([0., -1., -2.]))
        weights /= weights.sum()
        np.testing.assert_allclose(report["strategy_confidence_weights"], weights)
        expected = [weights[0] + weights[1], weights[0] + weights[2], 1.,
                    weights[0] + weights[1]]
        np.testing.assert_allclose(report["common_weighted_agreement"], expected)
        self.assertTrue(np.array_equal(before, model.network.scores))
        self.assertFalse(model.network.training)
        for column, value in fixed.items():
            anchor = model.bank.strategies[report["anchor_strategy_id"]].integer_values
            self.assertEqual(value, int(anchor[column]))
        json.dumps(report, allow_nan=False)

    def test_strategy_score_ties_preserve_retained_order(self):
        case, model, features = fixture()
        model.network = ConstantScoreNetwork([2., 2., 2.]).eval()
        fixed, report = propose_strategy_fixings(model, case, features, "li")
        self.assertEqual(report["anchor_strategy_id"], 2)
        self.assertEqual(report["strategy_score_ranking_positions"], [0, 1, 2])
        self.assertEqual(fixed, {6: 1, 1: 1})

    def test_K_boundary_zero_fixes_all_common_only(self):
        case, model, features = fixture(K=0)
        for kind in ("mlopt", "li"):
            fixed, report = propose_strategy_fixings(model, case, features, kind)
            self.assertEqual(fixed, {6: 1, 1: 1, 9: 0, 4: 0})
            self.assertEqual(set(fixed), set(case.commitment_indices))
            self.assertEqual(report["dimension_diagnostics"]["common_nominal_free_dimension"], 0)

    def test_K_boundary_ambient_fixes_nothing(self):
        case, model, features = fixture(K=4)
        for kind in ("mlopt", "li"):
            fixed, report = propose_strategy_fixings(model, case, features, kind)
            self.assertEqual(fixed, {})
            self.assertEqual(report["dimension_diagnostics"]["common_nominal_free_dimension"], 4)

    def test_single_strategy_ties_use_original_column(self):
        case, model, features = fixture()
        model.selected = np.asarray([2])
        model.network = ConstantScoreNetwork([0.]).eval()
        fixed, report = propose_strategy_fixings(model, case, features, "li")
        self.assertEqual(fixed, {1: 1, 4: 0})
        self.assertEqual(report["common_weighted_agreement"], [1., 1., 1., 1.])

    def test_integer_bank_reordering_maps_common_columns_correctly(self):
        case, model, features = fixture()
        old_order = model.bank.integer_indices.copy()
        new_order = old_order[::-1]
        model.bank.integer_indices = new_order
        for strategy in model.bank.strategies:
            strategy.integer_values = strategy.integer_values[::-1]
        fixed, report = propose_strategy_fixings(model, case, features, "mlopt")
        self.assertEqual(fixed, {6: 1, 1: 1})
        self.assertEqual(report["anchor_common_values"], [1, 0, 1, 0])

    def test_knapsack_all_item_columns_are_the_common_block(self):
        _, model, features = fixture()
        case = SimpleNamespace(
            family="knapsack", affine={"rank": 2},
            model=SimpleNamespace(A=sparse.csr_matrix((0, 4)),
                                  integrality=np.ones(4),
                                  col_lower=np.zeros(4), col_upper=np.ones(4)),
        )
        model.bank.integer_indices = np.arange(4)
        old_common = np.asarray([1, 4, 6, 9])
        for strategy in model.bank.strategies:
            strategy.integer_values = strategy.integer_values[old_common]
        fixed, report = propose_strategy_fixings(model, case, features, "li")
        self.assertEqual(fixed, {2: 1, 0: 1})
        self.assertEqual(report["dimension_diagnostics"]["common_ambient_dimension"], 4)
        self.assertEqual(report["dimension_diagnostics"]["common_nominal_free_dimension"], 2)

    def test_bad_feature_or_standardization_rejected(self):
        case, model, features = fixture()
        for malformed in (np.asarray([1.]), np.asarray([np.nan, 0.]), np.asarray([np.inf, 0.])):
            with self.assertRaises(ValueError):
                propose_strategy_fixings(model, case, malformed, "mlopt")
        model.scale[0] = 0.
        with self.assertRaises(ValueError):
            propose_strategy_fixings(model, case, features, "li")

    def test_bad_scores_or_strategy_bank_rejected(self):
        case, model, features = fixture()
        model.network = ConstantScoreNetwork([1., np.nan, 2.]).eval()
        with self.assertRaises(ValueError):
            propose_strategy_fixings(model, case, features, "li")
        model.network = ConstantScoreNetwork([1., 2.]).eval()
        with self.assertRaises(ValueError):
            propose_strategy_fixings(model, case, features, "li")
        model.selected = np.asarray([2, 2, 1])
        with self.assertRaises(ValueError):
            propose_strategy_fixings(model, case, features, "mlopt")
        case, model, features = fixture()
        model.bank.strategies[2].integer_values = model.bank.strategies[2].integer_values.astype(float)
        model.bank.strategies[2].integer_values[1] = .5
        with self.assertRaises(ValueError):
            propose_strategy_fixings(model, case, features, "mlopt")

    def test_training_mode_or_wrong_kind_rejected(self):
        case, model, features = fixture()
        model.network.train()
        with self.assertRaises(ValueError):
            propose_strategy_fixings(model, case, features, "li")
        with self.assertRaises(ValueError):
            propose_strategy_fixings(model, case, features, "other")


if __name__ == "__main__":
    unittest.main(verbosity=2)
