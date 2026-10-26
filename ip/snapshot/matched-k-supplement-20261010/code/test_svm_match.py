"""Synthetic interface tests; no real case, confirmation data, or cloud work."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
from scipy import sparse

PROJECT = Path(__file__).resolve().parents[2]
FROZEN_CODE = PROJECT / "audit/remote-attempt6/poststop-export-20261010/extracted-main/ip-paper-baselines-v2-attempt6-20261009/code"
sys.path.insert(0, str(FROZEN_CODE))

from controls.xqa_baseline import Equality
from svm_match import (
    CoordinateSVMModel,
    fit_coordinate_svm,
    propose_capped_equalities,
    propose_coordinate_fixings,
)


def fake_case(D=4, K=2, *, common=None, n_columns=None, family="knapsack"):
    common = np.arange(D, dtype=np.int64) if common is None else np.array(common, dtype=np.int64)
    n_columns = D if n_columns is None else n_columns
    model = SimpleNamespace(A=sparse.csr_matrix((0, n_columns)),
        integrality=np.ones(n_columns, dtype=np.int32),
        col_lower=np.zeros(n_columns), col_upper=np.ones(n_columns))
    return SimpleNamespace(model=model, commitment_indices=common,
        affine={"rank": K}, family=family, seed=123)


def native(equations, scores, statuses=None, D=4):
    statuses = ["accepted_classifier"] * len(equations) if statuses is None else statuses
    records = [SimpleNamespace(equality=e, status=s, frequency=p,
        oof_precision=p) for e, p, s in zip(equations, scores, statuses)]
    return SimpleNamespace(n_binary_=D, n_features_in_=2, records_=records,
        predict_equalities=lambda feature: (list(equations), {"selected": len(equations)}))


class FixedProbability:
    def __init__(self, p, classes=(0, 1)):
        self.p = p
        self.classes_ = np.array(classes)

    def predict_proba(self, features):
        return np.array([[self.p if c == 1 else 1.-self.p for c in self.classes_]])


def fixed_probability_model(p, common=None):
    common = np.arange(len(p)) if common is None else np.array(common)
    return CoordinateSVMModel([FixedProbability(value) for value in p],
        np.full(len(p), np.nan), common, 2, 123, 120)


class CappedTests(unittest.TestCase):
    def test_confidence_and_native_order_ties(self):
        equations = [Equality("fix0", i, rhs=0) for i in range(4)]
        model = native(equations, [.7, .95, .95, .9])
        chosen, d = propose_capped_equalities(model, fake_case(K=2), [1., 2.])
        self.assertEqual([e.i for e in chosen], [1, 2])
        self.assertEqual(d["actual_rank"], 2)
        self.assertEqual(d["actual_common_dimension"], 2)
        self.assertTrue(d["exact_match"])
        self.assertGreaterEqual(d["variant_seconds"], d["selection_seconds"])

    def test_dependent_rows_do_not_count_towards_rank(self):
        equations = [Equality("fix0", 0, rhs=0), Equality("next_equal", 0, 1),
                     Equality("fix0", 1, rhs=0), Equality("fix1", 2, rhs=1)]
        chosen, d = propose_capped_equalities(native(equations, [1., .99, .98, .97]),
                                             fake_case(K=1), [0., 0.])
        self.assertEqual(d["selected_indices"], [0, 1, 3])
        self.assertEqual(d["dependent_candidate_indices"], [2])
        self.assertEqual(len(chosen), 3)

    def test_insufficient_candidates_are_reported(self):
        equations = [Equality("fix0", 0, rhs=0)]
        chosen, d = propose_capped_equalities(native(equations, [1.]), fake_case(K=1), [0., 0.])
        self.assertEqual(chosen, equations)
        self.assertFalse(d["exact_match"])
        self.assertEqual(d["status"], "budget_not_reachable")
        self.assertEqual(d["actual_common_dimension"], 3)
        self.assertEqual(d["added_synthetic_equalities"], 0)

    def test_unconditional_uses_frequency(self):
        e = [Equality("fix0", 0, rhs=0), Equality("fix1", 1, rhs=1)]
        model = native(e, [.98, .99], ["unconditional", "accepted_classifier"])
        model.records_[0].oof_precision = None
        _, d = propose_capped_equalities(model, fake_case(K=3), [0., 0.])
        self.assertEqual(d["selected_indices"], [1])

    def test_rejects_unaccepted_candidate(self):
        model = native([Equality("fix0", 0, rhs=0)], [.9], ["skip_oof_gate"])
        with self.assertRaises(ValueError):
            propose_capped_equalities(model, fake_case(), [0., 0.])

    def test_zero_budget_returns_no_equations(self):
        model = native([Equality("fix0", 0, rhs=0)], [1.])
        chosen, d = propose_capped_equalities(model, fake_case(K=4), [0., 0.])
        self.assertEqual(chosen, [])
        self.assertTrue(d["exact_match"])


class CoordinateTests(unittest.TestCase):
    def test_real_sklearn_training_and_constants(self):
        case = fake_case(K=2)
        rng = np.random.default_rng(12)
        case.train_features = rng.normal(size=(120, 2))
        case.train_solutions = np.column_stack([
            np.zeros(120), np.ones(120),
            case.train_features[:, 0] > 0., case.train_features[:, 1] > 0.,
        ]).astype(float)
        original_features = case.train_features.copy()
        original_solutions = case.train_solutions.copy()
        model, metadata = fit_coordinate_svm(case)
        fixed, d = propose_coordinate_fixings(model, case, [1., -1.])
        self.assertEqual(metadata["training_samples"], 120)
        self.assertEqual(metadata["fitted_classifier_count"], 2)
        self.assertEqual(metadata["constant_coordinate_count"], 2)
        self.assertFalse(metadata["original_Table1_gate_used"])
        self.assertEqual(fixed, {0: 0, 1: 1})
        self.assertEqual(d["actual_common_dimension"], 2)
        self.assertTrue(d["dimension_diagnostics"]["matches_expected_common_dimension"])
        for predictor in model.predictors[2:]:
            estimator = predictor.steps[-1][1]
            self.assertEqual(estimator.kernel, "linear")
            self.assertEqual(estimator.C, 1.)
            self.assertTrue(estimator.probability)
            self.assertEqual(estimator.random_state, case.seed)
        np.testing.assert_array_equal(case.train_features, original_features)
        np.testing.assert_array_equal(case.train_solutions, original_solutions)

    def test_class_mapping_map_tie_and_column_ties(self):
        case = fake_case(K=0)
        model = fixed_probability_model([.5, .9, .1, .5])
        model.predictors[1] = FixedProbability(.9, classes=(1, 0))
        fixed, d = propose_coordinate_fixings(model, case, [0., 0.])
        self.assertEqual(list(fixed), [1, 2, 0, 3])
        self.assertEqual(fixed, {1: 1, 2: 0, 0: 1, 3: 1})
        self.assertAlmostEqual(d["common_probabilities"][1], .9)

    def test_only_common_columns_and_original_index_ties(self):
        common = [3, 1, 6, 4]
        case = fake_case(K=2, common=common, n_columns=8, family="uc")
        model = fixed_probability_model([.8]*4, common=common)
        fixed, d = propose_coordinate_fixings(model, case, [0., 0.])
        self.assertEqual(list(fixed), [1, 3])
        self.assertEqual(d["dimension_diagnostics"]["other_integer_fixed_count"], 0)

    def test_zero_fixings_at_full_K(self):
        case = fake_case(K=4)
        fixed, d = propose_coordinate_fixings(fixed_probability_model([.5]*4), case, [0., 0.])
        self.assertEqual(fixed, {})
        self.assertEqual(d["actual_common_dimension"], 4)

    def test_bad_probability_and_column_mapping_rejected(self):
        case = fake_case(K=2)
        with self.assertRaises(ValueError):
            propose_coordinate_fixings(fixed_probability_model([.5, .5, 1.1, .5]), case, [0., 0.])
        model = fixed_probability_model([.5]*4, common=[0, 1, 2, 3])
        with self.assertRaises(ValueError):
            propose_coordinate_fixings(model, fake_case(common=[1, 0, 2, 3]), [0., 0.])

    def test_training_rejects_wrong_count_and_nonbinary(self):
        case = fake_case()
        case.train_features = np.zeros((119, 2))
        case.train_solutions = np.zeros((119, 4))
        with self.assertRaises(ValueError):
            fit_coordinate_svm(case)
        case.train_features = np.zeros((120, 2))
        case.train_solutions = np.zeros((120, 4))
        case.train_solutions[0, 0] = .3
        with self.assertRaises(ValueError):
            fit_coordinate_svm(case)

    def test_nonfinite_feature_rejected(self):
        with self.assertRaises(ValueError):
            propose_coordinate_fixings(fixed_probability_model([.5]*4), fake_case(), [np.nan, 0.])


if __name__ == "__main__":
    unittest.main()
