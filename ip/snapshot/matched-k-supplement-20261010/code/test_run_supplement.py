"""Real HiGHS integration checks on a four-variable synthetic knapsack only.

No setup(), real dataset/model loader, checkpoint, confirmation cost, or cloud
helper is called. Predictors are deterministic toy fixtures, while both solver
paths and original/restriction validation use the frozen production interfaces.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import numpy as np
from scipy import sparse

PROJECT = Path(__file__).resolve().parents[2]
PARENT_CODE = PROJECT / "audit/remote-attempt6/poststop-export-20261010/extracted-main/ip-paper-baselines-v2-attempt6-20261009/code"
sys.path.insert(0, str(PARENT_CODE))
sys.path.insert(0, str(PARENT_CODE / "controls"))

import portable_adapter as adapter
import experiment_core
from xqa_baseline import Equality
import run_supplement
from svm_match import CoordinateSVMModel


class ProbabilityPredictor:
    classes_ = np.array([0, 1])

    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, features):
        return np.array([[1.-self.p, self.p]])


def coordinate_model(probabilities):
    return CoordinateSVMModel(
        [ProbabilityPredictor(p) for p in probabilities], np.full(4, np.nan),
        np.arange(4, dtype=np.int64), 2, 73, 120,
    )


def native_svm(equations, confidences):
    records = [SimpleNamespace(equality=e, status="accepted_classifier",
        frequency=.5, oof_precision=c) for e, c in zip(equations, confidences)]
    return SimpleNamespace(n_binary_=4, n_features_in_=2, records_=records,
        predict_equalities=lambda features: (list(equations), {"selected": len(equations)}))


def mlopt_model():
    def forbidden_decoder(*args, **kwargs):
        raise AssertionError("Matched variant must not invoke the native decoder")
    strategies = [SimpleNamespace(integer_values=np.array(y), representative_training_index=i)
                  for i, y in enumerate(([1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]))]
    return SimpleNamespace(
        mean=np.zeros(2), scale=np.ones(2),
        weights=[(np.zeros((3, 2)), np.array([3., 1., 0.]))],
        selected=np.arange(3), config=SimpleNamespace(n_best=1),
        bank=SimpleNamespace(integer_indices=np.arange(4), strategies=strategies),
        predict=forbidden_decoder,
    )


def case_and_cost(K=2):
    model = adapter.LinearModel(
        A=sparse.csr_matrix([[1., 1., 1., 1.]]),
        row_lower=np.array([-np.inf]), row_upper=np.array([2.]),
        col_lower=np.zeros(4), col_upper=np.ones(4),
        integrality=np.ones(4, dtype=np.int32),
    )
    base = dict(A=model.A, row_lower=model.row_lower, row_upper=model.row_upper,
        col_lower=model.col_lower, col_upper=model.col_upper, integrality=model.integrality,
        binary_indices=np.arange(4), n_times=1, n_units=4)
    case = SimpleNamespace(model=model, commitment_indices=np.arange(4),
        affine={"rank": K}, original_base=base, family="knapsack", seed=73)
    return case, np.array([-5., -4., -3., -2.])


def enumerated_restricted_optimum(case, cost, record):
    """Independent 16-point reference, computed only after runner prediction."""
    feasible = []
    for values in product((0, 1), repeat=4):
        x = np.asarray(values, dtype=float)
        if not adapter.validate_solution(case.model, x)["valid"]:
            continue
        if any(abs(float(x[int(i)])-value) > 1e-9 for i, value in (record["fixed"] or {}).items()):
            continue
        satisfied = True
        for e in record["equalities"]:
            equality = Equality(e["kind"], e["i"], e["j"], e["rhs"])
            row, rhs = equality.as_row(4)
            if abs(row @ x-rhs) > 1e-9:
                satisfied = False
                break
        if satisfied:
            feasible.append(float(cost @ x))
    return min(feasible) if feasible else None


class NewMethodIntegrationTests(unittest.TestCase):
    def execute(self, method, *, K=2, svm=None, coordinate=None, strategy=None):
        case, cost = case_and_cost(K)
        models = {}
        if svm is not None:
            models["svm_native"] = svm
        if strategy is not None:
            models["mlopt_native"] = strategy
        result = run_supplement.new_method(method, models, coordinate, case, cost,
            np.array([.2, -.4]), 73, adapter, {})
        return case, cost, result

    def check_success(self, case, cost, record, expected_dimension):
        self.assertTrue(record["valid"])
        self.assertTrue(record["original_feasible"])
        self.assertTrue(record["restriction_check"]["valid"])
        self.assertTrue(record["solved_model_certified"])
        self.assertFalse(record["original_global_solver_certified"])
        self.assertFalse(record["full_fallback_used"])
        self.assertFalse(record["method_wall_budget_exceeded"])
        self.assertIsNone(record["physics"])
        self.assertGreaterEqual(record["prediction_hint_seconds"], 0.)
        self.assertGreaterEqual(record["seconds"], record["prediction_hint_seconds"])
        self.assertLess(record["seconds"], 30.)
        self.assertLessEqual(record["restriction_check"]["residual"], 1e-6)
        self.assertAlmostEqual(record["objective"], enumerated_restricted_optimum(case, cost, record))
        self.assertTrue(adapter.validate_solution(case.model, np.array(record["solution"]))["valid"])
        dimensions = record["restriction_check"]["dimension_diagnostics"]
        dimension = (dimensions["affine_dimension"] if record["fixed"] is None
                     else dimensions["common_nominal_free_dimension"])
        self.assertEqual(dimension, expected_dimension)

    def test_coordinate_matched_real_submip(self):
        case, cost, r = self.execute("svm_coordinate_matched",
            coordinate=coordinate_model([.99, .98, .5, .5]))
        self.check_success(case, cost, r, 2)
        self.assertEqual(r["fixed"], {"0": 1, "1": 1})
        self.assertAlmostEqual(r["objective"], -9.)
        self.assertEqual(r["raw_method_result"]["n_fixed"], 2)
        self.assertEqual(r["raw_method_result"]["added_rows"], 0)
        self.assertTrue(r["supplement_diagnostic"]["exact_match"])

    def test_coordinate_local_optimum_is_not_original_certificate(self):
        case, cost, r = self.execute("svm_coordinate_matched",
            coordinate=coordinate_model([.99, .02, .5, .5]))
        self.check_success(case, cost, r, 2)
        self.assertEqual(r["fixed"], {"0": 1, "1": 0})
        self.assertAlmostEqual(r["objective"], -8.)
        self.assertFalse(r["original_global_solver_certified"])

    def test_capped_equalities_real_core_solve(self):
        svm = native_svm([Equality("fix1", 0, rhs=1), Equality("fix1", 1, rhs=1),
                          Equality("fix0", 2, rhs=0)], [.97, .99, .98])
        case, cost, r = self.execute("svm_rank_capped", svm=svm)
        self.check_success(case, cost, r, 2)
        self.assertEqual([e["i"] for e in r["equalities"]], [1, 2])
        self.assertEqual(r["raw_method_result"]["added_equalities"], 2)
        self.assertTrue(r["supplement_diagnostic"]["exact_match"])

    def test_capped_insufficient_rank_preserved(self):
        svm = native_svm([Equality("fix0", 2, rhs=0)], [.99])
        case, cost, r = self.execute("svm_rank_capped", K=1, svm=svm)
        self.check_success(case, cost, r, 3)
        self.assertEqual(r["supplement_diagnostic"]["target_common_dimension"], 1)
        self.assertEqual(r["supplement_diagnostic"]["actual_common_dimension"], 3)
        self.assertEqual(r["supplement_diagnostic"]["status"], "budget_not_reachable")
        self.assertFalse(r["supplement_diagnostic"]["exact_match"])

    def test_capped_no_candidates_reports_no_compression(self):
        case, cost, r = self.execute("svm_rank_capped", svm=native_svm([], []))
        self.check_success(case, cost, r, 4)
        self.assertEqual(r["equalities"], [])
        self.assertEqual(r["raw_method_result"]["added_equalities"], 0)
        self.assertFalse(r["supplement_diagnostic"]["exact_match"])
        self.assertEqual(r["supplement_diagnostic"]["actual_common_dimension"], 4)

    def test_capped_full_K_zero_rank_budget(self):
        svm = native_svm([Equality("fix0", 2, rhs=0)], [.99])
        case, cost, r = self.execute("svm_rank_capped", K=4, svm=svm)
        self.check_success(case, cost, r, 4)
        self.assertEqual(r["equalities"], [])
        self.assertTrue(r["supplement_diagnostic"]["exact_match"])

    def test_mlopt_guided_real_submip(self):
        case, cost, r = self.execute("mlopt_guided_matched", strategy=mlopt_model())
        self.check_success(case, cost, r, 2)
        self.assertEqual(r["fixed"], {"0": 1, "3": 0})
        self.assertAlmostEqual(r["objective"], -9.)
        self.assertEqual(r["raw_method_result"]["n_fixed"], 2)
        d = r["supplement_diagnostic"]
        self.assertFalse(d["decoder_called"])
        self.assertFalse(d["active_constraints_used"])
        self.assertFalse(d["native_candidate_limit_used"])
        self.assertEqual(len(d["all_retained_strategy_scores"]), 3)
        self.assertEqual(d["planned_submip_count"], 1)

    def test_infeasible_coordinate_restriction_returns_failure(self):
        case, cost, r = self.execute("svm_coordinate_matched", K=0,
            coordinate=coordinate_model([.99, .99, .99, .01]))
        self.assertIsNone(enumerated_restricted_optimum(case, cost, r))
        self.assertIsNone(r["solution"])
        self.assertIsNone(r["objective"])
        self.assertFalse(r["valid"])
        self.assertFalse(r["solved_model_certified"])
        self.assertFalse(r["original_global_solver_certified"])
        self.assertFalse(r["full_fallback_used"])
        self.assertEqual(r["restriction_check"]["dimension_diagnostics"]["common_nominal_free_dimension"], 0)
        self.assertEqual(r["raw_method_result"]["status"], "Infeasible")

    def test_outer_elapsed_budget_marks_result_invalid(self):
        case, cost = case_and_cost()
        with patch.object(run_supplement.time, "perf_counter", side_effect=[0., .1, 31.]):
            r = run_supplement.new_method("svm_coordinate_matched", {},
                coordinate_model([.99, .98, .5, .5]), case, cost,
                np.array([.2, -.4]), 73, adapter, {})
        self.assertTrue(r["original_feasible"])
        self.assertTrue(r["method_wall_budget_exceeded"])
        self.assertFalse(r["valid"])
        self.assertEqual(r["seconds"], 31.)
        self.assertFalse(r["full_fallback_used"])

    def test_unknown_supplemental_method_is_rejected(self):
        case, cost = case_and_cost()
        with self.assertRaises(ValueError):
            run_supplement.new_method("unknown_method", {}, None, case, cost,
                np.array([.2, -.4]), 73, adapter, {})


if __name__ == "__main__":
    unittest.main()
