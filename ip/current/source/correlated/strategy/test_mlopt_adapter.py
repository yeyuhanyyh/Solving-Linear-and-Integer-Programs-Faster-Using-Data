"""Targeted regression checks for sign, native winner, and dimension matching."""
import unittest
import numpy as np

from . import mlopt_adapter as adapter


class StrategyAdapterChecks(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
        self.case = adapter.make_case(np.ones(3), 2,
            np.array([[3, 2, 1], [3, 1, 2], [1, 3, 2]]), self.points,
            case_id="matching-regression")
        bank = adapter.runtime.StrategyBank(self.case)
        # The highest-logit candidate is deliberately NOT the native best-value
        # decoded candidate. The regression must anchor to native top-ten output.
        self.model = adapter.runtime.MLOPTModel(bank, np.arange(3),
            [(np.zeros((3, 3), dtype=np.float32), np.array([3, 2, 1], dtype=np.float32))],
            np.zeros(3), np.ones(3), adapter.runtime.Config(device="cpu"))

    def test_vendored_source_hash(self):
        adapter._verify_runtime()

    def test_positive_profit_conversion_and_native_winner(self):
        out = adapter.predict_native(self.model, [1, 4, 3])
        self.assertTrue(out["valid"])
        self.assertEqual(out["objective"], 7)
        self.assertEqual(out["native_minimization_objective"], -7)
        self.assertEqual(out["strategy_id"], 2)
        self.assertIsNone(out["K"])
        self.assertEqual(out["solver_seconds"], 0)

    def test_match_uses_native_winner_instead_of_raw_top_one(self):
        fixed, out = adapter.propose_fixed(self.model, [1, 4, 3], 1)
        self.assertEqual(out["native_anchor_strategy_id"], 2)
        self.assertEqual(out["raw_top_one_strategy_id"], 0)
        self.assertEqual(fixed, {1: 1, 2: 1})
        self.assertEqual(out["actual_K"], 1)
        self.assertFalse(out["original_algorithm"])

    def test_exact_matching_all_boundaries_and_feasible_witness(self):
        for K in range(4):
            fixed, out = adapter.propose_fixed(self.model, [1, 4, 3], K)
            self.assertEqual(len(fixed), 3-K)
            self.assertEqual(out["actual_K"], K)
            witness = out["native_anchor"]
            self.assertTrue(all(witness[j] == v for j, v in fixed.items()))
            self.assertLessEqual(sum(witness), 2)

    def test_invalid_input_is_rejected(self):
        for K in (-1, 4, 1.5, True):
            with self.assertRaises(ValueError):
                adapter.propose_fixed(self.model, [1, 4, 3], K)
        with self.assertRaises(ValueError):
            adapter.predict_native(self.model, [1, np.nan, 3])
        with self.assertRaises(ValueError):
            adapter.make_case(np.ones(3), 1, np.ones((3, 3)), self.points, case_id="invalid")


if __name__ == "__main__":
    unittest.main()
