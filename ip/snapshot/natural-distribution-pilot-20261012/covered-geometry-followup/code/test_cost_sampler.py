"""Independent sampler checks only; no optimization or model fitting."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr, ndtri

from cost_sampler import sample_diversity, sample_features, scenario_fields, select_scenarios


SCENARIOS = [f"{law}_{layout}"
             for layout in ("first4_whole_day", "first2_three_time_blocks")
             for law in ("log_normal_sigma1", "log_uniform_A5")]
PROTOCOL = {
    "version": "covered-geometry-followup-v1-20261012",
    "sampler_streams": {"train": 2026101207, "test": 2026101208},
    "scenarios": SCENARIOS,
}


class CoveredCostSamplerTests(unittest.TestCase):
    def test_four_scenarios_are_reproducible_positive_and_json_compatible(self):
        self.assertEqual(len(select_scenarios(PROTOCOL)), 4)
        for physical_seed in (401, 503, 607):
            for scenario in SCENARIOS:
                first, meta = sample_features(PROTOCOL, "train", physical_seed, 96, scenario)
                again, same_meta = sample_features(PROTOCOL, "train", physical_seed, 96,
                                                   scenario_fields(scenario))
                np.testing.assert_array_equal(first, again)
                self.assertEqual(meta, same_meta)
                self.assertEqual(first.shape, (96, 6))
                self.assertEqual(first.dtype, np.float64)
                self.assertTrue(np.isfinite(first).all())
                self.assertTrue(np.all(first > 0))
                for flag in ("clipping_applied", "truncation_applied", "resampling_applied"):
                    self.assertFalse(meta[flag])
                self.assertEqual(meta["sampler_stream"], 2026101207)
                self.assertEqual(len(meta["latent_uniform_sha256"]), 64)
                self.assertEqual(len(meta["feature_sha256"]), 64)
                json.dumps(meta, allow_nan=False)

    def test_common_quantiles_laws_layouts_and_training_prefixes(self):
        base, base_meta = sample_features(PROTOCOL, "train", 503, 96,
                                         "log_uniform_A5_first2_three_time_blocks")
        quantiles = (np.log(base) / math.log(5.0) + 1.0) / 2.0
        for scenario in SCENARIOS:
            draws, meta = sample_features(PROTOCOL, "train", 503, 96, scenario)
            short, _ = sample_features(PROTOCOL, "train", 503, 16, scenario)
            np.testing.assert_array_equal(short, draws[:16])
            self.assertEqual(meta["seed_sha256"], base_meta["seed_sha256"])
            self.assertEqual(meta["latent_uniform_sha256"], base_meta["latent_uniform_sha256"])
            ids = meta["varying_feature_ids"]
            if meta["distribution"] == "log_uniform":
                recovered = (np.log(draws[:, ids]) / math.log(5.0) + 1.0) / 2.0
            else:
                recovered = ndtr(np.log(draws[:, ids]))
            np.testing.assert_allclose(recovered, quantiles[:, ids], atol=3e-16, rtol=3e-15)
            fixed = meta["fixed_feature_ids"]
            if fixed:
                np.testing.assert_array_equal(draws[:, fixed], np.ones((96, len(fixed))))
        for law in ("log_normal_sigma1", "log_uniform_A5"):
            whole, _ = sample_features(PROTOCOL, "test", 607, 20, law + "_first4_whole_day")
            block, _ = sample_features(PROTOCOL, "test", 607, 20, law + "_first2_three_time_blocks")
            np.testing.assert_array_equal(whole[:, :4], block[:, :4])

    def test_layout_metadata_has_correct_generator_and_time_mapping(self):
        whole = scenario_fields("log_normal_sigma1_first4_whole_day")
        self.assertEqual(whole["varying_generator_ids"], [0, 1, 2, 3])
        self.assertEqual(whole["fixed_generator_ids"], [4, 5])
        self.assertEqual(whole["fixed_feature_ids"], [4, 5])
        self.assertEqual(whole["time_blocks"], [[0, 24]])
        for g, entry in enumerate(whole["feature_index_mapping"]):
            self.assertEqual(entry, {"feature_id": g, "generator_id": g,
                                     "period_start": 0, "period_stop_exclusive": 24})
        block = scenario_fields("log_uniform_A5_first2_three_time_blocks")
        self.assertEqual(block["varying_generator_ids"], [0, 1])
        self.assertEqual(block["fixed_generator_ids"], [2, 3, 4, 5])
        self.assertEqual(block["fixed_feature_ids"], [])
        self.assertEqual(block["time_blocks"], [[0, 8], [8, 16], [16, 24]])
        for feature, entry in enumerate(block["feature_index_mapping"]):
            self.assertEqual(entry, {"feature_id": feature, "generator_id": feature % 2,
                                     "period_start": 8 * (feature // 2),
                                     "period_stop_exclusive": 8 * (feature // 2 + 1)})

    def test_phase_seed_stream_and_version_separation(self):
        scenario = "log_normal_sigma1_first2_three_time_blocks"
        original, meta = sample_features(PROTOCOL, "train", 401, 96, scenario)
        changed_stream = deepcopy(PROTOCOL)
        changed_stream["sampler_streams"]["train"] += 1
        changed_version = dict(PROTOCOL, version=PROTOCOL["version"] + "-new")
        for protocol, phase, seed in ((PROTOCOL, "test", 401), (PROTOCOL, "train", 503),
                                      (changed_stream, "train", 401), (changed_version, "train", 401)):
            other, other_meta = sample_features(protocol, phase, seed, 96, scenario)
            self.assertNotEqual(other_meta["seed_sha256"], meta["seed_sha256"])
            self.assertFalse(np.array_equal(other, original))

    def test_confirmation_stream_does_not_change_training(self):
        confirmation = deepcopy(PROTOCOL)
        confirmation["sampler_streams"]["test"] = 2026101209
        scenario = "log_uniform_A5_first2_three_time_blocks"
        train, train_meta = sample_features(PROTOCOL, "train", 401, 96, scenario)
        confirmation_train, confirmation_train_meta = sample_features(confirmation, "train", 401, 96, scenario)
        np.testing.assert_array_equal(train, confirmation_train)
        self.assertEqual(train_meta, confirmation_train_meta)
        test, test_meta = sample_features(PROTOCOL, "test", 401, 20, scenario)
        fresh, fresh_meta = sample_features(confirmation, "test", 401, 80, scenario)
        self.assertFalse(np.array_equal(test, fresh[:20]))
        self.assertNotEqual(test_meta["seed_sha256"], fresh_meta["seed_sha256"])
        self.assertEqual(fresh_meta["sampler_stream"], 2026101209)

    def test_open_quantile_grid_excludes_both_endpoints_by_construction(self):
        class ExtremeGridRng:
            def integers(self, low, high, size, dtype):
                self_outer.assertEqual(low, 1)
                self_outer.assertEqual(high, 2**53)
                self_outer.assertEqual(dtype, np.int64)
                values = np.empty(size, dtype=dtype)
                values[0] = low
                values[1] = high - 1
                return values
        self_outer = self
        with patch("cost_sampler.np.random.Generator", return_value=ExtremeGridRng()):
            factors, metadata = sample_features(PROTOCOL, "test", 401, 2,
                                                "log_normal_sigma1_first2_three_time_blocks")
        expected = np.exp(ndtri(np.array([1.0 / 2**53, (2**53 - 1.0) / 2**53])))
        np.testing.assert_array_equal(factors[:, 0], expected)
        self.assertTrue(np.isfinite(factors).all())
        self.assertTrue(np.all(factors > 0))
        self.assertTrue(metadata["uniform_grid_is_discrete"])

    def test_lognormal_moments_and_nontruncated_support(self):
        draws, _ = sample_features(PROTOCOL, "test", 607, 40000,
                                   "log_normal_sigma1_first2_three_time_blocks")
        logs = np.log(draws)
        self.assertLess(abs(float(logs.mean())), 0.012)
        self.assertLess(abs(float(logs.var()) - 1), 0.02)
        self.assertLess(abs(float(np.median(draws)) - 1), 0.025)
        self.assertTrue(np.any(draws < 0.2))
        self.assertTrue(np.any(draws > 5))

    def test_loguniform_support_and_log_moments(self):
        draws, _ = sample_features(PROTOCOL, "test", 607, 40000,
                                   "log_uniform_A5_first2_three_time_blocks")
        logs = np.log(draws)
        self.assertTrue(np.all(draws >= 0.2))
        self.assertTrue(np.all(draws <= 5))
        self.assertLess(abs(float(logs.mean())), 0.012)
        self.assertLess(abs(float(logs.var()) - math.log(5)**2 / 3), 0.015)

    def test_unrepresentable_lognormal_fails_without_resampling(self):
        for pathological in (1000.0, -1000.0, np.inf, np.nan):
            with patch("cost_sampler.ndtri", side_effect=lambda u: np.full_like(u, pathological)) as inverse:
                with self.assertRaisesRegex(RuntimeError, "no resampling"):
                    sample_features(PROTOCOL, "train", 401, 16,
                                    "log_normal_sigma1_first4_whole_day")
                self.assertEqual(inverse.call_count, 1)

    def test_invalid_inputs_and_redundant_scientific_fields_fail(self):
        scenario = "log_uniform_A5_first4_whole_day"
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                sample_features(PROTOCOL, "train", 401, count, scenario)
        for phase in ("validation", "training", None):
            with self.assertRaises(ValueError):
                sample_features(PROTOCOL, phase, 401, 16, scenario)
        for bad in ("log_normal_sigma2_first4_whole_day",
                    {"scenario_id": scenario, "scale_A": 10},
                    {"scenario_id": scenario, "feature_layout": "first2_three_time_blocks"},
                    {"scenario_id": scenario, "varying_feature_ids": [0, 1]},
                    {"scenario_id": scenario, "sigma": 1},
                    {"scenario_id": "log_normal_sigma1_first4_whole_day", "scale_A": 5}):
            with self.assertRaises(ValueError):
                scenario_fields(bad)
        for bad_protocol in ({}, dict(PROTOCOL, sampler_streams={}),
                             dict(PROTOCOL, sampler_streams={"train": True}),
                             dict(PROTOCOL, sampler_streams={"train": -1})):
            with self.assertRaises(ValueError):
                sample_features(bad_protocol, "train", 401, 16, scenario)

    def test_select_scenarios_and_single_stream_api(self):
        selected = select_scenarios(PROTOCOL, [SCENARIOS[3], SCENARIOS[0]])
        self.assertEqual([s["scenario_id"] for s in selected], [SCENARIOS[3], SCENARIOS[0]])
        for bad in ([SCENARIOS[0], SCENARIOS[0]], ["unknown"], [], SCENARIOS[0], [1]):
            with self.assertRaises(ValueError):
                select_scenarios(PROTOCOL, bad)
        with self.assertRaises(ValueError):
            select_scenarios(dict(PROTOCOL, scenarios=[SCENARIOS[0], SCENARIOS[0]]))
        single = {"version": PROTOCOL["version"], "sampler_stream": 2026101207}
        first, meta = sample_features(PROTOCOL, "train", 401, 96, SCENARIOS[0])
        again, same_meta = sample_features(single, "train", 401, 96, SCENARIOS[0])
        np.testing.assert_array_equal(first, again)
        self.assertEqual(meta, same_meta)

    def test_diversity_retains_exact_duplicates_and_overlap_pairs(self):
        train = np.array([[1, 2, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1]], dtype=float)
        test = np.array([[1, 2, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1]], dtype=float)
        report = sample_diversity(train, test)
        self.assertEqual(report["training_count"], 3)
        self.assertEqual(report["training_duplicate_rows"], 1)
        self.assertEqual(report["test_duplicate_rows"], 1)
        self.assertEqual(report["shared_unique_full_vectors"], 1)
        self.assertEqual(report["test_rows_overlapping_training"], 2)
        self.assertEqual(report["train_test_overlap_pairs"], 4)
        self.assertTrue(report["iid_duplicates_retained"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
