"""Cost-map and physical-instance checks only; no optimization or fitting."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import portable_adapter as adapter
from uc_model import make_uc_base


LAYOUTS = ("first4_whole_day", "first2_three_time_blocks")
SEEDS = (401, 503, 607)
HERE = Path(__file__).resolve().parent


def physical_case(seed: int, layout: str | None = None) -> adapter.FrozenCase:
    base = make_uc_base(seed, n_units=6, n_times=24)
    metadata = dict(base["metadata"])
    if layout is not None:
        metadata["cost_feature_layout"] = layout
    model = adapter.LinearModel(
        A=base["A"], row_lower=base["row_lower"], row_upper=base["row_upper"],
        col_lower=base["col_lower"], col_upper=base["col_upper"],
        integrality=base["integrality"],
        variable_names=tuple(f"{kind}[{t},{g}]" for kind in ("u", "v", "w", "p")
                             for t in range(24) for g in range(6)),
        row_names=tuple(base["metadata"]["row_names"]))
    return adapter.FrozenCase(
        case_id=f"uc_seed{seed}", family="uc", seed=seed, amplitude=0.0,
        model=model, train_features=np.empty((0, 6)), train_costs=np.empty((0, 576)),
        train_solutions=np.empty((0, 576)), train_objectives=np.empty(0),
        commitment_indices=np.asarray(base["binary_indices"], dtype=np.int64),
        affine={}, original_base=base, metadata=metadata, source_hashes={},
        dataset_fingerprint="physical instance only; no training labels",
        source_directory=HERE)


class CoveredGeometryCostMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An accidental optimization call fails instead of silently running locally.
        cls.highs_guard = patch.object(adapter.highspy, "Highs", side_effect=AssertionError("No solver in these tests"))
        cls.solve_guard = patch.object(adapter, "solve", side_effect=AssertionError("No solves in these tests"))
        cls.highs_guard.start()
        cls.solve_guard.start()
        cls.addClassCleanup(cls.highs_guard.stop)
        cls.addClassCleanup(cls.solve_guard.stop)

    def test_fresh_physical_instances_have_original_dimensions(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                case = physical_case(seed)
                self.assertEqual(case.model.A.shape, (1176, 576))
                self.assertEqual(case.model.n_variables, 576)
                np.testing.assert_array_equal(case.model.integer_indices, np.arange(432))
                np.testing.assert_array_equal(case.commitment_indices, np.arange(144))
                self.assertEqual(case.original_base["n_units"], 6)
                self.assertEqual(case.original_base["n_times"], 24)
                self.assertEqual(case.model.variable_names[143], "u[23,5]")
                self.assertEqual(case.model.variable_names[144], "v[0,0]")
                self.assertEqual(case.model.variable_names[288], "w[0,0]")
                self.assertEqual(case.model.variable_names[432], "p[0,0]")

    def test_first_four_generator_map_for_every_cost_component(self):
        features = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
        for seed in SEEDS:
            case = physical_case(seed, "first4_whole_day")
            result = adapter.costs_from_features(case, features)
            for component in range(4):
                for time in range(24):
                    for generator in range(6):
                        j = component * 144 + time * 6 + generator
                        rho = features[generator] if generator < 4 else 1.0
                        self.assertEqual(result[j], case.original_base["base_cost"][j] * rho)

    def test_three_time_blocks_map_for_every_cost_component(self):
        features = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
        expected_blocks = ((0, 8, (2.0, 3.0)), (8, 16, (5.0, 7.0)), (16, 24, (11.0, 13.0)))
        for seed in SEEDS:
            case = physical_case(seed, "first2_three_time_blocks")
            result = adapter.costs_from_features(case, features)
            for component in range(4):
                for start, stop, pair in expected_blocks:
                    for time in range(start, stop):
                        for generator in range(6):
                            j = component * 144 + time * 6 + generator
                            rho = pair[generator] if generator < 2 else 1.0
                            self.assertEqual(result[j], case.original_base["base_cost"][j] * rho)

    def test_batched_and_single_sample_maps_agree(self):
        features = np.array([[2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
                             [0.2, 0.3, 0.5, 0.7, 1.1, 1.3]])
        for seed in SEEDS:
            for layout in LAYOUTS:
                with self.subTest(seed=seed, layout=layout):
                    case = physical_case(seed, layout)
                    batched = adapter.costs_from_features(case, features)
                    self.assertEqual(batched.shape, (2, 576))
                    for i in range(2):
                        single = adapter.costs_from_features(case, features[i])
                        self.assertEqual(single.shape, (576,))
                        np.testing.assert_array_equal(batched[i], single)

    def test_all_one_multipliers_reproduce_original_and_legacy_costs(self):
        for seed in SEEDS:
            legacy = physical_case(seed)
            original_cost = legacy.original_base["base_cost"]
            legacy_cost = adapter.costs_from_features(legacy, np.ones(6))
            np.testing.assert_array_equal(legacy_cost, original_cost)
            for layout in LAYOUTS:
                case = physical_case(seed, layout)
                np.testing.assert_array_equal(adapter.costs_from_features(case, np.ones(6)), original_cost)
                np.testing.assert_array_equal(adapter.costs_from_features(case, np.ones((2, 6))),
                                              np.vstack([legacy_cost, legacy_cost]))

    def test_legacy_generator_map_is_unchanged(self):
        features = np.array([[2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
                             [0.2, 0.3, 0.5, 0.7, 1.1, 1.3]])
        for seed in SEEDS:
            case = physical_case(seed)
            expected = np.stack([np.tile(row, 96) * case.original_base["base_cost"] for row in features])
            np.testing.assert_array_equal(adapter.costs_from_features(case, features), expected)
            np.testing.assert_array_equal(adapter.costs_from_features(case, features[0]), expected[0])

    def test_cost_mapping_does_not_change_constraints_or_domains(self):
        for layout in LAYOUTS:
            case = physical_case(401, layout)
            model = case.model
            names = ("row_lower", "row_upper", "col_lower", "col_upper", "integrality")
            before = {name: getattr(model, name).copy() for name in names}
            data, indices, indptr = model.A.data.copy(), model.A.indices.copy(), model.A.indptr.copy()
            base_cost = case.original_base["base_cost"].copy()
            features = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
            features_before = features.copy()
            adapter.costs_from_features(case, features)
            for name in names:
                np.testing.assert_array_equal(getattr(model, name), before[name])
            np.testing.assert_array_equal(model.A.data, data)
            np.testing.assert_array_equal(model.A.indices, indices)
            np.testing.assert_array_equal(model.A.indptr, indptr)
            np.testing.assert_array_equal(case.original_base["base_cost"], base_cost)
            np.testing.assert_array_equal(features, features_before)

    def test_new_layouts_reject_wrong_feature_shapes(self):
        for layout in LAYOUTS:
            case = physical_case(401, layout)
            for bad in (1.0, np.ones(5), np.ones(7), np.ones((2, 5)), np.ones((2, 7)), np.ones((1, 1, 6))):
                with self.subTest(layout=layout, shape=np.asarray(bad).shape):
                    with self.assertRaisesRegex(ValueError, "six features"):
                        adapter.costs_from_features(case, bad)

    def test_new_layouts_reject_nonfinite_and_nonpositive_features(self):
        for layout in LAYOUTS:
            case = physical_case(401, layout)
            for column in range(6):
                for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
                    bad = np.ones(6)
                    bad[column] = value
                    with self.subTest(layout=layout, column=column, value=value):
                        with self.assertRaisesRegex(ValueError, "finite and strictly positive"):
                            adapter.costs_from_features(case, bad)

    def test_unknown_layout_and_wrong_physical_dimensions_are_rejected(self):
        for layout in ("", "generator_major_blocks", "first2_two_time_blocks"):
            with self.assertRaisesRegex(ValueError, "Unknown UC cost feature layout"):
                adapter.costs_from_features(physical_case(401, layout), np.ones(6))
        case = physical_case(401, "first4_whole_day")
        for key, value in (("n_units", 4), ("n_times", 8)):
            altered = replace(case, original_base=dict(case.original_base, **{key: value}))
            with self.assertRaisesRegex(ValueError, "six generators and 24 periods"):
                adapter.costs_from_features(altered, np.ones(6))

    def test_positive_extreme_values_are_not_clipped(self):
        features = np.array([1e-8, 1e8, 1e-6, 1e6, 1e-4, 1e4])
        for layout in LAYOUTS:
            case = physical_case(401, layout)
            result = adapter.costs_from_features(case, features)
            self.assertEqual(result[0], case.original_base["base_cost"][0] * features[0])
            self.assertEqual(result[1], case.original_base["base_cost"][1] * features[1])


if __name__ == "__main__":
    unittest.main()
