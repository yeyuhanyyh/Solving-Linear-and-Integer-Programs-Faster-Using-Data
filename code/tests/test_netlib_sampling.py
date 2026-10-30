"""Regression checks for the SCAGR25/STAIR Gaussian sampling configuration.

Run with ``python -m unittest discover -s code/tests -p test_netlib_sampling.py``.
These checks generate three tiny objective samples per case. They neither read
Netlib data nor solve LPs. The basis test mocks MPS loading; the sampler test
mocks only nominal-prior selection.
"""

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "code" / "core"
sys.path.insert(0, str(CORE))

import compare_fixedX_family_suite as sampling  # noqa: E402
import final_projection_figure_suite as suite  # noqa: E402


class RecordingNormalRng:
    """Return specified standard-normal draws and record the requested scales."""

    def __init__(self, standardized_draws):
        self.draws = [np.asarray(draw, dtype=float) for draw in standardized_draws]
        self.calls = []

    def normal(self, loc, scale, size):
        index = len(self.calls)
        if index >= len(self.draws):
            raise AssertionError("Sampler requested an unexpected extra draw")
        draw = self.draws[index]
        if tuple(np.atleast_1d(size)) != draw.shape:
            raise AssertionError(f"Unexpected normal-draw shape: {size}")
        scale = np.asarray(scale, dtype=float)
        self.calls.append((float(loc), scale.copy(), size))
        return float(loc) + scale * draw


class NetlibGaussianSamplingTests(unittest.TestCase):
    CASES = {
        "scagr25": (24, 0.92, 0.044178889833553156),
        "stair": (10, 0.02, 1.6287031980838047e-6),
    }

    def cases(self):
        return {case.slug: case for case in suite.netlib_cases(ROOT)}

    def test_case_configuration_selects_gaussian_even_if_base_differs(self):
        base = suite.default_namespace()
        base.sample_mode = "multiplicative_factor"
        base.factor_scale_frac = 0.13
        base.factor_decay = 0.41
        cases = self.cases()
        for name, (rank, radius_fraction, _) in self.CASES.items():
            with self.subTest(case=name):
                args = suite.case_namespace(base, cases[name])
                self.assertEqual(args.sample_mode, "factor_gaussian")
                self.assertEqual(args.factor_scale_frac, 0.70)
                self.assertEqual(args.factor_decay, 0.82)
                self.assertEqual(args.netlib_cost_rank, rank)
                self.assertEqual(args.sample_radius_frac, radius_fraction)
        self.assertEqual(base.sample_mode, "multiplicative_factor")
        self.assertEqual(base.factor_scale_frac, 0.13)
        self.assertEqual(base.factor_decay, 0.41)
        self.assertEqual(suite.case_namespace(base, cases["stair"]).netlib_basis_mode, "topabs")

    def test_netlib_builder_constructs_all_requested_cost_directions(self):
        for name, (rank, _, _) in self.CASES.items():
            with self.subTest(case=name):
                case = self.cases()[name]
                args = suite.case_namespace(suite.default_namespace(), case)
                n_vars = rank + 2
                # Distinct magnitudes avoid ambiguity in the top-absolute-cost
                # ordering. Only MPS loading is replaced; basis construction is real.
                rewards = np.arange(1.0, n_vars + 1.0)
                data = {
                    "reward_base": rewards,
                    "A_ineq": np.eye(n_vars),
                    "b_ineq": np.ones(n_vars),
                    "A_eq": np.empty((0, n_vars)),
                    "b_eq": np.empty(0),
                    "x0": np.zeros(n_vars),
                }
                with mock.patch.object(suite, "load_mps_fixed_feasible_data", return_value=data) as loader:
                    problem = suite.make_netlib_problem(case, args, np.random.default_rng(17))
                loader.assert_called_once_with(case.mps_path)
                self.assertEqual(problem.U_reward.shape, (n_vars, rank))
                self.assertEqual(np.linalg.matrix_rank(problem.U_reward), rank)
                self.assertEqual(problem.metadata["netlib_basis_rank"], rank)
                self.assertEqual(problem.metadata["netlib_basis_mode"], "topabs")
                expected_indices = np.arange(n_vars - 1, n_vars - rank - 1, -1)
                np.testing.assert_array_equal(problem.metadata["netlib_topabs_indices"], expected_indices)
                np.testing.assert_array_equal(np.argmax(np.abs(problem.U_reward), axis=0), expected_indices)
                np.testing.assert_allclose(problem.U_reward.T @ problem.U_reward, np.eye(rank), atol=1e-14)

    @staticmethod
    def toy_problem(rank):
        # A dense basis makes sign, coordinate, and slack-padding errors visible.
        n_vars = rank + 2
        basis, _ = np.linalg.qr(np.random.default_rng(73).normal(size=(n_vars, rank)))
        center = np.linspace(-3.0, 5.0, n_vars)
        return sampling.OriginalFixedXProblem(
            name="sampling_regression",
            reward_center=center,
            A_ineq=np.ones((1, n_vars)),
            b_ineq=np.array([float(n_vars)]),
            A_eq=np.empty((0, n_vars)),
            b_eq=np.empty(0),
            ub=np.ones(n_vars),
            U_reward=2.0 * basis,
            metadata={},
        )

    def make_bundle(self, name, radius, standardized_draws, **overrides):
        args = suite.case_namespace(suite.default_namespace(), self.cases()[name])
        args.n_train = args.n_val = args.n_test = 1
        for key, value in overrides.items():
            setattr(args, key, value)
        problem = self.toy_problem(args.netlib_cost_rank)
        rng = RecordingNormalRng(standardized_draws)
        rho = radius / args.sample_radius_frac
        # Computing a nominal basis normally solves an LP. Its value is fixed
        # here so the real standard-form conversion and sampling remain tested.
        with mock.patch.object(
            sampling,
            "choose_prior_for_problem",
            return_value=(rho, np.array([rho]), np.empty(0, dtype=int)),
        ) as choose_prior:
            bundle = sampling.make_fixed_x_bundle(problem, args, rng)
        choose_prior.assert_called_once()
        return bundle, rng

    def test_actual_bundle_has_requested_scales_and_radial_cost_perturbations(self):
        for name, (rank, _, radius) in self.CASES.items():
            with self.subTest(case=name):
                inside = np.zeros(rank)
                inside[0] = 0.25
                outside = np.full(rank, 2.0)
                negative = np.zeros(rank)
                negative[-1] = -0.5
                standardized = [inside, outside, negative]
                bundle, rng = self.make_bundle(name, radius, standardized)

                # Check the scale passed to the real Gaussian draw, before
                # clipping. Post-clipping empirical SD is not the target law.
                scales = 0.70 * radius * 0.82 ** np.arange(rank)
                self.assertGreater(float(scales.min()), 1e-12)
                self.assertEqual(len(rng.calls), 3)
                for loc, actual_scales, size in rng.calls:
                    self.assertEqual(loc, 0.0)
                    self.assertEqual(size, rank)
                    np.testing.assert_allclose(actual_scales, scales, rtol=1e-14, atol=0.0)

                theta = np.asarray(bundle.truth["theta"])
                raw = np.asarray(standardized) * scales
                self.assertLess(np.linalg.norm(raw[0]), radius)
                self.assertGreater(np.linalg.norm(raw[1]), radius)
                self.assertLess(np.linalg.norm(raw[2]), radius)
                np.testing.assert_allclose(theta[[0, 2]], raw[[0, 2]], rtol=1e-14, atol=0.0)
                np.testing.assert_allclose(
                    theta[1], raw[1] * (radius / np.linalg.norm(raw[1])),
                    rtol=1e-14, atol=0.0,
                )
                self.assertAlmostEqual(np.linalg.norm(theta[1]) / radius, 1.0, places=13)

                costs = np.vstack([bundle.Ctrain, bundle.Cval, bundle.Ctest])
                center = np.asarray(bundle.truth["prior_c0"])
                basis = np.asarray(bundle.truth["U_cost"])
                reward_basis = np.asarray(bundle.truth["U_reward"])
                n_slack = bundle.stdlp.n_slack
                self.assertGreater(n_slack, 0)
                np.testing.assert_allclose(reward_basis.T @ reward_basis, np.eye(rank), atol=1e-14)
                np.testing.assert_array_equal(basis[:n_slack], 0.0)
                np.testing.assert_array_equal(basis[n_slack:], -reward_basis)
                np.testing.assert_array_equal(costs[:, :n_slack], 0.0)
                np.testing.assert_allclose(costs - center, theta @ basis.T, rtol=1e-9, atol=5e-14)
                recovered_theta = (costs - center) @ basis
                np.testing.assert_allclose(recovered_theta, theta, rtol=1e-9, atol=5e-14)
                self.assertTrue(np.all(np.linalg.norm(costs - center, axis=1) <= radius + 5e-14))
                self.assertEqual(bundle.truth["sample_mode"].tolist(), ["factor_gaussian"])

                for instance, cost in zip(bundle.train + bundle.val + bundle.test, costs):
                    np.testing.assert_allclose(instance.c, -cost[n_slack:], rtol=0, atol=0)
                    self.assertIsNone(instance.full_x)
                    self.assertIsNone(instance.full_obj)

                # Old multiplicative-only options must not leak into this law.
                other, _ = self.make_bundle(
                    name, radius, standardized, center_noise_frac=0.9, mult_log_scale=7.0,
                )
                np.testing.assert_array_equal(other.Ctrain, bundle.Ctrain)
                np.testing.assert_array_equal(other.Cval, bundle.Cval)
                np.testing.assert_array_equal(other.Ctest, bundle.Ctest)


if __name__ == "__main__":
    unittest.main()
