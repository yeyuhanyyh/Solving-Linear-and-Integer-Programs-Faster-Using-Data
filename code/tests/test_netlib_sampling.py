"""Regression checks for the SCAGR25/STAIR Gaussian sampling configuration.

Run with ``python -m unittest discover -s code/tests -p test_netlib_sampling.py``.
These checks generate three tiny objective samples per case. They neither read
Netlib data nor solve LPs. Basis tests mock MPS loading and coordinate-bound
solves; the sampler test mocks only nominal-prior selection.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
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
        self.assertEqual(suite.case_namespace(base, cases["stair"]).netlib_basis_mode, "bounded_topabs")

    def test_netlib_builder_constructs_all_requested_cost_directions(self):
        for name, (rank, _, _) in self.CASES.items():
            with self.subTest(case=name):
                case = self.cases()[name]
                args = suite.case_namespace(suite.default_namespace(), case)
                n_vars = rank + 3
                # Distinct magnitudes avoid ambiguity in coordinate ordering.
                # STAIR's highest-priority coordinate is unbounded and the next
                # is fixed. Neither may pad the requested varying cost rank.
                rewards = np.arange(1.0, n_vars + 1.0)
                inequalities = np.eye(n_vars)
                if name == "stair":
                    inequalities[:, -1] = 0.0
                equalities = np.zeros((2 if name == "stair" else 1, n_vars))
                equalities[0, 0] = 1.0
                equality_rhs = np.zeros(equalities.shape[0])
                feasible_anchor = np.zeros(n_vars)
                if name == "stair":
                    equalities[1, -2] = 1.0
                    equality_rhs[1] = 0.5
                    feasible_anchor[-2] = 0.5
                data = {
                    "reward_base": rewards,
                    "A_ineq": inequalities,
                    "b_ineq": np.ones(n_vars),
                    "A_eq": equalities,
                    "b_eq": equality_rhs,
                    "x0": feasible_anchor,
                }
                visited = []

                def coordinate_max(c, **kwargs):
                    c = np.asarray(c)
                    coordinate = int(np.argmax(np.abs(c)))
                    sign = float(c[coordinate])
                    self.assertIn(sign, [-1.0, 1.0])
                    np.testing.assert_array_equal(c, sign * np.eye(n_vars)[coordinate])
                    np.testing.assert_array_equal(kwargs["A_ub"], inequalities)
                    np.testing.assert_array_equal(kwargs["b_ub"], data["b_ineq"])
                    np.testing.assert_array_equal(kwargs["A_eq"], equalities)
                    np.testing.assert_array_equal(kwargs["b_eq"], data["b_eq"])
                    bounds = kwargs.get("bounds", (0, None))
                    if isinstance(bounds, tuple):
                        bounds = [bounds] * n_vars
                    self.assertEqual(list(bounds), [(0.0, None)] * n_vars)
                    visited.append((coordinate, sign))
                    if coordinate == n_vars - 1:
                        self.assertEqual(sign, -1.0)
                        return SimpleNamespace(success=False, status=3, fun=None, x=None, message="unbounded")
                    value = 0.5 if coordinate == n_vars - 2 else (1.0 if sign < 0 else 0.0)
                    return SimpleNamespace(success=True, status=0, fun=sign * value, x=None, message="optimal")

                with mock.patch.object(suite, "load_mps_fixed_feasible_data", return_value=data) as loader, \
                     mock.patch.object(suite, "linprog", side_effect=coordinate_max) as coordinate_solver:
                    problem = suite.make_netlib_problem(case, args, np.random.default_rng(17))
                loader.assert_called_once_with(case.mps_path)
                self.assertEqual(problem.U_reward.shape, (n_vars, rank))
                self.assertEqual(np.linalg.matrix_rank(problem.U_reward), rank)
                self.assertEqual(problem.metadata["netlib_basis_rank"], rank)
                if name == "stair":
                    self.assertEqual(problem.metadata["netlib_basis_mode"], "bounded_topabs")
                    expected_indices = np.arange(n_vars - 3, n_vars - rank - 3, -1)
                    expected_visits = [(n_vars - 1, -1.0), (n_vars - 2, -1.0), (n_vars - 2, 1.0)]
                    expected_visits += [(j, sign) for j in expected_indices.tolist() for sign in (-1.0, 1.0)]
                    self.assertEqual(visited, expected_visits)
                else:
                    self.assertEqual(problem.metadata["netlib_basis_mode"], "topabs")
                    expected_indices = np.arange(n_vars - 1, n_vars - rank - 1, -1)
                    coordinate_solver.assert_not_called()
                np.testing.assert_array_equal(problem.metadata["netlib_topabs_indices"], expected_indices)
                np.testing.assert_array_equal(np.argmax(np.abs(problem.U_reward), axis=0), expected_indices)
                np.testing.assert_allclose(problem.U_reward.T @ problem.U_reward, np.eye(rank), atol=1e-14)
                np.testing.assert_array_equal(problem.A_ineq, inequalities)
                np.testing.assert_array_equal(problem.b_ineq, data["b_ineq"])
                np.testing.assert_array_equal(problem.A_eq, equalities)
                np.testing.assert_array_equal(problem.b_eq, data["b_eq"])
                self.assertIsNone(problem.ub)

    def test_bounded_basis_rejects_failed_or_nonfinite_bound_checks(self):
        vec = np.array([4.0, -9.0, 3.0])
        args = (vec, 2, np.eye(3), np.ones(3), np.empty((0, 3)), np.empty(0))
        failures = [
            SimpleNamespace(success=False, status=2, fun=None, message="infeasible"),
            SimpleNamespace(success=False, status=4, fun=None, message="numerical failure"),
            SimpleNamespace(success=True, status=0, fun=np.inf, message="nonfinite optimum"),
            SimpleNamespace(success=True, status=0, fun=np.nan, message="nonfinite optimum"),
        ]
        for result in failures:
            for phase in ("maximum", "minimum"):
                with self.subTest(status=result.status, objective=result.fun, phase=phase):
                    responses = [result] if phase == "maximum" else [
                        SimpleNamespace(success=True, status=0, fun=-1.0, message="optimal"), result,
                    ]
                    with mock.patch.object(suite, "linprog", side_effect=responses) as solver:
                        with self.assertRaises(RuntimeError):
                            suite.bounded_topabs_coordinate_basis(*args)
                    self.assertEqual(solver.call_count, len(responses))

    def test_bounded_basis_rejects_insufficient_bounded_coordinates(self):
        def coordinate_max(c, **kwargs):
            coordinate = int(np.argmax(np.abs(c)))
            if coordinate == 0:
                value = -1.0 if c[coordinate] < 0 else 0.0
                return SimpleNamespace(success=True, status=0, fun=value, message="optimal")
            return SimpleNamespace(success=False, status=3, fun=None, message="unbounded")

        with mock.patch.object(suite, "linprog", side_effect=coordinate_max) as solver:
            with self.assertRaises(RuntimeError):
                suite.bounded_topabs_coordinate_basis(
                    np.array([4.0, -9.0, 3.0]), 2,
                    np.eye(3), np.ones(3), np.empty((0, 3)), np.empty(0),
                )
        self.assertEqual(solver.call_count, 4)

    def test_bounded_basis_does_not_count_numerically_fixed_coordinates(self):
        def bounds_for_coordinate(c, **kwargs):
            coordinate = int(np.argmax(np.abs(c)))
            sign = float(c[coordinate])
            if coordinate == 0:
                value = 1.0 + 1e-10 if sign < 0 else 1.0
            else:
                value = 1.0 if sign < 0 else 0.0
            return SimpleNamespace(success=True, status=0, fun=sign * value, message="optimal")

        with mock.patch.object(suite, "linprog", side_effect=bounds_for_coordinate) as solver:
            basis, indices = suite.bounded_topabs_coordinate_basis(
                np.array([9.0, 8.0]), 1,
                np.eye(2), np.ones(2), np.empty((0, 2)), np.empty(0),
            )
        np.testing.assert_array_equal(indices, [1])
        np.testing.assert_array_equal(basis, [[0.0], [1.0]])
        self.assertEqual(solver.call_count, 4)

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
