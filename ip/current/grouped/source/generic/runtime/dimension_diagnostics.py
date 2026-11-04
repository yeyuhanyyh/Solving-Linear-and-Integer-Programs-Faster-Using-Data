"""Common-coordinate restriction dimensions for the revised IP experiment.

Matching space: UC commitment u (144 original coordinates), or knapsack x
(200 coordinates). Ours' frozen K is the affine-hull rank in this space.
ND-matchedK is a labelled MAP coordinate-fixing variant, not native Neural
Diving: fix exactly D-K common coordinates, preserving v/w/p for UC.

No function reads test costs/solutions, solves a model, trains a learner, or
reports a solver's presolved size. Native MLOPT/Li complete strategies have no
comparable tunable K; their finite candidate bank must not be renamed an affine
search space just because its point cloud has a particular affine rank.

Paper definitions checked 2026-10-09:
ND, Section 6.2: https://arxiv.org/html/2012.13349v3
MLOPT, Section 3: https://arxiv.org/html/1907.02206v4
Li, Equations 4--7: https://arxiv.org/html/2501.00307v1
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse


def common_coordinate_indices(case: Any) -> np.ndarray:
    """Original u columns for UC, or all x columns for knapsack."""
    n = int(case.model.A.shape[1])
    if hasattr(case, "commitment_indices"):
        indices = np.asarray(case.commitment_indices, dtype=np.int64)
    elif case.family == "uc":
        # Portable cases may omit the alias; the frozen original variable order
        # is four equal blocks [u,v,w,p]. No 432-coordinate substitution is made.
        if n % 4:
            raise ValueError("UC original variables are not four equal blocks")
        indices = np.arange(n // 4, dtype=np.int64)
    elif case.family == "knapsack":
        indices = np.arange(n, dtype=np.int64)
    else:
        raise ValueError("Unknown common-coordinate space")
    if indices.ndim != 1 or len(indices) != len(np.unique(indices)) or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("Invalid original common-coordinate indices")
    if np.any(np.asarray(case.model.integrality)[indices] == 0):
        raise ValueError("Common coordinates must retain their integer domains")
    return indices.copy()


def target_common_dimension(case: Any) -> int:
    """Use the frozen training-derived Ours rank, without test-driven selection."""
    if not hasattr(case, "affine") or "rank" not in case.affine:
        raise ValueError("Case has no frozen Ours affine rank")
    value = case.affine["rank"]
    K = int(value)
    if K != value or not 0 <= K <= len(common_coordinate_indices(case)):
        raise ValueError("Invalid frozen Ours K")
    return K


def _checked_target(case: Any, target_K: int | None) -> tuple[np.ndarray, int]:
    common = common_coordinate_indices(case)
    K = target_common_dimension(case) if target_K is None else int(target_K)
    if target_K is not None and K != target_K:
        raise ValueError("Target dimension must be an integer")
    if not 0 <= K <= len(common):
        raise ValueError("Target dimension outside common coordinate space")
    return common, K


def _integer_indices(model: Any) -> np.ndarray:
    return np.flatnonzero(np.asarray(model.integrality) != 0)


def _common_array(case: Any, values: np.ndarray, name: str) -> np.ndarray:
    """Accept vectors over common, all original, or all integer columns."""
    common = common_coordinate_indices(case)
    values = np.asarray(values, dtype=float)
    if values.shape == (len(common),):
        out = values
    elif values.shape == (case.model.A.shape[1],):
        out = values[common]
    else:
        integer = _integer_indices(case.model)
        if values.shape != (len(integer),):
            raise ValueError(f"{name} must be common/full/integer-coordinate vector")
        location = {int(j): i for i, j in enumerate(integer)}
        out = values[[location[int(j)] for j in common]]
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return out.copy()


def diagnose_fixings(case: Any, fixed: Mapping[int, float], expected_K: int | None = None) -> dict[str, Any]:
    """Exact rank/count of explicit coordinate fixings in the common space.

    v/w fixings are recorded separately; they can propagate through UC
    constraints, so D-minus-direct-u-fixings is not called the actual feasible
    dimension for a native assignment containing such fixings. Inequalities,
    integrality and presolve can further restrict a feasible set.
    """
    common = common_coordinate_indices(case)
    common_set = set(map(int, common))
    integer_set = set(map(int, _integer_indices(case.model)))
    n = case.model.A.shape[1]
    normalized = {}
    for j, value in fixed.items():
        if not isinstance(j, (int, np.integer)) or not 0 <= int(j) < n:
            raise ValueError("Fixing key is not an original column index")
        j = int(j)
        if j not in integer_set or not np.isfinite(value) or abs(float(value) - round(float(value))) > 1e-9:
            raise ValueError("Fixings must assign finite integer values to integer variables")
        value = int(round(float(value)))
        if value < case.model.col_lower[j] - 1e-9 or value > case.model.col_upper[j] + 1e-9:
            raise ValueError("Fixed value violates an original bound")
        normalized[j] = value
    common_fixed = sorted(common_set & normalized.keys())
    other_fixed = sorted(normalized.keys() - common_set)
    K = len(common) - len(common_fixed)
    expected = None if expected_K is None else int(expected_K)
    if expected_K is not None and (expected != expected_K or not 0 <= expected <= len(common)):
        raise ValueError("Expected dimension outside common coordinate space")
    return dict(common_space="UC commitment u" if case.family == "uc" else "knapsack item x",
                common_ambient_dimension=len(common), common_fixed_count=len(common_fixed),
                common_fixing_rank=len(common_fixed), common_nominal_free_dimension=K,
                fixed_integer_count=len(normalized), other_integer_fixed_count=len(other_fixed),
                total_original_variables=int(n), total_original_integer_variables=len(integer_set),
                expected_common_dimension=expected,
                matches_expected_common_dimension=(None if expected is None else K == expected and not other_fixed),
                common_fixed_indices=common_fixed, other_integer_fixed_indices=other_fixed,
                metric_scope="affine space imposed by explicit common-coordinate equalities",
                actual_integer_feasible_set_dimension=None,
                solver_presolved_variables=None, solver_presolved_integer_variables=None,
                caveat=("Additional native integer fixings may propagate through original constraints; nominal common count is not a complete dimension comparison."
                        if other_fixed else "Original constraints/integrality and solver presolve may further reduce the actual feasible set; those dimensions are not inferred."))


def assert_matched_fixings(case: Any, fixed: Mapping[int, float], target_K: int | None = None) -> dict[str, Any]:
    """Require exactly D-K distinct common-coordinate fixings and no others."""
    _, K = _checked_target(case, target_K)
    metrics = diagnose_fixings(case, fixed, expected_K=K)
    if metrics["other_integer_fixed_count"]:
        raise ValueError("Matched variant may fix only common u/x coordinates")
    if not metrics["matches_expected_common_dimension"]:
        raise ValueError(f"Dimension mismatch: expected K={K}, got {metrics['common_nominal_free_dimension']}")
    return metrics


def matched_map_fixings(case: Any, probabilities: np.ndarray, target_K: int | None = None,
                        confidence_scores: np.ndarray | None = None) -> dict[int, int]:
    """Labelled matched-K variant: highest-confidence common-coordinate MAP.

    Default confidence is max(p,1-p). Values are MAP (p>=1/2 gives one).
    Equal confidence is resolved by original column index. Optional scores can
    use a predeclared learned acceptance head; changing that rule is a labelled
    variant, not an undisclosed native-algorithm replacement.
    """
    common, K = _checked_target(case, target_K)
    p = _common_array(case, probabilities, "probabilities")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilities must lie in [0,1]")
    confidence = np.maximum(p, 1 - p) if confidence_scores is None else _common_array(case, confidence_scores, "confidence")
    order = np.lexsort((common, -confidence))
    selected = order[:len(common) - K]
    fixed = {int(common[j]): int(p[j] >= .5) for j in selected}
    assert_matched_fixings(case, fixed, K)
    return fixed


class _RationalRowBasis:
    """Sparse exact elimination for candidate x_i=value/x_i=x_j equations."""
    def __init__(self, n_columns: int):
        self.n_columns = int(n_columns)
        self.rows: dict[int, tuple[dict[int, Fraction], Fraction]] = {}

    @property
    def rank(self) -> int:
        return len(self.rows)

    @staticmethod
    def _fraction(value: float) -> Fraction:
        if not np.isfinite(value):
            raise ValueError("Equation coefficients and RHS must be finite")
        # Exact for the integer ±1/0 candidate rows used by the baseline. For
        # general decimals this is explicitly the rational decimal input, not
        # an assertion that rounded numerical affine normals are exact.
        return Fraction(str(float(value)))

    def add(self, row: np.ndarray, rhs: float) -> str:
        row = np.asarray(row, dtype=float)
        if row.shape != (self.n_columns,):
            raise ValueError("Candidate equality has the wrong common-space shape")
        coefficients = {int(j): self._fraction(row[j]) for j in np.flatnonzero(row)}
        value = self._fraction(rhs)
        for pivot in sorted(self.rows):
            if pivot not in coefficients:
                continue
            scale = coefficients[pivot]
            basis, basis_rhs = self.rows[pivot]
            for j, a in basis.items():
                new = coefficients.get(j, Fraction(0)) - scale * a
                if new:
                    coefficients[j] = new
                else:
                    coefficients.pop(j, None)
            value -= scale * basis_rhs
        if not coefficients:
            return "dependent" if value == 0 else "inconsistent"
        pivot = min(coefficients)
        scale = coefficients[pivot]
        self.rows[pivot] = ({j: a / scale for j, a in coefficients.items()}, value / scale)
        return "independent"


def _candidate_row(candidate: Any, D: int) -> tuple[np.ndarray, float]:
    if hasattr(candidate, "as_row"):
        row, rhs = candidate.as_row(D)
    elif isinstance(candidate, Mapping):
        row, rhs = candidate["row"], candidate["rhs"]
    else:
        row, rhs = candidate
    return np.asarray(row, dtype=float), float(rhs)


def select_equalities_by_rank(candidates: Sequence[Any], ambient_dimension: int,
                              target_rank: int, confidence_scores: Sequence[float] | None = None) -> dict[str, Any]:
    """Optional SVM budget cap, selecting consistent independent candidates.

    It never invents additional equations to meet a target. If too few accepted
    independent candidates exist, status is budget_not_reachable and the actual
    imposed dimension exceeds the requested K. This is not native SVM.
    """
    D, budget = int(ambient_dimension), int(target_rank)
    if D != ambient_dimension or budget != target_rank or not 0 <= budget <= D:
        raise ValueError("Invalid common dimension or equality-rank budget")
    scores = np.zeros(len(candidates)) if confidence_scores is None else np.asarray(confidence_scores, dtype=float)
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
        raise ValueError("Candidate confidence scores have the wrong shape")
    order = np.lexsort((np.arange(len(candidates)), -scores))
    basis = _RationalRowBasis(D)
    selected_indices, dependent, inconsistent = [], [], []
    considered = 0
    for i in order:
        if basis.rank == budget:
            break
        considered += 1
        row, rhs = _candidate_row(candidates[int(i)], D)
        outcome = basis.add(row, rhs)
        if outcome == "independent":
            selected_indices.append(int(i))
        elif outcome == "dependent":
            dependent.append(int(i))
        else:
            inconsistent.append(int(i))
    reached = basis.rank == budget
    return dict(selected=[candidates[i] for i in selected_indices], selected_indices=selected_indices,
                actual_rank=basis.rank, target_rank=budget, actual_common_dimension=D - basis.rank,
                target_common_dimension=D - budget, exact_match=reached,
                status="budget_reached" if reached else "budget_not_reachable",
                inconsistent_candidate_indices=inconsistent, dependent_candidate_indices=dependent,
                considered_candidates=considered, unconsidered_candidates=len(candidates) - considered,
                rank_arithmetic="exact rational candidate coefficients and RHS",
                original_algorithm_label="SVM-budget-capped variant; no added synthetic equations")


def affine_equalities_metrics(rows: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    """Exact consistency/rank for supplied rational common-coordinate rows."""
    rows = rows.toarray() if sparse.issparse(rows) else np.asarray(rows, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if rows.ndim != 2 or rhs.shape != (len(rows),):
        raise ValueError("Invalid affine equality matrix/RHS")
    basis = _RationalRowBasis(rows.shape[1])
    inconsistent = []
    for i, (row, value) in enumerate(zip(rows, rhs)):
        if basis.add(row, value) == "inconsistent":
            inconsistent.append(i)
    return dict(ambient_dimension=rows.shape[1], rank=basis.rank,
                consistent=not inconsistent,
                affine_dimension=None if inconsistent else rows.shape[1] - basis.rank,
                inconsistent_row_indices=inconsistent,
                scope="supplied exact rational/decimal affine equalities, before inequalities/integrality/presolve")


def affine_projection_metrics(equality_matrix: np.ndarray, rhs: np.ndarray,
                              common_indices: np.ndarray, rank_rtol: float = 1e-10) -> dict[str, Any]:
    """Offline numerical audit of a supplied full-variable affine system.

    dim proj_J ker(E) = |J| - rank(E) + rank(E[:,J-complement]). Original base
    equalities can be supplied to expose effects of fixing UC start/stop
    variables. This is not an affine-hull computation for an integer feasible
    set. It is intentionally an offline diagnostic, outside benchmark timing.
    """
    E = equality_matrix.toarray() if sparse.issparse(equality_matrix) else np.asarray(equality_matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)
    J = np.asarray(common_indices, dtype=np.int64)
    if E.ndim != 2 or b.shape != (len(E),) or J.ndim != 1 or len(J) != len(np.unique(J)):
        raise ValueError("Invalid full equality system or projection indices")
    if np.any(J < 0) or np.any(J >= E.shape[1]) or not np.isfinite(E).all() or not np.isfinite(b).all():
        raise ValueError("Invalid indices or nonfinite affine system")
    if rank_rtol <= 0:
        raise ValueError("Rank tolerance must be positive")
    scale = np.maximum(np.max(np.abs(E), axis=1, initial=0.), 1.)
    E, b = E / scale[:, None], b / scale
    def matrix_rank(A):
        singular = la.svdvals(A, check_finite=False)
        return int(np.count_nonzero(singular > rank_rtol * max(float(singular[0]) if len(singular) else 0., 1.)))
    rank = matrix_rank(E)
    augmented_rank = matrix_rank(np.column_stack([E, b]))
    other = np.setdiff1d(np.arange(E.shape[1]), J, assume_unique=True)
    other_rank = matrix_rank(E[:, other])
    consistent = rank == augmented_rank
    dimension = len(J) - rank + other_rank
    if not 0 <= dimension <= len(J):
        raise RuntimeError("Numerical ranks violate the projection-dimension bounds")
    return dict(consistent=consistent, common_ambient_dimension=len(J), equality_rank=rank,
                augmented_rank=augmented_rank, complement_column_rank=other_rank,
                projected_affine_dimension=dimension if consistent else None,
                full_affine_dimension=E.shape[1] - rank if consistent else None,
                rank_rtol=rank_rtol, rank_arithmetic="row-scaled float64 SVD; offline numerical diagnostic",
                actual_integer_feasible_set_dimension=None, solver_presolved_variables=None,
                scope="supplied linear equality system, not inequalities/integrality or solver presolve")


def native_strategy_metrics(case: Any, retained_strategies: int | None = None,
                            evaluated_candidates: int | None = None) -> dict[str, Any]:
    """Honest N/A for native complete-strategy learners, not a fabricated K."""
    D = len(common_coordinate_indices(case))
    return dict(common_ambient_dimension=D, native_compressed_dimension=None,
                dimension_match_status="not_applicable_to_native_complete_strategy",
                per_strategy_common_integer_assignment_dimension=0,
                retained_strategies=retained_strategies, evaluated_candidates=evaluated_candidates,
                caveat="Complete strategies fix the integer assignment. A finite union of candidate assignments is not the learned affine search space of Ours; report strategy counts and candidate/time budgets instead.",
                solver_presolved_variables=None, solver_presolved_integer_variables=None)


def self_test() -> dict[str, Any]:
    """Small mathematical checks; no solvers, learned models or test data."""
    from types import SimpleNamespace
    case = SimpleNamespace(family="uc", commitment_indices=np.arange(4), affine={"rank": 2},
                           model=SimpleNamespace(A=sparse.csr_matrix((0, 16)),
                                                 integrality=np.r_[np.ones(12), np.zeros(4)],
                                                 col_lower=np.zeros(16), col_upper=np.ones(16)))
    fixed = matched_map_fixings(case, np.full(12, .5))
    assert fixed == {0: 1, 1: 1}
    assert assert_matched_fixings(case, fixed)["common_nominal_free_dimension"] == 2
    native = diagnose_fixings(case, {0: 1, 4: 1}, expected_K=2)
    assert native["other_integer_fixed_count"] == 1 and not native["matches_expected_common_dimension"]
    rejected = False
    try:
        assert_matched_fixings(case, {0: 1, 4: 1})
    except ValueError:
        rejected = True
    assert rejected
    candidates = [(np.array([1, 0, 0]), 0), (np.array([0, 1, 0]), 0),
                  (np.array([1, -1, 0]), 0), (np.array([1, 0, 0]), 1),
                  (np.array([0, 0, 1]), 0)]
    capped = select_equalities_by_rank(candidates, 3, 3, [5, 4, 3, 2, 1])
    assert capped["selected_indices"] == [0, 1, 4]
    assert capped["dependent_candidate_indices"] == [2]
    assert capped["inconsistent_candidate_indices"] == [3] and capped["exact_match"]
    short = select_equalities_by_rank(candidates[:1], 3, 2)
    assert short["status"] == "budget_not_reachable" and short["actual_common_dimension"] == 2
    contradictory = affine_equalities_metrics(np.array([[1, 0], [1, 0]]), np.array([0., 1.]))
    assert not contradictory["consistent"] and contradictory["affine_dimension"] is None
    projection = affine_projection_metrics(np.array([[1, 0, -1], [0, 1, 0]]), np.zeros(2), np.array([0, 1]))
    assert projection["projected_affine_dimension"] == 1
    return dict(passed=True, checks=["exact MAP cardinality and deterministic confidence ties",
                                    "no startup/shutdown fixings in matched UC variant",
                                    "dependent/inconsistent equality candidates and unreachable rank budget",
                                    "empty contradictory affine system",
                                    "full-variable affine projection formula"],
                solver_calls=0, training_calls=0, test_data_read=False)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test()))
