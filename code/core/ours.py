"""Default implementation of the paper's optimizer-span learner.

For continuously distributed costs, the optimizer is unique almost surely.
In that setting the learner solves each training LP once and computes a basis
of the differences between the resulting optimizers.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence

import numpy as np
from scipy.optimize import linprog


@dataclasses.dataclass(frozen=True)
class Representation:
    anchor: np.ndarray
    basis: np.ndarray
    raw_rank: int
    training_lp_solves: int
    training_solver_time_s: float
    learning_time_s: float


def orthonormal_columns(
    matrix: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    u, singular_values, _vh = np.linalg.svd(matrix, full_matrices=False)
    cutoff = tolerance * max(
        1.0,
        float(singular_values[0]) if singular_values.size else 1.0,
    )
    rank = int(np.sum(singular_values > cutoff))
    return np.asarray(u[:, :rank], dtype=float)


def solve_optimizer(instance: object) -> tuple[np.ndarray, float]:
    """Solve one compact max LP represented by common inequalities."""
    c = np.asarray(instance.c, dtype=float)
    scale = max(1.0, float(np.linalg.norm(c, ord=np.inf)))
    start = time.perf_counter()
    result = linprog(
        c=-(c / scale),
        A_ub=np.asarray(instance.A, dtype=float),
        b_ub=np.asarray(instance.b, dtype=float),
        bounds=[(None, None)] * c.size,
        method="highs",
        options={"presolve": True},
    )
    elapsed = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f"Training LP failed: {result.message}")
    return np.asarray(result.x, dtype=float), float(elapsed)


def solve_training_optimizers(
    instances: Sequence[object],
) -> tuple[list[np.ndarray], float]:
    optimizers: list[np.ndarray] = []
    solver_time = 0.0
    for instance in instances:
        optimizer, elapsed = solve_optimizer(instance)
        optimizers.append(optimizer)
        solver_time += elapsed
    return optimizers, float(solver_time)


def pad_basis(
    basis: np.ndarray,
    target_dimension: int,
    seed: int,
) -> np.ndarray:
    basis = orthonormal_columns(basis)
    dimension, rank = basis.shape
    if rank > int(target_dimension):
        raise ValueError(
            f"Learned rank {rank} exceeds target K={target_dimension}; "
            "truncating an exact span is not allowed."
        )
    if rank == int(target_dimension):
        return basis
    rng = np.random.default_rng(int(seed))
    needed = int(target_dimension) - rank
    candidates = rng.normal(size=(dimension, max(needed + 8, 2 * needed)))
    if rank:
        candidates -= basis @ (basis.T @ candidates)
    complement = orthonormal_columns(candidates)
    while complement.shape[1] < needed:
        extra = rng.normal(size=(dimension, needed + 8))
        if rank:
            extra -= basis @ (basis.T @ extra)
        if complement.shape[1]:
            extra -= complement @ (complement.T @ extra)
        complement = orthonormal_columns(
            np.column_stack([complement, extra])
        )
    return np.column_stack([basis, complement[:, :needed]])


def representation_from_optimizers(
    optimizers: Sequence[np.ndarray],
    *,
    target_dimension: int | None = None,
    seed: int = 0,
    solver_time_s: float = 0.0,
    learning_start: float | None = None,
) -> Representation:
    if not optimizers:
        raise ValueError("At least one training optimizer is required")
    start = time.perf_counter() if learning_start is None else learning_start
    anchor = np.asarray(optimizers[0], dtype=float)
    if len(optimizers) == 1:
        differences = np.zeros((anchor.size, 0), dtype=float)
    else:
        differences = np.column_stack(
            [
                np.asarray(optimizer, dtype=float) - anchor
                for optimizer in optimizers[1:]
            ]
        )
    basis = orthonormal_columns(differences)
    raw_rank = int(basis.shape[1])
    if target_dimension is not None:
        basis = pad_basis(basis, int(target_dimension), int(seed))
    return Representation(
        anchor=anchor,
        basis=basis,
        raw_rank=raw_rank,
        training_lp_solves=int(len(optimizers)),
        training_solver_time_s=float(solver_time_s),
        learning_time_s=float(time.perf_counter() - start),
    )


def fit(
    instances: Sequence[object],
    *,
    target_dimension: int | None = None,
    seed: int = 0,
) -> Representation:
    start = time.perf_counter()
    optimizers, solver_time = solve_training_optimizers(instances)
    return representation_from_optimizers(
        optimizers,
        target_dimension=target_dimension,
        seed=seed,
        solver_time_s=solver_time,
        learning_start=start,
    )


def common_reduced_constraints(
    test: Sequence[object],
    representation: Representation,
) -> tuple[np.ndarray, np.ndarray]:
    first = test[0]
    common_a = np.asarray(first.A, dtype=float)
    common_b = np.asarray(first.b, dtype=float)
    for instance in test[1:]:
        if (
            np.asarray(instance.A).shape != common_a.shape
            or np.asarray(instance.b).shape != common_b.shape
            or not np.allclose(np.asarray(instance.A, dtype=float), common_a)
            or not np.allclose(np.asarray(instance.b, dtype=float), common_b)
        ):
            raise RuntimeError("LP instances do not share one feasible region")
    reduced_a = common_a @ representation.basis
    reduced_b = common_b - common_a @ representation.anchor
    row_scale = (
        1.0
        + np.abs(common_b)
        + np.abs(common_a) @ np.abs(representation.anchor)
    )
    violation = np.maximum(-reduced_b, 0.0) / row_scale
    if float(np.max(violation, initial=0.0)) > 1e-7:
        raise RuntimeError("The learned affine anchor is infeasible")
    return reduced_a, np.maximum(reduced_b, 0.0)


def evaluate(
    test: Sequence[object],
    representation: Representation,
) -> list[dict[str, float]]:
    """Solve held-out reduced LPs and return per-instance records."""
    if not test:
        return []
    reduced_a, reduced_b = common_reduced_constraints(test, representation)
    rows: list[dict[str, float]] = []
    for instance in test:
        preparation_start = time.perf_counter()
        q = representation.basis.T @ np.asarray(instance.c, dtype=float)
        objective_scale = max(1.0, float(np.linalg.norm(q, ord=np.inf)))
        preparation_time = time.perf_counter() - preparation_start
        solver_start = time.perf_counter()
        result = linprog(
            c=-(q / objective_scale),
            A_ub=reduced_a,
            b_ub=reduced_b,
            bounds=[(None, None)] * representation.basis.shape[1],
            method="highs",
            options={"presolve": True},
        )
        solver_time = time.perf_counter() - solver_start
        if not result.success:
            raise RuntimeError(f"Reduced LP failed: {result.message}")
        reduced_solution = np.asarray(result.x, dtype=float)
        full_solution = (
            representation.anchor
            + representation.basis @ reduced_solution
        )
        objective = float(
            instance.objective_constant
            + np.asarray(instance.c, dtype=float) @ full_solution
        )
        full_objective = float(instance.full_obj)
        ratio = (
            float("nan")
            if abs(full_objective) <= 1e-8
            else objective / full_objective
        )
        rows.append(
            {
                "objective": objective,
                "objective_ratio": float(ratio),
                "solver_time_s": float(solver_time),
                "preparation_time_s": float(preparation_time),
                "online_time_s": float(preparation_time + solver_time),
                "success": 1.0,
            }
        )
    return rows


__all__ = [
    "Representation",
    "evaluate",
    "fit",
    "orthonormal_columns",
    "pad_basis",
    "representation_from_optimizers",
    "solve_optimizer",
    "solve_training_optimizers",
]
