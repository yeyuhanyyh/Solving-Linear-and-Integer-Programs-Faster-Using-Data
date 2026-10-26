"""Synthetic unit commitment for controlled fixed-feasibility experiments.

This is UC, not security-constrained UC: it has no transmission network,
contingencies, reserve constraint, or load shedding.  All units are initially
off and have already satisfied their minimum down time.  Minimum up/down times
are enforced within the modeled horizon.  A transition near the end may leave
a residual up/down obligation to be carried into the next horizon; no terminal
shutdown or cyclic condition is imposed.

Variables are [u, v, w, p], each time-major (t * G + g), where u is commitment,
v startup, w shutdown, and p generation.  The first three blocks are binary.
Only u is used by the affine-restriction experiments.  Historical u determines
v and w through the transition equations and mutual-exclusion inequalities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def make_uc_base(seed: int, n_units: int = 6, n_times: int = 8) -> dict[str, Any]:
    """Return a reproducible feasible UC instance in ranged-row matrix form.

    Demand, capacities, ramp limits and minimum times are fixed by ``seed``.
    An all-on proportional-dispatch schedule is guaranteed feasible, so the
    experiment never needs artificial shortage variables or shortage penalties.
    Costs are positive; time periods have unit duration.
    """
    if n_units < 1 or n_times < 1:
        raise ValueError("n_units and n_times must be positive")
    rng = np.random.default_rng(seed)
    G, T = int(n_units), int(n_times)
    q = G * T
    capacity = rng.uniform(75.0, 125.0, G)
    minimum_output = capacity * rng.uniform(0.12, 0.22, G)
    ramp_up = capacity * rng.uniform(0.35, 0.55, G)
    ramp_down = capacity * rng.uniform(0.35, 0.55, G)
    minimum_up_time = np.full(G, 2, dtype=int)
    minimum_down_time = np.full(G, 2, dtype=int)

    # Interpolation keeps adjacent fractions within the smallest ramp allowance.
    # A slight deterministic seed-dependent offset preserves this property.
    profile = np.array([0.32, 0.41, 0.55, 0.67, 0.61, 0.48, 0.36, 0.30])
    fraction = np.interp(np.linspace(0, 7, T), np.arange(8), profile)
    fraction += rng.uniform(-0.018, 0.018)
    demand = capacity.sum() * fraction
    # G=1 remains feasible: every fraction exceeds max minimum-output fraction.
    fuel_cost = rng.uniform(22.0, 45.0, G)
    no_load_cost = rng.uniform(80.0, 240.0, G)
    startup_cost = rng.uniform(220.0, 850.0, G)
    shutdown_cost = rng.uniform(5.0, 30.0, G)
    base_cost = np.concatenate([
        np.tile(no_load_cost, T),
        np.tile(startup_cost, T),
        np.tile(shutdown_cost, T),
        np.tile(fuel_cost, T),
    ])

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    row_names: list[str] = []

    def add(entries: list[tuple[int, float]], lo: float, hi: float, name: str) -> None:
        r = len(lowers)
        for j, value in entries:
            if value:
                rows.append(r)
                cols.append(j)
                values.append(float(value))
        lowers.append(float(lo))
        uppers.append(float(hi))
        row_names.append(name)

    for t in range(T):
        add([(3 * q + t * G + g, 1.0) for g in range(G)],
            demand[t], demand[t], f"balance[{t}]")
        for g in range(G):
            j = t * G + g
            u, v, w, p = j, q + j, 2 * q + j, 3 * q + j
            trans = [(u, 1.0), (v, -1.0), (w, 1.0)]
            if t:
                trans.append((u - G, -1.0))
            add(trans, 0.0, 0.0, f"transition[{t},{g}]")
            add([(v, 1.0), (w, 1.0)], -np.inf, 1.0,
                f"exclusive_transition[{t},{g}]")
            add([(p, 1.0), (u, -capacity[g])], -np.inf, 0.0,
                f"maximum_output[{t},{g}]")
            add([(p, 1.0), (u, -minimum_output[g])], 0.0, np.inf,
                f"minimum_output[{t},{g}]")
            add([(q + s * G + g, 1.0)
                 for s in range(max(0, t - minimum_up_time[g] + 1), t + 1)]
                + [(u, -1.0)], -np.inf, 0.0, f"minimum_up[{t},{g}]")
            add([(2 * q + s * G + g, 1.0)
                 for s in range(max(0, t - minimum_down_time[g] + 1), t + 1)]
                + [(u, 1.0)], -np.inf, 1.0, f"minimum_down[{t},{g}]")

            # Startup/shutdown can move between zero and the full capacity.
            # Normal ramp limits apply only when the unit stays on.
            up_entries = [(p, 1.0), (v, -capacity[g])]
            down_entries = [(p, -1.0), (u, -ramp_down[g]), (w, -capacity[g])]
            if t:
                up_entries += [(p - G, -1.0), (u - G, -ramp_up[g])]
                down_entries += [(p - G, 1.0)]
            add(up_entries, -np.inf, 0.0, f"ramp_up[{t},{g}]")
            add(down_entries, -np.inf, 0.0, f"ramp_down[{t},{g}]")

    A = sparse.coo_matrix((values, (rows, cols)), shape=(len(lowers), 4 * q)).tocsr()
    col_lower = np.zeros(4 * q)
    col_upper = np.concatenate([np.ones(3 * q), np.tile(capacity, T)])
    integrality = np.concatenate([np.ones(3 * q, dtype=np.int32),
                                 np.zeros(q, dtype=np.int32)])
    metadata = {
        "model": "synthetic unit commitment, no transmission or security constraints",
        "seed": int(seed), "capacity": capacity, "minimum_output": minimum_output,
        "ramp_up": ramp_up, "ramp_down": ramp_down,
        "minimum_up_time": minimum_up_time, "minimum_down_time": minimum_down_time,
        "demand": demand, "fuel_cost": fuel_cost, "no_load_cost": no_load_cost,
        "startup_cost": startup_cost, "shutdown_cost": shutdown_cost,
        "initial_commitment": np.zeros(G, dtype=int),
        "initial_generation": np.zeros(G),
        "initial_down_time": minimum_down_time.copy(),
        "terminal_treatment": "within-horizon minimum times; residual obligations carry forward",
        "variable_order": "u, v, w, p blocks, each in time-major t*n_units+g order",
        "row_names": row_names,
    }
    return {
        "A": A, "row_lower": np.asarray(lowers), "row_upper": np.asarray(uppers),
        "col_lower": col_lower, "col_upper": col_upper, "integrality": integrality,
        "base_cost": base_cost, "binary_indices": np.arange(q, dtype=np.int32),
        "n_times": T, "n_units": G, "metadata": metadata,
    }


def cost_for_uc(base: dict[str, Any], rng: np.random.Generator,
                amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Draw shared per-unit multipliers for every objective component.

    Return (full objective, feature vector of G multipliers).  These features
    exactly determine the changing objective; they omit no varying parameter.
    All physical constraints are unchanged.  Amplitude must be in [0,1).
    """
    if not 0 <= amplitude < 1:
        raise ValueError("amplitude must be in [0, 1)")
    G, T = base["n_units"], base["n_times"]
    factors = rng.uniform(1 - amplitude, 1 + amplitude, G)
    cost = base["base_cost"] * np.tile(factors, 4 * T)
    return cost, factors


def all_on_feasible_solution(base: dict[str, Any]) -> np.ndarray:
    """Construct a known feasible solution without invoking an optimizer."""
    G, T = base["n_units"], base["n_times"]
    u = np.ones((T, G))
    v = np.zeros((T, G))
    v[0, :] = 1.0
    w = np.zeros((T, G))
    capacity = base["metadata"]["capacity"]
    p = np.outer(base["metadata"]["demand"] / capacity.sum(), capacity)
    return np.concatenate([u.ravel(), v.ravel(), w.ravel(), p.ravel()])


def validate_uc_solution(base: dict[str, Any], x: np.ndarray,
                         atol: float = 1e-6) -> dict[str, Any]:
    """Check both matrix feasibility and independently evaluated UC physics.

    The independent checks use transitions and consecutive run lengths, not
    the matrix's minimum-time window equations.  Return named maximum absolute
    violations and a boolean valid flag.  No objective or optimality is tested.
    """
    x = np.asarray(x, dtype=float)
    G, T = base["n_units"], base["n_times"]
    q = G * T
    if x.shape != (4 * q,) or not np.isfinite(x).all():
        return {"valid": False, "violations": {"shape_or_nonfinite": float("inf")}}
    u, v, w, p = [part.reshape(T, G) for part in np.split(x, 4)]
    m = base["metadata"]
    activity = np.asarray(base["A"] @ x)

    def positive_max(a: np.ndarray) -> float:
        return float(max(0.0, np.max(a, initial=-np.inf)))

    prev_u = np.vstack([m["initial_commitment"], u[:-1]])
    prev_p = np.vstack([m["initial_generation"], p[:-1]])
    violations = {
        "matrix_lower": positive_max(base["row_lower"] - activity),
        "matrix_upper": positive_max(activity - base["row_upper"]),
        "variable_lower": positive_max(base["col_lower"] - x),
        "variable_upper": positive_max(x - base["col_upper"]),
        "integrality": float(np.max(np.abs(x[:3 * q] - np.rint(x[:3 * q])))),
        "balance": float(np.max(np.abs(p.sum(axis=1) - m["demand"]))),
        "maximum_output": positive_max(p - u * m["capacity"]),
        "minimum_output": positive_max(u * m["minimum_output"] - p),
        "startup_logic": float(np.max(np.abs(v - np.maximum(u - prev_u, 0)))),
        "shutdown_logic": float(np.max(np.abs(w - np.maximum(prev_u - u, 0)))),
        "ramp_up": positive_max(p - prev_p - m["ramp_up"] * prev_u
                                - m["capacity"] * v),
        "ramp_down": positive_max(prev_p - p - m["ramp_down"] * u
                                  - m["capacity"] * w),
    }
    min_time_violation = 0.0
    rounded_u = np.rint(u).astype(int)
    for g in range(G):
        last_state = 0
        run_length = int(m["initial_down_time"][g])
        for t in range(T):
            state = int(rounded_u[t, g])
            if state != last_state:
                required = (m["minimum_up_time"][g] if last_state
                            else m["minimum_down_time"][g])
                min_time_violation = max(min_time_violation, float(required - run_length))
                last_state = state
                run_length = 1
            else:
                run_length += 1
    violations["minimum_run_length"] = min_time_violation
    return {"valid": max(violations.values()) <= atol,
            "max_violation": max(violations.values()), "violations": violations}
