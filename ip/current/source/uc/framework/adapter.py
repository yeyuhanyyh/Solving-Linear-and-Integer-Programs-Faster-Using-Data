"""Portable synthetic models and single-thread HiGHS solves for the v2 study.

This module reads only synthetic training exports. Confirmation costs are
created separately, only after all models/configuration have been frozen.
Solver/model primitives preserve the v1 adapter semantics and tolerances.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from time import perf_counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
from types import SimpleNamespace
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key,"1")
import numpy as np
import scipy.sparse as sparse
import highspy
EXPERIMENT_ROOT=Path(__file__).resolve().parents[1]
FEASIBILITY_TOLERANCE=1e-6
ABS_RECOVERY_TOLERANCE=1e-6
REL_RECOVERY_TOLERANCE=1e-9


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+".writing")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    os.replace(partial, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+".writing")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(partial, path)
def recovery_tolerance(objective: float) -> float:
    return ABS_RECOVERY_TOLERANCE + REL_RECOVERY_TOLERANCE * abs(float(objective))


@dataclass
class LinearModel:
    A: sparse.csr_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    col_lower: np.ndarray
    col_upper: np.ndarray
    integrality: np.ndarray
    variable_names: tuple[str, ...] = ()
    row_names: tuple[str, ...] = ()

    def __post_init__(self):
        self.A = sparse.csr_matrix(self.A, dtype=float)
        self.row_lower = np.asarray(self.row_lower, dtype=float)
        self.row_upper = np.asarray(self.row_upper, dtype=float)
        self.col_lower = np.asarray(self.col_lower, dtype=float)
        self.col_upper = np.asarray(self.col_upper, dtype=float)
        self.integrality = np.asarray(self.integrality, dtype=np.int32)
        m, n = self.A.shape
        if self.row_lower.shape != (m,) or self.row_upper.shape != (m,):
            raise ValueError("Row bounds have the wrong shape")
        if any(v.shape != (n,) for v in (self.col_lower, self.col_upper, self.integrality)):
            raise ValueError("Column arrays have the wrong shape")
        if np.any(self.row_lower > self.row_upper) or np.any(self.col_lower > self.col_upper):
            raise ValueError("Inconsistent bounds")
        if not np.isfinite(self.A.data).all() or not np.isin(self.integrality, (0, 1)).all():
            raise ValueError("Expected finite matrix coefficients and continuous/integer domains")

    @property
    def n_variables(self) -> int:
        return self.A.shape[1]

    @property
    def integer_indices(self) -> np.ndarray:
        return np.flatnonzero(self.integrality)

    @property
    def continuous_indices(self) -> np.ndarray:
        return np.flatnonzero(self.integrality == 0)


@dataclass
class SolveResult:
    x: np.ndarray | None
    objective: float | None
    status: str
    optimal: bool
    certified: bool
    valid: bool
    dual_bound: float | None
    gap: float | None
    seconds: float
    build_seconds: float
    solve_seconds: float
    solver_internal_seconds: float
    nodes: int
    simplex_iterations: int
    original_violation: float | None
    integrality_residual: float | None
    restriction_violation: float | None
    n_fixed: int = 0
    added_rows: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def online_seconds(self) -> float:
        return self.seconds

    @property
    def solution(self) -> np.ndarray | None:
        return self.x

    def to_dict(self, include_solution: bool = True) -> dict[str, Any]:
        out = dict(self.__dict__)
        x = out.pop("x")
        if include_solution:
            out["solution"] = None if x is None else x.tolist()
        out["online_seconds"] = self.seconds
        return out


def validate_solution(model: LinearModel, x: np.ndarray, atol: float = FEASIBILITY_TOLERANCE) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    if x.shape != (model.n_variables,) or not np.isfinite(x).all():
        return dict(valid=False, original_violation=float("inf"), integrality_residual=float("inf"))
    ax = model.A @ x
    violation = max(float(np.max(model.row_lower - ax, initial=0.)),
                    float(np.max(ax - model.row_upper, initial=0.)),
                    float(np.max(model.col_lower - x, initial=0.)),
                    float(np.max(x - model.col_upper, initial=0.)))
    ii = model.integer_indices
    intres = float(np.max(np.abs(x[ii] - np.rint(x[ii])), initial=0.))
    return dict(valid=bool(violation <= atol and intres <= atol),
                original_violation=violation, integrality_residual=intres)


def build_solver(model: LinearModel, cost: np.ndarray, *, fixed: Mapping[int, float] | None = None,
                 additional_rows: tuple[sparse.spmatrix, np.ndarray, np.ndarray] | None = None,
                 seed: int = 0, time_limit: float = 30., warm_start: np.ndarray | None = None,
                 relax_integrality: bool = False) -> tuple[highspy.Highs, dict[str, Any]]:
    """Construct a fresh HiGHS model; fixed keys are original column indices."""
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (model.n_variables,) or not np.isfinite(cost).all():
        raise ValueError("Objective has the wrong shape or nonfinite entries")
    A = model.A
    rl, ru = model.row_lower.copy(), model.row_upper.copy()
    cl, cu = model.col_lower.copy(), model.col_upper.copy()
    fixed = {} if fixed is None else dict(fixed)
    for j, val in fixed.items():
        if not isinstance(j, (int, np.integer)) or not 0 <= j < model.n_variables:
            raise ValueError("Fixed variable index outside original model")
        if not np.isfinite(val) or not model.integrality[j] or abs(val - round(val)) > 1e-9:
            raise ValueError("Fixings must assign integer values to integer variables")
        if val < cl[j] - 1e-9 or val > cu[j] + 1e-9:
            raise ValueError("Fixed value outside original bounds")
        cl[j] = cu[j] = float(round(val))
    n_added = 0
    if additional_rows is not None:
        extra, lo, hi = additional_rows
        extra = sparse.csr_matrix(extra, dtype=float)
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        if extra.shape[1] != model.n_variables or lo.shape != (extra.shape[0],) or hi.shape != lo.shape:
            raise ValueError("Additional row dimensions differ from original model")
        if np.any(lo > hi) or not np.isfinite(extra.data).all():
            raise ValueError("Invalid additional row bounds or coefficients")
        A = sparse.vstack([A, extra], format="csr")
        rl, ru = np.r_[rl, lo], np.r_[ru, hi]
        n_added = extra.shape[0]
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = model.n_variables, A.shape[0]
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost.copy(), cl, cu
    lp.row_lower_, lp.row_upper_ = rl, ru
    lp.integrality_ = [highspy.HighsVarType.kInteger if i and not relax_integrality else highspy.HighsVarType.kContinuous
                       for i in model.integrality]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    h = highspy.Highs()
    for k, v in dict(output_flag=False, threads=1, random_seed=int(seed), time_limit=float(time_limit),
                     mip_rel_gap=0., mip_abs_gap=1e-8, primal_feasibility_tolerance=1e-7,
                     mip_feasibility_tolerance=1e-7).items():
        if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS option rejected: {k}={v}")
    pass_status = h.passModel(lp)
    if pass_status == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS rejected fresh model")
    if h.getNumRow() != A.shape[0] or h.getNumCol() != model.n_variables:
        raise RuntimeError("HiGHS loaded incomplete model dimensions")
    if warm_start is not None:
        ws = np.asarray(warm_start, dtype=float)
        if ws.shape != cost.shape:
            raise ValueError("Warm start shape differs")
        if h.setSolution(len(cost), np.arange(len(cost), dtype=np.int32), ws) != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS rejected warm start")
    return h, dict(A=A, row_lower=rl, row_upper=ru, col_lower=cl, col_upper=cu,
                  n_fixed=len(fixed), added_rows=n_added, relaxation=bool(relax_integrality),
                  pass_model_status=str(pass_status))


def _solve_highs(model: LinearModel, cost: np.ndarray, *, fixed: Mapping[int, float] | None = None,
          additional_rows: tuple[sparse.spmatrix, np.ndarray, np.ndarray] | None = None,
          seed: int = 0, time_limit: float = 30., warm_start: np.ndarray | None = None,
          incumbent_callback: Callable[[np.ndarray], None] | None = None,
          relax_integrality: bool = False) -> SolveResult:
    started = perf_counter()
    h, prepared = build_solver(model, cost, fixed=fixed, additional_rows=additional_rows,
                               seed=seed, time_limit=time_limit, warm_start=warm_start,
                               relax_integrality=relax_integrality)
    callback_events = 0
    if incumbent_callback is not None:
        def collect(event):
            nonlocal callback_events
            x = np.asarray(event.data_out.mip_solution, dtype=float).copy()
            if x.shape == (model.n_variables,):
                callback_events += 1
                incumbent_callback(x)
        h.cbMipSolution.subscribe(collect)
    build_seconds = perf_counter() - started
    h.setOptionValue("time_limit", max(0., float(time_limit) - build_seconds))
    run_started = perf_counter()
    run_status = h.run()
    solve_seconds = perf_counter() - run_started
    if run_status == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS run returned an error")
    status, info, sol = h.getModelStatus(), h.getInfo(), h.getSolution()
    x, objective, valid, violation, intres, restr = None, None, False, None, None, None
    if sol.value_valid:
        x = np.asarray(sol.col_value, dtype=float)
        v = validate_solution(model, x)
        violation, intres = v["original_violation"], v["integrality_residual"]
        ax = prepared["A"] @ x
        restr = max(float(np.max(prepared["row_lower"] - ax, initial=0.)),
                    float(np.max(ax - prepared["row_upper"], initial=0.)),
                    float(np.max(prepared["col_lower"] - x, initial=0.)),
                    float(np.max(x - prepared["col_upper"], initial=0.)))
        valid = v["valid"] and restr <= FEASIBILITY_TOLERANCE
        objective = float(np.asarray(cost) @ x)
    optimal = status == highspy.HighsModelStatus.kOptimal
    # LP relaxations have no MIP bound; their certificate concerns the relaxation.
    db = float(info.objective_function_value if relax_integrality else info.mip_dual_bound)
    dual_bound = db if np.isfinite(db) else None
    certified = bool(optimal and valid and objective is not None and dual_bound is not None
                     and -recovery_tolerance(objective) <= objective - dual_bound <= recovery_tolerance(objective))
    gap = float(info.mip_gap)
    seconds = perf_counter() - started
    return SolveResult(x, objective, h.modelStatusToString(status), optimal, certified, valid,
                       dual_bound, gap if np.isfinite(gap) else None, seconds, build_seconds,
                       solve_seconds, h.getRunTime(), int(info.mip_node_count),
                       int(info.simplex_iteration_count), violation, intres, restr,
                       prepared["n_fixed"], prepared["added_rows"],
                       dict(solver_seed=int(seed), callback_events=callback_events,
                            relaxation=relax_integrality,pass_model_status=prepared["pass_model_status"],
                            certificate_scope="relaxation" if relax_integrality else "solved_model"))


def solve(model, cost, *, backend=None, **kwargs):
    """Fresh, one-thread solve; backend does not change the shared matrix."""
    backend = backend or os.environ.get("FORMAL_SOLVER_BACKEND", "highs")
    if backend == "highs":
        result = _solve_highs(model,cost,**kwargs)
        result.metadata["backend"] = "highs"
        return result
    if backend != "scip": raise ValueError(f"Unknown solver backend: {backend}")
    if kwargs.get("incumbent_callback") is not None:
        raise ValueError("SCIP incumbent-pool callback has not been implemented")
    from uc.scip_solver import solve_csr
    started=perf_counter()
    relax=bool(kwargs.get("relax_integrality",False))
    domains=np.zeros_like(model.integrality) if relax else model.integrality
    raw=solve_csr(model.A,model.row_lower,model.row_upper,model.col_lower,model.col_upper,
        domains,np.asarray(cost),fixed=kwargs.get("fixed"),additional_rows=kwargs.get("additional_rows"),
        time_limit=float(kwargs.get("time_limit",30.)),gap=0.,seed=int(kwargs.get("seed",0)),threads=1,
        warm_start=kwargs.get("warm_start"))
    x=None if raw.get("x") is None else np.asarray(raw["x"],dtype=float)
    validation=validate_solution(model,x) if x is not None else dict(valid=False,original_violation=None,integrality_residual=None)
    restriction=0.
    if x is not None:
        for j,v in (kwargs.get("fixed") or {}).items():restriction=max(restriction,abs(x[int(j)]-v))
        if kwargs.get("additional_rows") is not None:
            a,lo,hi=kwargs["additional_rows"];ax=a@x
            restriction=max(restriction,float(np.max(np.asarray(lo)-ax,initial=0.)),float(np.max(ax-np.asarray(hi),initial=0.)))
    valid=bool(validation["valid"] and restriction<=FEASIBILITY_TOLERANCE)
    objective=None if x is None else float(np.asarray(cost)@x)
    bound=raw.get("dual_bound")
    certified=bool(raw.get("solver_optimal") and valid and objective is not None and bound is not None
        and np.isfinite(bound) and -recovery_tolerance(objective)<=objective-bound<=recovery_tolerance(objective))
    return SolveResult(x,objective,str(raw.get("status")),bool(raw.get("solver_optimal")),certified,valid,
        bound,raw.get("gap"),perf_counter()-started,float(raw.get("build_seconds",0.)),
        float(raw.get("solve_wall_seconds",0.)),float(raw.get("solve_wall_seconds",0.)),
        int(raw.get("nodes",-1)),int(raw.get("simplex_iterations",-1)),validation["original_violation"],
        validation["integrality_residual"],restriction,len(kwargs.get("fixed") or {}),
        0 if kwargs.get("additional_rows") is None else kwargs["additional_rows"][0].shape[0],
        dict(backend="scip",certificate_scope="relaxation" if relax else "solved_model",settings=raw.get("settings")))


def recovered(result: SolveResult, reference_objective: float, require_certificate: bool = True) -> bool:
    return bool(result.valid and (result.certified or not require_certificate)
                and result.objective is not None
                and abs(result.objective - reference_objective) <= recovery_tolerance(reference_objective))


@dataclass
class FrozenCase:
    """Training-only portable case; confirmation data have a separate loader."""
    case_id: str
    family: str
    seed: int
    amplitude: float
    model: LinearModel
    train_features: np.ndarray
    train_costs: np.ndarray
    train_solutions: np.ndarray
    train_objectives: np.ndarray
    commitment_indices: np.ndarray
    affine: dict[str, Any]
    original_base: dict[str, Any]
    metadata: dict[str, Any]
    source_hashes: dict[str, str]
    dataset_fingerprint: str
    source_directory: Path

    @property
    def n_train(self) -> int:
        return len(self.train_costs)


def training_case(case):
    """A test-free view, including no held-out references or objectives."""
    return SimpleNamespace(**{name: getattr(case, name) for name in (
        "case_id", "family", "seed", "amplitude", "model", "train_features", "train_costs",
        "train_solutions", "train_objectives", "commitment_indices", "affine", "metadata")})


def singleton_pools(case):
    """Certified-optimum-only ND pools; this disclosed choice adds no labels."""
    if not case.metadata.get("all_training_labels_certified", False):
        raise ValueError("Singleton pools require original globally certified labels")
    return [dict(train_index=i, solutions=x[None, :].copy(),
        incumbent_solutions=x[None, :].copy(), objectives=np.asarray([objective]),
        source_tags=["cached_optimum"], conditional_lp_completion_certified=True,
        collection_seconds=0., extra_conditional_lp_seconds=0.,
        source="shared certified global optimum")
        for i, (x, objective) in enumerate(zip(case.train_solutions, case.train_objectives))]


def collect_training_pools(case, indices=None, **kwargs):
    pools = singleton_pools(case)
    return pools if indices is None else [pools[int(i)] for i in indices]


load_training_pools = collect_training_pools


def complete_training_pools(case, pools=None, **kwargs):
    pools = singleton_pools(case) if pools is None else pools
    for pool in pools:
        if not pool.get("conditional_lp_completion_certified", False):
            raise ValueError("Uncertified extra pool candidates require explicit LP completion")
    return pools


