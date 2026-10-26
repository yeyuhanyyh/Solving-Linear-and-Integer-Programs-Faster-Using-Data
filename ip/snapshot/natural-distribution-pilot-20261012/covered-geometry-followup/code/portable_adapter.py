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
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS rejected fresh model")
    if warm_start is not None:
        ws = np.asarray(warm_start, dtype=float)
        if ws.shape != cost.shape:
            raise ValueError("Warm start shape differs")
        if h.setSolution(len(cost), np.arange(len(cost), dtype=np.int32), ws) != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS rejected warm start")
    return h, dict(A=A, row_lower=rl, row_upper=ru, col_lower=cl, col_upper=cu,
                  n_fixed=len(fixed), added_rows=n_added, relaxation=bool(relax_integrality))


def solve(model: LinearModel, cost: np.ndarray, *, fixed: Mapping[int, float] | None = None,
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
                     and abs(objective - dual_bound) <= recovery_tolerance(objective))
    gap = float(info.mip_gap)
    seconds = perf_counter() - started
    return SolveResult(x, objective, h.modelStatusToString(status), optimal, certified, valid,
                       dual_bound, gap if np.isfinite(gap) else None, seconds, build_seconds,
                       solve_seconds, h.getRunTime(), int(info.mip_node_count),
                       int(info.simplex_iteration_count), violation, intres, restr,
                       prepared["n_fixed"], prepared["added_rows"],
                       dict(solver_seed=int(seed), callback_events=callback_events,
                            relaxation=relax_integrality,
                            certificate_scope="relaxation" if relax_integrality else "solved_model"))


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


def list_case_ids() -> list[str]:
    manifest = _read_json(EXPERIMENT_ROOT / "data" / "dataset-manifest.json")
    return [row["case_id"] for row in manifest["cases"]]


def list_cases() -> list[str]:
    return list_case_ids()


def _physical_metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {k: np.asarray(v) if isinstance(v, list) and k != "row_names" else v for k, v in value.items()}


def dataset_manifest(verify: bool = True) -> dict[str, Any]:
    manifest = _read_json(EXPERIMENT_ROOT / "data" / "dataset-manifest.json")
    hashes = manifest["file_sha256"]
    canonical = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if canonical != manifest["dataset_fingerprint"]:
        raise RuntimeError("Portable dataset manifest fingerprint is inconsistent")
    if verify:
        for name, expected in hashes.items():
            path = (EXPERIMENT_ROOT / name).resolve()
            if not path.is_relative_to(EXPERIMENT_ROOT):
                raise RuntimeError("Dataset manifest path escapes the synthetic package")
            if _sha256(path) != expected:
                raise RuntimeError(f"Portable synthetic file changed: {name}")
    return manifest


def load_case(case_id: str, verify: bool = True) -> FrozenCase:
    manifest = dataset_manifest(verify=False)
    if case_id not in [row["case_id"] for row in manifest["cases"]]:
        raise ValueError(f"Unknown exported synthetic case: {case_id}")
    folder = EXPERIMENT_ROOT / "data" / case_id
    if verify:
        for name in (f"data/{case_id}/training.npz", f"data/{case_id}/metadata.json"):
            if _sha256(EXPERIMENT_ROOT / name) != manifest["file_sha256"][name]:
                raise RuntimeError(f"Exported case file changed: {name}")
    meta = _read_json(folder / "metadata.json")
    with np.load(folder / "training.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    A = sparse.csr_matrix((arrays["A_data"], arrays["A_indices"], arrays["A_indptr"]), shape=tuple(arrays["A_shape"]))
    model = LinearModel(A=A, **{name: arrays[name] for name in (
        "row_lower", "row_upper", "col_lower", "col_upper", "integrality")},
        variable_names=tuple(meta["variable_names"]), row_names=tuple(meta["row_names"]))
    affine = dict(meta["affine"])
    affine.update(anchor=np.asarray(arrays["affine_anchor"], dtype=np.int64),
                  basis=np.asarray(arrays["affine_basis"], dtype=float),
                  normals=np.asarray(arrays["affine_normals"], dtype=float),
                  normals_exact=np.asarray([[int(q) for q in row] for row in meta["affine"]["normals_exact"]], dtype=object))
    physical = _physical_metadata(meta["metadata"])
    base = dict(A=model.A, row_lower=model.row_lower, row_upper=model.row_upper,
                col_lower=model.col_lower, col_upper=model.col_upper, integrality=model.integrality,
                base_cost=arrays["base_cost"], binary_indices=arrays["commitment_indices"],
                n_times=int(meta["n_times"]), n_units=int(meta["n_units"]), metadata=physical)
    case = FrozenCase(case_id=case_id, family=meta["family"], seed=int(meta["seed"]),
        amplitude=float(meta["amplitude"]), model=model,
        train_features=arrays["train_features"], train_costs=arrays["train_costs"],
        train_solutions=arrays["train_solutions"], train_objectives=arrays["train_objectives"],
        commitment_indices=np.asarray(arrays["commitment_indices"], dtype=np.int64),
        affine=affine, original_base=base, metadata=physical, source_hashes=meta["source_hashes"],
        dataset_fingerprint=manifest["dataset_fingerprint"], source_directory=folder)
    if verify:
        if case.n_train != 120 or case.train_features.shape != (120, 6 if case.family == "uc" else 8):
            raise RuntimeError("Portable training sample count/feature dimensions differ from Appendix C.3")
        expected_n = 576 if case.family == "uc" else 200
        expected_i = 432 if case.family == "uc" else 200
        if case.model.n_variables != expected_n or len(case.model.integer_indices) != expected_i:
            raise RuntimeError("Portable model dimensions differ from Appendix C.3")
        if case.train_costs.shape != case.train_solutions.shape or case.train_costs.shape != (120, expected_n):
            raise RuntimeError("Portable cost/solution shapes differ")
        for cost, x, objective in zip(case.train_costs, case.train_solutions, case.train_objectives):
            if not validate_solution(model, x)["valid"] or abs(float(cost@x)-objective) > recovery_tolerance(objective):
                raise RuntimeError("Invalid exported certified training optimizer")
        reproduced = costs_from_features(case, case.train_features)
        if not np.allclose(reproduced, case.train_costs, atol=1e-12, rtol=1e-14):
            raise RuntimeError("Portable cost-parameter map differs from original training objectives")
    return case


def training_case(case: FrozenCase) -> SimpleNamespace:
    # No confirmation test attributes exist in FrozenCase or this namespace.
    return SimpleNamespace(**{name: getattr(case, name) for name in (
        "case_id", "family", "seed", "amplitude", "model", "train_features", "train_costs",
        "train_solutions", "train_objectives", "commitment_indices", "affine", "metadata")})


def costs_from_features(case: FrozenCase, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    scalar = features.ndim == 1
    if scalar:
        features = features[None, :]
    if case.family == "uc":
        layout = case.metadata.get("cost_feature_layout")
        if layout is not None:
            if layout not in ("first4_whole_day", "first2_three_time_blocks"):
                raise ValueError(f"Unknown UC cost feature layout: {layout!r}")
            if features.ndim != 2 or features.shape[1] != 6:
                raise ValueError("UC cost feature layouts require six features per sample")
            if not np.isfinite(features).all() or np.any(features <= 0):
                raise ValueError("UC cost features must be finite and strictly positive; no clipping is applied")
            if (case.original_base["n_units"] != 6 or case.original_base["n_times"] != 24
                    or case.model.n_variables != 576):
                raise ValueError("UC cost feature layouts require six generators and 24 periods")
            # All four component blocks use time-major t*6+g ordering.
            columns = np.arange(case.model.n_variables)
            generators = columns % 6
            times = (columns // 6) % 24
            multipliers = np.ones((len(features), case.model.n_variables), dtype=float)
            if layout == "first4_whole_day":
                varying = generators < 4
                multipliers[:, varying] = features[:, generators[varying]]
            else:
                varying = generators < 2
                # Feature order: (block0,g0), (block0,g1), ..., (block2,g1).
                feature_indices = (times[varying] // 8) * 2 + generators[varying]
                multipliers[:, varying] = features[:, feature_indices]
            costs = multipliers * np.asarray(case.original_base["base_cost"])[None, :]
            return costs[0] if scalar else costs
        groups = np.arange(case.model.n_variables) % case.original_base["n_units"]
    else:
        groups = np.asarray(case.metadata["category_ids"], dtype=int)
    costs = features[:, groups] * np.asarray(case.original_base["base_cost"])[None, :]
    return costs[0] if scalar else costs


def draw_confirmation_costs(case: FrozenCase, count: int = 40, stream: int = 2026100919) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Called by the benchmark only AFTER its fitted-model freeze guard.

    The prescribed stream is shared in name, but hashing the physical case ID
    yields independent family/seed/amplitude streams. No frozen old test input
    is present in the portable training pack.
    """
    freeze_path = EXPERIMENT_ROOT / "results" / "confirmation-model-freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("Confirmation generation requires the completed fitted-model cohort freeze")
    guard = _read_json(freeze_path)
    if case.case_id not in guard["cases"] or guard["dataset_fingerprint"] != case.dataset_fingerprint:
        raise RuntimeError("Case is outside the frozen confirmation cohort")
    if _sha256(EXPERIMENT_ROOT/"models"/"training-freeze.json") != guard["training_freeze_sha256"]:
        raise RuntimeError("Training configuration changed before confirmation generation")
    seed_material = f"fresh-confirmation-v2|{stream}|{case.family}|{case.seed}|{case.amplitude:g}"
    digest = hashlib.sha256(seed_material.encode()).digest()
    seed_words = np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    dimension = 6 if case.family == "uc" else 8
    if case.family == "uc":
        features = rng.uniform(1-case.amplitude, 1+case.amplitude, size=(count, dimension))
    else:
        features = 1+case.amplitude*rng.uniform(-1., 1., size=(count, dimension))
    costs = costs_from_features(case, features)
    return costs, features, dict(stream=stream, seed_material=seed_material, seed_words=seed_words,
        count=count, independent_from_original_training_stream=True,
        generated_only_after_fitted_model_freeze=True,
        cost_sha256=hashlib.sha256(costs.tobytes()).hexdigest(),
        parameter_sha256=hashlib.sha256(features.tobytes()).hexdigest())


def load_training_pools(case: FrozenCase, indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
    manifest = dataset_manifest(verify=False)
    indices = list(range(case.n_train)) if indices is None else list(indices)
    pools = []
    for i in indices:
        if not 0 <= int(i) < case.n_train:
            raise ValueError("Pool index must belong to the training set")
        path = case.source_directory / "pools" / f"{case.case_id}_train{int(i):03d}.npz"
        relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        if _sha256(path) != manifest["file_sha256"][relative]:
            raise RuntimeError(f"Portable source pool changed: {relative}")
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            solutions = data["solutions"].copy()
            objectives = data["objectives"].copy()
            tags = data["source_tags"].tolist()
        if meta["case_id"] != case.case_id or int(meta["train_index"]) != i or not meta["cached_optimum_included"]:
            raise RuntimeError("Pool identity/optimum flag mismatch")
        if meta["max_solutions"] != 64 or meta["time_limit"] != 30.:
            raise RuntimeError("Training pool settings differ from original frozen pool protocol")
        for x, obj in zip(solutions, objectives):
            if not validate_solution(case.model, x)["valid"] or abs(float(case.train_costs[i]@x)-obj)>recovery_tolerance(obj):
                raise RuntimeError("Exported training pool has an invalid candidate")
        pools.append(dict(meta, solutions=solutions, incumbent_solutions=solutions,
                          objectives=objectives, source_tags=tags, exported_pool_sha256=_sha256(path)))
    return pools


def collect_training_pools(case: FrozenCase, indices: Sequence[int] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Compatibility loader: no training costs are resampled or MIPs rerun."""
    return load_training_pools(case, indices=indices)


def complete_training_pools(case: FrozenCase, pools: list[dict[str, Any]] | None = None,
                            per_lp_time_limit: float = 10., progress: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    """Certify continuous completions before weighted Neural Diving fitting.

    All integer coordinates are fixed to each original incumbent assignment.
    Cached globally optimal training solutions already certify their conditional
    optimum and need no redundant solve. For knapsack all coordinates are
    integer, so conditional completion is the same point. Every other UC pool
    candidate receives a genuine original-constraint LP solve. Failed completion
    blocks training rather than dropping a label or replacing the pool.
    """
    pools = load_training_pools(case) if pools is None else pools
    completed = []
    integer = case.model.integer_indices
    for source in pools:
        i = int(source["train_index"])
        path = EXPERIMENT_ROOT / "completed-pools" / case.case_id / f"train{i:03d}.npz"
        guard = dict(case_id=case.case_id, train_index=i, dataset_fingerprint=case.dataset_fingerprint,
            source_pool_sha256=source["exported_pool_sha256"], per_lp_time_limit=float(per_lp_time_limit),
            completion_version="fix-all-original-integers-lp-v1")
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
                if metadata["guard"] != guard:
                    raise RuntimeError("Conditional LP pool cache data/settings changed")
                solutions = data["solutions"].copy()
                objectives = data["objectives"].copy()
                old_objectives = data["original_incumbent_objectives"].copy()
        else:
            begin = perf_counter()
            solutions, objectives, records = [], [], []
            cost = case.train_costs[i]
            for j, (incumbent, tag) in enumerate(zip(source["solutions"], source["source_tags"])):
                if case.family == "knapsack":
                    x = incumbent.copy()
                    result_meta = dict(certified=True, status="all coordinates integer; conditional point unique", seconds=0.)
                elif tag == "cached_optimum":
                    x = incumbent.copy()
                    if abs(float(cost@x)-case.train_objectives[i])>recovery_tolerance(case.train_objectives[i]):
                        raise RuntimeError("Cached training optimum label does not certify conditional completion")
                    result_meta = dict(certified=True, status="frozen globally certified optimum also conditionally optimal", seconds=0.)
                else:
                    assignment = np.rint(incumbent[integer]).astype(np.int64)
                    fixed = {int(index): int(value) for index, value in zip(integer, assignment)}
                    solved = solve(case.model, cost, fixed=fixed, seed=case.seed+i,
                                   time_limit=per_lp_time_limit, relax_integrality=True)
                    if not (solved.certified and solved.valid and solved.x is not None):
                        raise RuntimeError(f"Uncertified conditional pool LP: {case.case_id}/train{i}/candidate{j}: {solved.status}")
                    x = solved.x.copy()
                    result_meta = solved.to_dict(include_solution=False)
                objective = float(cost@x)
                if not validate_solution(case.model, x)["valid"] or not np.array_equal(np.rint(x[integer]), np.rint(incumbent[integer])):
                    raise RuntimeError("Conditional LP changed integer assignment or original feasibility")
                if objective-source["objectives"][j] > recovery_tolerance(source["objectives"][j]):
                    raise RuntimeError("A conditionally optimal LP completion is worse than its feasible source incumbent")
                solutions.append(x)
                objectives.append(objective)
                records.append(dict(pool_index=j, source_tag=tag, source_objective=float(source["objectives"][j]),
                                    completed_objective=objective, completion=result_meta))
            solutions = np.asarray(solutions, dtype=float)
            objectives = np.asarray(objectives, dtype=float)
            old_objectives = np.asarray(source["objectives"], dtype=float)
            metadata = dict(guard=guard, conditional_lp_completion_certified=True,
                            extra_conditional_lp_seconds=perf_counter()-begin, records=records)
            _atomic_npz(path, solutions=solutions, objectives=objectives,
                original_incumbent_objectives=old_objectives, source_tags=np.asarray(source["source_tags"]),
                metadata=json.dumps(metadata, ensure_ascii=True, allow_nan=False))
        if solutions.shape != source["solutions"].shape or not np.array_equal(np.rint(solutions[:, integer]), np.rint(source["solutions"][:, integer])):
            raise RuntimeError("Completed pool shape/integer assignments differ from source")
        for x, obj in zip(solutions, objectives):
            if not validate_solution(case.model, x)["valid"] or abs(float(case.train_costs[i]@x)-obj)>recovery_tolerance(obj):
                raise RuntimeError("Invalid cached conditional LP completion")
        completed.append(dict(source, solutions=solutions, incumbent_solutions=solutions, objectives=objectives,
            original_incumbent_objectives=old_objectives, conditional_lp_completion_certified=True,
            conditional_lp_completion_records=metadata["records"],
            extra_conditional_lp_seconds=metadata["extra_conditional_lp_seconds"],
            completed_pool_sha256=_sha256(path)))
        if progress is not None:
            progress(case_id=case.case_id, train_index=i, pool_size=len(solutions), completed=len(completed))
    return completed

