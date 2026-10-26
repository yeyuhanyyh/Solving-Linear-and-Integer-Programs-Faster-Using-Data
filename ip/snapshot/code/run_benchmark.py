"""Serial paired timing after all requested offline checkpoints are complete.

Every test objective is retained, including infeasible restrictions, timeouts,
and failed strategy predictions. There is no Full fallback. Fresh Full solves
must match the frozen, independently certified original reference objectives.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
import pickle
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits

import adapter
from runner_common import (ROOT, CODE, METHODS, MODEL_KEYS, MODEL_FOR_METHOD, PhaseLock, clean, dump,
    event, read, utc, freeze, selected, checkpoint, object_digest, ensure_cpu, runtime, digest)
from dimension_diagnostics import (diagnose_fixings, assert_matched_fixings, target_common_dimension,
    affine_equalities_metrics, native_strategy_metrics)
import experiment_core
from uc_model import validate_uc_solution


def active_offline_processes() -> list[dict[str, Any]]:
    """Check independently launched Python computation on Windows.

    The shared lock prevents normal runner overlap. This startup check also
    catches independently started collectors or benchmark jobs. The process
    snapshot uses the Win32 API directly; CIM is unavailable in this sandbox.
    Ancestors include the virtual-environment Python launcher and are ignored.
    Other Python processes conservatively block a timing run until inspected.
    This check is outside every measurement.
    """
    if os.name != "nt":
        inventory={}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat=(entry/"stat").read_text()
                parent=int(stat[stat.rfind(")")+2:].split()[1])
                command=(entry/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
                inventory[int(entry.name)]=dict(pid=int(entry.name),parent_pid=parent,command=command)
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                continue
        ancestors=set()
        pid=os.getpid()
        while pid and pid not in ancestors:
            ancestors.add(pid)
            pid=inventory.get(pid,{}).get("parent_pid",0)
        return [v for pid,v in inventory.items() if pid not in ancestors and
                any(name in v["command"] for name in ("run_training.py","run_benchmark.py","collect_pools.py"))]
    import ctypes
    from ctypes import wintypes
    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR*260)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    inventory = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        success = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())
        while success:
            inventory[int(entry.th32ProcessID)] = dict(pid=int(entry.th32ProcessID),
                parent_pid=int(entry.th32ParentProcessID), name=str(entry.szExeFile))
            success = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(snapshot)
    ancestors = set()
    pid = os.getpid()
    while pid and pid not in ancestors:
        ancestors.add(pid)
        pid = inventory.get(pid, {}).get("parent_pid", 0)
    return [value for pid, value in inventory.items() if pid not in ancestors
            and value["name"].lower().startswith(("python", "pypy"))]


def load_models(case: adapter.FrozenCase, methods: list[str], frozen: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    models, markers = {}, {}
    fitcase = adapter.training_case(case)
    for method in methods:
        if method == "full":
            continue
        marker = checkpoint(case, MODEL_FOR_METHOD[method], frozen)
        if marker is None:
            raise RuntimeError(f"Offline checkpoint not complete: {case.case_id}/{method}")
        key=MODEL_FOR_METHOD[method]
        markers[method] = marker
        if key in models:
            models[method]=models[key]
            continue
        path = ROOT / "models" / case.case_id / key
        if method in ("ours", "svm_native"):
            with (path / ("affine.pkl" if method == "ours" else "svm.pkl")).open("rb") as stream:
                models[method] = pickle.load(stream)
        elif method == "mlopt_native":
            from mlopt_baseline import MLOPTModel
            models[method] = MLOPTModel.load(path / "mlopt.pkl")
        elif method.startswith("neural_diving_"):
            import neural_diving_baseline
            models[method] = neural_diving_baseline.load(fitcase, path / "neural-diving.pt")
        elif method == "li_bounds_residual":
            from preference_baseline import load_fitted
            models[method] = load_fitted(fitcase, path)
        else:
            raise ValueError(method)
        models[key]=models[method]
        ensure_cpu(models[method])
    return models, markers


def restriction_check(method: str, model: Any, case: adapter.FrozenCase,
                      x: np.ndarray | None, result: dict[str, Any], eq: list[Any]) -> dict[str, Any]:
    if x is None:
        return dict(valid=False, residual=None, check="no returned solution")
    if method == "full":
        return dict(valid=True, residual=0., check="original formulation",
                    dimension_diagnostics=dict(common_dimension=len(case.commitment_indices),presolved_dimension=None))
    if method == "ours":
        binary = np.rint(x[case.commitment_indices]).astype(np.int64)
        contained = experiment_core.exact_slice_contains(model, binary)
        residual = float(np.max(np.abs(model["normals"] @ (binary-model["anchor"])), initial=0.))
        return dict(valid=bool(contained), residual=residual, check="exact rational affine membership on rounded integer coordinates",
                    dimension_diagnostics=dict(common_dimension=model["rank"],nominal_affine_dimension=model["rank"],presolved_dimension=None))
    if method == "svm_native":
        binary = np.rint(x[case.commitment_indices]).astype(np.int64)
        residual = max((abs(float(e.as_row(len(binary))[0] @ binary)-e.rhs) for e in eq), default=0.)
        return dict(valid=residual <= adapter.FEASIBILITY_TOLERANCE, residual=residual, check="all predicted equalities")
    if method.startswith("neural_diving_"):
        fixed = result.get("best_fixed")
        if fixed is None:
            fixed = result.get("best_partial_assignment")
        if fixed is None:
            raise RuntimeError("Neural Diving returned a solution without the winning fixed assignment for independent audit")
        residual = max((abs(float(x[int(i)])-float(value)) for i, value in fixed.items()), default=0.)
        dimensions=(assert_matched_fixings(case,fixed,target_K=target_common_dimension(case))
                    if method=="neural_diving_matched" else diagnose_fixings(case,fixed))
        return dict(valid=residual <= adapter.FEASIBILITY_TOLERANCE, residual=residual,
                    check="winning partial integer assignment", fixed_count=len(fixed),dimension_diagnostics=dimensions)
    strategy_id = result.get("strategy_id")
    if strategy_id is None:
        raise RuntimeError(f"{method} returned a solution without a strategy identifier")
    bank = model.bank
    strategy = bank.strategies[int(strategy_id)]
    integer_residual = float(np.max(np.abs(x[bank.integer_indices]-strategy.integer_values), initial=0.))
    equality_residual = float(np.max(np.abs(bank.E @ x-bank.beq), initial=0.))
    activity = np.asarray(bank.G[strategy.active_rows] @ x).ravel()
    rhs = bank.h[strategy.active_rows]
    if method == "mlopt_native":
        learned_residual = float(np.max(np.abs(activity-rhs), initial=0.))
        check = "complete integer assignment and active equality strategy"
    elif method == "li_bounds_residual":
        learned_residual = float(np.max(activity-rhs, initial=0.))
        check = "complete integer assignment, permanent equalities, selected strategy inequalities"
    else:
        raise ValueError(method)
    residual = max(integer_residual, equality_residual, learned_residual)
    return dict(valid=residual <= adapter.FEASIBILITY_TOLERANCE, residual=residual,
                integer_assignment_residual=integer_residual, permanent_equality_residual=equality_residual,
                strategy_residual=learned_residual, check=check,
                dimension_diagnostics=native_strategy_metrics(case,retained_strategies=len(model.selected),
                                        evaluated_candidates=raw_candidates(result)))


def raw_candidates(result):
    return result.get("candidate_count",result.get("diagnostics",{}).get("n_candidates"))


def run_method(case: adapter.FrozenCase, method: str, models: dict[str, Any],
               cost: np.ndarray, features: np.ndarray, solver_seed: int,
               neural_seed: int) -> dict[str, Any]:
    # This timer encloses prediction, construction, every strategy/subproblem
    # solve, and common original plus learned-constraint validation. Scoring
    # against the reference, physics audit and output serialization follow it.
    started = time.perf_counter()
    model = models.get(method)
    equalities, hint, kwargs = [], {}, {}
    if method == "ours":
        kwargs["affine"] = model
    elif method == "svm_native":
        equalities, hint = model.predict_equalities(features)
        kwargs["equalities"] = equalities
    hint_seconds = time.perf_counter()-started
    if method in ("full", "ours", "svm_native"):
        # Preserve the original control solver implementation, while charging
        # prediction against the same outer method budget as every baseline.
        remaining = max(1.e-6, 30.-hint_seconds)
        raw, x = experiment_core.solve(case.original_base, cost, seed=solver_seed, time_limit=remaining, **kwargs)
        raw = dict(raw, x=x, solver_seed=solver_seed)
    elif method == "mlopt_native":
        raw = model.predict(features, cost)
    elif method.startswith("neural_diving_"):
        from neural_diving_baseline import solve_cost
        raw = solve_cost(model, adapter.training_case(case), cost, seed=neural_seed, total_time_limit=30.,
                         mode=("matched" if method=="neural_diving_matched" else "native"),
                         K=(target_common_dimension(case) if method=="neural_diving_matched" else None),solve_fn=adapter.solve)
    elif method == "li_bounds_residual":
        raw = model.solve(cost, features, total_time_limit=30.)
    else:
        raise ValueError(method)
    x = raw.get("x")
    x = None if x is None else np.asarray(x, dtype=float)
    original = (dict(valid=False, original_violation=None, integrality_residual=None) if x is None
                else adapter.validate_solution(case.model, x))
    restriction = restriction_check(method, model, case, x, raw, equalities)
    valid = bool(raw.get("valid", False) and original["valid"] and restriction["valid"])
    objective = None if x is None or not np.isfinite(x).all() else float(cost @ x)
    seconds = time.perf_counter() - started
    budget_exceeded = seconds > 30.
    valid = bool(valid and not budget_exceeded)
    # Exact-rank and candidate-distribution diagnostics explain comparisons;
    # they are not required to produce/validate a solution and stay outside the
    # algorithm timer, alongside independent physics and serialization.
    if method=="svm_native":
        rows=[e.as_row(len(case.commitment_indices)) for e in equalities]
        restriction["dimension_diagnostics"]=affine_equalities_metrics(
            np.asarray([r for r,_ in rows],dtype=float).reshape(-1,len(case.commitment_indices)),
            np.asarray([b for _,b in rows],dtype=float))
    if method.startswith("neural_diving_"):
        raw["all_subproblem_dimension_diagnostics"]=[]
        for subproblem in raw.get("submip_records",[]):
            fixed=subproblem["fixed"]
            raw["all_subproblem_dimension_diagnostics"].append(
                assert_matched_fixings(case,fixed,target_K=target_common_dimension(case))
                if method=="neural_diving_matched" else diagnose_fixings(case,fixed))
    raw_local = bool(raw.get("certified", False))
    if method == "li_bounds_residual":
        raw_local = bool(raw.get("reduced_certified", raw.get("reduced_optimal", False)))
    local_certified = bool(raw_local and restriction["valid"])
    # Only an original-formulation solve has its own original global solver
    # certificate. Reference-based optimal value recovery is separate.
    global_original_certified = bool(method == "full" and local_certified)
    physics = None
    if case.family == "uc" and x is not None:
        physics = validate_uc_solution(case.original_base, x)
        if bool(physics["valid"]) != bool(original["valid"]):
            # Physics is a second, independent formulation audit; disagreement
            # is an implementation/data problem, not an ordinary failure.
            raise RuntimeError(f"UC independent physics audit disagrees with original validation for {case.case_id}/{method}")
    raw = dict(raw)
    raw.pop("x", None)
    raw.pop("solution", None)
    return clean(dict(method=method, seconds=seconds, online_seconds=seconds,
        prediction_hint_seconds=hint_seconds, raw_method_result=raw,
        solution=x, objective=objective, original_feasible=original["valid"],
        valid=valid, original_check=original, restriction_check=restriction,
        method_time_limit_seconds=30., method_wall_budget_exceeded=budget_exceeded,
        solved_model_certified=local_certified, raw_solved_model_certificate=raw_local,
        original_global_solver_certified=global_original_certified,
        certificate_scope=("original" if method == "full" else "selected strategy equality system" if method == "mlopt_native"
                           else "restricted/subproblem model"),
        status=raw.get("status"), physics=physics,
        equalities=[asdict(e) for e in equalities], hint_diagnostic=hint,
        solver_seed=solver_seed, neural_sampling_seed=(neural_seed if method.startswith("neural_diving_") else None)))


def existing_trials(path: Path, case_id: str, fingerprint: str) -> dict[tuple[int, str], dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"Blank line in retained raw results at {path}:{number}")
        row = json.loads(line)
        if row["case"] != case_id or row["freeze_fingerprint"] != fingerprint:
            raise RuntimeError(f"Raw results belong to different frozen data/code at {path}:{number}")
        key = (int(row["test_index"]), row["method"])
        if key in records or key[0] not in range(40) or key[1] not in METHODS:
            raise RuntimeError(f"Duplicated or invalid raw trial: {path}:{number}")
        records[key] = row
    return records


def freeze_confirmation_cohort(frozen):
    # Configuration is frozen before training. This second freeze binds every
    # final learned model and gate BEFORE the first new confirmation draw.
    cases=adapter.list_case_ids()
    markers={}
    for cid in cases:
        case=adapter.load_case(cid,verify=True)
        for key in MODEL_KEYS:
            markers[f"{cid}/{key}"]=object_digest(checkpoint(case,key,frozen))
    proof=dict(cases=cases,dataset_fingerprint=frozen["dataset_fingerprint"],
        training_freeze_sha256=digest(ROOT/"models"/"training-freeze.json"),
        training_freeze_fingerprint=frozen["fingerprint"],model_checkpoint_fingerprints=markers,
        all_train_validation_sanity_gates_passed=True,all_methods_frozen_before_test_generation=True)
    proof["fingerprint"]=object_digest(proof)
    path=ROOT/"results"/"confirmation-model-freeze.json"
    if path.exists() and read(path)!=proof:
        raise RuntimeError("Learned models changed after independent confirmation was frozen")
    if not path.exists():
        dump(path,proof)
    return proof


def confirmation_data(case, frozen, protocol):
    folder=ROOT/"confirmation"/case.case_id
    folder.mkdir(parents=True,exist_ok=True)
    costs,features,sampler=adapter.draw_confirmation_costs(case,
        count=int(protocol.get("confirmation_count",40)),stream=int(protocol.get("fresh_test_stream",2026100919)))
    guard=dict(case=case.case_id,freeze_fingerprint=frozen["fingerprint"],sampler=sampler,
               model_cohort_fingerprint=read(ROOT/"results"/"confirmation-model-freeze.json")["fingerprint"])
    path=folder/"costs.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if not np.array_equal(saved["costs"],costs) or not np.array_equal(saved["features"],features):
                raise RuntimeError("Independent confirmation costs changed")
    else:
        with path.with_name(path.name+".writing").open("wb") as stream:
            np.savez_compressed(stream,costs=costs,features=features)
        os.replace(path.with_name(path.name+".writing"),path)
    guard["costs_npz_sha256"]=digest(path)
    meta=folder/"guard.json"
    if meta.exists() and read(meta)!=guard:
        raise RuntimeError("Independent confirmation sampler/model guard changed")
    if not meta.exists():
        dump(meta,guard)
    references={}
    refpath=folder/"full-references.jsonl"
    if refpath.exists():
        for line in refpath.read_text(encoding="utf-8").splitlines():
            row=json.loads(line)
            index=int(row["test_index"])
            if index in references or index not in range(40) or row["guard_fingerprint"]!=object_digest(guard):
                raise RuntimeError("Duplicated or mismatched fresh Full reference")
            if not row["result"]["certified"] or not row["result"]["valid"]:
                raise RuntimeError("Retained blocked fresh Full reference requires investigation")
            x=np.asarray(row["solution"],dtype=float)
            if not adapter.validate_solution(case.model,x)["valid"] or abs(float(costs[index]@x)-row["result"]["objective"])>adapter.recovery_tolerance(row["result"]["objective"]):
                raise RuntimeError("Invalid retained independent Full reference")
            references[index]=row
    with refpath.open("a",encoding="utf-8") as stream:
        for i,cost in enumerate(costs):
            if i in references:
                continue
            result,x=experiment_core.solve(case.original_base,cost,seed=case.seed+550000+i,time_limit=30.)
            physics=validate_uc_solution(case.original_base,x) if case.family=="uc" and x is not None else None
            row=clean(dict(case=case.case_id,test_index=i,guard_fingerprint=object_digest(guard),
                result=result,solution=x,physics=physics,not_passed_to_any_fitter=True,utc=utc()))
            stream.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
            references[i]=row
            if not result["certified"] or not result["valid"] or (physics is not None and not physics["valid"]):
                raise RuntimeError(f"Fresh Full reference uncertified: {case.case_id}/{i}; raw failure retained")
            if (i+1)%10==0:
                event(event="fresh_reference_progress",case=case.case_id,completed=i+1)
    objectives=np.asarray([references[i]["result"]["objective"] for i in range(40)],dtype=float)
    return costs,features,objectives,guard


def run_case(case: adapter.FrozenCase, methods: list[str], frozen: dict[str, Any], protocol) -> None:
    costs,features,reference_objectives,confirmation_guard=confirmation_data(case,frozen,protocol)
    load_started = time.perf_counter()
    models, markers = load_models(case, methods, frozen)
    model_load_seconds = time.perf_counter()-load_started
    folder = ROOT / "results" / case.case_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "case-manifest.json"
    manifest = dict(case=case.case_id, source_hashes=case.source_hashes,
                    freeze_fingerprint=frozen["fingerprint"], method_checkpoints={})
    if manifest_path.exists():
        saved = read(manifest_path)
        if any(saved[key] != manifest[key] for key in ("case", "source_hashes", "freeze_fingerprint")):
            raise RuntimeError(f"Benchmark source manifest mismatch: {case.case_id}")
        manifest["method_checkpoints"] = saved["method_checkpoints"]
    for method, marker in markers.items():
        marker_hash = object_digest(marker)
        old = manifest["method_checkpoints"].get(method)
        if old is not None and old != marker_hash:
            raise RuntimeError(f"Model changed after timed results: {case.case_id}/{method}")
        manifest["method_checkpoints"][method] = marker_hash
    dump(manifest_path, manifest)
    # Each method gets exactly one identical TRAIN objective; no test solution
    # or parameter influences warming, fitting, or hyperparameter selection.
    warm = []
    for method in methods:
        record = run_method(case, method, models, case.train_costs[0], case.train_features[0],
                            solver_seed=case.seed + 88000, neural_seed=case.seed + 88000)
        record["warm_training_index"] = 0
        warm.append(record)
    stamp = utc().replace(":", "-")
    dump(folder / f"warm-{stamp}.json", dict(model_load_seconds=model_load_seconds, records=warm,
                                           train_only=True, not_used_in_summary=True))
    path = folder / "trials.jsonl"
    existing = existing_trials(path, case.case_id, frozen["fingerprint"])
    for (i, method), retained in existing.items():
        if retained["reference_objective"] != float(reference_objectives[i]):
            raise RuntimeError(f"Retained reference differs from frozen source: {case.case_id}/{i}/{method}")
        if method == "full" and not (retained["original_global_solver_certified"] and retained["quality_recovered"]):
            # Failed references were intentionally retained before raising.
            # Resumption cannot skip that hard failure and publish a complete
            # marker; investigate the preserved record in a separate run.
            raise RuntimeError(f"Retained blocked Full reference prevents resumption: {case.case_id}/{i}")
    order_rng = np.random.default_rng(np.random.SeedSequence([case.seed, 20261009, 31009]))
    with path.open("a", encoding="utf-8") as stream:
        for i, (cost, feature) in enumerate(zip(costs, features)):
            order = [str(m) for m in order_rng.permutation(METHODS)]
            for position, method in enumerate(order):
                if method not in methods or (i, method) in existing:
                    continue
                begun = utc()
                try:
                    row = run_method(case, method, models, cost, feature, solver_seed=case.seed+i,
                                     neural_seed=case.seed + 104729 * (i+1))
                    cached_reference = float(reference_objectives[i])
                    tolerance = adapter.recovery_tolerance(cached_reference)
                    quality_recovered = bool(row["valid"] and row["objective"] is not None and
                                             abs(row["objective"]-cached_reference) <= tolerance)
                    row.update(case=case.case_id, family=case.family, seed=case.seed, amplitude=case.amplitude,
                        test_index=i, freeze_fingerprint=frozen["fingerprint"], reference_objective=cached_reference,
                        test_kind="independent_fresh_confirmation",confirmation_guard_fingerprint=object_digest(confirmation_guard),
                        recovery_tolerance=tolerance, objective_error=(None if row["objective"] is None else row["objective"]-cached_reference),
                        quality_recovered=quality_recovered,
                        certified_recovered=bool(quality_recovered and row["solved_model_certified"]),
                        table_compatible_recovered=bool(quality_recovered and (row["solved_model_certified"] if method in ("full", "ours", "svm_native") else True)),
                        method_order=order, method_order_position=position, started_utc=begun, completed_utc=utc(),
                        ambient_integer_dimension=len(case.model.integer_indices),
                        affine_coordinate_dimension=len(case.commitment_indices), affine_rank=int(case.affine["rank"]),
                        full_fallback_used=False)
                    row = clean(row)
                    stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    existing[i, method] = row
                    if method == "full" and (not row["original_global_solver_certified"] or not quality_recovered):
                        raise RuntimeError(f"Fresh Full not certified or differs from frozen original reference: {case.case_id}/{i}; raw record retained")
                except Exception:
                    dump(folder / "blocked.json", dict(case=case.case_id, method=method, test_index=i,
                         utc=utc(), exception=traceback.format_exc(), all_previous_raw_records_preserved=True))
                    raise
            if (i+1) % 5 == 0 or i == 0:
                dump(ROOT / "logs" / "benchmark-status.json", dict(case=case.case_id, phase="serial_test",
                     completed_test_objectives=i+1, total=40, utc=utc()))
                event(event="test_progress", case=case.case_id, completed=i+1, records=len(existing))
    expected = {(i, m) for i in range(40) for m in methods}
    if not expected <= set(existing):
        raise RuntimeError(f"Incomplete requested raw tests: {case.case_id}")
    # This marker can be extended by staged methods, but original raw records
    # and model hashes never change. An all-method marker requires all 280.
    completed_methods = [m for m in METHODS if all((i, m) in existing for i in range(40))]
    dump(folder / "complete.json", dict(case=case.case_id, records=len(existing),
         completed_methods=completed_methods, all_seven_complete=set(completed_methods)==set(METHODS),
         fingerprint=frozen["fingerprint"], utc=utc()))
    event(event="case_complete", case=case.case_id, records=len(existing), completed_methods=completed_methods)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cases = selected(args.cases, adapter.list_case_ids())
    methods = selected(args.methods, METHODS)
    if "full" not in methods:
        raise ValueError("Every timed selection must include Full for paired fresh-reference timing")
    frozen = freeze(create=False)
    # Check EVERY requested checkpoint before the first timed solve.
    case_data = []
    for cid in cases:
        case = adapter.load_case(cid, verify=True)
        case_data.append(case)
        for method in methods:
            if method != "full" and checkpoint(case, MODEL_FOR_METHOD[method], frozen) is None:
                raise RuntimeError(f"All offline training must finish before timing: missing {cid}/{method}")
    if args.validate_only:
        event(event="benchmark_inputs_validated", cases=cases, methods=methods, fingerprint=frozen["fingerprint"])
        return
    with PhaseLock("serial_benchmark"):
        active = active_offline_processes()
        if active:
            raise RuntimeError(f"Offline pool collection/training processes still active: {active}")
        freeze_confirmation_cohort(frozen)
        protocol=read(ROOT/"protocol.json")
        current_runtime=runtime()
        runtime_path=ROOT/"logs"/"benchmark-runtime.json"
        if runtime_path.exists() and read(runtime_path)!=current_runtime:
            raise RuntimeError("The serial paired benchmark runtime changed between resumed measurements")
        if not runtime_path.exists():
            dump(runtime_path,current_runtime)
        dump(ROOT / "logs" / "benchmark-start.json", dict(cases=cases, methods=methods,
             fingerprint=frozen["fingerprint"], pid=os.getpid(), utc=utc(), single_process=True,
             training_complete_before_first_measurement=True, offline_process_check=active))
        with threadpool_limits(limits=1):
            for case in case_data:
                run_case(case, methods, frozen, protocol)
        event(event="serial_benchmark_complete", cases=cases, methods=methods)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
