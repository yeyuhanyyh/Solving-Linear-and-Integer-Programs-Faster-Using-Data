"""Small, source-bound CPU SVM/MLOPT controls for new arbitrary-size UC cases.

No old data/model/result is opened. Imports alone never fit or predict. Only
the caller's certified training arrays are used. Common dimension is commitment
u; startup/shutdown remain binary and dispatch remains continuous. The MLOPT
matching anchor is its native top-ten feasible objective winner, not score top1.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import importlib
import json
import math
import pickle
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent.parent
CODE = EXPERIMENT / "code"
CONTROLS = CODE / "controls"
MATCHED = EXPERIMENT / "matched-k-supplement-20261010/code"
NATURAL = EXPERIMENT / "natural-distribution-pilot-20261012/covered-geometry-followup/code"
VERSION = "new-uc-common-u-svm-native-anchor-mlopt-v1-20261025"
EXPECTED_SOURCES = {
    CODE / "portable_adapter.py": "aa2cbdb7a06f5955a8a030401a5ffe3fbe43e13fd3e9719bb68bf569c6e85850",
    CODE / "dimension_diagnostics.py": "adcec8c49e24fa373cdf92532e1a3e1c980fd10dea722d1225223f50e398c099",
    CODE / "mlopt_baseline.py": "ee35123e6212a72c663e3af98baf04fafd9d89b81ea64027934658706b57a068",
    CONTROLS / "uc_model.py": "4e0b4b0d24fa1b2e54e3f399569976e1a53f429bd0ff8196976fa0713bacbf38",
    MATCHED / "svm_match.py": "4324295873b116df85b965efe275d4754f43bac7a3e7cbb7a5c6798881312d67",
    MATCHED / "strategy_match.py": "74ecd5b97480d3d50a5c0c1d5c128c33fc756f016409523d442950d1c6bd0434",
    NATURAL / "fast_baselines.py": "d9e36195d8dc7663def225fc87fdac5571f8ee9996a2699df92395c24c8618be",
    NATURAL / "native_anchor.py": "2924cf3204c9ed8cfe89db0f0b05e91f500602f190fd426ce9dd9051e945c972",
    NATURAL / "mlopt_bridge.py": "443e5873cd30b6549a453e5c055caf933ed601aee31e8541335c6e5069f22a68",
    NATURAL / "mlopt_impl.py": "4886b0283fbec2e59d23f4901bfcd3c1a9d46fe2fb8a54e515e177e07665a322",
}
_RUNTIME = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    for path, expected in EXPECTED_SOURCES.items():
        if _sha(path) != expected:
            raise ValueError("Read-only baseline source changed: " + str(path))
    # The first entry has precedence. The canonical generic portable adapter
    # and MLOPT fit module must win over older experiment-local copies.
    for path in reversed((CODE, CONTROLS, MATCHED, NATURAL)):
        while str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
    expected_modules = {
        "portable_adapter": CODE / "portable_adapter.py",
        "dimension_diagnostics": CODE / "dimension_diagnostics.py",
        "mlopt_baseline": CODE / "mlopt_baseline.py",
        "uc_model": CONTROLS / "uc_model.py",
        "svm_match": MATCHED / "svm_match.py",
        "strategy_match": MATCHED / "strategy_match.py",
        "fast_baselines": NATURAL / "fast_baselines.py",
        "native_anchor": NATURAL / "native_anchor.py",
        "mlopt_bridge": NATURAL / "mlopt_bridge.py",
        "mlopt_impl": NATURAL / "mlopt_impl.py",
    }
    modules = {}
    for name, path in expected_modules.items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != path.resolve():
            raise ValueError("An old/shadowed baseline module was imported: " + name)
        modules[name] = module
    import numpy as np
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
    _RUNTIME = dict(modules, np=np, threadpool_limits=threadpool_limits, torch=torch)
    return _RUNTIME


def _array_hash(*arrays):
    r = _runtime()
    np = r["np"]
    result = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        result.update(str(value.shape).encode())
        result.update(str(value.dtype).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def _case_binding(case):
    r = _runtime()
    np = r["np"]
    matrix = case.model.A.tocsr()
    return dict(version=VERSION, helper_source_sha256=_sha(__file__), case_id=str(case.case_id),
        n_train=len(case.train_features), target_K=int(case.affine["rank"]),
        supplied_dataset_fingerprint=str(case.dataset_fingerprint), supplied_source_hashes=dict(case.source_hashes),
        common_indices=np.asarray(case.commitment_indices, dtype=int).tolist(),
        training_hash=_array_hash(case.train_features, case.train_costs, case.train_solutions, case.train_objectives),
        original_model_hash=_array_hash(matrix.data, matrix.indices, matrix.indptr, np.asarray(matrix.shape),
            case.model.row_lower, case.model.row_upper, case.model.col_lower, case.model.col_upper, case.model.integrality),
        base_cost_hash=_array_hash(case.original_base["base_cost"]),
        physical_metadata_hash=_array_hash(*[case.original_base["metadata"][key] for key in (
            "capacity", "minimum_output", "ramp_up", "ramp_down", "minimum_up_time", "minimum_down_time", "demand",
            "initial_commitment", "initial_generation", "initial_down_time")]),
        source_hashes={str(path.resolve()): expected for path, expected in EXPECTED_SOURCES.items()},
        original_labels_generation_included=False, test_information_used=False)


def _validate_training(case):
    r = _runtime()
    np, adapter = r["np"], r["portable_adapter"]
    if case.family != "uc":
        raise ValueError("This new screen is for UC only")
    G, T = int(case.original_base["n_units"]), int(case.original_base["n_times"])
    q, D, n = G * T, case.model.n_variables, len(case.train_features)
    if D != 4 * q or n < 2:
        raise ValueError("Need complete four-block UC variables and at least two training examples")
    if not np.array_equal(case.commitment_indices, np.arange(q)):
        raise ValueError("Common dimension must be u only, in time-major order")
    if not np.array_equal(case.model.integrality, np.r_[np.ones(3 * q, dtype=int), np.zeros(q, dtype=int)]):
        raise ValueError("Original u/v/w must be binary; original p must remain continuous")
    if not 0 <= int(case.affine["rank"]) <= q:
        raise ValueError("Training-derived common K is outside the u dimension")
    if np.asarray(case.train_features).ndim != 2 or np.asarray(case.train_features).shape[1] < 1:
        raise ValueError("Need all varying objective parameters as training features")
    if np.asarray(case.train_costs).shape != (n, D) or np.asarray(case.train_solutions).shape != (n, D):
        raise ValueError("Training optimizers/costs must include all original UC variables")
    if np.asarray(case.train_objectives).shape != (n,):
        raise ValueError("Need one supplied original objective for every training solution")
    for cost, x, reference in zip(case.train_costs, case.train_solutions, case.train_objectives):
        if not adapter.validate_solution(case.model, x)["valid"] or not r["uc_model"].validate_uc_solution(case.original_base, x)["valid"]:
            raise ValueError("An archived training optimizer fails original UC physics")
        if abs(float(np.asarray(cost) @ x) - reference) > adapter.recovery_tolerance(float(reference)):
            raise ValueError("Training objective does not match the supplied optimizer")
    return n, q, D


def fit_baselines(case):
    """Return (SVM, MLOPT, metadata), using only the supplied n training rows.

    Timers exclude common training-label production/certification, this initial
    source/runtime/training-data validation, and later serialization. The SVM
    interval includes its lazy sklearn import. No ND or Li network is fitted.
    """
    setup_wall, setup_cpu = time.perf_counter(), time.process_time()
    r = _runtime()
    n, q, D = _validate_training(case)
    binding = _case_binding(case)
    setup = dict(wall_seconds=time.perf_counter() - setup_wall, cpu_seconds=time.process_time() - setup_cpu)
    started, cpu_started = time.perf_counter(), time.process_time()
    with r["threadpool_limits"](limits=1):
        svm, svm_raw = r["fast_baselines"].fit_coordinate_svm_small(case)
    svm_wall, svm_cpu = time.perf_counter() - started, time.process_time() - cpu_started
    svm._new_uc_source_binding = binding
    seed = int(case.metadata.get("baseline_seed", case.seed)) % (2**31 - 1)
    config = r["mlopt_baseline"].Config(device="cpu", seed=seed, n_best=10,
        hidden_candidates=((32, 32, 32), (64, 64, 64)), learning_rates=(1e-3, 1e-2),
        epoch_candidates=(50, 150), dropout=0.1, batch_size=32, validation_fraction=0.2)
    started, cpu_started = time.perf_counter(), time.process_time()
    with r["threadpool_limits"](limits=1):
        mlopt = r["mlopt_baseline"].fit(r["portable_adapter"].training_case(case), config=config)
    mlopt_wall, mlopt_cpu = time.perf_counter() - started, time.process_time() - cpu_started
    mlopt._new_uc_source_binding = binding
    if len(mlopt.metadata["validation_trials"]) != 8 or mlopt.metadata["final_bank"]["n_samples"] != n:
        raise ValueError("MLOPT did not retain the full eight-configuration search and all-n refit")
    parameter_count = sum(int(weight.size + bias.size) for weight, bias in mlopt.weights)
    metadata = dict(version=VERSION, case_id=case.case_id, n_train=n, full_original_dimension=D,
        common_u_dimension=q, target_K=int(case.affine["rank"]), original_binary_dimension=3 * q,
        source_binding=binding, setup_and_shared_validation=setup,
        fit_timer_scope="learning only; shared setup/training-label generation/certification/model serialization excluded",
        training_label_count=n, no_external_test_information=True, device="cpu", torch_threads=1,
        svm=dict(fit_seconds=svm_wall, fit_cpu_seconds=svm_cpu, training_label_count=n,
            neural_parameter_count=0, fitted_classifier_count=svm_raw["fitted_classifier_count"], raw_metadata=svm_raw),
        mlopt=dict(fit_seconds=mlopt_wall, fit_cpu_seconds=mlopt_cpu, training_label_count=n,
            trainable_parameter_count=parameter_count, selected_configuration=mlopt.metadata["selected_hyperparameters"],
            input_dimension=int(mlopt.weights[0][0].shape[1]), retained_strategy_count=len(mlopt.selected),
            neural_fit_grid_count=8, final_refit_count=1, raw_metadata=mlopt.metadata))
    # Persist only new case-specific artifacts. Serialization and source/model
    # hashing below are outside both reported learning intervals.
    folder = Path(case.source_directory).resolve() / "baseline-models"
    if folder.parent != Path(case.source_directory).resolve():
        raise ValueError("New model artifacts must stay in this case directory")
    folder.mkdir(parents=False, exist_ok=False)
    save_started, save_cpu = time.perf_counter(), time.process_time()
    artifacts = {}
    for name, model in (("svm.pkl", svm), ("mlopt.pkl", mlopt)):
        path = folder / name
        with path.open("xb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        artifacts[name] = _sha(path)
    serialization = dict(wall_seconds=time.perf_counter() - save_started, cpu_seconds=time.process_time() - save_cpu,
        scope="model pickle writes and SHA256 hashing; fit clocks and receipt serialization excluded")
    receipt = dict(schema="new-uc-small-baselines-complete-v1", source_binding=binding,
        artifact_sha256=artifacts, fit_metadata=metadata, serialization=serialization,
        original_training_label_generation_included=False, no_confirmation_information_used=True)
    with (folder / "complete.json").open("x", encoding="utf-8") as stream:
        json.dump(_clean(receipt), stream, indent=2, allow_nan=False)
        stream.write("\n")
    metadata["saved_models"] = dict(directory=str(folder), artifact_sha256=artifacts,
        complete_path=str(folder / "complete.json"), complete_sha256=_sha(folder / "complete.json"),
        serialization=serialization)
    return svm, mlopt, _clean(metadata)


def _query(model, case, cost, feature, budget):
    r = _runtime()
    np = r["np"]
    if getattr(model, "_new_uc_source_binding", None) != _case_binding(case):
        raise ValueError("The baseline model, source, training sample or original UC case changed")
    c, f = np.asarray(cost, dtype=float), np.asarray(feature, dtype=float)
    if c.shape != (case.model.n_variables,) or f.shape != (case.train_features.shape[1],):
        raise ValueError("Query must include the full original cost and all varying features")
    if not np.isfinite(c).all() or not np.isfinite(f).all() or not math.isfinite(float(budget)) or budget <= 0:
        raise ValueError("Nonfinite query or nonpositive wall budget")
    return r, c, f


def _finish(case, cost, fixed, raw, solution, diagnostic, started, cpu_started, budget):
    r = _runtime()
    np, adapter = r["np"], r["portable_adapter"]
    original = adapter.validate_solution(case.model, solution) if solution is not None else dict(valid=False)
    physics = r["uc_model"].validate_uc_solution(case.original_base, solution) if solution is not None else dict(valid=False)
    original_valid = bool(original["valid"] and physics["valid"])
    residual = None if solution is None or fixed is None else max((abs(float(solution[j]) - v) for j, v in fixed.items()), default=0.0)
    restriction_valid = bool(original_valid and fixed is not None and residual <= 1e-6)
    actual_K = None if fixed is None else len(case.commitment_indices) - len(fixed)
    objective = float(cost @ solution) if original_valid else None
    cpu_seconds = time.process_time() - cpu_started
    seconds = time.perf_counter() - started
    timely = seconds <= budget
    valid = bool(raw.get("valid", False) and original_valid and restriction_valid and timely
                 and actual_K == int(case.affine["rank"]) and objective is not None and math.isfinite(objective))
    return _clean(dict(solution=solution, valid=valid, objective=objective if valid else None,
        raw_objective=objective, actualK=actual_K, actual_K=actual_K, target_K=int(case.affine["rank"]),
        seconds=seconds, cpu_seconds=cpu_seconds, budget_seconds=float(budget), timely=timely,
        original_feasible=original_valid, restriction_valid=restriction_valid, restriction_residual=residual,
        fixed=fixed, raw=raw, diagnostic=diagnostic, original_check=original, physics=physics,
        restricted_model_certified=bool(valid and raw.get("certified", False)),
        certificate_scope="restricted original UC IP; no original-global optimum certificate",
        original_global_solver_certified=False, full_fallback_used=False, warm_start_used=False,
        timer_scope="source/data guard checks, prediction/matching, cold restricted IP and independent UC validation"))


def predict_svm(model, case, cost, feature, seed, budget=30.0):
    started, cpu_started = time.perf_counter(), time.process_time()
    r, c, f = _query(model, case, cost, feature, budget)
    with r["threadpool_limits"](limits=1):
        fixed, diagnostic = r["svm_match"].propose_coordinate_fixings(model, case, f)
        remaining = float(budget) - (time.perf_counter() - started)
        if remaining <= 0:
            raw, solution = dict(valid=False, certified=False, status="prediction_budget_exhausted"), None
        else:
            result = r["portable_adapter"].solve(case.model, c, fixed=fixed, seed=int(seed), time_limit=remaining)
            raw, solution = result.to_dict(include_solution=False), result.x
        return _finish(case, c, fixed, raw, solution, diagnostic, started, cpu_started, float(budget))


def _native_predict(model, case, cost, feature):
    """Only bridge the unchanged native MLOPT result; no external optimum."""
    raw = dict(model.predict(feature, cost))
    x = raw.pop("x", None)
    return raw, x


def predict_mlopt(model, case, cost, feature, seed, budget=30.0):
    started, cpu_started = time.perf_counter(), time.process_time()
    r, c, f = _query(model, case, cost, feature, budget)
    with r["threadpool_limits"](limits=1):
        fixed, diagnostic, native_solution = r["native_anchor"].propose_native_anchor(
            model, case, c, f, native_predict_fn=_native_predict)
        remaining = float(budget) - (time.perf_counter() - started)
        if fixed is None or remaining <= 0:
            raw, solution = dict(valid=False, certified=False,
                status="no_valid_native_winner" if fixed is None else "prediction_budget_exhausted"), None
        else:
            result = r["portable_adapter"].solve(case.model, c, fixed=fixed, seed=int(seed), time_limit=remaining)
            raw, solution = result.to_dict(include_solution=False), result.x
        diagnostic["native_returned_solution"] = native_solution
        diagnostic["native_returned_objective"] = None if native_solution is None else float(c @ native_solution)
        diagnostic["native_output_optimality_certified"] = False
        return _finish(case, c, fixed, raw, solution, diagnostic, started, cpu_started, float(budget))
