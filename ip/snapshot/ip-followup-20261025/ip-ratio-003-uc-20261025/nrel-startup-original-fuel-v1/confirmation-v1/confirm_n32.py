"""One fixed three-seed UC confirmation: freeze, then generate/evaluate.

No experiment/RNG/solver executes on import. No retraining or setting search.
The whole three-case source/model population is frozen before fresh costs.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import re
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
VARIANT = HERE.parent
SCREEN = VARIANT / "run-v1"
UC_ROOT = VARIANT.parent
HELPER = UC_ROOT / "baseline_helpers.py"
HELPER_SHA = "cad89cffdfff790005520be922dcf246b0c3052c34ad78d34a8ddab6ca0b197e"
PROTOCOL = HERE / "protocol.json"
SEEDS = (101, 211, 307)
METHODS = ("Ours", "SVM", "MLOPT")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if hasattr(value, "tolist"): return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False); stream.write("\n")


def load_helpers():
    if sha(HELPER) != HELPER_SHA: raise ValueError("Archived source helper changed")
    spec = importlib.util.spec_from_file_location("fixed_uc_confirmation_helpers", HELPER)
    helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
    runtime = helper._runtime()
    return helper, runtime


def load_case(seed, helper, runtime):
    """Restore saved model state exactly; never rerun a physical/sample producer."""
    np = runtime["np"]; adapter = runtime["portable_adapter"]
    from scipy import sparse
    folder = SCREEN / "settings" / f"G4T12_all_units_nrel_startup_seed{seed}" / "n32"
    names = ("training.npz", "metadata.json", "affine.json")
    hashes = {name: sha(folder / name) for name in names}
    meta = read(folder / "metadata.json"); saved = read(folder / "affine.json")
    with np.load(folder / "training.npz", allow_pickle=False) as stream:
        arrays = {key: stream[key].copy() for key in stream.files}
    if meta["training_npz_sha256"] != hashes["training.npz"]: raise ValueError("Training NPZ binding changed")
    if (meta["n_units"], meta["n_times"], meta["n_train"], meta["seed"]) != (4, 12, 32, seed): raise ValueError("Fixed n32 cohort differs")
    physical = {key: np.asarray(value) if isinstance(value, list) and key != "row_names" else value for key, value in meta["metadata"].items()}
    base_cost = np.concatenate([np.tile(physical[key], 12) for key in ("no_load_cost", "startup_cost", "shutdown_cost", "fuel_cost")])
    if not np.all((physical["fuel_cost"] >= 22) & (physical["fuel_cost"] <= 45)): raise ValueError("Original fuel range changed")
    rate = physical["startup_cost"] / physical["capacity"]
    if not np.all((rate >= 22 - 1e-12) & (rate <= 60 + 1e-12)): raise ValueError("Capacity-scaled startup range changed")
    A = sparse.csr_matrix((arrays["A_data"], arrays["A_indices"], arrays["A_indptr"]), shape=tuple(arrays["A_shape"]))
    model = adapter.LinearModel(A=A, **{key: arrays[key] for key in ("row_lower", "row_upper", "col_lower", "col_upper", "integrality")}, row_names=tuple(physical["row_names"]))
    q, K = 48, int(saved["rank"])
    if model.n_variables != 192 or not np.array_equal(arrays["commitment_indices"], np.arange(q)): raise ValueError("Original/common dimensions differ")
    affine = dict(saved, anchor=np.asarray(saved["anchor"], dtype=np.int64), basis=np.asarray(saved["basis"], dtype=float).reshape(q, K),
        normals=np.asarray(saved["normals"], dtype=float).reshape(q-K, q), normals_exact=np.asarray(saved["normals_exact"], dtype=object).reshape(q-K, q))
    base = dict(A=A, **{key: arrays[key] for key in ("row_lower", "row_upper", "col_lower", "col_upper", "integrality")},
                base_cost=base_cost, binary_indices=arrays["binary_indices"], n_units=4, n_times=12, metadata=physical)
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    case = adapter.FrozenCase(case_id=meta["case_id"], family="uc", seed=seed, amplitude=0., model=model,
        train_features=arrays["train_features"], train_costs=arrays["train_costs"], train_solutions=arrays["train_solutions"],
        train_objectives=arrays["train_objectives"], commitment_indices=arrays["commitment_indices"], affine=affine,
        original_base=base, metadata=deepcopy(physical), source_hashes=hashes, dataset_fingerprint=fingerprint, source_directory=folder)
    helper._validate_training(case)
    receipt_path = folder / "baseline-models/complete.json"; receipt = read(receipt_path)
    fit_path = folder / "baseline-fit-metadata.json"; fit = read(fit_path)
    binding = helper._case_binding(case)
    if receipt["source_binding"] != binding or fit["source_binding"] != binding: raise ValueError("Reconstructed training/source binding differs")
    if {key: value for key, value in fit.items() if key != "saved_models"} != receipt["fit_metadata"]: raise ValueError("Learning metadata differs from model receipt")
    if fit["saved_models"]["complete_sha256"] != sha(receipt_path): raise ValueError("Fit/model completion digest differs")
    files = {str((folder / name).resolve()): digest for name, digest in hashes.items()}
    files.update({str(receipt_path.resolve()): sha(receipt_path), str(fit_path.resolve()): sha(fit_path)})
    for name, expected in receipt["artifact_sha256"].items():
        path = folder / "baseline-models" / name
        if path.resolve().parent != (folder / "baseline-models").resolve() or sha(path) != expected: raise ValueError("Saved model artifact hash differs")
        files[str(path.resolve())] = expected
    models = {}
    for method, name in (("SVM", "svm.pkl"), ("MLOPT", "mlopt.pkl")):
        with (folder / "baseline-models" / name).open("rb") as stream: models[method] = pickle.load(stream)
        if getattr(models[method], "_new_uc_source_binding", None) != binding: raise ValueError("Loaded model source/training binding differs")
        helper._query(models[method], case, case.train_costs[0], case.train_features[0], 30.)
    certificates = folder.parent / "training-reference-certificates.json"
    cert = read(certificates)
    for index, reference in enumerate(case.train_objectives):
        raw = cert["records"][index]; tol = adapter.recovery_tolerance(float(reference))
        if not (raw["certified"] and raw["valid"] and raw["independent_UC_physics_valid"]) or abs(raw["objective"]-reference) > tol or abs(raw["dual_bound"]-reference) > tol:
            raise ValueError("Original prefix training optimum certificate differs")
    files[str(certificates.resolve())] = sha(certificates)
    return case, models, fit, arrays["cost_offset"], arrays["cost_feature_coefficients"], files


def validate_population(freeze=None):
    """Read source/training/model artifacts only, never a confirmation bank."""
    summary_path = SCREEN / "summary.json"; summary = read(summary_path)
    if summary["status"] != "COMPLETE": raise ValueError("All three original-fuel screen cases must finish before confirmation")
    screen_protocol = read(SCREEN / "protocol.json")
    sources = {str(path.resolve()): sha(path) for path in (Path(__file__), PROTOCOL, HELPER, VARIANT / "nrel_startup_variant.py", UC_ROOT / "uc_reference_screen.py", SCREEN / "protocol.json", summary_path)}
    for name, expected in screen_protocol["source_sha256"].items():
        if sha(name) != expected: raise ValueError("Screen scientific source changed")
        sources[str(Path(name).resolve())] = expected
    helper, runtime = load_helpers()
    cases = {}; entries = {}
    for seed in SEEDS:
        case, models, fit, offset, coefficients, files = load_case(seed, helper, runtime)
        marker = case.source_directory.parent / "complete.json"
        if read(marker)["status"] != "COMPLETE": raise ValueError("A physical screen did not complete")
        files[str(marker.resolve())] = sha(marker)
        cases[seed] = (case, models, fit, offset, coefficients)
        entries[case.case_id] = dict(seed=seed, K=int(case.affine["rank"]), n_train=32, roles=list(METHODS), artifact_sha256=files,
            source_binding=helper._case_binding(case), learning_timer_source=str((case.source_directory / "baseline-fit-metadata.json").resolve()))
        sources.update(helper._case_binding(case)["source_hashes"])
    for name, expected in sources.items():
        if sha(name) != expected: raise ValueError("Source hash differs before global freeze")
    population = dict(schema="uc-nrel-g4all-n32-three-case-global-freeze-v1", case_count=3, model_role_count=9,
        source_sha256=sources, protocol_sha256=sha(PROTOCOL), cases=entries, confirmation_generated=False)
    if freeze is not None:
        for key in population:
            if population[key] != freeze.get(key): raise ValueError("Global frozen population/source differs: " + key)
    return population, cases, helper, runtime


def run(args):
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_freeze_sha256 or "") or sha(args.freeze_path) != args.expected_freeze_sha256:
        raise ValueError("Explicit expected global-freeze SHA256 is required")
    freeze = read(args.freeze_path)
    population, cases, helper, runtime = validate_population(freeze)
    if sha(args.freeze_path) != args.expected_freeze_sha256: raise ValueError("Freeze changed while validating")
    # No output/bank inspection or objective RNG precedes the whole-population gate.
    if str(args.out.resolve()) != freeze["confirmation_output_directory"]: raise ValueError("Frozen confirmation output target differs")
    args.out.mkdir(parents=True, exist_ok=False)
    np, adapter, uc = runtime["np"], runtime["portable_adapter"], runtime["uc_model"]
    import experiment_core as core
    records = []; per_case = []
    for seed in SEEDS:
        # Recheck immutable source/model bytes at each predeclared seed boundary.
        for name, expected in freeze["source_sha256"].items():
            if sha(name) != expected: raise ValueError("Globally frozen source changed")
        for entry in freeze["cases"].values():
            for name, expected in entry["artifact_sha256"].items():
                if sha(name) != expected: raise ValueError("Globally frozen model/label artifact changed")
        case, models, fit, offset, coefficients = cases[seed]
        label = read(PROTOCOL)["rng_namespace"] + "|physical_seed=" + str(seed)
        seed128 = int.from_bytes(hashlib.sha256(label.encode()).digest()[:16], "little")
        random = np.random.default_rng(seed128)
        features = np.exp(random.uniform(-np.log(3.), np.log(3.), (120, 12)))
        costs = offset + features @ coefficients.T
        if not np.all(costs > 0): raise ValueError("Original positive-cost law differs")
        folder = args.out / case.case_id; folder.mkdir(exist_ok=False)
        with (folder / "planned-costs.npz").open("xb") as stream:
            np.savez_compressed(stream, features=features, costs=costs)
        write_new(folder / "generation-start.json", dict(case_id=case.case_id, count=120,
            source_freeze_sha256=args.expected_freeze_sha256, rng_namespace=read(PROTOCOL)["rng_namespace"],
            rng_seed128=seed128, planned_costs_sha256=sha(folder / "planned-costs.npz")))
        full_records = []; full_solutions = []; full_objectives = []; case_rows = []
        for index, (feature, cost) in enumerate(zip(features, costs)):
            started, cpu_started = time.perf_counter(), time.process_time()
            full, full_x = core.solve(case.original_base, cost, seed=seed+100000+index, time_limit=30.)
            if not (full["certified"] and full["valid"] and uc.validate_uc_solution(case.original_base, full_x)["valid"]):
                write_new(folder / f"failed-Full-reference-{index:03d}.json", dict(index=index, feature=feature,
                    cost=cost, raw=full, solution=full_x, source_freeze_sha256=args.expected_freeze_sha256))
                raise ValueError("Cold original Full reference not certified/physical; planned costs and failure retained")
            reference = float(cost @ full_x); tol = adapter.recovery_tolerance(reference)
            if abs(full["objective"]-reference) > tol or abs(full["dual_bound"]-reference) > tol: raise ValueError("Original Full primal/dual mismatch")
            full.update(cold_wall_seconds=time.perf_counter()-started, cold_cpu_seconds=time.process_time()-cpu_started,
                index=index, original_UC_physics_valid=True, source_freeze_sha256=args.expected_freeze_sha256)
            full_records.append(full); full_solutions.append(full_x); full_objectives.append(reference)
            started, cpu_started = time.perf_counter(), time.process_time()
            raw, x = core.solve(case.original_base, cost, affine=case.affine, seed=seed+200000+index, time_limit=30.)
            feasible = bool(x is not None and raw["valid"] and uc.validate_uc_solution(case.original_base, x)["valid"])
            seconds, cpu_seconds = time.perf_counter()-started, time.process_time()-cpu_started
            ours = dict(solution=x, raw=raw, actual_K=int(case.affine["rank"]), seconds=seconds, cpu_seconds=cpu_seconds,
                valid=bool(feasible and seconds<=30.), objective=float(cost@x) if feasible and seconds<=30. else None,
                raw_objective=float(cost@x) if feasible else None, original_feasible=feasible, timely=seconds<=30.)
            predictions = dict(Ours=ours, SVM=helper.predict_svm(models["SVM"], case, cost, feature, seed+300000+index, 30.),
                MLOPT=helper.predict_mlopt(models["MLOPT"], case, cost, feature, seed+400000+index, 30.))
            for method, row in predictions.items():
                valid = bool(row["valid"] and row.get("objective") is not None and row["objective"]>0)
                row.update(method=method, case_id=case.case_id, seed=seed, index=index, n_train=32,
                    reference_objective=reference, confirmation=True, phase="independent_confirmation",
                    recovered=bool(valid and abs(row["objective"]-reference)<=tol),
                    objective_ratio=reference/row["objective"] if valid else None,
                    source_freeze_sha256=args.expected_freeze_sha256, protocol_sha256=population["protocol_sha256"],
                    learning_timer_source=freeze["cases"][case.case_id]["learning_timer_source"], timing_not_isolated=True)
                case_rows.append(row); records.append(row)
            write_new(folder / "raw-records" / f"test{index:03d}.json", dict(index=index,
                feature=feature, cost=cost, Full=full, Full_solution=full_x, methods=predictions,
                source_freeze_sha256=args.expected_freeze_sha256))
        with (folder / "confirmation.npz").open("xb") as stream:
            np.savez_compressed(stream, features=features, costs=costs, solutions=np.asarray(full_solutions), objectives=np.asarray(full_objectives))
        write_new(folder / "Full-reference-certificates.json", full_records)
        write_new(folder / "method-records.json", case_rows)
        write_new(folder / "metadata.json", dict(case_id=case.case_id, seed=seed, n_train=32, K=case.affine["rank"], count=120,
            phase="independent_confirmation", rng_namespace=read(PROTOCOL)["rng_namespace"], rng_seed128=seed128,
            source_freeze_sha256=args.expected_freeze_sha256, learning_metadata=fit,
            files={path.name:sha(path) for path in folder.iterdir() if path.is_file()}))
        per_case.append(dict(case_id=case.case_id, seed=seed, K=case.affine["rank"], count=120))
        print(json.dumps(dict(event="confirmation_seed_complete", seed=seed, count=120, returns=360)), flush=True)
    cells = []
    for method in METHODS:
        rows = [row for row in records if row["method"]==method]
        ratios = [row["objective_ratio"] for row in rows if row["objective_ratio"] is not None]
        cells.append(dict(method=method, total=360, recovered=sum(row["recovered"] for row in rows),
            recovery_percent=100*sum(row["recovered"] for row in rows)/360, ratio_denominator=len(ratios),
            objective_ratio_mean=float(np.mean(ratios)) if ratios else None,
            return_seconds_mean=float(np.mean([row["seconds"] for row in rows]))))
    write_new(args.out / "summary.json", dict(status="COMPLETE", case_count=3, Full_reference_count=360, method_return_count=1080,
        source_freeze_sha256=args.expected_freeze_sha256, protocol_sha256=population["protocol_sha256"], cases=per_case,
        cells=cells, ratio_aggregation="pooled arithmetic mean over valid raw tests, never a mean of seed means",
        original_total_cost_retained=True, no_new_training_or_setting_search=True, timing_not_isolated=True,
        files={str(path.relative_to(args.out)):sha(path) for path in args.out.rglob("*") if path.is_file()}))
    print(json.dumps(dict(status="COMPLETE", cells=cells)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-path", type=Path, default=HERE / "global-freeze.json")
    parser.add_argument("--out", type=Path, default=HERE / "results-v1")
    parser.add_argument("--expected-freeze-sha256")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--freeze", action="store_true")
    operation.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        population, _, _, _ = validate_population()
        if args.out.exists(): raise FileExistsError("Confirmation output already exists before global freeze")
        population.update(frozen_at_utc=utc(), confirmation_output_directory=str(args.out.resolve()))
        write_new(args.freeze_path, population)
        print(json.dumps(dict(status="ALL_THREE_SEEDS_NINE_ROLES_FROZEN", freeze_sha256=sha(args.freeze_path))), flush=True)
    else:
        run(args)


if __name__ == "__main__": main()
