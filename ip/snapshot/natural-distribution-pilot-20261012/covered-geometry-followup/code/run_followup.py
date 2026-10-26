"""Serial training-geometry follow-up; one CPU thread, frozen fresh tests.

Run only on the authorized experimental server. --validate-only performs no
optimization or learning and creates no experiment outputs.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
from time import perf_counter, process_time
IMPORT_STARTED = perf_counter()
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import uuid
import warnings
warnings.filterwarnings("once", message="The `probability` parameter was deprecated", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
import scipy
import sklearn
import highspy
from threadpoolctl import threadpool_info, threadpool_limits
import portable_adapter as adapter
import experiment_core as core
from uc_model import make_uc_base, all_on_feasible_solution, validate_uc_solution
from xqa_baseline import XQAAffineBaseline
from dimension_diagnostics import affine_equalities_metrics
from svm_match import propose_coordinate_fixings
from fast_baselines import fit_coordinate_svm_small, prepare_historical, propose_historical_matched, solve_best_observed
from mlopt_bridge import fit_mlopt, predict_native
from native_anchor import execute_native_anchor
from cost_sampler import sample_features, scenario_fields, sample_diversity
IMPORT_SECONDS = perf_counter()-IMPORT_STARTED


def utc(): return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def seed(material): return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "little") % (2**31-1)


def publish_new(path, writer):
    """Publish complete bytes atomically without overwriting an existing artifact."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+".writing-"+uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        writer(stream);stream.flush();os.fsync(stream.fileno())
    if os.name=="nt":
        # Windows rename is atomic and refuses an existing destination.
        os.rename(temporary,path)
    else:
        # POSIX link atomically creates a missing destination; unlink leaves
        # one regular file. Interrupted temporary artifacts are preserved.
        os.link(temporary,path);temporary.unlink()


def write_new(path,value):
    raw=(json.dumps(clean(value),indent=2,ensure_ascii=True,allow_nan=False)+"\n").encode("utf-8")
    publish_new(path,lambda stream:stream.write(raw))


def save_npz_new(path,**arrays):
    publish_new(path,lambda stream:np.savez_compressed(stream,**arrays))


def event(**fields): print(json.dumps(clean(dict(utc=utc(), **fields))), flush=True)


def bindings():
    paths = [ROOT / "protocol.json", ROOT / "provenance.json"]+[p for p in sorted((ROOT / "code").glob("*.py")) if not p.name.startswith("test_")]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}


class Deadline:
    def __init__(self, seconds): self.end = IMPORT_STARTED+float(seconds)
    def remaining(self):
        remaining = self.end-perf_counter()
        if remaining <= 1.: raise TimeoutError("Recorded process wall limit reached; retained results are partial")
        return remaining
    def budget(self, maximum):
        remaining = self.remaining()
        if float(maximum) == 30. and remaining < 31.:
            raise TimeoutError("Insufficient global time for another full 30-second method budget")
        return min(float(maximum), remaining)


def physical_template(physical_seed, gid, source_binding, scenario=None):
    base = make_uc_base(physical_seed, n_units=6, n_times=24)
    names = tuple(f"{kind}[{t},{g}]" for kind in ("u", "v", "w", "p") for t in range(24) for g in range(6))
    model = adapter.LinearModel(A=base["A"], row_lower=base["row_lower"], row_upper=base["row_upper"],
        col_lower=base["col_lower"], col_upper=base["col_upper"], integrality=base["integrality"],
        variable_names=names, row_names=tuple(base["metadata"]["row_names"]))
    return adapter.FrozenCase(case_id=gid, family="uc", seed=int(physical_seed), amplitude=0., model=model,
        train_features=np.empty((0, 6)), train_costs=np.empty((0, 576)), train_solutions=np.empty((0, 576)),
        train_objectives=np.empty(0), commitment_indices=np.asarray(base["binary_indices"], dtype=np.int64),
        affine={}, original_base=base, metadata=dict(base["metadata"], **({} if scenario is None else
            {"cost_feature_layout": scenario["feature_layout"]})), source_hashes=source_binding,
        dataset_fingerprint="fresh physical model; no training labels yet", source_directory=ROOT)


def new_case(template, gid, n, data, affine):
    h = hashlib.sha256()
    for key in ("features", "costs", "solutions"): h.update(np.ascontiguousarray(data[key][:n]).tobytes())
    return replace(template, case_id=f"{gid}_n{n}", train_features=data["features"][:n].copy(),
        train_costs=data["costs"][:n].copy(), train_solutions=data["solutions"][:n].copy(),
        train_objectives=data["objectives"][:n].copy(), affine=affine, dataset_fingerprint=h.hexdigest())


def measured(function):
    wall, cpu = perf_counter(), process_time()
    value = function()
    return value, dict(fit_wall_seconds=perf_counter()-wall, fit_cpu_seconds=process_time()-cpu)


def verify_files(folder, files):
    folder = Path(folder).resolve()
    for relative, digest in files.items():
        raw = folder / relative
        path = raw.resolve()
        if raw.is_symlink() or not path.is_relative_to(folder) or not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"Committed artifact changed: {relative}")


def geometry_record(template, solutions, n):
    common = np.rint(solutions[:n, template.commitment_indices]).astype(np.int8)
    whole = np.rint(solutions[:n, template.model.integer_indices]).astype(np.int8)
    affine, timing = measured(lambda: core.fit_affine(common))
    K = int(affine["rank"])
    unique_whole = int(len(np.unique(whole, axis=0)))
    return dict(training_count=n, exact_K=K, common_D=144,
        unique_common_commitment_count=int(len(np.unique(common, axis=0))),
        unique_complete_integer_strategy_count=unique_whole,
        unique_strategy_count_over_K=None if K==0 else unique_whole/K,
        strategy_definition="Unique original432 integer u/v/w assignments; continuous dispatch not counted",
        exact_rational_rank=True, global_coverage_certified=False,
        geometry_fit_wall_seconds=timing["fit_wall_seconds"],geometry_fit_cpu_seconds=timing["fit_cpu_seconds"])


def label_group(out, protocol, physical_seed, scenario, sources, deadline):
    fields=scenario_fields(scenario); gid=f"uc_seed{physical_seed}_"+fields["scenario_id"]
    folder=out/"training"/gid; folder.mkdir(parents=True,exist_ok=True)
    template=physical_template(physical_seed,gid,sources,scenario)
    if (folder/"labels-complete.json").exists():
        verify_files(folder,read(folder/"labels-complete.json")["files"])
        return read(folder/"geometry.json")
    features,sampling=sample_features(protocol,"train",physical_seed,protocol["maximum_training_count"],scenario)
    costs=adapter.costs_from_features(template,features)
    plan=folder/"training-plan.npz"
    if not plan.exists():
        save_npz_new(plan,features=features,costs=costs)
        write_new(folder/"training-plan.json",dict(group_id=gid,sampler=sampling,plan_sha256=sha(plan),source_binding=sources))
    else:
        with np.load(plan,allow_pickle=False) as stream:
            if not np.array_equal(features,stream["features"]) or not np.array_equal(costs,stream["costs"]):
                raise RuntimeError("Training sampler/prefix changed on resume")
        if not (folder/"training-plan.json").exists():
            write_new(folder/"training-plan.json",dict(group_id=gid,sampler=sampling,plan_sha256=sha(plan),
                source_binding=sources,receipt_recovered_after_interrupted_publication=True))
        receipt=read(folder/"training-plan.json")
        if receipt["source_binding"]!=sources or receipt["plan_sha256"]!=sha(plan): raise RuntimeError("Training plan changed")
    checkpoints=[]
    for marker in (folder/"checkpoints").glob("*/checkpoint-complete.json"):
        checkpoint=read(marker); verify_files(marker.parent,checkpoint["files"])
        checkpoints.append((int(checkpoint["training_count"]),marker,checkpoint))
    xs,objectives,labels,diagnostics=[],[],[],[]
    if checkpoints:
        n,marker,checkpoint=max(checkpoints,key=lambda x:x[0])
        if checkpoint["source_binding"]!=sources: raise RuntimeError("Checkpoint source binding changed")
        with np.load(marker.parent/"arrays.npz",allow_pickle=False) as stream:
            if not np.array_equal(features[:n],stream["features"]) or not np.array_equal(costs[:n],stream["costs"]):
                raise RuntimeError("Checkpoint is not the fixed training prefix")
            xs=list(stream["solutions"].copy());objectives=list(stream["objectives"].copy())
        labels=read(marker.parent/"label-records.json")["records"]
        diagnostics=read(marker.parent/"geometry.json")["prefixes"]
        if len(xs)!=n or len(labels)!=n or not all(r["certified"] for r in labels):
            raise RuntimeError("Checkpoint does not contain certified complete labels")
        for x,c,obj in zip(xs,costs[:n],objectives):
            if not adapter.validate_solution(template.model,x)["valid"] or abs(float(c@x)-obj)>adapter.recovery_tolerance(obj):
                raise RuntimeError("Checkpoint label failed original constraints/objective")
    for i in range(len(xs),len(costs)):
        wall,cpu=perf_counter(),process_time()
        result=adapter.solve(template.model,costs[i],seed=seed(f"{gid}|train|{i}"),
            time_limit=deadline.budget(protocol["solver"]["training_label_limit_seconds"]))
        record=result.to_dict(include_solution=False)
        record.update(train_index=i,outer_label_wall_seconds=perf_counter()-wall,outer_label_cpu_seconds=process_time()-cpu)
        if not result.certified:
            write_new(folder/f"uncertified-label-{i}-{utc().replace(':','')}.json",record)
            raise RuntimeError("Fresh training label not certified; committed prefix retained")
        xs.append(result.x);objectives.append(float(costs[i]@result.x));labels.append(record)
        n=i+1
        if n%protocol["checkpointing"]["training_labels_per_complete_snapshot"]==0:
            diagnostic=geometry_record(template,np.asarray(xs),n);diagnostics.append(diagnostic)
            directory=folder/"checkpoints"/f"n{n:03d}"
            if directory.exists():directory=folder/"checkpoints"/f"n{n:03d}-retry-{utc().replace(':','')}"
            directory.mkdir(parents=True)
            save_npz_new(directory/"arrays.npz",features=features[:n],costs=costs[:n],solutions=np.asarray(xs),objectives=np.asarray(objectives))
            write_new(directory/"label-records.json",dict(group_id=gid,sampler=sampling,records=labels))
            write_new(directory/"geometry.json",dict(group_id=gid,prefixes=diagnostics))
            write_new(directory/"checkpoint-complete.json",dict(group_id=gid,training_count=n,source_binding=sources,
                files={name:sha(directory/name) for name in ("arrays.npz","label-records.json","geometry.json")},committed_utc=utc()))
            event(event="training_checkpoint_complete",group=gid,n=n,K=diagnostic["exact_K"],
                unique_integer_strategies=diagnostic["unique_complete_integer_strategy_count"],path=str(directory.relative_to(out)))
    if len(xs)!=96:raise RuntimeError("The protocol requires all96 labels")
    data=dict(features=features,costs=costs,solutions=np.asarray(xs),objectives=np.asarray(objectives))
    if not (folder/"training.npz").exists():
        save_npz_new(folder/"training.npz",**data)
    else:
        with np.load(folder/"training.npz",allow_pickle=False) as stream:
            if set(stream.files)!=set(data) or any(not np.array_equal(stream[k],v) for k,v in data.items()):
                raise RuntimeError("Interrupted final training publication differs from the verified96-label checkpoint")
    label_receipt=dict(group_id=gid,sampler=sampling,records=labels)
    if not (folder/"label-records.json").exists():write_new(folder/"label-records.json",label_receipt)
    elif read(folder/"label-records.json")!=clean(label_receipt):raise RuntimeError("Final training label receipt changed")
    prefixes={d["training_count"]:d for d in diagnostics}
    for n in protocol["geometry_counts"]:
        if n not in prefixes:raise RuntimeError("Missing prescribed exact-rank prefix")
    K96=prefixes[96]["exact_K"]; unique96=prefixes[96]["unique_complete_integer_strategy_count"]
    covered=bool(K96==prefixes[64]["exact_K"] and K96<=40 and unique96>=2*K96+1)
    geometry=dict(group_id=gid,physical_seed=physical_seed,**fields,prefixes=diagnostics,
        observed_covered_candidate=covered,criterion_K96_equals_K64=K96==prefixes[64]["exact_K"],
        criterion_K96_at_most40=K96<=40,criterion_unique96_at_least2K_plus1=unique96>=2*K96+1,
        K48_equals_K96=prefixes[48]["exact_K"]==K96,coverage_is_empirical_only=True,
        global_coverage_certified=False,all96_label_wall_seconds=sum(r["outer_label_wall_seconds"] for r in labels),
        all96_label_cpu_seconds=sum(r["outer_label_cpu_seconds"] for r in labels),
        exact_geometry_diagnostic_wall_seconds=sum(d["geometry_fit_wall_seconds"] for d in diagnostics),
        exact_geometry_diagnostic_cpu_seconds=sum(d["geometry_fit_cpu_seconds"] for d in diagnostics),test_information_used=False)
    if not (folder/"geometry.json").exists():write_new(folder/"geometry.json",geometry)
    elif read(folder/"geometry.json")!=clean(geometry):raise RuntimeError("Final training geometry receipt changed")
    write_new(folder/"labels-complete.json",dict(group_id=gid,training_count=96,
        files={name:sha(folder/name) for name in ("training.npz","label-records.json","geometry.json")},committed_utc=utc()))
    return geometry


def freeze_training_screen(out,protocol,groups):
    by_scenario=[]
    for scenario in protocol["scenarios"]:
        selected=[g for g in groups if g["scenario_id"]==scenario["scenario_id"]]
        eligible=[g for g in selected if g["observed_covered_candidate"]]
        candidate=len(eligible)>=2;main_n=None
        if candidate:
            for n in protocol["training_counts"]:
                plateau=sum(next(p["exact_K"] for p in g["prefixes"] if p["training_count"]==n)==
                            next(p["exact_K"] for p in g["prefixes"] if p["training_count"]==96) for g in eligible)
                if plateau>=2:main_n=n;break
        by_scenario.append(dict(**scenario,candidate=candidate,eligible_group_count=len(eligible),main_training_count=main_n,
                                eligible_physical_seeds=[g["physical_seed"] for g in eligible]))
    result=dict(status="frozen-before-tests",policy=protocol["coverage_screening"],groups=groups,scenarios=by_scenario,
        any_candidate=any(s["candidate"] for s in by_scenario),all12groups_recorded=True,
        tests_generated=False,test_information_used=False,training_sampler_stream=2026101207,
        all96_label_wall_seconds=sum(g["all96_label_wall_seconds"] for g in groups),
        all96_label_cpu_seconds=sum(g["all96_label_cpu_seconds"] for g in groups),frozen_utc=utc())
    path=out/"training-screening.json"
    if path.exists():
        previous=read(path)
        if previous["groups"]!=groups or previous["scenarios"]!=by_scenario:raise RuntimeError("Training screen changed")
        return previous
    write_new(path,result);return result


def fit_group(out, protocol, physical_seed, scenario, source_binding, deadline):
    fields = scenario_fields(scenario); gid = f"uc_seed{physical_seed}_"+fields["scenario_id"]
    template = physical_template(physical_seed, gid, source_binding, scenario)
    folder = out / "training" / gid
    verify_files(folder, read(folder / "labels-complete.json")["files"])
    with np.load(folder / "training.npz", allow_pickle=False) as stream: data = {k:stream[k].copy() for k in stream.files}
    features = data["features"]; labels = read(folder / "label-records.json")["records"]
    geometry = read(folder / "geometry.json")
    if (folder / "complete.json").exists():
        completed = read(folder / "complete.json"); verify_files(out, completed["files"])
        return dict(group_id=gid, files=completed["files"], complete_sha256=sha(folder / "complete.json"),
                    models=[read(folder / f"n{n}/metadata.json") for n in protocol["training_counts"]])
    metadata_rows = []
    for n in protocol["training_counts"]:
        deadline.remaining()
        model_folder = folder / f"n{n}"
        if (model_folder / "model-complete.json").exists():
            verify_files(model_folder, read(model_folder / "model-complete.json")["files"])
            metadata_rows.append(read(model_folder / "metadata.json")); continue
        if model_folder.exists():
            model_folder.rename(folder / f"n{n}-interrupted-{utc().replace(':','').replace('+','')}")
        binary = np.rint(data["solutions"][:n, template.commitment_indices]).astype(np.int8)
        affine, affine_time = measured(lambda: core.fit_affine(binary))
        case = new_case(template, gid, n, data, affine)
        native, native_time = measured(lambda: XQAAffineBaseline(C=1., n_splits=5,
            random_state=seed(f"{gid}|n{n}|svm")).fit(features[:n], binary, n_times=24, n_units=6))
        coordinate_result, coordinate_time = measured(lambda: fit_coordinate_svm_small(case))
        coordinate, coordinate_metadata = coordinate_result
        bank, historical_time = measured(lambda: prepare_historical(case))
        deadline.remaining()
        mlopt_result, mlopt_time = measured(lambda: fit_mlopt(case, seed=seed(f"{gid}|n{n}|mlopt")))
        mlopt, mlopt_metadata = mlopt_result
        deadline.remaining()
        models = dict(affine=affine, native=native, coordinate=coordinate, historical=bank, mlopt=mlopt)
        model_folder.mkdir()
        publish_new(model_folder/"models.pkl",lambda stream:pickle.dump(models,stream,pickle.HIGHEST_PROTOCOL))
        label_wall = sum(r["outer_label_wall_seconds"] for r in labels[:n])
        label_cpu = sum(r["outer_label_cpu_seconds"] for r in labels[:n])
        all_label_wall = sum(r["outer_label_wall_seconds"] for r in labels)
        all_label_cpu = sum(r["outer_label_cpu_seconds"] for r in labels)
        fit_times = {"ours": affine_time, "svm_native": native_time, "coordinate_svm_matched": coordinate_time,
                     "historical_strategy_matched": historical_time, "best_observed_strategy": historical_time,
                     "mlopt_native": mlopt_time, "mlopt_native_anchor_matched": mlopt_time}
        offline = {method: dict(training_samples=n, label_wall_seconds=label_wall, label_cpu_seconds=label_cpu,
                               **timing, total_offline_wall_seconds=label_wall+timing["fit_wall_seconds"],
                               total_offline_cpu_seconds=label_cpu+timing["fit_cpu_seconds"])
                   for method, timing in fit_times.items()}
        for timing in offline.values():
            timing.update(geometry_acquisition_samples=len(labels), geometry_acquisition_label_wall_seconds=all_label_wall,
                geometry_acquisition_label_cpu_seconds=all_label_cpu, extra_geometry_label_samples=len(labels)-n,
                extra_geometry_label_wall_seconds=all_label_wall-label_wall, extra_geometry_label_cpu_seconds=all_label_cpu-label_cpu,
                total_with_geometry_acquisition_wall_seconds=all_label_wall+timing["fit_wall_seconds"],
                total_with_geometry_acquisition_cpu_seconds=all_label_cpu+timing["fit_cpu_seconds"],
                exact_geometry_diagnostic_wall_seconds=geometry["exact_geometry_diagnostic_wall_seconds"],
                exact_geometry_diagnostic_cpu_seconds=geometry["exact_geometry_diagnostic_cpu_seconds"],
                total_with_geometry_labels_and_diagnostics_wall_seconds=all_label_wall+geometry["exact_geometry_diagnostic_wall_seconds"]+timing["fit_wall_seconds"],
                total_with_geometry_labels_and_diagnostics_cpu_seconds=all_label_cpu+geometry["exact_geometry_diagnostic_cpu_seconds"]+timing["fit_cpu_seconds"])
        for method in ("mlopt_native", "mlopt_native_anchor_matched"):
            offline[method]["mlopt_timing_phases"] = mlopt_metadata["timing_phases"]
        offline["full"] = dict(training_samples=0, label_wall_seconds=0., label_cpu_seconds=0.,
            fit_wall_seconds=0., fit_cpu_seconds=0., total_offline_wall_seconds=0., total_offline_cpu_seconds=0.)
        meta = dict(group_id=gid, physical_seed=physical_seed, **fields, training_count=n, target_K=int(affine["rank"]),
                    common_D=144, total_original_variables=576, original_integer_variables=432,
                    offline_times=offline, coordinate_svm=coordinate_metadata, mlopt=mlopt_metadata,
                    historical_strategy=bank.metadata, native_svm=native.diagnostics_,
                    geometry_diagnostics=geometry, coverage_is_empirical_only=True,
                    training_feature_diversity=sample_diversity(features[:n]),
                    shared_preparation={"historical_bank": ["historical_strategy_matched", "best_observed_strategy"],
                                        "mlopt_model": ["mlopt_native", "mlopt_native_anchor_matched"]},
                    old_training_labels_used=False, test_information_used=False)
        write_new(model_folder / "metadata.json", meta); metadata_rows.append(meta)
        write_new(model_folder / "model-complete.json", dict(files={name:sha(model_folder/name) for name in
                  ("models.pkl", "metadata.json")}, completed_utc=utc()))
        event(event="pilot_models_fitted", group=gid, n=n, K=int(affine["rank"]))
    files = {str(p.relative_to(out)).replace("\\", "/"): sha(p) for p in folder.rglob("*") if p.is_file()}
    write_new(folder / "complete.json", dict(group_id=gid, files=files, completed_utc=utc()))
    return dict(group_id=gid, files=files, complete_sha256=sha(folder / "complete.json"), models=metadata_rows)


def execute(method, models, case, cost, features, solver_seed, budget):
    if method == "mlopt_native_anchor_matched":
        return execute_native_anchor(models["mlopt"], case, cost, features, solver_seed=solver_seed, budget=budget)
    start, cpu_start = perf_counter(), process_time(); fixed, equalities, diagnostic = None, [], {}
    if method == "svm_native": equalities, diagnostic = models["native"].predict_equalities(features)
    elif method == "coordinate_svm_matched": fixed, diagnostic = propose_coordinate_fixings(models["coordinate"], case, features)
    elif method == "historical_strategy_matched": fixed, diagnostic = propose_historical_matched(models["historical"], case, cost)
    prediction = perf_counter()-start
    remaining = budget-prediction
    if remaining <= 0: raw, x = dict(valid=False, certified=False, status="prediction_budget_exhausted"), None
    elif method in ("full", "coordinate_svm_matched", "historical_strategy_matched"):
        solved = adapter.solve(case.model, cost, fixed=fixed, seed=solver_seed, time_limit=remaining)
        raw, x = solved.to_dict(include_solution=False), solved.x
    elif method in ("ours", "svm_native"):
        raw, x = core.solve(case.original_base, cost, affine=case.affine if method == "ours" else None,
                            equalities=equalities, seed=solver_seed, time_limit=remaining)
    elif method == "best_observed_strategy":
        raw, x = solve_best_observed(models["historical"], case, cost, adapter, solver_seed, remaining)
    elif method == "mlopt_native":
        raw, x = predict_native(models["mlopt"], case, cost, features)
        prediction = perf_counter()-start
    else: raise ValueError("Unknown fixed pilot method")
    original = adapter.validate_solution(case.model, x) if x is not None else dict(valid=False)
    restriction = bool(raw.get("valid", False)); actual_K = None
    if method == "ours":
        actual_K = int(case.affine["rank"])
        if x is not None: restriction = restriction and core.exact_slice_contains(case.affine, np.rint(x[case.commitment_indices]))
    elif method == "full": actual_K = len(case.commitment_indices)
    elif method == "svm_native":
        if x is not None:
            binary = np.rint(x[case.commitment_indices]); restriction = restriction and all(int(eq.as_row(144)[0]@binary) == int(eq.rhs) for eq in equalities)
    elif fixed is not None:
        actual_K = 144-len(fixed)
        if x is not None: restriction = restriction and all(abs(float(x[j])-v) <= 1e-6 for j,v in fixed.items())
    objective = None if x is None else float(cost@x)
    elapsed, cpu = perf_counter()-start, process_time()-cpu_start
    valid = bool(original["valid"] and restriction and objective is not None and np.isfinite(objective) and elapsed <= budget)
    build, solves = float(raw.get("build_seconds", 0.)), float(raw.get("solve_seconds", 0.))
    if method == "best_observed_strategy":
        build = sum(c["solver_result"].get("build_seconds",0.) for c in raw["candidates"])
        solves = sum(c["solver_result"].get("solve_seconds",0.) for c in raw["candidates"])
    if method == "svm_native":
        pairs = [eq.as_row(144) for eq in equalities]
        dim = affine_equalities_metrics(np.asarray([r for r,b in pairs]).reshape(-1,144), np.asarray([b for r,b in pairs]))
        actual_K = dim["affine_dimension"] if dim["consistent"] else None
        diagnostic["dimension_diagnostics"] = dim
    physics = None
    if x is not None:
        physics = validate_uc_solution(case.original_base, x)
        if physics["valid"] != original["valid"]: raise RuntimeError("Independent UC physics disagrees")
    return clean(dict(method=method, seconds=elapsed, process_cpu_seconds=cpu, prediction_seconds=prediction,
        construction_seconds=build, solve_seconds=solves, validation_seconds=max(0.,elapsed-prediction-build-solves),
        other_required_seconds=max(0.,elapsed-prediction-build-solves),
        valid=valid, original_feasible=original["valid"], restriction_valid=restriction,
        objective=objective, solution=x, fixed=fixed, equalities=[dict(kind=e.kind,i=e.i,j=e.j,rhs=e.rhs) for e in equalities],
        actual_K=actual_K, solved_model_certified=bool(raw.get("certified",False)), status=raw.get("status"),
        certificate_scope=raw.get("certificate_scope", "original IP" if method=="full" else "restricted IP"),
        original_global_solver_certified=bool(method=="full" and valid and raw.get("certified",False)),
        raw_method_result=raw, diagnostic=diagnostic, original_check=original, physics=physics,
        method_wall_budget_exceeded=elapsed>budget, full_fallback_used=False, warm_start_used=False))


def benchmark(out, protocol, frozen, deadline):
    sources = frozen["source_binding"]; freeze_sha = sha(out / "model-freeze.json"); records = 0
    for group in frozen["groups"]:
        deadline.remaining()
        for rel,h in group["files"].items():
            if sha(out / rel) != h: raise RuntimeError("Frozen fitted artifact changed")
        meta = group["models"][0]; gid = group["group_id"]; physical_seed = meta["physical_seed"]
        scenario = next(s for s in protocol["scenarios"] if s["scenario_id"] == meta["scenario_id"])
        template = physical_template(physical_seed, gid, sources, scenario)
        with np.load(out / "training" / gid / "training.npz", allow_pickle=False) as stream: data = {k:stream[k].copy() for k in stream.files}
        fitted = {}
        for n in protocol["training_counts"]:
            with (out / "training" / gid / f"n{n}/models.pkl").open("rb") as stream: models = pickle.load(stream)
            fitted[n] = (new_case(template,gid,n,data,models["affine"]), models)
        features, sampling = sample_features(protocol,"test",physical_seed,protocol["test_count"],scenario)
        costs = adapter.costs_from_features(template,features)
        folder = out / "results" / gid; folder.mkdir(parents=True)
        save_npz_new(folder/"test-costs.npz",features=features,costs=costs)
        write_new(folder / "test-guard.json",dict(model_freeze_sha256=freeze_sha,sampler=sampling,generated_after_all_fitting=True,
            costs_sha256=sha(folder / "test-costs.npz"),
            diversity_by_training_count={n:sample_diversity(data["features"][:n],features) for n in protocol["training_counts"]}))
        for n,(case,models) in fitted.items():
            for method in protocol["methods"]:
                execute(method,models,case,case.train_costs[-1],case.train_features[-1],seed(f"{gid}|n{n}|warm"),deadline.budget(30))
        with (folder / "trials.jsonl").open("x",encoding="utf-8") as stream:
            for i,(cost,feature) in enumerate(zip(costs,features)):
                first_case, first_models = fitted[protocol["training_counts"][0]]
                full = execute("full",first_models,first_case,cost,feature,seed(f"{gid}|test{i}|full"),deadline.budget(30))
                reference, extra_reference = full["objective"], None
                if not full["valid"] or not full["solved_model_certified"]:
                    certified = adapter.solve(template.model,cost,seed=seed(f"{gid}|test{i}|reference"),
                                               time_limit=deadline.budget(protocol["solver"]["reference_certification_extension_seconds"]))
                    extra_reference = certified.to_dict()
                    if not certified.certified:
                        write_new(folder / f"uncertified-reference-{i}.json",extra_reference)
                        raise RuntimeError("Reference not certified; timed Full retained separately")
                    reference = float(cost@certified.x)
                for n,(case,models) in fitted.items():
                    other = [m for m in protocol["methods"] if m != "full"]
                    np.random.default_rng(seed(f"{gid}|n{n}|test{i}|order")).shuffle(other)
                    rows = [dict(full,reference_timing_reused_across_n=True)]+[execute(m,models,case,cost,feature,
                            seed(f"{gid}|n{n}|test{i}|solver"),deadline.budget(30)) for m in other]
                    for row in rows:
                        eligible = row["valid"] and row["objective"] is not None
                        objective = row["objective"]
                        row.update(group_id=gid,family="uc",physical_seed=physical_seed,**scenario_fields(scenario),
                            training_count=n,test_index=i,target_K=int(case.affine["rank"]),common_D=144,
                            total_original_variables=576,original_integer_variables=432,
                            reference_objective=reference,recovery_tolerance=adapter.recovery_tolerance(reference),
                            quality_recovered=bool(eligible and abs(objective-reference)<=adapter.recovery_tolerance(reference)),
                            objective_ratio=None if not eligible else objective/reference,
                            relative_gap=None if not eligible else (objective-reference)/abs(reference),
                            extra_reference_solver_result=extra_reference,model_freeze_sha256=freeze_sha,completed_utc=utc())
                        stream.write(json.dumps(clean(row),allow_nan=False)+"\n"); stream.flush(); records += 1
                if (i+1)%5 == 0: event(event="pilot_tests",group=gid,completed=i+1,records=records)
        write_new(folder / "complete.json",dict(records=protocol["test_count"]*len(protocol["training_counts"])*len(protocol["methods"]),
            trials_sha256=sha(folder / "trials.jsonl"),model_freeze_sha256=freeze_sha,completed_utc=utc()))
    expected = len(frozen["groups"])*len(protocol["training_counts"])*protocol["test_count"]*len(protocol["methods"])
    if records != expected or bindings() != sources: raise RuntimeError("Final trial count/source binding changed")
    write_new(out / "benchmark-complete.json",dict(records=records,expected_records=expected,all_groups_complete=True,
                                                   model_freeze_sha256=freeze_sha,completed_utc=utc()))


def validate_only(protocol):
    source_binding = bindings()
    summaries = []
    for physical_seed in protocol["physical_seeds"]:
        for scenario in protocol["scenarios"]:
            template = physical_template(physical_seed,"validation-only",source_binding,scenario)
            x = all_on_feasible_solution(template.original_base)
            if not validate_uc_solution(template.original_base,x)["valid"]: raise RuntimeError("Physical generator witness invalid")
            features,_ = sample_features(protocol,"train",physical_seed,96,scenario)
            costs = adapter.costs_from_features(template,features)
            if costs.shape != (96,576) or not np.isfinite(costs).all(): raise RuntimeError("Invalid cost mapping")
        summaries.append(dict(seed=physical_seed,variables=576,binary_variables=432,common_D=144,rows=template.model.A.shape[0]))
    event(event="pilot_validation_passed",physical_instances=summaries,new_solves=0,new_fits=0,source_binding=source_binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "run")
    parser.add_argument("--wall-seconds",type=float,default=1500.)
    parser.add_argument("--phase",choices=("all","geometry","fit","test"),default="all")
    parser.add_argument("--resume",action="store_true",help="Resume only SHA-verified committed label/model phases; never read old runs")
    parser.add_argument("--validate-only",action="store_true")
    args = parser.parse_args(); protocol = read(ROOT / "protocol.json")
    selected_scenarios=protocol["scenarios"];selected_ids=[s["scenario_id"] for s in selected_scenarios]
    expected = len(protocol["physical_seeds"])*len(selected_scenarios)*len(protocol["training_counts"])*protocol["test_count"]*len(protocol["methods"])
    if args.validate_only: validate_only(protocol); return
    if args.wall_seconds <= 0 or args.wall_seconds > 1800: raise ValueError("Process wall budget must be in (0,1800]")
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT) or args.out == ROOT: raise ValueError("Output must remain below this follow-up")
    if args.out.exists() and not args.resume:raise FileExistsError("Existing outputs require explicit --resume")
    if not args.out.exists() and args.phase in ("fit","test"):raise RuntimeError("Fit/test phases require committed fresh training from this follow-up")
    deadline = Deadline(args.wall_seconds); deadline.remaining()
    torch_begin, torch_cpu = perf_counter(),process_time()
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    optimizer_setup = torch.optim.Adam(torch.nn.Linear(1,1).parameters())
    del optimizer_setup
    setup = dict(import_setup_wall_seconds=perf_counter()-torch_begin,import_setup_cpu_seconds=process_time()-torch_cpu,
                 module_import_seconds=IMPORT_SECONDS,not_training=True)
    sources = bindings(); args.out.mkdir(parents=True,exist_ok=True)
    runtime = dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                   torch=torch.__version__,highs=highspy.Highs().version(),threads=1,threadpools=threadpool_info())
    source_path=args.out/"source-freeze.json"
    if source_path.exists():
        previous=read(source_path)
        if previous["source_binding"]!=sources or previous["protocol"]!=protocol:raise RuntimeError("Resume source/protocol changed")
        for field in ("python","numpy","scipy","sklearn","torch","highs"):
            if previous["runtime"][field]!=runtime[field]:raise RuntimeError("Resume runtime version changed")
    else:
        write_new(source_path,dict(protocol=protocol,source_binding=sources,runtime=runtime,torch_setup=setup,
            started_utc=utc(),process_wall_limit_seconds=args.wall_seconds,serial_training=True,tests_generated=False,
            selected_scenario_ids=selected_ids,selected_cohort_expected_records=expected,full_predeclared_grid_selected=True))
    invocation=utc().replace(':','').replace('+','')
    write_new(args.out/f"phase-start-{invocation}.json",dict(phase=args.phase,resume=args.resume,runtime=runtime,
        process_wall_limit_seconds=args.wall_seconds,source_binding=sources,started_utc=utc()))
    try:
        with threadpool_limits(limits=1):
            if args.phase in ("all","geometry"):
                geometry=[]
                for physical_seed in protocol["physical_seeds"]:
                    for scenario in selected_scenarios:
                        deadline.remaining();geometry.append(label_group(args.out,protocol,physical_seed,scenario,sources,deadline))
                freeze_training_screen(args.out,protocol,geometry)
            if args.phase in ("all","fit"):
                if not (args.out/"training-screening.json").exists():raise RuntimeError("Training-only screening must be frozen before any fitting/test selection")
                groups=[]
                for physical_seed in protocol["physical_seeds"]:
                    for scenario in selected_scenarios:
                        deadline.remaining();groups.append(fit_group(args.out,protocol,physical_seed,scenario,sources,deadline))
                if bindings()!=sources:raise RuntimeError("Sources changed before model freeze")
                model_path=args.out/"model-freeze.json"
                groups=clean(groups)
                frozen=dict(source_binding=sources,groups=groups,completed_utc=utc(),test_accessed=False,
                    training_screening_sha256=sha(args.out/"training-screening.json"),all96_label_acquisition_costs_reported=True,
                    sampler_streams=protocol["sampler_streams"])
                if model_path.exists():
                    previous=read(model_path)
                    if previous["source_binding"]!=sources or previous["groups"]!=groups:raise RuntimeError("Frozen models changed")
                else:write_new(model_path,frozen)
            if args.phase in ("all","test"):
                frozen=read(args.out/"model-freeze.json")
                if frozen["training_screening_sha256"]!=sha(args.out/"training-screening.json"):raise RuntimeError("Pretest training screen changed")
                if (args.out/"results").exists():raise RuntimeError("Testing has already started; preserve its outputs and audit partial state before any new run")
                benchmark(args.out,protocol,frozen,deadline)
        write_new(args.out/f"phase-receipt-{invocation}.json",dict(status="complete",phase=args.phase,
            process_wall_seconds=perf_counter()-IMPORT_STARTED,source_binding=sources,completed_utc=utc()))
        if args.phase in ("all","test"):
            fit_sum=0.;label_sum=0.;cpu_labels=0.;geometry_sum=0.
            for group in frozen["groups"]:
                meta0=group["models"][0]
                label_sum+=meta0["offline_times"]["ours"]["geometry_acquisition_label_wall_seconds"]
                cpu_labels+=meta0["offline_times"]["ours"]["geometry_acquisition_label_cpu_seconds"]
                geometry_sum+=meta0["offline_times"]["ours"]["exact_geometry_diagnostic_wall_seconds"]
                for meta in group["models"]:
                    fit_sum+=sum(meta["offline_times"][m]["fit_wall_seconds"] for m in
                        ("ours","svm_native","coordinate_svm_matched","historical_strategy_matched","mlopt_native"))
            write_new(args.out/"completion-receipt.json",dict(status="complete",records=expected,expected_records=expected,
                new_training_labels=1152,source_binding=sources,all96_shared_label_wall_seconds=label_sum,
                all96_shared_label_cpu_seconds=cpu_labels,all_distinct_model_fit_wall_seconds=fit_sum,
                shared_label_plus_distinct_fit_wall_seconds=label_sum+fit_sum,
                all_exact_geometry_diagnostic_wall_seconds=geometry_sum,
                shared_label_plus_geometry_diagnostic_plus_distinct_fit_wall_seconds=label_sum+geometry_sum+fit_sum,
                serial_training=True,serial_testing=True,old_data_read=False,unverified_screen06_used=False,
                new_physical_seeds=protocol["physical_seeds"],selected_scenario_ids=selected_ids,
                full_predeclared_grid_complete=True,completed_utc=utc()))
    except Exception as exc:
        write_new(args.out / f"failure-receipt-{invocation}.json",dict(status="incomplete",phase=args.phase,exception=repr(exc),completed_utc=utc(),
            process_wall_seconds=perf_counter()-IMPORT_STARTED,outputs_preserved=True))
        raise


if __name__ == "__main__": main()
