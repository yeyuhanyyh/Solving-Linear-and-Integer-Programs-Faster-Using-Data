"""New exploratory UC reference/SVM/MLOPT screen, preserving all fixed seeds.

Does not access old/fresh confirmation banks, change original sources, or train
Neural Diving/Li. All objective comparisons retain real total UC costs.
"""
from __future__ import annotations
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
from datetime import datetime,timezone
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from time import perf_counter,process_time
import numpy as np
import sympy as sp

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parents[1]
CODE=EXPERIMENT/"code"
for path in [CODE,CODE/"controls"]:sys.path.insert(0,str(path))
import experiment_core as core
import uc_model as uc
import portable_adapter as adapter

SEEDS=[101,211,307]
SETTINGS=[dict(name=f"G{G}T12_{scope}",G=G,T=12,scope=scope) for G in [4,6] for scope in ["two_units","all_units"]]
PREFIXES=[20,32]
EXPLORATORY_LABEL_COUNT=64
DEVELOPMENT_COUNT=24


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:json.dump(clean(value),stream,indent=2,allow_nan=False);stream.write("\n")
def rng(label):return np.random.default_rng(int.from_bytes(hashlib.sha256(label.encode()).digest()[:16],"little"))


def physical(setting,seed):
    G,T=setting["G"],setting["T"];q=G*T
    base=uc.make_uc_base(seed,n_units=G,n_times=T)
    old_fuel=np.asarray(base["metadata"]["fuel_cost"]).copy()
    fuel=2.+(old_fuel-22.)*3./23.
    base["metadata"]["fuel_cost"]=fuel
    base["base_cost"][3*q:4*q]=np.tile(fuel,T)
    old_fraction=base["metadata"]["demand"]/sum(base["metadata"]["capacity"])
    fraction=.30+.20*(old_fraction-old_fraction.min())/(old_fraction.max()-old_fraction.min())
    demand=fraction*sum(base["metadata"]["capacity"])
    for t,value in enumerate(demand):
        row=base["metadata"]["row_names"].index(f"balance[{t}]")
        base["row_lower"][row]=base["row_upper"][row]=value
    base["metadata"].update(demand=demand,original_fuel_cost_before_mapping=old_fuel,
        new_fuel_cost_mapping="2+(oldfuel-22)*3/23; preserves old physical RNG",
        demand_fraction=fraction,demand_mapping="Original smooth profile mapped to total-capacity fractions[0.30,0.50]",
        objective_note="Original total operating, no-load, startup and shutdown costs; no constant subtracted")
    feasible=uc.all_on_feasible_solution(base)
    if not uc.validate_uc_solution(base,feasible)["valid"]:raise ValueError("Modified UC lost all-on physical feasibility")
    selected=list(range(2)) if setting["scope"]=="two_units" else list(range(G))
    offset=base["base_cost"].copy();coeff=np.zeros((4*q,3*len(selected)))
    for component in range(4):
        for t in range(T):
            for local,g in enumerate(selected):
                j=component*q+t*G+g;feature=3*local+t//4
                coeff[j,feature]=base["base_cost"][j];offset[j]=0
    return base,offset,coeff,selected


def draw(offset,coeff,count,random):
    features=np.exp(random.uniform(-np.log(3),np.log(3),(count,coeff.shape[1])))
    costs=offset+features@coeff.T
    if not np.all(costs>0):raise ValueError("UC requires positive original costs")
    return features,costs


def solve_reference(base,cost,seed):
    start=perf_counter();cpu=process_time();raw,x=core.solve(base,cost,time_limit=30,seed=seed)
    valid=bool(x is not None and raw["valid"] and uc.validate_uc_solution(base,x)["valid"])
    raw.update(outer_wall_seconds=perf_counter()-start,outer_cpu_seconds=process_time()-cpu,
               independent_UC_physics_valid=valid,timing_not_isolated=True)
    if not raw["certified"] or not valid:raise ValueError("UC original reference did not certify optimum")
    return raw,x


def make_case(folder,setting,seed,n,base,offset,coeff,features,costs,solutions,objectives):
    G,T=setting["G"],setting["T"];common=base["binary_indices"]
    affine=core.fit_affine(np.rint(solutions[:n,common]).astype(np.int64))
    folder.mkdir(parents=True,exist_ok=False)
    A=base["A"]
    np.savez_compressed(folder/"training.npz",A_data=A.data,A_indices=A.indices,A_indptr=A.indptr,A_shape=np.asarray(A.shape),
        **{name:base[name] for name in ["row_lower","row_upper","col_lower","col_upper","integrality"]},
        commitment_indices=common,binary_indices=common,train_features=features[:n],train_costs=costs[:n],
        train_solutions=solutions[:n],train_objectives=objectives[:n],cost_offset=offset,cost_feature_coefficients=coeff)
    cid=f"uc_startupheavy_{setting['name']}_seed{seed}_n{n}"
    metadata=dict(case_id=cid,family="uc",seed=seed,n_train=n,n_units=G,n_times=T,D=G*T,
        original_variables=4*G*T,original_integer_variables=3*G*T,K=affine["rank"],feature_dimension=coeff.shape[1],
        exploratory_training_labels_generated=EXPLORATORY_LABEL_COUNT,
        fitting_uses_only_prefix=n,global_optimizer_hull_K_not_known=True,
        source_purpose="Exploratory screen; no confirmation claim",metadata=base["metadata"],
        cost_offset=offset,cost_feature_coefficients=coeff,
        feature_order="generator-major: feature=3*selected_generator_position+four_period_block",
        training_npz_sha256=sha(folder/"training.npz"))
    write_new(folder/"metadata.json",metadata);write_new(folder/"affine.json",affine)
    model=adapter.LinearModel(A=base["A"],**{name:base[name] for name in ["row_lower","row_upper","col_lower","col_upper","integrality"]},row_names=tuple(base["metadata"]["row_names"]))
    hashes={name:sha(folder/name) for name in ["training.npz","metadata.json","affine.json"]}
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    case=adapter.FrozenCase(case_id=cid,family="uc",seed=seed,amplitude=0.,model=model,
        train_features=features[:n].copy(),train_costs=costs[:n].copy(),train_solutions=solutions[:n].copy(),
        train_objectives=objectives[:n].copy(),commitment_indices=common.copy(),affine=affine,
        original_base=base,metadata=deepcopy(base["metadata"]),source_hashes=hashes,dataset_fingerprint=fingerprint,source_directory=folder)
    return case


def run_setting(out,setting,seed,helpers):
    key=f"{setting['name']}_seed{seed}";folder=out/"settings"/key
    folder.mkdir(parents=True,exist_ok=False)
    base,offset,coeff,varying=physical(setting,seed)
    train_features,train_costs=draw(offset,coeff,EXPLORATORY_LABEL_COUNT,rng(f"uc-ratio003-training-v1|{key}"))
    refs=[];solutions=[];objectives=[]
    for i,cost in enumerate(train_costs):
        raw,x=solve_reference(base,cost,seed+i)
        refs.append(raw);solutions.append(x);objectives.append(raw["objective"])
    solutions=np.asarray(solutions);objectives=np.asarray(objectives)
    common=base["binary_indices"]
    rank_by_n={n:len(sp.Matrix((np.rint(solutions[1:n,common])-np.rint(solutions[0,common])).T.astype(int).tolist()).rref()[1]) for n in [20,32,40,64]}
    write_new(folder/"training-reference-certificates.json",dict(count=64,records=refs,rank_by_prefix=rank_by_n,
        total_original_label_IP_wall_seconds=sum(r["outer_wall_seconds"] for r in refs),
        scope="64-label exploratory diagnostics, not only baseline fitting prefix"))
    fitted=[]
    for n in PREFIXES:
        case=make_case(folder/f"n{n}",setting,seed,n,base,offset,coeff,train_features,train_costs,solutions,objectives)
        svm,mlopt,metadata=helpers.fit_baselines(case)
        write_new(case.source_directory/"baseline-fit-metadata.json",metadata)
        fitted.append((case,svm,mlopt))
    # Independent exploratory development is sampled only after both fixed
    # prefix models have been fitted. It is never used by baseline fitting.
    dev_features,dev_costs=draw(offset,coeff,DEVELOPMENT_COUNT,rng(f"uc-ratio003-development-v1|{key}"))
    dev_refs=[];dev_solutions=[];dev_objectives=[]
    for i,cost in enumerate(dev_costs):
        raw,x=solve_reference(base,cost,seed+10000+i)
        dev_refs.append(raw);dev_solutions.append(x);dev_objectives.append(raw["objective"])
    dev_solutions=np.asarray(dev_solutions);dev_objectives=np.asarray(dev_objectives)
    np.savez_compressed(folder/"exploratory-development.npz",features=dev_features,costs=dev_costs,solutions=dev_solutions,objectives=dev_objectives)
    write_new(folder/"development-reference-certificates.json",dev_refs)
    rows=[]
    for case,svm,mlopt in fitted:
        records=[];contained=[]
        for i,(feature,cost,reference,xfull) in enumerate(zip(dev_features,dev_costs,dev_objectives,dev_solutions)):
            contained.append(core.exact_slice_contains(case.affine,np.rint(xfull[common]).astype(np.int64)))
            start=perf_counter();cpu=process_time();raw,x=core.solve(base,cost,affine=case.affine,time_limit=30,seed=seed+20000+i)
            original_valid=bool(x is not None and raw["valid"] and uc.validate_uc_solution(base,x)["valid"])
            outer_wall=perf_counter()-start;outer_cpu=process_time()-cpu
            ours=dict(valid=bool(original_valid and outer_wall<=30.),
                objective=None if x is None else float(cost@x),solution=x,actual_K=case.affine["rank"],
                seconds=outer_wall,cpu_seconds=outer_cpu,raw=raw,original_feasible=original_valid,
                method_wall_budget_exceeded=outer_wall>30.)
            predictions={"Ours":ours,"SVM":helpers.predict_svm(svm,case,cost,feature,seed+30000+i,30.),
                         "MLOPT":helpers.predict_mlopt(mlopt,case,cost,feature,seed+40000+i,30.)}
            for method,prediction in predictions.items():
                value=prediction.get("objective")
                valid=bool(prediction.get("valid") and value is not None and value>0 and np.isfinite(value))
                prediction.update(method=method,index=i,case_id=case.case_id,reference_objective=reference,
                    recovered=bool(valid and abs(value-reference)<=adapter.recovery_tolerance(reference)),
                    objective_ratio=None if not valid else float(reference/value),
                    comparison_uses_original_total_cost=True,confirmation=False)
                records.append(prediction)
        write_new(case.source_directory/"exploratory-development-results.json",records)
        for method in ["Ours","SVM","MLOPT"]:
            subset=[r for r in records if r["method"]==method]
            ratios=[r["objective_ratio"] for r in subset if r["objective_ratio"] is not None]
            rows.append(dict(case_id=case.case_id,setting=setting["name"],seed=seed,n_train=case.n_train,
                common_D=len(common),original_variables=base["A"].shape[1],original_binary_variables=int(base["integrality"].sum()),
                feature_dimension=coeff.shape[1],K=case.affine["rank"],rank_by_prefix=rank_by_n,
                method=method,count=24,valid_count=len(ratios),recovery_count=sum(r["recovered"] for r in subset),
                objective_ratio_mean=None if not ratios else float(np.mean(ratios)),ratio_denominator=len(ratios),
                return_wall_seconds_mean=float(np.mean([r["seconds"] for r in subset])),
                restricted_IP_build_seconds_mean=float(np.mean([r["raw"].get("build_seconds",0.) for r in subset])),
                restricted_IP_solve_seconds_mean=float(np.mean([r["raw"].get("solve_seconds",0.) for r in subset])),
                reference_wall_seconds_mean=float(np.mean([r["outer_wall_seconds"] for r in dev_refs])),
                reference_IP_solve_seconds_mean=float(np.mean([r.get("solve_seconds",0.) for r in dev_refs])),
                heldout_optimizer_in_learned_u_affine_fraction=float(np.mean(contained)),
                global_optimizer_hull_K_proven=False,exploratory_only=True,timing_not_isolated=True,
                fit_metadata_sha256=sha(case.source_directory/"baseline-fit-metadata.json")))
    write_new(folder/"complete.json",dict(status="COMPLETE",rows=rows,source_files={str(p.resolve()):sha(p) for p in folder.rglob("*") if p.is_file()}))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--settings",nargs="*",choices=[r["name"] for r in SETTINGS]);args=parser.parse_args()
    selected=SETTINGS if not args.settings else [r for r in SETTINGS if r["name"] in args.settings]
    out=HERE/"run-v2"
    if out.exists():raise FileExistsError("Immutable UC screen already exists")
    helpers_path=HERE/"baseline_helpers.py"
    if not helpers_path.is_file():raise FileNotFoundError("Wait for reviewed baseline helper before sampling")
    out.mkdir()
    sources=[Path(__file__).resolve(),helpers_path,CODE/"controls"/"uc_model.py",CODE/"controls"/"experiment_core.py",CODE/"portable_adapter.py",CODE/"mlopt_baseline.py"]
    protocol=dict(created_utc=now(),settings=selected,all_predeclared_settings=SETTINGS,seeds=SEEDS,training_prefixes=PREFIXES,
        exploratory_original_training_label_count=64,rank_diagnostic_prefixes=[20,32,40,64],development_count=24,
        cost_law="Independent LogUniform[1/3,3] cost multipliers for each selected generator and four-period block, shared by u/v/w/p components",
        feature_order="generator-major: feature=3*selected_generator_position+four_period_block",
        physical_modifications="Original UC preserved except smooth demand mapped to total capacity0.30–0.50 and original fuelUniform22–45 mapped to2–5; on/start/stop untouched",
        retained_UC_integrality="u,v,w binary; only u participates in common compressionK; continuous dispatch reoptimized",
        metric="Original optimal totalcost / returned totalcost; higher is better, ideally<=1; no objective constant removed",
        all_fixed_seeds_retained=True,source_sha256={str(p):sha(p) for p in sources},
        no_large_neural_training=True,no_NeuralDiving_or_Li_training=True,
        MLOPT="Original small eight-grid classifier; native top-ten winner anchors partial common-u matching",
        global_rank_not_proven=True,no_confirmation_claim=True,no_old_or_fresh_bank_read=True,timing_not_isolated=True)
    write_new(out/"protocol.json",protocol)
    spec=importlib.util.spec_from_file_location("uc_screen_baseline_helpers",helpers_path)
    helpers=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helpers;spec.loader.exec_module(helpers)
    helpers._runtime()  # Validate every transitive read-only module before sampling.
    rows=[]
    for setting in selected:
        for seed in SEEDS:
            try:
                new=run_setting(out,setting,seed,helpers);rows.extend(new)
                print(json.dumps(dict(event="uc_screen_complete",setting=setting["name"],seed=seed,
                    means=[dict(n=r["n_train"],method=r["method"],K=r["K"],ratio=r["objective_ratio_mean"],valid=r["valid_count"]) for r in new])),flush=True)
            except Exception as error:
                write_new(out/f"{setting['name']}_seed{seed}-failure.json",dict(setting=setting,seed=seed,error=repr(error),
                    retained=True,no_seed_or_cost_replacement=True,created_utc=now()))
                print(json.dumps(dict(event="uc_screen_failure",setting=setting["name"],seed=seed,error=repr(error))),flush=True)
    write_new(out/"summary.json",dict(status="COMPLETE_WITH_RETAINED_FAILURES" if list(out.glob("*-failure.json")) else "COMPLETE",
        created_utc=now(),rows=rows,protocol_sha256=sha(out/"protocol.json"),
        source_sha256={str(p.resolve()):sha(p) for p in out.rglob("*") if p.is_file()},exploratory_only=True))
    print(json.dumps(dict(status="UC_SCREEN_FINISHED",rows=len(rows),summary_sha256=sha(out/"summary.json"))))


if __name__=="__main__":main()
