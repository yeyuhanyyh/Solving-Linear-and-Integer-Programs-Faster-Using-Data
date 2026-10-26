"""Paired stream2 control: no generation, training or bank access on import.

All five method roles at all three physical seeds are frozen before any bank
read. Original/continued ND retain the sealed scientific evaluator; their role
identities live in guards and output directories, never in model selection.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

TOOLS=Path(__file__).resolve().parent
CONTROL=TOOLS.parent
DESIGNS=TOOLS.parents[2]
FOLLOWUP=DESIGNS.parent
PROTOCOL=TOOLS/"stream2-protocol.json"
BASE_PATH=DESIGNS/"coupled-bundle/confirmation-v1/confirmation_runner.py"
BASE_SHA="d5805a5f0b214ce7ce078f06046403334878f97f8afdfe2b9890ebd9d514ec51"
OLD_GENERATOR=BASE_PATH.parent/"fresh_bank_cli.py"
OLD_GENERATOR_SHA="62a6557c74bfcc791c74bc462252f3dc8cd7ce67d4e98c2eb5e9f2f4908eab09"
EXTRA_ROOT=FOLLOWUP/"objective-gap-followup-20261025/extra-baselines-coupled-v1"
EXTRA_DRIVER=EXTRA_ROOT/"coupled_svm_mlopt_driver.py"
EXTRA_SHA="6b492cd46f13645b3076a35f6a0769e747443988dd3eda3046243465b57c530a"
DEFAULT_RUN=FOLLOWUP/"objective-gap-followup-20261025/uniform-gross-stream2-training-budget-control"
DEFAULT_FREEZE=DEFAULT_RUN/"all-five-role-models-freeze.json"
METHODS=("ours","neural_diving_original","neural_diving_continued","svm","mlopt")
PRIMARY=METHODS[:3]
_BASE=_CONTROL=_EXTRA=None

def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def _import(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def enable():
    global _BASE,_CONTROL,_EXTRA
    if sha(BASE_PATH)!=BASE_SHA or sha(OLD_GENERATOR)!=OLD_GENERATOR_SHA or sha(EXTRA_DRIVER)!=EXTRA_SHA:raise ValueError("A frozen reused source changed")
    if _BASE is None:_BASE=_import(BASE_PATH,"stream2_original_guard")
    d=_BASE.enable()
    if _CONTROL is None:_CONTROL=_import(CONTROL/"continue_training.py","stream2_training_control")
    if _EXTRA is None:
        sys.path.insert(0,str(EXTRA_ROOT));_EXTRA=_import(EXTRA_DRIVER,"stream2_extra_baselines")
    _EXTRA.enable()
    if Path(_EXTRA.components.__file__).resolve()!=EXTRA_ROOT/"svm_mlopt_components.py":raise ValueError("Wrong extra-baseline component module")
    return d
def clean(value):return enable().clean(value)
def write_new(path,value):return enable().write_new(path,value)
def canonical(value):return json.dumps(clean(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def payload_sha(value):return hashlib.sha256(canonical(value)).hexdigest()
def protocol():
    p=read(PROTOCOL)
    expected=dict(expected_case_count=3,expected_physical_bank_count=3,expected_model_role_count=15,method_ids=list(METHODS),physical_seeds=[1009,1013,1019],G=32,D=96,K=32,n_train=36,test_count_per_case=120,xi_lower=.1,xi_upper=2.5,evaluation_budget_seconds=30.)
    if any(p.get(k)!=v for k,v in expected.items()) or p.get("rng_namespace")!="coupled-bundle-independent-uniform-gross-stream2-fixed-budget-v1-20261025":raise ValueError("Fixed stream2 protocol changed")
    return p
def _files(folder,marker,required):
    if not set(required)<=set(marker["artifact_sha256"]):raise ValueError("Missing actual model artifact")
    result={str((folder/"complete.json").resolve()):sha(folder/"complete.json")}
    for name,expected in marker["artifact_sha256"].items():
        path=folder/name
        if not path.resolve().is_relative_to(folder.resolve()) or sha(path)!=expected:raise ValueError("Model artifact changed or escaped its folder")
        result[str(path.resolve())]=expected
    return result
def _live_cases():
    d=enable();p=protocol();_CONTROL.validate_source();cp=_CONTROL.protocol();records=_CONTROL.originals()
    manifest=d.validate_selection(cp["manifest_path"]);byid={e["case_id"]:e for e in manifest["cases"]}
    cases={};entries={}
    for cid,record in records.items():
        case=record["case"];entry=byid[cid]
        if case.n_train!=36 or case.train_features.shape[1]!=32 or int(case.affine["rank"])!=32 or case.seed not in p["physical_seeds"]:raise ValueError("Wrong stream2 training cohort")
        continued_folder,continued=_CONTROL.validate_final(cid,CONTROL/"continued-models")
        if continued.get("training_only") is not True:raise ValueError("Continued checkpoint was not training only")
        cf=_files(continued_folder,continued,{"continued-predictor.pt","metadata.json"})
        cm=read(continued_folder/"metadata.json")
        if any(cm.get(k)!=v for k,v in dict(optimizer_steps=2000,final_checkpoint_step=2000,all_training_count=36,optimizer_state_reset=True,initial_Adam_state_empty=True,training_only=True,validation_or_test_read=False,checkpoint_selected_by="fixed final update2000 only").items()):raise ValueError("Continued final-budget semantics changed")
        methods={"ours":record["original_roles"]["ours"],"neural_diving_original":record["original_roles"]["neural_diving"],"neural_diving_continued":dict(guard=continued["guard"],files=cf)}
        for method in ("svm","mlopt"):
            folder,marker=_EXTRA.validate_checkpoint(case,method,EXTRA_ROOT/"results",cp["manifest_path"])
            if marker.get("sanity_gate_passed") is not True:raise ValueError("Extra baseline readiness gate failed")
            methods[method]=dict(guard=marker["guard"],files=_files(folder,marker,{"model.pkl","metadata.json"}))
        training={str((Path(entry["directory"])/name).resolve()):expected for name,expected in entry["files"].items()}
        entries[cid]=dict(case_id=cid,G=32,D=96,K=32,n_train=36,seed=case.seed,physical_id=f"stream2_G32_seed{case.seed}",
            manifest_path=str(Path(cp["manifest_path"]).resolve()),model_root=str(Path(cp["original_model_root"]).resolve()),continued_root=str((CONTROL/"continued-models").resolve()),extra_model_root=str((EXTRA_ROOT/"results").resolve()),training_files=training,methods=methods,
            physical_model_sha256=d.array_hash(case.model.A.data,case.model.A.indices,case.model.A.indptr,case.model.row_lower,case.model.row_upper,case.model.col_lower,case.model.col_upper,case.model.integrality,np.asarray(case.metadata["cost_offset"]),np.asarray(case.metadata["cost_feature_coefficients"])))
        cases[cid]=case
    if len(cases)!=3 or {e["seed"] for e in entries.values()}!=set(p["physical_seeds"]) or sum(len(e["methods"]) for e in entries.values())!=15:raise ValueError("Need all three cases / fifteen method roles")
    return cases,entries
def freeze_all_models(freeze_path=DEFAULT_FREEZE,bank_root=DEFAULT_RUN/"banks"):
    freeze_path=Path(freeze_path)
    if freeze_path.exists():return validate_global_freeze(freeze_path)[0]
    if Path(bank_root).exists() and any(p.is_file() for p in Path(bank_root).rglob("*")):raise ValueError("Stream2 banks precede the all-role freeze")
    protocol();cases,entries=_live_cases()
    helper=BASE_PATH.parent/"verify_evaluator_equivalence.py"
    eq=_import(helper,"stream2_static_equivalence").verify(enable().ROOT/"generic_binary_pilot.py",Path(__file__))
    if not eq["passed"]:raise ValueError("Original ND/Ours scientific evaluator changed")
    source={str(p.resolve()):sha(p) for p in (Path(__file__),PROTOCOL,TOOLS/"fresh_stream2_cli.py",BASE_PATH,OLD_GENERATOR,helper,CONTROL/"continue_training.py",CONTROL/"control-protocol.json",CONTROL/"control-source-seal.json")}
    source.update(_CONTROL.validate_source()["source_sha256"])
    extra_seal=EXTRA_ROOT/"source-seal.json";extra_freeze=EXTRA_ROOT/"g32-model-freeze.json"
    if sha(extra_freeze)!="58dca742d64884833db5f7c10c62bf08cb2173d42e1b6428411066b4e3f6b9c9":raise ValueError("The separately frozen six SVM/MLOPT checkpoints changed")
    for p in (extra_seal,extra_freeze):source[str(p.resolve())]=sha(p)
    for name,expected in read(extra_seal)["source_sha256"].items():
        p=EXTRA_ROOT/name
        if sha(p)!=expected:raise ValueError("Frozen SVM/MLOPT source changed")
        source[str(p.resolve())]=expected
    result=dict(schema="stream2-all-five-role-models-freeze-v1",frozen=True,frozen_at_utc=datetime.now(timezone.utc).isoformat(),case_count=3,model_role_count=15,physical_bank_count=3,
        protocol_path=str(PROTOCOL.resolve()),protocol_sha256=sha(PROTOCOL),bank_root=str(Path(bank_root).resolve()),source_sha256=source,cases=entries,scientific_evaluation_equivalence=eq,confirmation_not_generated_before_freeze=True)
    write_new(freeze_path,result);validate_global_freeze(freeze_path);return result
def validate_global_freeze(freeze_path=DEFAULT_FREEZE,expected_sha=None):
    if expected_sha is not None and sha(freeze_path)!=expected_sha:raise ValueError("Explicit stream2 freeze digest differs")
    f=read(freeze_path)
    if any(f.get(k)!=v for k,v in dict(schema="stream2-all-five-role-models-freeze-v1",frozen=True,case_count=3,model_role_count=15,physical_bank_count=3).items()):raise ValueError("Need the completed fifteen-role stream2 freeze")
    if Path(f["protocol_path"]).resolve()!=PROTOCOL.resolve() or f["protocol_sha256"]!=sha(PROTOCOL):raise ValueError("Stream2 protocol binding differs")
    for name,expected in f["source_sha256"].items():
        if sha(name)!=expected:raise ValueError("Frozen source/artifact changed: "+name)
    cases,entries=_live_cases()
    if entries!=f["cases"]:raise ValueError("Live complete model/label/training registry differs")
    if expected_sha is not None and sha(freeze_path)!=expected_sha:raise ValueError("Freeze changed during validation")
    return f,cases
def load_confirmation_bank(case,freeze_path=DEFAULT_FREEZE):
    f,_=validate_global_freeze(freeze_path);e=f["cases"][case.case_id];folder=Path(f["bank_root"])/e["physical_id"]
    meta=read(folder/"metadata.json");npz=folder/"confirmation.npz";refpath=folder/"reference-certificates.json"
    bindings={case.case_id:dict(K=32,n_train=36,training_files=e["training_files"],model_role_files={role:e["methods"][role]["files"] for role in METHODS})}
    required=dict(schema="stream2-shared-physical-confirmation-bank-v1",phase="independent_confirmation",confirmation=True,physical_id=e["physical_id"],case_ids=[case.case_id],case_bindings=bindings,G=32,D=96,seed=e["seed"],count=120,model_freeze_sha256=sha(freeze_path),protocol_sha256=f["protocol_sha256"],npz_sha256=sha(npz),reference_certificates_sha256=sha(refpath),rng_namespace=protocol()["rng_namespace"],method_ids=list(METHODS),model_role_ids=list(METHODS))
    for key,value in required.items():
        if meta.get(key)!=value:raise ValueError("Stream2 bank context differs: "+key)
    with np.load(npz,allow_pickle=False) as stream:bank={key:stream[key].copy() for key in stream.files}
    if set(bank)!={"features","costs","solutions","objectives"}:raise ValueError("Stream2 array schema differs")
    for key,shape in dict(features=(120,32),costs=(120,96),solutions=(120,96),objectives=(120,)).items():
        if bank[key].shape!=shape or not np.isfinite(bank[key]).all():raise ValueError("Stream2 array shape/finite failure")
    d=enable();import adapter
    if not np.all((bank["features"]>=.1)&(bank["features"]<=2.5)) or not np.all(bank["objectives"]<0) or not np.allclose(d.signed_costs_from_features(case,bank["features"]),bank["costs"],rtol=1e-14,atol=1e-12):raise ValueError("Stream2 original law/map differs")
    certificates=read(refpath);guard=dict(model_freeze_sha256=sha(freeze_path),protocol_sha256=f["protocol_sha256"])
    required_cert=dict(schema="stream2-confirmation-reference-certificates-v1",physical_id=e["physical_id"],case_ids=[case.case_id],count=120,guard=guard,all_original_IP_certified=True,all_independent_references_agree=True)
    if any(certificates.get(k)!=v for k,v in required_cert.items()) or set(certificates["record_file_sha256"])!={f"reference-records/ref{i:03d}.json" for i in range(120)}:raise ValueError("Stream2 reference certificate population differs")
    for i,(feature,cost,x,reference) in enumerate(zip(bank["features"],bank["costs"],bank["solutions"],bank["objectives"])):
        relative=f"reference-records/ref{i:03d}.json";path=folder/relative
        if sha(path)!=certificates["record_file_sha256"][relative]:raise ValueError("Reference record changed")
        row=read(path);_BASE.verify_payload(row);oracle=row.get("highs",{});tol=adapter.recovery_tolerance(float(reference))
        if row.get("physical_id")!=e["physical_id"] or row.get("index")!=i or row.get("guard")!=guard:raise ValueError("Reference identity differs")
        if not np.array_equal(row["feature"],feature) or not np.array_equal(row["cost"],cost) or not np.array_equal(row["solution"],x):raise ValueError("Reference arrays differ")
        if not adapter.validate_solution(case.model,x)["valid"] or abs(float(cost@x)-reference)>tol or not (oracle.get("valid") is True and oracle.get("certified") is True):raise ValueError("Invalid or uncertified original reference")
        for value in (row["objective"],row["independent_objective"],oracle["objective"],oracle["dual_bound"]):
            if not _BASE._reference_equal(value,reference,tol):raise ValueError("Reference primal/dual/MWIS differs")
    return bank,meta
def verify_confirmation_record(*args,**kwargs):return _BASE.verify_confirmation_record(*args,**kwargs)
def confirmation_guard(case,role,meta,freeze_path,budget):
    f,_=validate_global_freeze(freeze_path);e=f["cases"][case.case_id]
    return dict(training=enable().pilot_guard(case,"ours" if role=="ours" else "neural_diving" if role in PRIMARY else role),method_role=role,phase="independent_confirmation",stream="fixed_training_budget_stream2",model_freeze_sha256=sha(freeze_path),protocol_sha256=f["protocol_sha256"],confirmation_bank_sha256=meta["npz_sha256"],confirmation_metadata_sha256=sha(Path(f["bank_root"])/e["physical_id"]/"metadata.json"),reference_certificates_sha256=meta["reference_certificates_sha256"],wrapper_sha256=sha(__file__),model_role_files_sha256=e["methods"][role]["files"],budget=float(budget))

ROOT=DESIGNS/"adapter/generic-binary-v2"

def evaluate_case_confirmation(case,out,methods,budget,freeze_path,results_root,role):
    import adapter
    import experiment_core as core
    import train_uc_baselines as nn
    import neural_diving_baseline as nd
    # Fail closed: complete training guard precedes all development-bank access.
    nn.validate_checkpoint(out,case,"neural_diving",ROOT)
    bank,metadata=load_confirmation_bank(case,freeze_path)
    bank_hash=metadata["npz_sha256"]
    model=None
    if "neural_diving" in methods:
        folder,_=nn.validate_checkpoint(out,case,"neural_diving",ROOT)
        if role=="neural_diving_continued":
            continued,_=_CONTROL.validate_final(case.case_id,CONTROL/"continued-models")
            model=nn.load_nd_complete(case,continued/"continued-predictor.pt")
        else:model=nn.load_nd_complete(case,folder/"complete-predictor.pt")
    K=int(case.affine["rank"]);results=[]
    for method in methods:
        folder=Path(results_root)/case.case_id/role;folder.mkdir(parents=True,exist_ok=True)
        guard=confirmation_guard(case,role,metadata,freeze_path,budget)
        rows=[]
        for i,(cost,reference) in enumerate(zip(bank["costs"],bank["objectives"])):
            path=folder/f"test{i:03d}.json"
            if path.exists():
                row=read(path)
                verify_confirmation_record(row,case,method,i,cost,reference,guard,budget,bank["features"][i])
                rows.append(row);continue
            begin=time.perf_counter()
            if method=="ours":
                raw,x=core.solve(case.original_base,cost,affine=case.affine,time_limit=budget,seed=case.seed+i)
                fixed=None
            elif method=="neural_diving":
                raw=nd.solve_matched_cost(model,case,cost,seed=case.seed+i,K=K,total_time_limit=budget,solve_fn=adapter.solve)
                x=raw.pop("x",None);fixed=raw.get("best_fixed")
            else:raise ValueError(method)
            seconds=time.perf_counter()-begin
            original_valid=bool(x is not None and adapter.validate_solution(case.model,x)["valid"])
            actual_K=K if method=="ours" else (case.model.n_variables-len(fixed) if fixed is not None else None)
            if fixed is not None and actual_K!=K:raise ValueError("ND returned a different common-coordinate K")
            restriction_valid=bool(original_valid and (core.exact_slice_contains(case.affine,np.rint(x[case.commitment_indices]).astype(int)) if method=="ours"
                else fixed is not None and all(abs(x[int(j)]-v)<=1e-6 for j,v in fixed.items())))
            raw_objective=float(cost@x) if original_valid else None
            timely=seconds<=budget
            valid=bool(original_valid and restriction_valid and timely and actual_K==K)
            objective=raw_objective if valid else None
            ratio=objective/float(reference) if objective is not None else None
            if ratio is not None and ratio>1.+adapter.recovery_tolerance(reference)/abs(reference):
                raise ValueError("Restricted result profit exceeds original certified optimum")
            row=dict(guard=guard,case_id=case.case_id,method=method,index=i,n_train=case.n_train,K=K,actual_K=actual_K,D=case.model.n_variables,
                cost=cost,feature=bank["features"][i],budget_seconds=float(budget),
                original_feasible=original_valid,restriction_valid=restriction_valid,timely=timely,valid=valid,
                objective=objective,reference_objective=float(reference),profit_ratio=ratio,
                recovered=bool(valid and abs(objective-reference)<=adapter.recovery_tolerance(reference)),
                raw_objective=raw_objective,seconds=seconds,raw=raw,solution=x,fixed=fixed,
                experiment_phase="independent_confirmation",confirmation=True)
            row=clean(row)
            row["record_payload_sha256"]=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            verify_confirmation_record(row,case,method,i,cost,reference,guard,budget,bank["features"][i])
            write_new(path,row);rows.append(row)
        ratios=[row["profit_ratio"] for row in rows if row["profit_ratio"] is not None]
        summary=dict(case_id=case.case_id,method=method,phase="independent_confirmation",n_train=case.n_train,K=K,D=case.model.n_variables,
            count=len(rows),recovery_count=sum(row["recovered"] for row in rows),
            recovery_percent=100.*sum(row["recovered"] for row in rows)/len(rows),
            profit_ratio_mean_valid=float(np.mean(ratios)) if ratios else None,ratio_denominator=len(ratios),
            negative_returned_profit_count=sum(value<0 for value in ratios),
            independent_confirmation=True)
        complete=dict(guard=guard,summary=summary,test_record_sha256={f"test{i:03d}.json":sha(folder/f"test{i:03d}.json") for i in range(len(rows))})
        write_new(folder/"complete.json",complete);results.append(summary)
    return results


def _job(job):
    cid,role,freeze_path,results_root=job
    f,cases=validate_global_freeze(freeze_path);entry=f["cases"][cid]
    results_root=Path(results_root).resolve()
    for protected in (Path(entry["model_root"]),Path(entry["continued_root"]),Path(entry["extra_model_root"]),Path(f["bank_root"])):
        protected=protected.resolve()
        if results_root==protected or protected in results_root.parents or results_root in protected.parents:raise ValueError("Results must be separate from all frozen models and banks")
    if role not in PRIMARY:raise ValueError("This initial evaluator accepts the three primary paired roles only; frozen SVM/MLOPT use this unchanged bank in the supplemental evaluator")
    method="ours" if role=="ours" else "neural_diving"
    return evaluate_case_confirmation(cases[cid],Path(entry["model_root"]),[method],30.,freeze_path,results_root,role)


def evaluate_all(freeze_path=DEFAULT_FREEZE,results_root=DEFAULT_RUN/"results",roles=PRIMARY,workers=3):
    import concurrent.futures
    f,_=validate_global_freeze(freeze_path)
    if not roles or len(set(roles))!=len(roles) or not set(roles)<=set(PRIMARY):raise ValueError("Unknown or duplicate primary evaluation role")
    jobs=[(cid,role,str(freeze_path),str(results_root)) for cid in sorted(f["cases"]) for role in roles]
    summaries=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for job,result in zip(jobs,pool.map(_job,jobs)):
            summaries.extend([dict(method_role=job[1],summary=row) for row in result])
            print(json.dumps(clean(dict(event="stream2_case_role_complete",case_id=job[0],method_role=job[1],summaries=result)),allow_nan=False),flush=True)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","validate","evaluate"))
    parser.add_argument("--freeze",type=Path,default=DEFAULT_FREEZE)
    parser.add_argument("--bank-root",type=Path,default=DEFAULT_RUN/"banks")
    parser.add_argument("--results-root",type=Path,default=DEFAULT_RUN/"results")
    parser.add_argument("--roles",nargs="+",choices=PRIMARY,default=list(PRIMARY))
    parser.add_argument("--workers",type=int,choices=(1,2,3),default=3)
    args=parser.parse_args()
    if args.command=="freeze":
        f=freeze_all_models(args.freeze,args.bank_root)
        result=dict(status="FROZEN_ALL_FIFTEEN_ROLES",case_count=f["case_count"],model_role_count=f["model_role_count"],freeze_path=str(args.freeze.resolve()),freeze_sha256=sha(args.freeze))
    elif args.command=="validate":
        f,_=validate_global_freeze(args.freeze);result=dict(status="PASS",case_count=f["case_count"],model_role_count=f["model_role_count"],bank_read=False)
    else:result=evaluate_all(args.freeze,args.results_root,args.roles,args.workers)
    print(json.dumps(clean(result),allow_nan=False),flush=True)


if __name__=="__main__":main()

