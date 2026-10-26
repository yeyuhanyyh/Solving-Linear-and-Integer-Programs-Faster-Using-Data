"""Independent twelve-case confirmation, reusing the frozen v2 scientific path.

No RNG or fresh generation lives in this file. The global source/24-role
freeze is validated before any fresh bank read or solver/model evaluation.
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
FOLLOWUP=TOOLS.parents[2]
ADAPTER=FOLLOWUP/"objective-gap-designs-20261025/adapter/generic-binary-v2"
DRIVER_PATH=ADAPTER/"generic_binary_pilot.py"
DRIVER_SHA="d7ff552f55775f90387444e8cbc6290ee371ca94960f7280f42c762c8768e8c2"
DEFAULT_REGISTRY=TOOLS/"twelve-case-registry.json"
DEFAULT_PROTOCOL=TOOLS/"fresh-confirmation-protocol.json"
DEFAULT_RUN=FOLLOWUP/"objective-gap-followup-20261025/uniform-gross-confirmation"
DEFAULT_FREEZE=DEFAULT_RUN/"root_all_models_freeze.json"
METHODS=("ours","neural_diving")
_DRIVER=None


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def _reference_equal(value,reference,tolerance):
    value=float(value)
    return bool(np.isfinite(value) and abs(value-reference)<=tolerance)


def enable():
    global _DRIVER
    if sha(DRIVER_PATH)!=DRIVER_SHA:raise ValueError("Original frozen v2 driver changed")
    if _DRIVER is None:
        spec=importlib.util.spec_from_file_location("confirmation_frozen_generic_v2",DRIVER_PATH)
        _DRIVER=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_DRIVER;spec.loader.exec_module(_DRIVER)
    _DRIVER.enable_runtime();_DRIVER.bind_signed_cost_map()
    import torch
    if torch.get_num_threads()!=1:torch.set_num_threads(1)
    import train_uc_baselines as nn
    nn.model_guard=_DRIVER.pilot_guard
    return _DRIVER


def clean(value):return enable().clean(value)
def write_new(path,value):return enable().write_new(path,value)
def canonical(value):return json.dumps(clean(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def payload_sha(value):return hashlib.sha256(canonical(value)).hexdigest()


def verify_payload(row):
    expected=row.get("record_payload_sha256")
    if expected!=payload_sha({key:value for key,value in row.items() if key!="record_payload_sha256"}):
        raise ValueError("Canonical record checksum differs")


def _source_seal(path):
    seal=read(path);mapping=seal.get("source_sha256",seal.get("file_sha256"))
    if not mapping:raise ValueError("Source seal has no explicit hash registry")
    result={str(Path(path).resolve()):sha(path)}
    for name,expected in mapping.items():
        file=Path(name)
        if not file.is_absolute():file=Path(path).parent/file
        if sha(file)!=expected:raise ValueError("Frozen source seal mismatch: "+str(file))
        result[str(file.resolve())]=expected
    return result


def _protocol(path):
    p=read(path)
    if p.get("test_count_per_case")!=120 or p.get("expected_case_count")!=12 or p.get("method_ids")!=list(METHODS) or not p.get("rng_namespace"):
        raise ValueError("Need the fixed twelve-case, 120-cost, Ours/ND protocol")
    return p


def _case_sources(registry_path):
    registry=read(registry_path)
    if registry.get("schema")!="coupled-bundle-twelve-case-confirmation-registry-v1":raise ValueError("Unknown case registry")
    d=enable();cases={};entries={}
    for cohort in registry["cohorts"]:
        manifest_path=Path(cohort["manifest_path"]);manifest=d.validate_selection(manifest_path)
        for entry in manifest["cases"]:
            cid=entry["case_id"]
            if cid in cases:raise ValueError("Duplicate confirmation case")
            case=d.load_case(entry["directory"],manifest_path)
            G=case.train_features.shape[1]
            if G not in (16,20,32) or case.seed not in (1009,1013,1019) or case.n_train not in {16:(37,),20:(45,),32:(36,69)}[G] or int(case.affine["rank"])!=G:
                raise ValueError("Pre-registered physical/sample cohort changed")
            cases[cid]=case
            entries[cid]=dict(case_id=cid,manifest_path=str(manifest_path.resolve()),
                training_directory=str(Path(entry["directory"]).resolve()),model_root=str(Path(cohort["model_root"]).resolve()),
                G=G,D=case.model.n_variables,K=G,n_train=case.n_train,seed=case.seed,
                physical_id=f"physical_G{G}_seed{case.seed}",
                physical_model_sha256=d.array_hash(case.model.A.data,case.model.A.indices,case.model.A.indptr,case.model.row_lower,case.model.row_upper,case.model.col_lower,case.model.col_upper,case.model.integrality,np.asarray(case.metadata["cost_offset"]),np.asarray(case.metadata["cost_feature_coefficients"])),
                training_files={str((Path(entry["directory"])/name).resolve()):expected for name,expected in entry["files"].items()})
    expected={(G,n,seed) for G,n in ((16,37),(20,45),(32,36),(32,69)) for seed in (1009,1013,1019)}
    if len(cases)!=12 or {(e["G"],e["n_train"],e["seed"]) for e in entries.values()}!=expected:
        raise ValueError("All twelve physical/sample cases are required")
    for physical in {entry["physical_id"] for entry in entries.values()}:
        if len({entry["physical_model_sha256"] for entry in entries.values() if entry["physical_id"]==physical})!=1:
            raise ValueError("Paired training prefixes do not share the identical original physical model")
    return registry,cases,entries


def _semantic_roles(case,entry):
    d=enable()
    import train_uc_baselines as nn
    import adapter
    import experiment_core as core
    out=Path(entry["model_root"]);cid=case.case_id
    folder,marker=nn.validate_checkpoint(out,case,"neural_diving",ADAPTER)
    if marker.get("sanity_gate_passed") is not True or marker.get("training_label_count")!=case.n_train:
        raise ValueError("ND training readiness/label budget gate failed")
    nd_files={str((folder/"complete.json").resolve()):sha(folder/"complete.json")}
    for name,expected in marker["artifact_sha256"].items():
        path=folder/name
        if sha(path)!=expected:raise ValueError("ND artifact changed")
        nd_files[str(path.resolve())]=expected
    if not any(Path(name).name=="complete-predictor.pt" for name in nd_files):raise ValueError("Missing actual ND weights")
    receipt_path=out/"training-case-receipts"/(cid+".json");receipt=read(receipt_path)
    if receipt.get("training_guard")!=d.pilot_guard(case,"case"):raise ValueError("Ours training receipt guard differs")
    for key in ("anchor","basis","normals_exact","normals","rank","representation"):
        if clean(receipt["exact_affine"][key])!=clean(case.affine[key]):raise ValueError("Stored learned affine space differs")
    G=entry["G"]
    for C in np.vstack((np.zeros((1,G),dtype=int),np.eye(G,dtype=int))):
        witness=np.column_stack((1-C,1-C,C)).reshape(-1)
        if not core.exact_slice_contains(case.affine,witness):raise ValueError("Learned affine space misses global empty/unit geometry witness")
    certfolder=out/"training-certificates"/cid;complete=read(certfolder/"complete.json")
    oracle_guard=d.pilot_guard(case,"training_oracle")
    if complete.get("guard")!=oracle_guard or complete.get("certified_label_count")!=case.n_train or complete.get("all_labels_globally_certified") is not True:
        raise ValueError("Ours label certification marker differs")
    ours_files={str(receipt_path.resolve()):sha(receipt_path),str((certfolder/"complete.json").resolve()):sha(certfolder/"complete.json")}
    for i,reference in enumerate(case.train_objectives):
        path=certfolder/f"train{i:03d}.json";row=read(path);oracle=row.get("oracle",{})
        tol=adapter.recovery_tolerance(float(reference))
        if row.get("guard")!=oracle_guard or row.get("index")!=i or not _reference_equal(row["source_objective"],reference,tol):
            raise ValueError("Training certificate identity/source differs")
        if not (oracle.get("valid") is True and oracle.get("certified") is True) or not _reference_equal(oracle["objective"],reference,tol) or not _reference_equal(oracle["dual_bound"],reference,tol):
            raise ValueError("Training label primal/dual optimality certificate differs")
        ours_files[str(path.resolve())]=sha(path)
    return dict(ours=dict(guard=d.pilot_guard(case,"ours"),files=ours_files),
        neural_diving=dict(guard=d.pilot_guard(case,"neural_diving"),files=nd_files))


def freeze_all_models(registry_path=DEFAULT_REGISTRY,protocol_path=DEFAULT_PROTOCOL,freeze_path=DEFAULT_FREEZE,
                      bank_root=DEFAULT_RUN/"banks",extra_sources=()):
    freeze_path=Path(freeze_path)
    if freeze_path.exists():return validate_global_freeze(freeze_path,semantic=True)[0]
    if Path(bank_root).exists() and any(p.is_file() for p in Path(bank_root).rglob("*")):
        raise ValueError("Fresh costs already exist before the all-model freeze")
    protocol=_protocol(protocol_path);registry,cases,entries=_case_sources(registry_path)
    from verify_evaluator_equivalence import verify
    equivalence=verify(DRIVER_PATH,Path(__file__))
    if equivalence.get("passed") is not True:raise ValueError("Original prediction/solve/metrics AST differs")
    sources={str(Path(path).resolve()):sha(path) for path in (Path(__file__),Path(registry_path),Path(protocol_path),TOOLS/"verify_evaluator_equivalence.py")}
    for path in registry["source_seals"]:sources.update(_source_seal(path))
    for path in extra_sources:sources[str(Path(path).resolve())]=sha(path)
    if not any(Path(path).name=="fresh_bank_cli.py" for path in sources):raise ValueError("The actual fresh generation CLI must be frozen too")
    for cid,case in cases.items():entries[cid]["methods"]=_semantic_roles(case,entries[cid])
    if sum(len(entry["methods"]) for entry in entries.values())!=24:raise ValueError("Need all 24 roles / twelve ND models")
    result=dict(schema="coupled-bundle-all-twelve-models-freeze-v1",frozen=True,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),registry_path=str(Path(registry_path).resolve()),
        protocol_path=str(Path(protocol_path).resolve()),protocol_sha256=sha(protocol_path),bank_root=str(Path(bank_root).resolve()),
        source_sha256=sources,cases=entries,case_count=12,model_role_count=24,ND_model_count=12,physical_bank_count=9,
        scientific_evaluation_equivalence=equivalence,
        confirmation_not_generated_before_freeze=True)
    write_new(freeze_path,result);validate_global_freeze(freeze_path,semantic=True)
    return result


def validate_global_freeze(freeze_path=DEFAULT_FREEZE,semantic=True):
    freeze=read(freeze_path)
    if freeze.get("schema")!="coupled-bundle-all-twelve-models-freeze-v1" or freeze.get("frozen") is not True or freeze.get("case_count")!=12 or freeze.get("model_role_count")!=24:
        raise ValueError("Need the completed all-twelve-model global freeze")
    _protocol(freeze["protocol_path"])
    if freeze["protocol_sha256"]!=sha(freeze["protocol_path"]):raise ValueError("Fresh protocol changed")
    for name,expected in freeze["source_sha256"].items():
        if sha(name)!=expected:raise ValueError("Globally frozen source changed: "+name)
    registry,cases,entries=_case_sources(freeze["registry_path"])
    for cid,entry in entries.items():
        for name,expected in entry["training_files"].items():
            if sha(name)!=expected:raise ValueError("Globally frozen training data changed")
        stored=freeze["cases"].get(cid)
        if stored is None:raise ValueError("Global case population changed")
        for method in METHODS:
            for name,expected in stored["methods"][method]["files"].items():
                if sha(name)!=expected:raise ValueError("Globally frozen model/label artifact changed")
        entry["methods"]=_semantic_roles(cases[cid],entry) if semantic else stored["methods"]
        if entry!=stored:raise ValueError("Live all-role semantic registry differs from global freeze")
    if set(entries)!=set(freeze["cases"]):raise ValueError("Extra global cases")
    return freeze,cases


def load_confirmation_bank(case,freeze_path=DEFAULT_FREEZE):
    freeze,_=validate_global_freeze(freeze_path,semantic=True)
    entry=freeze["cases"][case.case_id];folder=Path(freeze["bank_root"])/entry["physical_id"]
    meta=read(folder/"metadata.json");npz=folder/"confirmation.npz";refpath=folder/"reference-certificates.json"
    physical_entries={cid:value for cid,value in freeze["cases"].items() if value["physical_id"]==entry["physical_id"]}
    bindings={cid:dict(K=value["K"],n_train=value["n_train"],training_files=value["training_files"],model_role_files={method:value["methods"][method]["files"] for method in METHODS}) for cid,value in physical_entries.items()}
    required=dict(schema="shared-physical-confirmation-bank-v2",phase="independent_confirmation",confirmation=True,
        physical_id=entry["physical_id"],case_ids=sorted(physical_entries),case_bindings=bindings,
        G=entry["G"],D=entry["D"],seed=entry["seed"],count=120,
        model_freeze_sha256=sha(freeze_path),protocol_sha256=freeze["protocol_sha256"],npz_sha256=sha(npz),
        reference_certificates_sha256=sha(refpath),rng_namespace=_protocol(freeze["protocol_path"])["rng_namespace"])
    for key,value in required.items():
        if meta.get(key)!=value:raise ValueError("Fresh bank metadata guard differs: "+key)
    with np.load(npz,allow_pickle=False) as stream:bank={key:stream[key].copy() for key in stream.files}
    if set(bank)!={"features","costs","solutions","objectives"}:raise ValueError("Fresh bank array schema differs")
    for key,shape in dict(features=(120,entry["G"]),costs=(120,entry["D"]),solutions=(120,entry["D"]),objectives=(120,)).items():
        if bank[key].shape!=shape or not np.isfinite(bank[key]).all():raise ValueError("Fresh array shape/finite check failed")
    d=enable()
    import adapter
    if not np.all((bank["features"]>=.1)&(bank["features"]<=2.5)) or not np.all(bank["costs"]<0) or not np.all(bank["objectives"]<0):
        raise ValueError("Fixed original positive-gross-profit law differs")
    if not np.allclose(d.signed_costs_from_features(case,bank["features"]),bank["costs"],rtol=1e-14,atol=1e-12):raise ValueError("Fresh signed affine costs differ")
    certificates=read(refpath);refguard=dict(model_freeze_sha256=sha(freeze_path),protocol_sha256=freeze["protocol_sha256"])
    if certificates.get("schema")!="coupled-bundle-confirmation-reference-certificates-v1" or certificates.get("physical_id")!=entry["physical_id"] or certificates.get("case_ids")!=sorted(physical_entries) or certificates.get("count")!=120 or certificates.get("guard")!=refguard or certificates.get("all_original_IP_certified") is not True or certificates.get("all_independent_references_agree") is not True:
        raise ValueError("Incomplete fresh reference certification")
    expected_files={f"reference-records/ref{i:03d}.json" for i in range(120)}
    if set(certificates["record_file_sha256"])!=expected_files:raise ValueError("Fresh reference population differs")
    for i,(feature,cost,x,reference) in enumerate(zip(bank["features"],bank["costs"],bank["solutions"],bank["objectives"])):
        path=folder/f"reference-records/ref{i:03d}.json"
        if sha(path)!=certificates["record_file_sha256"][f"reference-records/ref{i:03d}.json"]:raise ValueError("Fresh reference record changed")
        row=read(path);verify_payload(row);oracle=row.get("highs",{});tol=adapter.recovery_tolerance(float(reference))
        if row.get("physical_id")!=entry["physical_id"] or row.get("index")!=i or row.get("guard")!=refguard:raise ValueError("Fresh reference identity differs")
        if not np.array_equal(row["feature"],feature) or not np.array_equal(row["cost"],cost) or not np.array_equal(row["solution"],x):raise ValueError("Fresh reference arrays differ")
        if not adapter.validate_solution(case.model,x)["valid"] or abs(float(cost@x)-reference)>tol:raise ValueError("Invalid original fresh reference")
        if not (oracle.get("valid") is True and oracle.get("certified") is True):raise ValueError("Uncertified original fresh optimum")
        for value in (row["objective"],row["independent_objective"],oracle["objective"],oracle["dual_bound"]):
            if not _reference_equal(value,reference,tol):raise ValueError("Fresh primal/dual/independent reference differs")
    return bank,meta


def verify_confirmation_record(row,case,method,index,cost,reference,guard,budget,feature):
    verify_payload(row)
    if row.get("confirmation") is not True or row.get("experiment_phase")!="independent_confirmation":
        raise ValueError("A development record cannot be used as confirmation")
    shadow={key:value for key,value in row.items() if key!="record_payload_sha256"}
    shadow.update(confirmation=False,experiment_phase="development")
    shadow["record_payload_sha256"]=payload_sha(shadow)
    enable().verify_development_record(shadow,case,method,index,cost,reference,guard,budget,feature)


def confirmation_guard(case,method,meta,freeze_path,budget):
    freeze,_=validate_global_freeze(freeze_path,semantic=False)
    return dict(training=enable().pilot_guard(case,method),model_freeze_sha256=sha(freeze_path),
        protocol_sha256=freeze["protocol_sha256"],confirmation_bank_sha256=meta["npz_sha256"],
        confirmation_metadata_sha256=sha(Path(freeze["bank_root"])/freeze["cases"][case.case_id]["physical_id"]/"metadata.json"),
        reference_certificates_sha256=meta["reference_certificates_sha256"],
        wrapper_sha256=sha(__file__),model_role_files_sha256=freeze["cases"][case.case_id]["methods"][method]["files"],
        budget=float(budget),phase="independent_confirmation")


ROOT=ADAPTER


def evaluate_case_confirmation(case,out,methods,budget,freeze_path,results_root):
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
        model=nn.load_nd_complete(case,folder/"complete-predictor.pt")
    K=int(case.affine["rank"]);results=[]
    for method in methods:
        folder=Path(results_root)/case.case_id/method;folder.mkdir(parents=True,exist_ok=True)
        guard=confirmation_guard(case,method,metadata,freeze_path,budget)
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


def _evaluate_job(job):
    case_id,freeze_path,results_root=job
    freeze,cases=validate_global_freeze(freeze_path,semantic=True)
    entry=freeze["cases"][case_id]
    results_root=Path(results_root).resolve();model_root=Path(entry["model_root"]).resolve();bank_root=Path(freeze["bank_root"]).resolve()
    if results_root==model_root or model_root in results_root.parents or results_root==bank_root or bank_root in results_root.parents:
        raise ValueError("Fresh results must be separate from frozen checkpoints and banks")
    return evaluate_case_confirmation(cases[case_id],model_root,METHODS,30.,freeze_path,results_root)


def evaluate_all(freeze_path=DEFAULT_FREEZE,results_root=DEFAULT_RUN/"results",workers=4,case_ids=None):
    import concurrent.futures
    freeze,_=validate_global_freeze(freeze_path,semantic=True)
    selected=sorted(freeze["cases"]) if case_ids is None else list(case_ids)
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(freeze["cases"]):
        raise ValueError("Unknown or duplicate fresh evaluation case")
    jobs=[(cid,str(freeze_path),str(results_root)) for cid in selected];summaries=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_evaluate_job,jobs):
            summaries.extend(result)
            print(json.dumps(clean(dict(event="fresh_case_complete",summaries=result)),allow_nan=False),flush=True)
    return summaries


def self_test():
    from types import SimpleNamespace
    from scipy import sparse
    d=enable()
    import adapter
    import experiment_core as core
    from verify_evaluator_equivalence import verify
    equivalence=verify(DRIVER_PATH,Path(__file__))
    if not equivalence["passed"]:raise AssertionError("Scientific AST changed")
    model=adapter.LinearModel(A=sparse.csr_matrix([[1.,1.]]),row_lower=np.array([-np.inf]),row_upper=np.array([1.]),
        col_lower=np.zeros(2),col_upper=np.ones(2),integrality=np.ones(2,dtype=int))
    case=SimpleNamespace(case_id="zero-fresh-toy",n_train=2,model=model,affine=core.fit_affine(np.array([[1,0],[0,1]])),
        metadata=dict(cost_offset=[0.,0.],cost_feature_coefficients=[[-1.,0.],[0.,-1.]]))
    cost=np.array([-2.,-1.]);feature=np.array([2.,1.]);guard=dict(toy=True)
    row=dict(guard=guard,case_id=case.case_id,method="ours",index=0,n_train=2,K=1,actual_K=1,D=2,
        cost=cost,feature=feature,budget_seconds=30.,original_feasible=True,restriction_valid=True,timely=True,valid=True,
        objective=-2.,reference_objective=-2.,profit_ratio=1.,recovered=True,raw_objective=-2.,seconds=.1,
        solution=[1.,0.],fixed=None,experiment_phase="independent_confirmation",confirmation=True)
    row=clean(row);row["record_payload_sha256"]=payload_sha(row)
    verify_confirmation_record(row,case,"ours",0,cost,-2.,guard,30.,feature)
    rejected=[]
    for key,value in (("confirmation",False),("profit_ratio",.8),("actual_K",0),("seconds",31.)):
        changed={k:v for k,v in row.items() if k!="record_payload_sha256"};changed[key]=value
        changed["record_payload_sha256"]=payload_sha(changed)
        try:verify_confirmation_record(changed,case,"ours",0,cost,-2.,guard,30.,feature)
        except ValueError:rejected.append(key)
        else:raise AssertionError("Confirmation tamper was accepted: "+key)
    return dict(status="PASS",scientific_evaluation_equivalence=equivalence,legacy_shadow_never_published=True,
        rejected_recomputed_checksum_tampers=rejected,models_loaded=False,original_solver_run=False,
        training_or_development_banks_opened=False,confirmation_generated=False,confirmation_read=False,
        wrapper_sha256=sha(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","validate","evaluate","self-test"))
    parser.add_argument("--registry",type=Path,default=DEFAULT_REGISTRY)
    parser.add_argument("--protocol",type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument("--freeze",type=Path,default=DEFAULT_FREEZE)
    parser.add_argument("--bank-root",type=Path,default=DEFAULT_RUN/"banks")
    parser.add_argument("--results-root",type=Path,default=DEFAULT_RUN/"results")
    parser.add_argument("--include-source",type=Path,nargs="*",default=[])
    parser.add_argument("--case-id",nargs="+")
    parser.add_argument("--workers",type=int,choices=(1,2,3,4),default=4)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.command=="self-test":result=self_test()
    elif args.command=="freeze":
        result=freeze_all_models(args.registry,args.protocol,args.freeze,args.bank_root,args.include_source)
        result=dict(status="FROZEN",case_count=result["case_count"],model_role_count=result["model_role_count"],freeze_sha256=sha(args.freeze))
    elif args.command=="validate":
        freeze,_=validate_global_freeze(args.freeze,semantic=True)
        result=dict(status="PASS",case_count=freeze["case_count"],model_role_count=freeze["model_role_count"],confirmation_read=False)
    else:result=evaluate_all(args.freeze,args.results_root,args.workers,args.case_id)
    if args.output:write_new(args.output,result)
    print(json.dumps(clean(result),allow_nan=False),flush=True)


if __name__=="__main__":main()

