"""Minimal training-only SVM/MLOPT adapter for the existing coupled-bundle cohort.

No development or confirmation bank is read. The original component and
eight-grid MLOPT sources are copied byte-for-byte into this new sibling folder.
The caller schedules independent case/method processes under its CPU-slot cap.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import argparse
import importlib.util
from pathlib import Path
import sys
import time
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parent
FOLLOWUP=ROOT.parents[1]
GENERIC_ROOT=FOLLOWUP/"objective-gap-designs-20261025/adapter/generic-binary-v2"
GENERIC_PATH=GENERIC_ROOT/"generic_binary_pilot.py"
sys.path.insert(0,str(GENERIC_ROOT))
import generic_binary_pilot as generic
import svm_mlopt_components as components


def enable():
    generic.enable_runtime()
    generic.bind_signed_cost_map()


def selected_entry(manifest_path,case_dir):
    manifest=generic.validate_selection(manifest_path)
    selected=[e for e in manifest["cases"] if Path(e["directory"]).resolve()==Path(case_dir).resolve()]
    if len(selected)!=1:raise ValueError("Case is outside its frozen training manifest")
    entry=selected[0]
    if (entry.get("case_schema")!="coupled-bundle-uniform-gross-v1"
            or entry.get("local_formulation")!="capacity"):
        raise ValueError("This adapter is for the original coupled capacity formulation")
    G=int(entry["D"])//3
    if (G,int(entry["n_train"])) not in ((32,36),(16,37)):
        raise ValueError("Only root-fixed G32/n36 or G16/n37 cohorts are authorized here")
    if int(entry["K"])!=G or int(entry["physical_seed"]) not in (1009,1013,1019):
        raise ValueError("Original complete-K cohort or physical seeds changed")
    if "training_reference_certificate" in entry:
        certificate=Path(entry["training_reference_certificate"])
        if generic.sha(certificate)!=entry["training_reference_certificate_sha256"]:
            raise ValueError("Frozen original training optimizer certificate changed")
    return entry


def guard(case,method,manifest_path):
    return dict(schema="coupled-extra-baselines-v1-20261025",method=method,
        generic_training_guard=generic.pilot_guard(case,"shared_case"),
        component_binding=components.source_binding(case),
        driver_sha256=generic.sha(__file__),manifest_sha256=generic.sha(manifest_path),
        development_opened=False,confirmation_opened=False)


def validate_checkpoint(case,method,out,manifest_path):
    folder=Path(out)/"models"/case.case_id/method
    complete=generic.read(folder/"complete.json")
    if complete["guard"]!=guard(case,method,manifest_path):
        raise ValueError("Checkpoint source/cohort guard changed")
    for name,expected in complete["artifact_sha256"].items():
        if generic.sha(folder/name)!=expected:raise ValueError("Checkpoint bytes changed")
    return folder,complete


def load_model(case,method,out,manifest_path):
    folder,_=validate_checkpoint(case,method,out,manifest_path)
    return components.load_component(folder/"model.pkl",case,method)


def fit_one(case,method,out,manifest_path,device):
    final=Path(out)/"models"/case.case_id/method
    if final.exists():
        validate_checkpoint(case,method,out,manifest_path)
        print(generic.json.dumps(dict(event="already_complete",case_id=case.case_id,method=method)),flush=True)
        return
    partial=final.with_name(method+".partial-"+uuid.uuid4().hex)
    partial.mkdir(parents=True)
    generic.write_new(partial/"start.json",dict(guard=guard(case,method,manifest_path),utc=generic.utc()))
    started,cpu_started=time.perf_counter(),time.process_time()
    if method=="svm":model,metadata=components.fit_svm(case)
    elif method=="mlopt":model,metadata=components.fit_mlopt(case,device=device)
    else:raise ValueError(method)
    fit_wall,fit_cpu=time.perf_counter()-started,time.process_time()-cpu_started
    # One *training* query checks the signed map, original-coordinate budget,
    # and checkpoint round-trip. No held-out features or outcomes are opened.
    propose=components.propose_svm if method=="svm" else components.propose_mlopt
    fixed,diagnostics=propose(model,case,case.train_costs[0],case.train_features[0])
    if fixed is None or len(fixed)!=case.model.n_variables-int(case.affine["rank"]):
        raise ValueError("Training-only checkpoint interface smoke did not reach target K")
    components.save_component(model,partial/"model.pkl")
    loaded=components.load_component(partial/"model.pkl",case,method)
    reloaded_fixed,_=propose(loaded,case,case.train_costs[0],case.train_features[0])
    if fixed!=reloaded_fixed:raise ValueError("Checkpoint round-trip changes coordinate proposal")
    generic.write_new(partial/"metadata.json",dict(method=method,metadata=metadata,
        wrapper_fit_wall_seconds=fit_wall,wrapper_fit_cpu_seconds=fit_cpu,
        label_generation_included=False,training_smoke=diagnostics,
        development_information_used=False))
    files={str(p.relative_to(partial)).replace("\\","/"):generic.sha(p)
           for p in partial.rglob("*") if p.is_file()}
    generic.write_new(partial/"complete.json",dict(guard=guard(case,method,manifest_path),
        artifact_sha256=files,fit_wall_seconds=fit_wall,fit_cpu_seconds=fit_cpu,
        sanity_gate_passed=True,utc=generic.utc()))
    # Existing model directories are immutable. Windows rename cannot replace
    # one; this single-job-per-case/method driver never schedules duplicates.
    if final.exists():raise FileExistsError(final)
    os.rename(partial,final)
    print(generic.json.dumps(dict(event="fit_complete",case_id=case.case_id,method=method,
        fit_wall_seconds=fit_wall,fit_cpu_seconds=fit_cpu,confirmation=False)),flush=True)


def smoke(case,manifest_path,out):
    """One training cost: real SVM fit and an unfitted MLOPT interface probe."""
    started=time.perf_counter()
    svm,svm_metadata=components.fit_svm(case)
    svm_fixed,svm_info=components.propose_svm(svm,case,case.train_costs[0],case.train_features[0])
    runtime=components._mlopt_runtime()
    # No neural fit is started before root confirms the worker slots. The
    # original decoder/pruning is exercised with deterministic zero logits.
    config=runtime.Config(device="cpu",n_best=10)
    bank=runtime.StrategyBank(case,config.active_tolerance,config.feasibility_tolerance,config.rank_tolerance)
    selected,_,pruning=runtime.prune_strategies(bank,config)
    weights=[(np.zeros((len(selected),case.train_features.shape[1]),dtype=np.float32),
              np.zeros(len(selected),dtype=np.float32))]
    mlopt=runtime.MLOPTModel(bank,selected,weights,np.zeros(case.train_features.shape[1]),
                            np.ones(case.train_features.shape[1]),config)
    mlopt.component_method="mlopt";mlopt.component_binding=components.source_binding(case)
    mlopt_fixed,mlopt_info=components.propose_mlopt(mlopt,case,case.train_costs[0],case.train_features[0])
    D=case.model.n_variables;K=int(case.affine["rank"])
    if mlopt_fixed is None or len(mlopt_fixed)!=D-K or len(svm_fixed)!=D-K:
        raise ValueError("Training-only interface budget check failed")
    receipt=dict(schema="coupled-extra-baselines-one-training-query-smoke-v1",
        guard=guard(case,"smoke",manifest_path),case_id=case.case_id,n_train=case.n_train,D=D,K=K,
        training_query_index=0,development_accessed=False,confirmation_accessed=False,
        svm_fit_seconds=svm_metadata["fit_seconds"],svm_fix_count=len(svm_fixed),
        mlopt_fix_count=len(mlopt_fixed),mlopt_native_candidates=mlopt_info["native_candidate_count"],
        mlopt_strategy_count=len(selected),mlopt_pruning=pruning,
        mlopt_neural_training_performed=False,
        mlopt_probe_weights="deterministic zero logits, solely interface smoke; never an experiment model",
        original_MLP_grid_unchanged=True,seconds=time.perf_counter()-started)
    generic.write_new(Path(out)/"training-interface-smoke.json",receipt)
    print(generic.json.dumps(generic.clean(receipt)),flush=True)
    return receipt


def main():
    sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--case-dir",type=Path,nargs="+",required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"results")
    parser.add_argument("--methods",choices=("svm","mlopt"),nargs="+",default=["svm","mlopt"])
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--fit",action="store_true")
    args=parser.parse_args()
    if not any((args.validate_only,args.smoke,args.fit)):parser.error("Select an explicit operation")
    if args.smoke and (args.fit or len(args.case_dir)!=1):
        parser.error("Smoke is one separate training-only case; do not combine with fitting")
    enable()
    for folder in args.case_dir:
        selected_entry(args.manifest,folder)
        case=generic.load_case(folder,args.manifest)
        print(generic.json.dumps(dict(event="case_verified",case_id=case.case_id,
            n_train=case.n_train,D=case.model.n_variables,K=case.affine["rank"],confirmation=False)),flush=True)
        if args.smoke:smoke(case,args.manifest,args.out)
        if args.fit:
            for method in args.methods:fit_one(case,method,args.out,args.manifest,args.device)


if __name__=="__main__":main()
