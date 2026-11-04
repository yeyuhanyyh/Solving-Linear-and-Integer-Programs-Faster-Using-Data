"""Independent fit/evaluate CLI. Label-bank generation belongs to uc/.

Example: python framework/run.py fit --bank BANK --out RUN --n 256 --method ours
Evaluation opens test only after validating the frozen training checkpoint.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"framework"))
import argparse
import importlib
import json
import pickle
import platform
import socket
import time
from datetime import datetime,timezone
import numpy as np
import scipy.sparse as sp
import adapter
import ours
from cases import load_bank_split,file_hash,model_hash,assert_same_physics
from dimension_diagnostics import assert_matched_fixings

METHODS=("ours","full","svm","svm_native","mlopt","mlopt_native","neural_diving","preference_learning")


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".writing")
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def runtime():
    import scipy,highspy
    from importlib.metadata import version,PackageNotFoundError
    packages={}
    for name in ("numpy","scipy","highspy","sympy","python-flint","scikit-learn","torch","threadpoolctl","pyscipopt"):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return dict(hostname=socket.gethostname(),platform=platform.platform(),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,highs=highspy.Highs().version(),
        packages=packages,
        thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})


def path_for(args,method=None):return Path(args.out)/f"n{args.n}"/(method or args.method)


def core_source_hashes():
    files=list((ROOT/"framework").glob("*.py"))
    files.extend((ROOT/"uc").glob("*.py"))
    files.extend((ROOT/"fallback_uc").glob("*.py"))
    files.extend(ROOT.glob("protocol*.json"))
    for folder in (ROOT/"baselines").iterdir():
        if folder.is_dir():files.extend(folder.rglob("*.py"))
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files) if "__pycache__" not in str(p)}


def guard(case,validation,args,K):
    return dict(n_train=case.n_train,n_validation=validation.n_train,target_K=K,method=args.method,
        physical_family=case.family,commitment_indices=case.commitment_indices.tolist(),
        train_bank_sources=case.source_hashes,validation_bank_sources=validation.source_hashes,
        adapter_backend=getattr(args,"backend","highs"),
        train_sha256=case.dataset_fingerprint,validation_sha256=validation.dataset_fingerprint,
        physical_model_sha256=model_hash(case.model),source_sha256=core_source_hashes())


def load_neural_api():
    sys.path.insert(0,str(ROOT/"baselines"/"neural"))
    return importlib.import_module("neural_api")


def load_training(args):
    case=load_bank_split(args.bank,"train",n=args.n)
    validation=load_bank_split(args.bank,"validation",n=args.validation_n)
    assert_same_physics(case,validation)
    return case,validation


def fit(args):
    os.environ["FORMAL_SOLVER_BACKEND"]=getattr(args,"backend","highs")
    case,validation=load_training(args)
    folder=path_for(args);folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():
        raise FileExistsError("Checkpoint already complete; do not silently overwrite")
    config={} if args.config is None else json.loads(Path(args.config).read_text())
    start=time.perf_counter();cpu=time.process_time()
    if args.method=="ours":
        model=ours.fit(case);K=model.K;model.save(folder/"model.npz");meta=model.manifest
    elif args.method=="full":raise ValueError("Full baseline needs no fit; evaluate it after Ours")
    else:
        compression=ours.AffineCompression.load(path_for(args,"ours")/"model.npz")
        if compression.manifest["training_array_sha256"]!=case.dataset_fingerprint:raise ValueError("Ours prefix mismatch")
        K=compression.K;case.affine={"rank":K};validation.affine={"rank":K}
        if args.method in ("svm","svm_native"):
            module=importlib.import_module("baselines.svm.wrapper")
            config["mode"]="native" if args.method=="svm_native" else "matched"
            model=module.fit(case,config=config,validation_case=validation);meta=model.manifest
            with (folder/"model.pkl").open("wb") as f:pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
        elif args.method in ("mlopt","mlopt_native"):
            module=importlib.import_module("baselines.mlopt.api")
            config.setdefault("device",args.device)
            model=module.fit(case,validation,config=config);meta=model.manifest
            model.save(folder/"model.pkl")
        else:
            module=load_neural_api()
            model=module.fit(case,validation,args.method,folder,device=args.device,config=config or None)
            meta=model.metadata
    if args.device=="cuda":
        import torch
        torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    write_json(folder/"fit.json",dict(metadata=meta,method=args.method,K=K,fit_wall_seconds=elapsed,
        fit_cpu_seconds=time.process_time()-cpu,training_label_seconds_counted_separately=True,
        runtime=runtime(),completed_utc=datetime.now(timezone.utc).isoformat()))
    artifacts={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".writing")}
    write_json(folder/"complete.json",dict(guard=guard(case,validation,args,K),artifacts=artifacts))
    print(json.dumps(dict(event="fit_complete",method=args.method,N=args.n,K=K,seconds=elapsed)),flush=True)


def verify_checkpoint(args,case,validation,method=None):
    name=method or args.method;folder=path_for(args,name)
    marker=json.loads((folder/"complete.json").read_text())
    old=args.method;args.method=name
    expected=guard(case,validation,args,marker["guard"]["target_K"])
    args.method=old
    if marker["guard"]!=expected:raise ValueError("Checkpoint source/data/config identity changed")
    for relative,sha in marker["artifacts"].items():
        if file_hash(folder/relative)!=sha:raise ValueError(f"Checkpoint artifact changed: {relative}")
    return folder,marker


def physical_validator(bank,case):
    root=Path(bank);manifest=json.loads((root/"manifest.json").read_text())
    if case.family=="ucjl_static":
        from fallback_uc.import_export import load_export,objective,validate_solution
        from cases import as_linear_model
        name="export.json"
        if file_hash(root/name)!=manifest["file_sha256"][name]:raise ValueError("Unsealed UC.jl export")
        physical,aux=load_export(root/name)
        if model_hash(as_linear_model(physical))!=model_hash(case.model):raise ValueError("Physical UC.jl export mismatch")
        if not np.array_equal(physical["commitment_indices"],case.commitment_indices):
            raise ValueError("Physical UC.jl commitment layout mismatch")
        def check(x,features,cost):
            expected=objective(aux,features)
            if not np.allclose(expected,cost,rtol=1e-13,atol=1e-12):raise ValueError("UC.jl feature-to-objective mismatch")
            result=validate_solution(physical,aux,x,expected,tolerance=adapter.FEASIBILITY_TOLERANCE)
            result["objective"]=result.get("objective_recomputed")
            return result
        return check
    if case.family!="uc":raise ValueError(f"Unsupported physical case family: {case.family}")
    from uc.model import load_case,build_model
    from uc.validate_uc import validate_solution
    name="physical_case.json"
    if file_hash(root/name)!=manifest["file_sha256"][name]:raise ValueError("Unsealed physical case")
    physical=build_model(load_case(root/name))
    from cases import as_linear_model
    if model_hash(as_linear_model(physical))!=model_hash(case.model):raise ValueError("Physical reconstruction mismatch")
    generators=list(physical.data["thermal_generators"])
    def check(x,features,cost):
        multipliers={g:float(v) for g,v in zip(generators,features)}
        if len(features)!=len(generators):raise ValueError("Physical feature layout mismatch")
        if not np.allclose(physical.objective(multipliers),cost,rtol=1e-13,atol=1e-12):raise ValueError("PGLib feature-to-objective mismatch")
        return validate_solution(physical.data,physical.unpack(x),multipliers,tol=1e-5)
    return check


def evaluate(args):
    if not args.epoch_id:raise ValueError("Evaluation requires a common hardware epoch id")
    case,validation=load_training(args)
    _,omarker=verify_checkpoint(args,case,validation,"ours")
    K=omarker["guard"]["target_K"];case.affine={"rank":K}
    compression=ours.AffineCompression.load(path_for(args,"ours")/"model.npz")
    affine_descriptor=dict(kind="integer_affine_extended",basis_artifact=str(path_for(args,"ours")/"model.npz"),
        basis_artifact_sha256=file_hash(path_for(args,"ours")/"model.npz"),
        affine_data_sha256=compression.manifest["affine_data_sha256"],
        common_indices=compression.common.tolist(),rank_K=compression.K,
        original_variables=case.model.n_variables,added_continuous_variables=compression.K,
        added_equalities=len(compression.common),basis_nnz=int(compression.basis.nnz))
    model=None
    if args.method not in ("full","ours"):
        folder,marker=verify_checkpoint(args,case,validation)
        if args.method in ("neural_diving","preference_learning"):
            model=load_neural_api().load(case,args.method,folder)
        else:
            with (folder/"model.pkl").open("rb") as f:model=pickle.load(f)
    # This is deliberately the first test-bank access in the evaluation path.
    test=load_bank_split(args.bank,"test",n=args.test_n,target_K=K)
    assert_same_physics(case,test);assert_same_physics(validation,test)
    validate_physics=physical_validator(args.bank,case)
    os.environ["FORMAL_SOLVER_BACKEND"]=args.backend
    folder=path_for(args);folder.mkdir(parents=True,exist_ok=True)
    query_index=getattr(args,"query_index",None)
    if query_index is not None and not 0<=query_index<test.n_train:raise ValueError("Query index outside test prefix")
    query_indices=range(test.n_train) if query_index is None else [query_index]
    shard="" if query_index is None else f"-query{query_index:05d}"
    ledger=folder/f"queries-{args.epoch_id}{shard}.jsonl"
    if ledger.exists():raise FileExistsError("Query ledger exists; select a new epoch or audit/resume explicitly")
    host=runtime();method_native=args.method.endswith("_native")
    for i in query_indices:
        features,cost,reference=test.train_features[i],test.train_costs[i],test.train_objectives[i]
        started=time.perf_counter();prediction_seconds=0.;diagnostic={};restriction=None
        fixed_for_audit=None;rows_for_audit=None
        result=None
        if args.method=="full":result=adapter.solve(case.model,cost,seed=args.solver_seed,time_limit=args.query_budget)
        elif args.method=="ours":
            result=ours.query(compression,case,cost,seed=args.solver_seed,time_limit=args.query_budget)
            restriction=dict(affine_descriptor)
        elif args.method=="mlopt_native":
            raw=model.predict_native(cost,features);prediction_seconds=raw["seconds"]
            result=dict(x=raw["x"],objective=raw.get("objective"),valid=raw["valid"],status=raw["status"],
                certified=False,dual_bound=None,solve_seconds=0.,build_seconds=0.)
            diagnostic=dict(raw["diagnostics"],winning_strategy_id=raw.get("strategy_id"))
        else:
            if args.method in ("neural_diving","preference_learning"):
                proposal=model.predict_fixed(cost,features,target_K=K,time_limit=args.query_budget)
            else:proposal=model.predict_restriction(cost,features=features,target_K=K)
            prediction_seconds=proposal["prediction_seconds"]
            diagnostic=proposal.get("diagnostics",proposal.get("diagnostic",{}))
            fixed=proposal.get("fixed");additional=proposal.get("additional_rows")
            fixed_for_audit=fixed
            if additional is None and proposal.get("extra_equalities") is not None:
                a,rhs=proposal["extra_equalities"];additional=(a,rhs,rhs)
            rows_for_audit=additional
            remaining=args.query_budget-(time.perf_counter()-started)
            if fixed is not None and remaining>0:
                if not method_native:restriction=assert_matched_fixings(case,fixed,K)
                result=adapter.solve(case.model,cost,fixed=fixed,additional_rows=additional,
                                     seed=args.solver_seed,time_limit=remaining)
        if result is None:
            result=dict(x=None,objective=None,valid=False,status="no_restriction_or_budget_exhausted",
                        certified=False,dual_bound=None,solve_seconds=0.,build_seconds=0.)
        raw=dict(result.__dict__) if hasattr(result,"__dict__") else result
        x=raw.pop("x",None);physical=None
        valid=bool(raw.get("valid"))
        if x is not None:
            generic=adapter.validate_solution(case.model,x)
            physical=validate_physics(x,features,cost)
            physical_ok=bool(physical.get("valid",physical.get("feasible",False)))
            valid=bool(valid and generic["valid"] and physical_ok)
            objective=float(cost@x)
            physical_objective=physical.get("objective")
            valid=bool(valid and physical_objective is not None and
                abs(objective-float(physical_objective))<=adapter.recovery_tolerance(objective))
        else:objective=None
        total=time.perf_counter()-started
        if fixed_for_audit is not None:
            if restriction is None:restriction={}
            restriction.update(kind="coordinate_or_native_equalities",fixed_global_indices=list(map(int,fixed_for_audit)),
                               fixed_values=[float(v) for v in fixed_for_audit.values()])
        if rows_for_audit is not None:
            a,lo,hi=rows_for_audit;a=sp.csr_matrix(a)
            restriction_file=folder/"restrictions"/args.epoch_id/f"query_{i:05d}.npz"
            restriction_file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(restriction_file,A_data=a.data,A_indices=a.indices,A_indptr=a.indptr,
                A_shape=np.asarray(a.shape),row_lower=lo,row_upper=hi)
            if restriction is None:restriction={}
            restriction.update(additional_rows_file=str(restriction_file.relative_to(folder)),
                additional_rows_sha256=file_hash(restriction_file),additional_rows_nnz=int(a.nnz))
        solution_name=solution_sha=None
        if x is not None:
            solution_file=folder/"returned_solutions"/args.epoch_id/f"query_{i:05d}.npz"
            solution_file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(solution_file,x=x)
            solution_name=str(solution_file.relative_to(folder));solution_sha=file_hash(solution_file)
        within=total<=args.query_budget
        tol=adapter.recovery_tolerance(reference)
        below_reference=bool(valid and objective is not None and objective<reference-tol)
        if below_reference:valid=False
        exact=bool(valid and objective is not None and abs(objective-reference)<=tol)
        record=dict(schema="formal-uc-query-v1",method=args.method,n_train=args.n,n_validation=args.validation_n,
            query_index=i,test_array_sha256=test.dataset_fingerprint,train_array_sha256=case.dataset_fingerprint,
            physical_model_sha256=model_hash(case.model),epoch_id=args.epoch_id,host=host,
            query_budget_seconds=args.query_budget,solver_backend=args.backend,solver_threads=1,
            solver_seed=args.solver_seed,warm_start=False,common_D=len(case.commitment_indices),
            target_K=None if method_native or args.method=="full" else K,
            K_scope="commitment affine dimension before original constraints",restriction=restriction,
            reference_objective=float(reference),reference_certified=True,reference_solve_is_timed_full=False,
            returned_objective=objective,valid=valid,within_budget=within,exact_recovery=exact,
            numerical_data_blocker=below_reference,
            blocker_reason="valid primal below certified reference beyond tolerance" if below_reference else None,
            returned_solution_file=solution_name,returned_solution_sha256=solution_sha,
            primary_exact_recovery=bool(exact and within),total_query_seconds=total,
            prediction_seconds=prediction_seconds,solve=raw,physical_validation=physical,
            relative_excess=None if not valid else (objective-reference)/max(abs(reference),1e-12),
            full_over_returned_cost=None if not valid or objective==0 else reference/objective,
            diagnostics=diagnostic)
        with ledger.open("a",encoding="utf-8") as f:f.write(json.dumps(clean(record),allow_nan=False)+"\n")
        print(json.dumps(dict(event="query_complete",method=args.method,i=i,valid=valid,exact=exact,seconds=total)),flush=True)
    write_json(folder/f"evaluation-{args.epoch_id}{shard}.json",dict(queries=len(query_indices),
        expected_test_count=test.n_train,query_indices=list(query_indices),ledger_sha256=file_hash(ledger),
        epoch_id=args.epoch_id,runtime=host,all_methods_require_same_epoch=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("fit","evaluate"));p.add_argument("--bank",required=True)
    p.add_argument("--out",required=True);p.add_argument("--n",type=int,choices=(256,512),required=True)
    p.add_argument("--method",choices=METHODS,required=True);p.add_argument("--validation-n",type=int,default=64)
    p.add_argument("--test-n",type=int,default=128);p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--backend",choices=("highs","scip"),default="highs");p.add_argument("--query-budget",type=float,default=20.)
    p.add_argument("--solver-seed",type=int,default=42001);p.add_argument("--epoch-id");p.add_argument("--config")
    p.add_argument("--query-index",type=int,help="Evaluate one immutable test index; final audit must cover every index")
    args=p.parse_args()
    if args.action=="fit":fit(args)
    else:evaluate(args)


if __name__=="__main__":main()
