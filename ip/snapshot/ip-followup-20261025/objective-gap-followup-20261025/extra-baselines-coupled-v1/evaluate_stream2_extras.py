"""SVM/MLOPT on the shared all-fifteen-role-frozen stream2 bank.

Source-only preparation is safe: import never fits models, generates RNG, or
reads a confirmation bank. This later supplemental evaluator is separately
sealed against the completed global model freeze before its first bank access.
All methods and model roles were frozen before RNG. It never selects models.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
FOLLOWUP=ROOT.parents[1]
STREAM2_PATH=FOLLOWUP/"objective-gap-designs-20261025/coupled-bundle/training-budget-control-v1/stream2/stream2_runner.py"
_STREAM2=None
ROLES=("svm","mlopt")


def stream2():
    global _STREAM2
    if _STREAM2 is None:
        spec=importlib.util.spec_from_file_location("coupled_stream2_extra_shared_runner",STREAM2_PATH)
        _STREAM2=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=_STREAM2;spec.loader.exec_module(_STREAM2)
    return _STREAM2


def validate(freeze_path):
    shared=stream2();f,cases=shared.validate_global_freeze(freeze_path)
    seal=shared.read(ROOT/"stream2-extra-evaluator-seal.json")
    if (seal.get("model_freeze_sha256")!=shared.sha(freeze_path)
            or seal.get("evaluator_sha256")!=shared.sha(__file__)
            or seal.get("prepared_after_global_model_freeze") is not True
            or seal.get("no_frozen_method_or_model_changes") is not True):
        raise ValueError("Supplemental evaluator/global model source binding changed")
    return shared,f,cases


def evaluate_case(case,role,freeze_path,results_root):
    shared,f,cases=validate(freeze_path)
    if role not in ROLES or case.case_id not in cases:raise ValueError("Unknown frozen extra role")
    entry=f["cases"][case.case_id]
    results_root=Path(results_root).resolve()
    for name in ("model_root","continued_root","extra_model_root"):
        protected=Path(entry[name]).resolve()
        if results_root==protected or protected in results_root.parents or results_root in protected.parents:
            raise ValueError("Results must be separate from frozen models")
    bank_root=Path(f["bank_root"]).resolve()
    if results_root==bank_root or bank_root in results_root.parents or results_root in bank_root.parents:
        raise ValueError("Results must be separate from the stream2 bank")
    extra=shared._EXTRA
    # Global all-fifteen-role validation and own checkpoint validation precede
    # bank loading, even when another process has already evaluated NN roles.
    model=extra.load_model(case,role,Path(entry["extra_model_root"]),Path(entry["manifest_path"]))
    bank,metadata=shared.load_confirmation_bank(case,freeze_path)
    import adapter
    budget=30.;K=int(case.affine["rank"])
    guard=shared.confirmation_guard(case,role,metadata,freeze_path,budget)
    guard.update(supplemental_evaluator_sha256=shared.sha(__file__),
        supplemental_source_seal_sha256=shared.sha(ROOT/"stream2-extra-evaluator-seal.json"),
        extra_checkpoint_guard=extra.guard(case,role,Path(entry["manifest_path"])),
        no_parameter_or_checkpoint_selection=True)
    folder=results_root/case.case_id/role;folder.mkdir(parents=True,exist_ok=True)
    proposer=extra.components.propose_svm if role=="svm" else extra.components.propose_mlopt
    rows=[]
    for i,(feature,cost,reference) in enumerate(zip(bank["features"],bank["costs"],bank["objectives"])):
        path=folder/f"test{i:03d}.json"
        if path.exists():
            row=shared.read(path)
            shared.verify_confirmation_record(row,case,role,i,cost,reference,guard,budget,feature)
            rows.append(row);continue
        begin=time.perf_counter()
        fixed,diagnostics=proposer(model,case,cost,feature)
        remaining=budget-(time.perf_counter()-begin)
        if fixed is None or remaining<=0:
            raw=dict(valid=False,certified=False,status="no_native_anchor_or_prediction_budget_exhausted");x=None
        else:
            solved=adapter.solve(case.model,cost,fixed=fixed,seed=case.seed+i,time_limit=remaining)
            raw=solved.to_dict(include_solution=False);x=solved.x
        seconds=time.perf_counter()-begin
        original=bool(x is not None and adapter.validate_solution(case.model,x)["valid"])
        actual=None if fixed is None else case.model.n_variables-len(fixed)
        if fixed is not None and actual!=K:raise ValueError("SVM/MLOPT returned a different common-coordinate K")
        restriction=bool(original and fixed is not None and all(abs(x[int(j)]-v)<=1e-6 for j,v in fixed.items()))
        raw_objective=float(cost@x) if original else None
        timely=seconds<=budget
        valid=bool(original and restriction and timely and actual==K)
        objective=raw_objective if valid else None
        ratio=objective/float(reference) if objective is not None else None
        if ratio is not None and ratio>1.+adapter.recovery_tolerance(reference)/abs(reference):
            raise ValueError("Restricted result profit exceeds original certified optimum")
        row=dict(guard=guard,case_id=case.case_id,method=role,index=i,n_train=case.n_train,
            K=K,actual_K=actual,D=case.model.n_variables,cost=cost,feature=feature,budget_seconds=budget,
            original_feasible=original,restriction_valid=restriction,timely=timely,valid=valid,
            objective=objective,reference_objective=float(reference),profit_ratio=ratio,
            recovered=bool(valid and abs(objective-reference)<=adapter.recovery_tolerance(reference)),
            raw_objective=raw_objective,seconds=seconds,raw=raw,solution=x,fixed=fixed,
            diagnostics=diagnostics,experiment_phase="independent_confirmation",confirmation=True)
        row=shared.clean(row);row["record_payload_sha256"]=shared.payload_sha(row)
        shared.verify_confirmation_record(row,case,role,i,cost,reference,guard,budget,feature)
        shared.write_new(path,row);rows.append(row)
    ratios=[row["profit_ratio"] for row in rows if row["profit_ratio"] is not None]
    summary=dict(case_id=case.case_id,method=role,phase="independent_confirmation",n_train=case.n_train,
        K=K,D=case.model.n_variables,count=len(rows),recovery_count=sum(row["recovered"] for row in rows),
        recovery_percent=100.*sum(row["recovered"] for row in rows)/len(rows),
        profit_ratio_mean_valid=float(np.mean(ratios)) if ratios else None,ratio_denominator=len(ratios),
        negative_returned_profit_count=sum(value<0 for value in ratios),independent_confirmation=True)
    complete=dict(guard=guard,summary=summary,
        test_record_sha256={f"test{i:03d}.json":shared.sha(folder/f"test{i:03d}.json") for i in range(len(rows))})
    shared.write_new(folder/"complete.json",complete)
    return summary


def main():
    sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    shared=stream2()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",type=Path,default=shared.DEFAULT_FREEZE)
    parser.add_argument("--results",type=Path,default=shared.DEFAULT_RUN/"results")
    parser.add_argument("--roles",nargs="+",choices=ROLES,default=list(ROLES))
    parser.add_argument("--case-id",nargs="+")
    args=parser.parse_args()
    shared,f,cases=validate(args.freeze)
    chosen=sorted(cases) if args.case_id is None else args.case_id
    if not chosen or len(set(chosen))!=len(chosen) or not set(chosen)<=set(cases):raise ValueError("Unknown or duplicate cases")
    if len(set(args.roles))!=len(args.roles):raise ValueError("Duplicate role")
    summaries=[]
    for cid in chosen:
        for role in args.roles:
            summary=evaluate_case(cases[cid],role,args.freeze,args.results)
            summaries.append(summary);print(shared.json.dumps(summary),flush=True)
    if set(chosen)==set(cases) and set(args.roles)==set(ROLES):
        shared.write_new(args.results/"svm-mlopt-summary.json",dict(
            schema="stream2-svm-mlopt-shared-bank-summary-v1",model_freeze_sha256=shared.sha(args.freeze),
            evaluator_sha256=shared.sha(__file__),summaries=summaries,independent_confirmation=True))


if __name__=="__main__":main()
