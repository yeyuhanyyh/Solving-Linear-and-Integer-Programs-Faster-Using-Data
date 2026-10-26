"""Post-fit diagnostics on the unchanged existing development banks only.

All twelve new checkpoints must pass guards before the first bank is read.
Existing original-IP certificates are reused and validated; no fresh RNG or
confirmation bank is accessed. This file never selects parameters or epochs.
"""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import coupled_svm_mlopt_driver as driver
from run_six_serial import G32,G16

ROOT=Path(__file__).resolve().parent


def canonical_hash(payload):
    return hashlib.sha256(driver.generic.json.dumps(driver.generic.clean(payload),
        sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def jobs():
    result=[]
    for manifest,G,n in ((G32,32,36),(G16,16,37)):
        for entry in driver.generic.validate_selection(manifest)["cases"]:
            if int(entry["D"])==3*G and int(entry["n_train"])==n:
                result.append((manifest,entry))
    if len(result)!=6:raise ValueError("Expected all six fixed cases")
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,default=ROOT/"results")
    parser.add_argument("--freeze",type=Path,default=ROOT/"fit-logs/complete.json")
    parser.add_argument("--out",type=Path,default=ROOT/"existing-development")
    parser.add_argument("--budget",type=float,default=30.)
    args=parser.parse_args()
    freeze=driver.generic.read(args.freeze)
    if freeze.get("all_exit_zero") is not True:raise ValueError("All twelve fit roles must complete first")
    driver.enable()
    import adapter
    all_cases=[];role_files={}
    # No development-bank access occurs in this global all-twelve-role gate.
    for manifest,entry in jobs():
        driver.selected_entry(manifest,entry["directory"])
        case=driver.generic.load_case(entry["directory"],manifest)
        for method in ("svm","mlopt"):
            folder,complete=driver.validate_checkpoint(case,method,args.models,manifest)
            role_files[str(folder/"complete.json")]=driver.generic.sha(folder/"complete.json")
        all_cases.append((manifest,case))
    global_guard=dict(fit_freeze_sha256=driver.generic.sha(args.freeze),role_marker_sha256=role_files,
        evaluator_sha256=driver.generic.sha(__file__),phase="existing_development_diagnostic",
        prior_fresh_stream_not_read=True,no_parameter_or_epoch_selection=True,
        confirmation=False,budget_seconds=args.budget)
    gate_path=args.out/"global-model-gate.json"
    if gate_path.exists():
        if driver.generic.read(gate_path)["guard"]!=global_guard:
            raise ValueError("Prior global model gate changed")
    else:
        driver.generic.write_new(gate_path,dict(guard=global_guard,
            verified_utc=datetime.now(timezone.utc).isoformat()))
    summaries=[]
    for manifest,case in all_cases:
        bank,bank_hash=driver.generic.load_development(case)
        prior="uniform-gross-g32-n36-pilot-results" if case.model.n_variables==96 else "uniform-gross-pilot-results"
        references={}
        for i,reference in enumerate(bank["objectives"]):
            path=ROOT.parent/prior/"development-certificates"/case.case_id/f"dev{i:03d}.json"
            certificate=driver.generic.read(path)
            expected=dict(training=driver.generic.pilot_guard(case,"development_oracle"),development_bank_sha256=bank_hash)
            if certificate["guard"]!=expected or certificate["index"]!=i or certificate["reference"]!=float(reference):
                raise ValueError("Existing development optimum certificate source changed")
            if certificate["oracle"].get("certified") is not True or certificate["oracle"].get("valid") is not True:
                raise ValueError("Existing development original optimizer is not certified")
            references[str(path)]=driver.generic.sha(path)
        for method in ("svm","mlopt"):
            model=driver.load_model(case,method,args.models,manifest)
            propose=driver.components.propose_svm if method=="svm" else driver.components.propose_mlopt
            guard=dict(global_model_gate=global_guard,training=driver.guard(case,method,manifest),
                       development_bank_sha256=bank_hash,reference_certificates_sha256=references)
            folder=args.out/case.case_id/method;rows=[]
            for i,(feature,cost,reference) in enumerate(zip(bank["features"],bank["costs"],bank["objectives"])):
                path=folder/f"dev{i:03d}.json"
                if path.exists():
                    row=driver.generic.read(path)
                    driver.generic.verify_development_record(row,case,method,i,cost,reference,guard,args.budget,feature)
                    rows.append(row);continue
                started=time.perf_counter()
                fixed,diagnostics=propose(model,case,cost,feature)
                remaining=args.budget-(time.perf_counter()-started)
                if fixed is None or remaining<=0:
                    x=None;raw=dict(valid=False,status="no_fixings_or_prediction_budget")
                else:
                    solved=adapter.solve(case.model,cost,fixed=fixed,seed=case.seed+i,time_limit=remaining)
                    x=solved.x;raw=solved.to_dict(include_solution=False)
                seconds=time.perf_counter()-started
                original=bool(x is not None and adapter.validate_solution(case.model,x)["valid"])
                actual=None if fixed is None else case.model.n_variables-len(fixed)
                restriction=bool(original and fixed is not None and all(abs(x[j]-v)<=1e-6 for j,v in fixed.items()))
                valid=bool(original and restriction and seconds<=args.budget and actual==int(case.affine["rank"]))
                objective=float(cost@x) if valid else None
                row=dict(guard=guard,case_id=case.case_id,method=method,index=i,
                    n_train=case.n_train,D=case.model.n_variables,K=int(case.affine["rank"]),actual_K=actual,
                    budget_seconds=float(args.budget),feature=feature,cost=cost,
                    original_feasible=original,restriction_valid=restriction,timely=seconds<=args.budget,
                    valid=valid,objective=objective,raw_objective=float(cost@x) if original else None,
                    reference_objective=float(reference),profit_ratio=objective/float(reference) if valid else None,
                    recovered=bool(valid and abs(objective-reference)<=adapter.recovery_tolerance(reference)),
                    seconds=seconds,solution=x,fixed=fixed,raw=raw,diagnostics=diagnostics,
                    phase="existing_development_diagnostic",confirmation=False)
                row=driver.generic.clean(row);row["record_payload_sha256"]=canonical_hash(row)
                driver.generic.verify_development_record(row,case,method,i,cost,reference,guard,args.budget,feature)
                driver.generic.write_new(path,row);rows.append(row)
            ratios=[r["profit_ratio"] for r in rows if r["profit_ratio"] is not None]
            summary=dict(case_id=case.case_id,method=method,n_train=case.n_train,D=case.model.n_variables,
                K=int(case.affine["rank"]),count=len(rows),recovery_count=sum(r["recovered"] for r in rows),
                recovery_percent=100.*sum(r["recovered"] for r in rows)/len(rows),
                mean_profit_ratio_valid=float(np.mean(ratios)) if ratios else None,ratio_denominator=len(ratios),
                fit_seconds=float(model.metadata["fit_seconds"]),phase="existing_development_diagnostic",confirmation=False)
            driver.generic.write_new(folder/"complete.json",dict(guard=guard,summary=summary))
            summaries.append(summary);print(driver.generic.json.dumps(summary),flush=True)
    driver.generic.write_new(args.out/"summary.json",dict(guard=global_guard,summaries=summaries,
        no_tuning_or_model_reselection_performed=True,completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=="__main__":main()
