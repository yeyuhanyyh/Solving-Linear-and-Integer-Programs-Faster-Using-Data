"""Budget-aligned Li evaluation on all360 unchanged existing UC test costs.

This separate evaluator leaves the already running fit-bound source and
checkpoints untouched. Its outer clock includes prediction/native decoding,
restricted solve, original CSR/UC-physics/fixing validation and cost@x. The
entire fixed bank is rerun under this rule, never selected against v1 returns.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import uc_li_supplement as driver

ROOT=Path(__file__).resolve().parent
SEAL=ROOT/"evaluation-v2-source-seal.json"


def verify_source():
    seal=driver.read(SEAL)
    if (seal["evaluator_sha256"]!=driver.sha(__file__)
            or seal["fit_driver_sha256"]!=driver.sha(driver.__file__)
            or seal["original_bank_model_freeze_sha256"]!=driver.ORIGINAL_FREEZE_SHA
            or seal["prepared_before_first_Li_bank_prediction"] is not True):
        raise ValueError("Budget-aligned evaluator source binding changed")
    return seal


def evaluate(out=ROOT/"run-v1",freeze_path=ROOT/"li-model-freeze.json",results=ROOT/"same-bank-results-v2"):
    verify_source()
    frozen,cases=driver.validate_freeze(out,freeze_path)
    confirm,helper,runtime,nn=driver.enable()
    import adapter,preference_baseline as li
    from dimension_diagnostics import assert_matched_fixings
    summaries=[]
    for seed,case in cases.items():
        folder,marker=driver.validate_checkpoint(case,out)
        model=li.load_fitted(adapter.training_case(case),folder)
        bank,bank_files=driver.load_existing_bank(case)
        guard=dict(li_model_freeze_sha256=driver.sha(freeze_path),checkpoint_sha256=driver.sha(folder/"complete.json"),
            fit_source_sha256=driver.source_hashes(),evaluator_sha256=driver.sha(__file__),
            evaluator_source_seal_sha256=driver.sha(SEAL),existing_bank_file_sha256=bank_files,
            original_bank_predates_this_model=True,selection_used_only_training_prefix=True,budget_seconds=30.,
            budget_includes="native prediction/decoding + IP build/solve + original CSR, UC physics, fixing and cost@x checks",
            evaluation_rule="v2 on all360 costs, no per-cost or per-case selection against v1")
        rows=[];result_dir=Path(results)/case.case_id
        for i,(feature,cost,reference) in enumerate(zip(bank["features"],bank["costs"],bank["objectives"])):
            path=result_dir/f"test{i:03d}.json"
            if path.exists():
                row=driver.read(path)
                payload={k:v for k,v in row.items() if k!="record_payload_sha256"}
                if row["guard"]!=guard or row["index"]!=i or row["record_payload_sha256"]!=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest():
                    raise ValueError("Prior budget-aligned result identity/payload changed")
                rows.append(row);continue
            begin=time.perf_counter()
            if not np.allclose(driver.signed_cost_map(case,feature),cost,rtol=1e-14,atol=1e-12):
                raise ValueError("Current cost does not follow the explicit original affine map")
            fixed,diagnostic=nn.propose_li_native_anchor(model,case,cost,feature,30.)
            remaining=30.-(time.perf_counter()-begin)
            if fixed is None or remaining<=0:
                raw=dict(valid=False,certified=False,status="no_native_anchor_or_budget");x=None
            else:
                assert_matched_fixings(case,fixed,target_K=int(case.affine["rank"]))
                remaining=30.-(time.perf_counter()-begin)
                if remaining<=0:raw=dict(valid=False,certified=False,status="fixing_validation_budget_exhausted");x=None
                else:
                    solved=adapter.solve(case.model,cost,fixed=fixed,seed=seed+500000+i,time_limit=remaining)
                    raw=solved.to_dict(include_solution=False);x=solved.x
            original_check=adapter.validate_solution(case.model,x) if x is not None else dict(valid=False)
            physics=runtime["uc_model"].validate_uc_solution(case.original_base,x) if x is not None else dict(valid=False)
            original=bool(x is not None and original_check["valid"] and physics["valid"])
            actual_K=None if fixed is None else 48-len(fixed)
            fixing_residual=None if x is None or fixed is None else max((abs(float(x[int(j)])-v) for j,v in fixed.items()),default=0.)
            restriction=bool(original and fixing_residual is not None and fixing_residual<=1e-6)
            raw_objective=float(cost@x) if original else None
            # All original feasibility/physics/fixing/objective validation
            # above belongs to the method's shared outer30second budget.
            seconds=time.perf_counter()-begin
            timely=seconds<=30.
            valid=bool(original and restriction and timely and actual_K==int(case.affine["rank"]))
            objective=raw_objective if valid else None
            ratio=float(reference)/objective if objective is not None and objective>0 else None
            if ratio is not None and ratio>1.+adapter.recovery_tolerance(reference)/abs(reference):
                raise ValueError("Restricted Li return beats certified original optimum")
            row=driver.clean(dict(guard=guard,method="Li",case_id=case.case_id,seed=seed,index=i,n_train=32,
                K=int(case.affine["rank"]),actual_K=actual_K,common_u=48,original_variables=192,
                original_integer_uvw=144,continuous_p=48,feature=feature,cost=cost,reference_objective=float(reference),
                objective=objective,raw_objective=raw_objective,objective_ratio=ratio,
                recovered=bool(valid and abs(objective-reference)<=adapter.recovery_tolerance(reference)),
                original_feasible=original,restriction_valid=restriction,valid=valid,timely=timely,
                seconds=seconds,solution=x,fixed=fixed,fixing_residual=fixing_residual,raw=raw,diagnostic=diagnostic,
                original_check=original_check,UC_physics=physics,
                phase="posthoc_baseline_on_existing_confirmation_bank",confirmation=True,
                original_confirmation_bank_generated_before_Li_fit=True,evaluation_version=2))
            row["record_payload_sha256"]=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            driver.write_new(path,row);rows.append(row)
        ratios=[r["objective_ratio"] for r in rows if r["objective_ratio"] is not None]
        summary=dict(method="Li",case_id=case.case_id,seed=seed,n_train=32,K=int(case.affine["rank"]),count=len(rows),
            recovered=sum(r["recovered"] for r in rows),recovery_percent=100.*sum(r["recovered"] for r in rows)/len(rows),
            objective_ratio_mean=float(np.mean(ratios)) if ratios else None,ratio_denominator=len(ratios),
            fit_seconds=marker["fit_seconds"],fit_cpu_seconds=marker["fit_cpu_seconds"],
            posthoc_existing_bank_baseline=True,no_test_parameter_selection=True,evaluation_version=2)
        driver.write_new(result_dir/"complete.json",dict(guard=guard,summary=summary,
            test_record_sha256={f"test{i:03d}.json":driver.sha(result_dir/f"test{i:03d}.json") for i in range(len(rows))}))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    rows=[driver.read(Path(results)/case.case_id/f"test{i:03d}.json") for case in cases.values() for i in range(120)]
    ratios=[r["objective_ratio"] for r in rows if r["objective_ratio"] is not None]
    value=dict(status="COMPLETE",method="Li",case_count=3,total=360,recovered=sum(r["recovered"] for r in rows),
        recovery_percent=100.*sum(r["recovered"] for r in rows)/360,
        objective_ratio_mean=float(np.mean(ratios)) if ratios else None,ratio_denominator=len(ratios),
        per_case=summaries,li_model_freeze_sha256=driver.sha(freeze_path),posthoc_existing_bank_baseline=True,
        no_test_parameter_or_epoch_selection=True,evaluation_version=2,
        ratio_aggregation="pooled arithmetic mean over valid raw returns",
        previous_v1_preserved_but_superseded=True,no_selection_between_v1_and_v2=True)
    driver.write_new(Path(results)/"summary.json",value)
    return value


def main():
    sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"run-v1")
    parser.add_argument("--freeze",type=Path,default=ROOT/"li-model-freeze.json")
    parser.add_argument("--results",type=Path,default=ROOT/"same-bank-results-v2")
    args=parser.parse_args();evaluate(args.out,args.freeze,args.results)


if __name__=="__main__":main()
