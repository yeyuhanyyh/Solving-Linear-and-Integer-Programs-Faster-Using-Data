"""Full native-code Li supplement for the fixed n32 startup-cost UC cohort.

No training or confirmation access occurs on import. Original source/model/
record files remain read-only. All three Li models are frozen after fitting
and before their first evaluation on the existing 360-cost bank. The bank was
generated earlier; this is explicitly a later baseline supplement.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
UCROOT=ROOT.parent
FOLLOWUP=ROOT.parents[2]
EXPERIMENT=FOLLOWUP.parent
CONFIRM_PATH=UCROOT/"confirmation-v1/confirm_n32.py"
TRAIN_PATH=FOLLOWUP/"train_uc_baselines.py"
DATA_PATH=FOLLOWUP/"uc_adapters.py"
LI_PATH=EXPERIMENT/"code/preference_baseline.py"
ORIGINAL_FREEZE=UCROOT/"confirmation-v1/global-freeze.json"
ORIGINAL_FREEZE_SHA="a47236f3e39465c1b6053d2c84282facc51ab7e1a281af247735ca6c0979beb3"
EXPECTED={CONFIRM_PATH:"5dfb457bfac7df3a1cef073a99a942a1d29e1985b3bef47e4d1c5db68650f8c1",
    TRAIN_PATH:"f9613f57036869e59d607ba62fa470df62893501183023d2e802f137777f01dd",
    DATA_PATH:"b0f5cbd42a2099289c89cbd0b28bf41a6c2bbeed673af02abf9540862edb9f40",
    LI_PATH:"3825e64f3daa34884721cef260895d5e587da4366150f4b3ca17f6594c2ff9ba",
    ORIGINAL_FREEZE:ORIGINAL_FREEZE_SHA}
SEEDS=(101,211,307)
TARGET_K={101:14,211:14,307:17}
_CONFIRM=_HELPER=_RUNTIME=_NN=None


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(clean(value),indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    if path.exists():
        if path.read_bytes()!=raw:raise FileExistsError(path)
        return
    with path.open("xb") as stream:stream.write(raw)


def import_path(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def signed_cost_map(case,features):
    return np.asarray(case.metadata["cost_offset"])+np.asarray(features)@np.asarray(case.metadata["cost_feature_coefficients"]).T


def enable():
    global _CONFIRM,_HELPER,_RUNTIME,_NN
    for path,expected in EXPECTED.items():
        if sha(path)!=expected:raise ValueError("Read-only source changed: "+str(path))
    if _CONFIRM is None:
        _CONFIRM=import_path(CONFIRM_PATH,"uc_li_original_confirmation")
        _HELPER,_RUNTIME=_CONFIRM.load_helpers()
        sys.path.insert(0,str(FOLLOWUP))
        # The existing native runner is reused without changing any class,
        # network, epoch, optimizer, grid, or final-state selection rule.
        _NN=import_path(TRAIN_PATH,"uc_li_native_training_runner")
        import adapter,portable_adapter,uc_adapters,preference_baseline
        if Path(preference_baseline.__file__).resolve()!=LI_PATH.resolve():
            raise ValueError("Shadowed Li source module")
        for module in (adapter,portable_adapter,uc_adapters):module.costs_from_features=signed_cost_map
        _NN.model_guard=model_guard
    return _CONFIRM,_HELPER,_RUNTIME,_NN


def load_case(seed):
    if seed not in SEEDS:raise ValueError("Retain only the three fixed physical seeds")
    confirm,helper,runtime,nn=enable()
    original,_,_,offset,coefficient,files=confirm.load_case(seed,helper,runtime)
    metadata=dict(original.metadata,baseline_seed=int(original.seed),
        cost_offset=np.asarray(offset).tolist(),cost_feature_coefficients=np.asarray(coefficient).tolist())
    case=replace(original,metadata=metadata)
    if case.n_train!=32 or case.model.n_variables!=192 or case.train_features.shape!=(32,12):
        raise ValueError("Fixed n32 full-original UC data shape changed")
    if int(case.affine["rank"])!=TARGET_K[seed] or not np.array_equal(case.commitment_indices,np.arange(48)):
        raise ValueError("Fixed common-u K or column order changed")
    if not np.array_equal(case.model.integrality,np.r_[np.ones(144,dtype=int),np.zeros(48,dtype=int)]):
        raise ValueError("Startup/shutdown binary or dispatch continuous domains changed")
    if np.asarray(coefficient).shape!=(192,12) or not np.allclose(signed_cost_map(case,case.train_features),case.train_costs,rtol=1e-14,atol=1e-12):
        raise ValueError("Explicit generator/time-block objective map changed")
    return case,files


def source_hashes():
    return {str(path.resolve()):sha(path) for path in (Path(__file__),*EXPECTED)}


def model_guard(case,method,bundle=None):
    if method!="li":raise ValueError("This supplement fits Li only")
    confirm,helper,runtime,nn=enable()
    return dict(schema="uc-nrel-n32-li-guard-v1",case_id=case.case_id,method="li",
        original_training_binding=helper._case_binding(case),source_sha256=source_hashes(),
        source_hashes=dict(case.source_hashes),training_array_hash=case.dataset_fingerprint,
        target_K=int(case.affine["rank"]),common_indices=case.commitment_indices.tolist(),
        explicit_objective_map_sha256=hashlib.sha256(np.asarray(case.metadata["cost_offset"],dtype=float).tobytes()+np.asarray(case.metadata["cost_feature_coefficients"],dtype=float).tobytes()).hexdigest(),
        baseline_seed=int(case.metadata["baseline_seed"]),
        original_confirmation_bank_generated_before_Li_fit=True,test_information_used=False)


def validate_checkpoint(case,out=ROOT/"run-v1"):
    nn=enable()[3]
    folder,marker=nn.validate_checkpoint(out,case,"li",ROOT)
    if marker.get("sanity_gate_passed") is not True or marker.get("training_label_count")!=32:
        raise ValueError("Li training readiness/count gate failed")
    meta=read(folder/"metadata.json")["metadata"]
    cfg=meta["configuration"]
    if any(cfg.get(k)!=v for k,v in dict(hidden_dim=128,attention_layers=2,epochs=1200,batch_size=128,
        learning_rates=[.0001,.001],lambda1_values=[.8,.9],lambda2_values=[.1,.2]).items()):
        raise ValueError("Full original Li architecture/epoch/grid changed")
    if meta.get("training_count")!=32 or meta.get("test_information_used") is not False:
        raise ValueError("Li fitting sample scope changed")
    return folder,marker


def train_one(seed,out,device):
    case,_=load_case(seed);nn=enable()[3]
    nn.train_one(case,"li",out,ROOT,device)
    folder,marker=validate_checkpoint(case,out)
    print(json.dumps(dict(event="uc_li_complete",seed=seed,case_id=case.case_id,
        K=int(case.affine["rank"]),fit_seconds=marker["fit_seconds"],checkpoint=str(folder))),flush=True)


def freeze_all(out=ROOT/"run-v1",freeze_path=ROOT/"li-model-freeze.json"):
    if Path(freeze_path).exists():return validate_freeze(out,freeze_path)[0]
    roles={}
    for seed in SEEDS:
        case,_=load_case(seed);folder,marker=validate_checkpoint(case,out)
        roles[case.case_id]=dict(seed=seed,K=int(case.affine["rank"]),n_train=32,
            guard=marker["guard"],checkpoint_dir=str(folder.resolve()),
            complete_sha256=sha(folder/"complete.json"),artifact_sha256=marker["artifact_sha256"])
    value=dict(schema="uc-nrel-n32-li-all-three-models-freeze-v1",roles=roles,
        source_sha256=source_hashes(),frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        original_confirmation_bank_generated_before_these_fits=True,
        own_confirmation_evaluation_not_started=True,no_test_parameter_selection=True)
    write_new(freeze_path,value);validate_freeze(out,freeze_path);return value


def validate_freeze(out=ROOT/"run-v1",freeze_path=ROOT/"li-model-freeze.json"):
    value=read(freeze_path)
    if value.get("schema")!="uc-nrel-n32-li-all-three-models-freeze-v1" or len(value.get("roles",{}))!=3:
        raise ValueError("All three Li checkpoints must be frozen before bank access")
    if value["source_sha256"]!=source_hashes():raise ValueError("Frozen Li source changed")
    cases={}
    for seed in SEEDS:
        case,_=load_case(seed);folder,marker=validate_checkpoint(case,out);role=value["roles"][case.case_id]
        if role["guard"]!=marker["guard"] or role["artifact_sha256"]!=marker["artifact_sha256"] or role["complete_sha256"]!=sha(folder/"complete.json"):
            raise ValueError("Frozen Li checkpoint bytes/context changed")
        cases[seed]=case
    return value,cases


def smoke(seed=101):
    case,files=load_case(seed);confirm,helper,runtime,nn=enable()
    import preference_baseline as li,adapter
    cfg=li.PreferenceConfig(seed=case.seed,device="cpu",top_k_values=(1,3,5,10,30))
    model=li.PreferenceBaseline(adapter.training_case(case),config=cfg)
    # Native constructor and one certified training query; no neural update,
    # pair-generation solve, validation bank or held-out input is requested.
    cost=signed_cost_map(case,case.train_features[0]);x=case.train_solutions[0]
    if not adapter.validate_solution(case.model,x)["valid"] or not runtime["uc_model"].validate_uc_solution(case.original_base,x)["valid"]:
        raise ValueError("Training-only smoke fails original UC feasibility")
    if abs(float(cost@x)-case.train_objectives[0])>adapter.recovery_tolerance(case.train_objectives[0]):
        raise ValueError("Training-only signed objective query differs")
    value=dict(schema="uc-li-one-training-query-smoke-v1",seed=seed,case_id=case.case_id,
        n_train=32,D=192,common_u=48,K=int(case.affine["rank"]),integer_uvw=144,continuous_p=48,
        explicit_affine_map_reproduces_all_32_training_costs=True,query_index=0,
        native_Li_constructor_passed=True,epochs=cfg.epochs,network_hidden_dim=cfg.hidden_dim,
        attention_layers=cfg.attention_layers,requested_top_k=[1,3,5,10,30],
        neural_fit_performed=False,confirmation_bank_accessed=False,
        original_training_files=files,source_sha256=source_hashes())
    write_new(ROOT/"training-smoke.json",value)
    print(json.dumps(clean(value)),flush=True)


def load_existing_bank(case):
    confirm,helper,runtime,nn=enable()
    # This function is called only after the separate all-three-Li-model gate.
    freeze=read(ORIGINAL_FREEZE);population,_,_,_=confirm.validate_population(freeze)
    summary_path=UCROOT/"confirmation-v1/results-v1/summary.json";summary=read(summary_path)
    if summary.get("status")!="COMPLETE" or summary.get("Full_reference_count")!=360 or summary.get("source_freeze_sha256")!=ORIGINAL_FREEZE_SHA:
        raise ValueError("Need the unchanged complete 360-cost original UC bank")
    folder=summary_path.parent/case.case_id
    required=[folder/"confirmation.npz",folder/"Full-reference-certificates.json",folder/"metadata.json"]
    summary_files={str(name).replace("\\","/"):digest for name,digest in summary["files"].items()}
    for path in required:
        relative=str(path.relative_to(summary_path.parent)).replace("\\","/")
        if summary_files.get(relative)!=sha(path):raise ValueError("Existing confirmation bank artifact changed")
    meta=read(folder/"metadata.json")
    if meta["source_freeze_sha256"]!=ORIGINAL_FREEZE_SHA or meta["count"]!=120 or meta["K"]!=TARGET_K[case.seed]:
        raise ValueError("Existing confirmation population changed")
    with np.load(folder/"confirmation.npz",allow_pickle=False) as stream:bank={key:stream[key].copy() for key in stream.files}
    if (bank["features"].shape,bank["costs"].shape,bank["solutions"].shape,bank["objectives"].shape)!=((120,12),(120,192),(120,192),(120,)):
        raise ValueError("Existing confirmation shapes changed")
    if not np.allclose(signed_cost_map(case,bank["features"]),bank["costs"],rtol=1e-14,atol=1e-12):
        raise ValueError("Existing confirmation generator/time-block map changed")
    certificates=read(folder/"Full-reference-certificates.json")
    import adapter
    for i,(c,x,ref) in enumerate(zip(bank["costs"],bank["solutions"],bank["objectives"])):
        raw=certificates[i];tol=adapter.recovery_tolerance(ref)
        if not (raw["valid"] and raw["certified"] and raw["original_UC_physics_valid"] and raw["index"]==i):
            raise ValueError("Original confirmation reference lacks certificate")
        if not adapter.validate_solution(case.model,x)["valid"] or not runtime["uc_model"].validate_uc_solution(case.original_base,x)["valid"]:
            raise ValueError("Original reference violates UC constraints")
        if max(abs(float(c@x)-ref),abs(raw["objective"]-ref),abs(raw["dual_bound"]-ref))>tol:
            raise ValueError("Original reference primal/dual objective differs")
    return bank,{str(p.resolve()):sha(p) for p in [summary_path,*required]}


def evaluate_existing(out=ROOT/"run-v1",freeze_path=ROOT/"li-model-freeze.json",results=ROOT/"same-bank-results-v1"):
    frozen,cases=validate_freeze(out,freeze_path)
    confirm,helper,runtime,nn=enable()
    import adapter,preference_baseline as li
    from dimension_diagnostics import assert_matched_fixings
    summaries=[]
    for seed,case in cases.items():
        folder,marker=validate_checkpoint(case,out)
        model=li.load_fitted(adapter.training_case(case),folder)
        bank,bank_files=load_existing_bank(case)
        guard=dict(li_model_freeze_sha256=sha(freeze_path),checkpoint_sha256=sha(folder/"complete.json"),
            source_sha256=source_hashes(),existing_bank_file_sha256=bank_files,
            original_bank_predates_this_model=True,selection_used_only_training_prefix=True,budget_seconds=30.)
        rows=[];result_dir=Path(results)/case.case_id
        for i,(feature,cost,reference) in enumerate(zip(bank["features"],bank["costs"],bank["objectives"])):
            path=result_dir/f"test{i:03d}.json"
            if path.exists():
                row=read(path)
                if row["guard"]!=guard or row["index"]!=i or row.get("record_payload_sha256")!=hashlib.sha256(json.dumps({k:v for k,v in row.items() if k!="record_payload_sha256"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest():
                    raise ValueError("Prior same-bank result identity/payload changed")
                rows.append(row);continue
            begin=time.perf_counter();fixed,diagnostic=nn.propose_li_native_anchor(model,case,cost,feature,30.)
            remaining=30.-(time.perf_counter()-begin)
            if fixed is None or remaining<=0:raw=dict(valid=False,certified=False,status="no_native_anchor_or_budget");x=None
            else:
                assert_matched_fixings(case,fixed,target_K=int(case.affine["rank"]))
                solved=adapter.solve(case.model,cost,fixed=fixed,seed=seed+500000+i,time_limit=remaining)
                raw=solved.to_dict(include_solution=False);x=solved.x
            seconds=time.perf_counter()-begin
            original=bool(x is not None and adapter.validate_solution(case.model,x)["valid"] and runtime["uc_model"].validate_uc_solution(case.original_base,x)["valid"])
            actual_K=None if fixed is None else 48-len(fixed)
            restriction=bool(original and fixed is not None and all(abs(x[int(j)]-v)<=1e-6 for j,v in fixed.items()))
            valid=bool(original and restriction and seconds<=30. and actual_K==int(case.affine["rank"]))
            objective=float(cost@x) if valid else None;ratio=float(reference)/objective if objective is not None and objective>0 else None
            if ratio is not None and ratio>1.+adapter.recovery_tolerance(reference)/abs(reference):raise ValueError("Restricted Li solution beats certified Full optimum")
            row=clean(dict(guard=guard,method="Li",case_id=case.case_id,seed=seed,index=i,n_train=32,K=int(case.affine["rank"]),actual_K=actual_K,
                common_u=48,original_variables=192,original_integer_uvw=144,continuous_p=48,
                feature=feature,cost=cost,reference_objective=float(reference),objective=objective,
                objective_ratio=ratio,recovered=bool(valid and abs(objective-reference)<=adapter.recovery_tolerance(reference)),
                original_feasible=original,restriction_valid=restriction,valid=valid,timely=seconds<=30.,
                seconds=seconds,solution=x,fixed=fixed,raw=raw,diagnostic=diagnostic,
                phase="posthoc_baseline_on_existing_confirmation_bank",confirmation=True,
                original_confirmation_bank_generated_before_Li_fit=True))
            row["record_payload_sha256"]=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            write_new(path,row);rows.append(row)
        ratios=[r["objective_ratio"] for r in rows if r["objective_ratio"] is not None]
        summary=dict(method="Li",case_id=case.case_id,seed=seed,n_train=32,K=int(case.affine["rank"]),count=len(rows),
            recovered=sum(r["recovered"] for r in rows),recovery_percent=100.*sum(r["recovered"] for r in rows)/len(rows),
            objective_ratio_mean=float(np.mean(ratios)) if ratios else None,ratio_denominator=len(ratios),
            fit_seconds=marker["fit_seconds"],fit_cpu_seconds=marker["fit_cpu_seconds"],
            posthoc_existing_bank_baseline=True,no_test_parameter_selection=True)
        write_new(result_dir/"complete.json",dict(guard=guard,summary=summary,
            test_record_sha256={f"test{i:03d}.json":sha(result_dir/f"test{i:03d}.json") for i in range(len(rows))}))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    all_rows=[read(Path(results)/case.case_id/f"test{i:03d}.json") for case in cases.values() for i in range(120)]
    ratios=[r["objective_ratio"] for r in all_rows if r["objective_ratio"] is not None]
    write_new(Path(results)/"summary.json",dict(status="COMPLETE",method="Li",case_count=3,total=360,
        recovered=sum(r["recovered"] for r in all_rows),recovery_percent=100.*sum(r["recovered"] for r in all_rows)/360,
        objective_ratio_mean=float(np.mean(ratios)) if ratios else None,ratio_denominator=len(ratios),
        per_case=summaries,li_model_freeze_sha256=sha(freeze_path),posthoc_existing_bank_baseline=True,
        no_test_parameter_or_epoch_selection=True,ratio_aggregation="pooled arithmetic mean over valid raw returns"))


def main():
    sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",type=int,nargs="+",choices=SEEDS,default=list(SEEDS))
    parser.add_argument("--out",type=Path,default=ROOT/"run-v1")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--fit",action="store_true")
    parser.add_argument("--freeze-all",action="store_true")
    parser.add_argument("--evaluate-existing",action="store_true")
    args=parser.parse_args()
    if not any((args.validate_only,args.smoke,args.fit,args.freeze_all,args.evaluate_existing)):parser.error("Select an explicit operation")
    if args.smoke:
        if args.fit or args.evaluate_existing:parser.error("Smoke must be separate from fit/evaluation")
        smoke(args.seeds[0]);return
    if args.validate_only:
        for seed in args.seeds:
            case,_=load_case(seed);print(json.dumps(dict(seed=seed,n_train=case.n_train,K=case.affine["rank"],common_u=48)),flush=True)
    if args.fit:
        for seed in args.seeds:train_one(seed,args.out,args.device)
    if args.freeze_all:freeze_all(args.out)
    if args.evaluate_existing:evaluate_existing(args.out)


if __name__=="__main__":main()
