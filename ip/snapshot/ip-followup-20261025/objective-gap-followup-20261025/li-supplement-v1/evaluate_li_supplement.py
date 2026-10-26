"""Evaluate frozen later Li models on the existing shared360 stream2 costs.

No fitting, test-driven selection, new cost generation or original IP solve is
performed. Native Li top-k/Eq21 and the existing common-K proposer are reused.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import sympy as sym

ROOT=Path(__file__).resolve().parent
TRAINER=ROOT/'train_li_supplement.py'
LI_FREEZE=ROOT/'li-models-freeze.json'
STREAM2=ROOT.parent/'uniform-gross-stream2-training-budget-control'
CHECK=STREAM2/'independent-lightweight-check.json'
CHECK_SHA='e1404485b8b16641a020ef0481386392d0dbe1b4a204cd07dbfc6ca10a40d7f5'
MATH=ROOT.parent/'audit_uniform_gross_confirmation_v2.py'
MATH_SHA='b9493eb3dcb2f58da66924945b0be1f08fc9e17b50e6d127ebb394e22668588f'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def utc():return datetime.now(timezone.utc).isoformat()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def filemap(mapping):
    for path,expected in mapping.items():
        if sha(path)!=expected:raise ValueError('Frozen artifact changed: '+path)


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace');sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
    trainer=module(TRAINER,'later_li_frozen_trainer');d=trainer.enable()
    frozen=read(LI_FREEZE)
    if frozen.get('frozen') is not True or frozen.get('model_count')!=3:raise ValueError('All three Li models must be frozen')
    filemap(frozen['source_sha256'])
    for entry in frozen['cases'].values():filemap(entry['files']);filemap(entry['training_files'])
    if sha(CHECK)!=CHECK_SHA or sha(MATH)!=MATH_SHA:raise ValueError('Existing independent reference/math check changed')
    original=read(trainer.OLD_FREEZE);proof=read(CHECK)
    if proof['discrepancy_count']!=0 or proof['reference_count']!=360:raise ValueError('Shared references lack the existing independent check')
    pure=module(MATH,'later_li_independent_scoring')
    import adapter
    import preference_baseline as li
    import train_uc_baselines as nn
    started=utc();evaluator_sha=sha(__file__);all_ratios=[];all_scores=[];summaries=[];source={}
    d.write_new(ROOT/'evaluation-source-receipt.json',dict(evaluator_sha256=evaluator_sha,started_at_utc=started,
        li_model_freeze_sha256=sha(LI_FREEZE),original_stream2_freeze_sha256=sha(trainer.OLD_FREEZE),
        existing_360_reference_check_sha256=CHECK_SHA,models_frozen_before_Li_evaluation=True,
        Li_fitted_after_original_test_generation=True,test_used_for_selection=False))
    for case in trainer.cases():
        cid=case.case_id;entry=frozen['cases'][cid]
        folder,marker=nn.validate_checkpoint(ROOT,case,'li',trainer.GENERIC)
        if entry['guard']!=trainer.guard(case,'li'):raise ValueError('Li training guard changed')
        bank_folder=Path(original['bank_root'])/original['cases'][cid]['physical_id']
        meta=read(bank_folder/'metadata.json');npz=bank_folder/'confirmation.npz'
        for path in (bank_folder/'metadata.json',npz,bank_folder/'reference-certificates.json'):
            expected=proof['source_and_artifact_sha256'].get(str(path.resolve()))
            if expected is None or sha(path)!=expected:raise ValueError('Previously checked shared bank changed')
            source[str(path.resolve())]=expected
        with np.load(npz,allow_pickle=False) as stream:bank={key:stream[key].copy() for key in stream.files}
        if bank['features'].shape!=(120,32) or bank['costs'].shape!=(120,96):raise ValueError('Wrong shared bank shape')
        if not np.array_equal(d.signed_costs_from_features(case,bank['features']),bank['costs']):raise ValueError('Original gross costs changed')
        for cost,x,reference in zip(bank['costs'],bank['solutions'],bank['objectives']):
            if not adapter.validate_solution(case.model,x)['valid'] or abs(float(cost@x)-reference)>adapter.recovery_tolerance(reference):
                raise ValueError('Checked original reference no longer matches')
        model=li.load_fitted(adapter.training_case(case),folder)
        N=np.asarray([[sym.Rational(v) for v in row] for row in case.affine['normals_exact']],dtype=object)
        math_case=dict(G=32,D=96,K=32,A=case.model.A.toarray(),lower=case.model.row_lower,
            upper=case.model.row_upper,N=N,anchor=np.asarray([sym.Rational(v) for v in case.affine['anchor']],dtype=object))
        output=ROOT/'results'/cid/'li';output.mkdir(parents=True,exist_ok=True)
        guard=dict(case_id=cid,method_role='li',training=trainer.guard(case,'li'),
            li_model_freeze_sha256=sha(LI_FREEZE),original_stream2_freeze_sha256=sha(trainer.OLD_FREEZE),
            confirmation_bank_sha256=meta['npz_sha256'],model_role_files_sha256=entry['files'],
            evaluator_sha256=evaluator_sha,existing_reference_check_sha256=CHECK_SHA,budget=30.,
            chronology='Later baseline: fitted after stream2 generation; frozen before Li evaluation; training-only selection.')
        rows=[];scores=[]
        for i,(feature,cost,reference) in enumerate(zip(bank['features'],bank['costs'],bank['objectives'])):
            path=output/f'test{i:03d}.json'
            if path.exists():row=read(path)
            else:
                begin=time.perf_counter();fixed,diagnostic=nn.propose_li_native_anchor(model,case,cost,feature,30.)
                remaining=30.-(time.perf_counter()-begin)
                if fixed is None or remaining<=0:
                    raw=dict(valid=False,certified=False,status='no_native_anchor_or_prediction_budget_exhausted');x=None
                else:
                    solved=adapter.solve(case.model,cost,fixed=fixed,seed=case.seed+i,time_limit=remaining)
                    raw=solved.to_dict(include_solution=False);x=solved.x
                original_valid=bool(x is not None and adapter.validate_solution(case.model,x)['valid'])
                actual_K=None if fixed is None else 96-len(fixed)
                if fixed is not None and actual_K!=32:raise ValueError('Li returned a different common-coordinate K')
                restriction=bool(original_valid and fixed is not None and all(abs(x[int(j)]-v)<=1e-6 for j,v in fixed.items()))
                raw_objective=float(cost@x) if original_valid else None
                seconds=time.perf_counter()-begin;timely=seconds<=30.
                valid=bool(original_valid and restriction and timely and actual_K==32);objective=raw_objective if valid else None
                ratio=objective/float(reference) if valid else None
                row=dict(guard=guard,case_id=cid,method='li',index=i,n_train=36,K=32,actual_K=actual_K,D=96,
                    feature=feature,cost=cost,budget_seconds=30.,solution=x,fixed=fixed,seconds=seconds,
                    original_feasible=original_valid,restriction_valid=restriction,timely=timely,valid=valid,
                    raw_objective=raw_objective,objective=objective,reference_objective=float(reference),profit_ratio=ratio,
                    recovered=bool(valid and abs(objective-reference)<=adapter.recovery_tolerance(reference)),
                    raw=raw,diagnostics=diagnostic,experiment_phase='later_baseline_same_stream2',confirmation=False)
                row=d.clean(row);row['record_payload_sha256']=hashlib.sha256(pure.canonical(row)).hexdigest()
                d.write_new(path,row)
            pure.verify_payload(row,'later Li return')
            if row['guard']!=guard or row['index']!=i or not np.array_equal(row['cost'],cost) or not np.array_equal(row['feature'],feature):
                raise ValueError('Li row provenance changed')
            score=pure.score(math_case,'neural_diving',row,cost,feature,reference)
            for key in ('actual_K','original_feasible','restriction_valid','timely','valid','recovered'):
                if row[key]!=score[key]:raise ValueError('Independent returned status differs: '+key)
            for key in ('raw_objective','objective','reference_objective','profit_ratio'):
                pure.same_number(row[key],score[key],'Independent original objective/ratio differs')
            rows.append(row);scores.append(score);source[str(path.resolve())]=sha(path)
        good=[score for score in scores if score['valid']];ratios=[score['profit_ratio'] for score in good]
        summary=dict(case_id=cid,seed=case.seed,method='Li',G=32,n=36,D=96,K=32,total=120,
            recovered=sum(score['recovered'] for score in scores),recovery_percent=100.*sum(score['recovered'] for score in scores)/120,
            ratio_denominator=len(good),objective_ratio=math.fsum(ratios)/len(good) if good else None,
            canonical_completion_ratio=math.fsum(score['completion_ratio'] for score in good)/len(good) if good else None,
            invalid_count=120-len(good),overbudget_count=sum(not score['timely'] for score in scores),fit_seconds=marker['fit_seconds'])
        d.write_new(output/'complete.json',dict(guard=guard,summary=summary,
            test_record_sha256={f'test{i:03d}.json':sha(output/f'test{i:03d}.json') for i in range(120)}))
        summaries.append(summary);all_ratios+=ratios;all_scores+=scores
        print(json.dumps(dict(event='later_Li_case_complete',summary=summary)),flush=True)
    if len(all_scores)!=360:raise ValueError('Require every fixed seed/cost')
    good=[score for score in all_scores if score['valid']]
    pooled=dict(method='Li',G=32,n=36,D=96,K=32,total=360,recovered=sum(score['recovered'] for score in all_scores),
        recovery_percent=100.*sum(score['recovered'] for score in all_scores)/360,ratio_denominator=len(good),
        objective_ratio=math.fsum(all_ratios)/len(good) if good else None,
        canonical_completion_ratio=math.fsum(score['completion_ratio'] for score in good)/len(good) if good else None,
        invalid_count=360-len(good),overbudget_count=sum(not score['timely'] for score in all_scores))
    filemap(frozen['source_sha256'])
    for entry in frozen['cases'].values():filemap(entry['files'])
    d.write_new(ROOT/'results/pooled-quality.json',dict(status='COMPLETE_CHECKED_LATER_BASELINE',cells=[pooled],per_seed=summaries,
        evaluator_sha256=evaluator_sha,independent_math_sha256=MATH_SHA,source_sha256=source,
        li_model_freeze_sha256=sha(LI_FREEZE),original_stream2_model_freeze_sha256=sha(trainer.OLD_FREEZE),
        evaluation_started_at_utc=started,evaluation_completed_at_utc=utc(),no_new_costs_or_original_IP_references=True,
        chronology=frozen['chronology'],test_used_for_selection=False))
    print(json.dumps(dict(event='all360_later_Li_returns_checked',summary=pooled)),flush=True)


if __name__=='__main__':main()
