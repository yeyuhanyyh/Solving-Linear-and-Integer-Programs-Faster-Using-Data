"""Evaluate all three frozen supplemental ND models on the existing UC bank.

The original bank was generated and used for the original three methods before
ND was added. This sidecar freezes ND before its first access to that bank and
does not claim that ND was frozen before the original bank generation.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('supplemental_uc_nd_training',ROOT/'train_uc_nd.py')
train=importlib.util.module_from_spec(spec);sys.modules[spec.name]=train;spec.loader.exec_module(train)
FREEZE=ROOT/'all-three-ND-sidecar-freeze.json'
RESULTS=ROOT/'results'

def sha(path):return train.sha(path)
def read(path):return train.read(path)
def payload(row):return hashlib.sha256(json.dumps(train.clean(row),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def validate_freeze():
    f=read(FREEZE)
    if f.get('model_count')!=3 or not f.get('bank_predates_ND_supplement'):raise ValueError('Need the three-model sidecar freeze')
    for p,digest in f['source_and_artifact_sha256'].items():
        if sha(p)!=digest:raise ValueError('Frozen ND sidecar/original artifact changed: '+p)
    cases={}
    for seed in train.SEEDS:
        case,folder,marker=train.validate(seed)
        if f['cases'][str(seed)]['guard']!=marker['guard']:raise ValueError('ND guard changed')
        cases[seed]=(case,folder)
    return f,cases
def freeze():
    if FREEZE.exists():return validate_freeze()[0]
    files={str(p.resolve()):sha(p) for p in (Path(__file__),ROOT/'train_uc_nd.py',ROOT/'continue_uc_nd.py',train.ORIGINAL_FREEZE)}
    cases={}
    for seed in train.SEEDS:
        case,folder,marker=train.validate(seed)
        files.update(marker['guard']['training_files']);files.update(marker['guard']['source_sha256'])
        for p in folder.iterdir():
            if p.is_file():files[str(p.resolve())]=sha(p)
        # The original bank is read only after all models have completed: its
        # bytes and original certificates are bound before prediction.
        bank=train.UC/'confirmation-v1/results-v1'/case.case_id/'confirmation.npz'
        cases[str(seed)]=dict(case_id=case.case_id,guard=marker['guard'],bank_path=str(bank.resolve()))
    # Only after the loop has validated all three final checkpoints.
    for row in cases.values():
        p=Path(row['bank_path']);files[str(p)]=sha(p)
        row['bank_sha256']=files[str(p)]
    f=dict(schema='UC-ND-three-model-existing-bank-sidecar-v1',model_count=3,bank_predates_ND_supplement=True,
        no_bank_used_in_training_or_selection=True,original_global_freeze_sha256=train.ORIGINAL_FREEZE_SHA,
        sources_frozen_before_ND_predictions=True,source_and_artifact_sha256=files,cases=cases)
    train.write(FREEZE,f);validate_freeze();return f
def evaluate(seed):
    f,cases=validate_freeze();case,folder=cases[seed];_,_,rt,nn,_,nd=train.enable()
    import adapter
    import dimension_diagnostics as dims
    model=nn.load_nd_complete(case,folder/'complete-predictor.pt')
    with np.load(f['cases'][str(seed)]['bank_path'],allow_pickle=False) as s:bank={k:s[k].copy() for k in s.files}
    if (bank['features'].shape,bank['costs'].shape,bank['objectives'].shape)!=((120,12),(120,192),(120,)):raise ValueError('Existing confirmation population changed')
    if not np.allclose(train.costs_from_features(case,bank['features']),bank['costs'],rtol=1e-14,atol=1e-10) or not np.all(bank['objectives']>0):raise ValueError('Existing UC cost/reference map changed')
    out=RESULTS/case.case_id;out.mkdir(parents=True,exist_ok=True);rows=[];K=int(case.affine['rank'])
    guard=dict(sidecar_freeze_sha256=sha(FREEZE),bank_sha256=f['cases'][str(seed)]['bank_sha256'],method='Neural Diving',n_train=32,common_u_D=48,K=K)
    for i,(cost,reference) in enumerate(zip(bank['costs'],bank['objectives'])):
        path=out/f'test{i:03d}.json'
        if path.exists():
            row=read(path);copy=dict(row);saved=copy.pop('record_payload_sha256')
            if payload(copy)!=saved or row['guard']!=guard or row['index']!=i:raise ValueError('Saved ND evaluation changed')
            rows.append(row);continue
        begin=time.perf_counter()
        raw=nd.solve_matched_cost(model,case,cost,seed=case.seed+i,K=K,total_time_limit=30.,solve_fn=adapter.solve)
        x=raw.pop('x',None);fixed=raw.get('best_fixed')
        original=bool(x is not None and rt['portable_adapter'].validate_solution(case.model,x)['valid'])
        physics=bool(original and rt['uc_model'].validate_uc_solution(case.original_base,x)['valid'])
        matching=bool(fixed is not None)
        if fixed is not None:dims.assert_matched_fixings(case,fixed,K)
        residual=max((abs(float(x[int(j)])-float(v)) for j,v in fixed.items()),default=0.) if original and fixed is not None else None
        raw_objective=float(cost@x) if original else None
        seconds=time.perf_counter()-begin;timely=seconds<=30.
        valid=bool(original and physics and matching and residual<=1e-6 and timely and raw_objective is not None and np.isfinite(raw_objective) and raw_objective>0)
        objective=raw_objective if valid else None;ratio=float(reference)/objective if valid else None
        tolerance=rt['portable_adapter'].recovery_tolerance(float(reference))
        if valid and objective<reference-tolerance:raise ValueError('ND beat a certified original optimum')
        row=dict(guard=guard,case_id=case.case_id,method='Neural Diving',index=i,seed=seed,n_train=32,K=K,actual_K=None if fixed is None else 48-len(fixed),D=192,common_u_D=48,
            feature=bank['features'][i],cost=cost,solution=x,fixed=fixed,raw=raw,original_feasible=original,physics_valid=physics,restriction_valid=bool(matching and residual is not None and residual<=1e-6),
            fixed_residual=residual,timely=timely,valid=valid,seconds=seconds,budget_seconds=30.,raw_objective=raw_objective,objective=objective,reference_objective=float(reference),objective_ratio=ratio,
            recovered=bool(valid and abs(objective-reference)<=tolerance),phase='existing-independent-confirmation-bank-ND-supplement',bank_predates_ND_supplement=True)
        row=train.clean(row);row['record_payload_sha256']=payload(row);train.write(path,row);rows.append(row)
    ratios=[x['objective_ratio'] for x in rows if x['objective_ratio'] is not None]
    summary=dict(case_id=case.case_id,method='Neural Diving',count=120,K=K,recovered=sum(x['recovered'] for x in rows),recovery_percent=100*sum(x['recovered'] for x in rows)/120,
        objective_ratio=float(np.mean(ratios)) if ratios else None,N=len(ratios),overbudget_count=sum(not x['timely'] for x in rows),bank_predates_ND_supplement=True)
    train.write(out/'complete.json',dict(guard=guard,summary=summary,test_record_sha256={f'test{i:03d}.json':sha(out/f'test{i:03d}.json') for i in range(120)}))
    print(json.dumps(summary),flush=True);return summary
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','evaluate'));p.add_argument('--seed',type=int,choices=train.SEEDS);a=p.parse_args()
    if a.command=='freeze':freeze();print(json.dumps(dict(status='THREE_ND_MODELS_FROZEN',sha256=sha(FREEZE))))
    else:
        if a.seed is None:p.error('Supply --seed')
        evaluate(a.seed)
if __name__=='__main__':main()
