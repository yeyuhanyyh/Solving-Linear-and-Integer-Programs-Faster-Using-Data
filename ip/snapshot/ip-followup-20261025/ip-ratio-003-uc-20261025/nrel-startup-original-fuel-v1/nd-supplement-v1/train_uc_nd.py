"""Training-only ND sidecar for the fixed n32 three-seed mixed UC cases.

The existing confirmation bank predates this supplemental training. This
module never opens it. Original complete-predictor architecture and training
rules are reused without changing their sources or reducing their schedules.
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
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parent
UC=ROOT.parent
FOLLOWUP=ROOT.parents[2]
CONFIRM=UC/'confirmation-v1/confirm_n32.py'
CONFIRM_SHA='5dfb457bfac7df3a1cef073a99a942a1d29e1985b3bef47e4d1c5db68650f8c1'
ORIGINAL_FREEZE=UC/'confirmation-v1/global-freeze.json'
ORIGINAL_FREEZE_SHA='a47236f3e39465c1b6053d2c84282facc51ab7e1a281af247735ca6c0979beb3'
RUNTIME=FOLLOWUP/'uc_bundle_frozen/runtime'
SEEDS=(101,211,307)
EXPECTED_K={101:14,211:14,307:17}
_C=_H=_RT=_NN=_DATA=_ND=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def imp(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def enable():
    global _C,_H,_RT,_NN,_DATA,_ND
    if sha(CONFIRM)!=CONFIRM_SHA or sha(ORIGINAL_FREEZE)!=ORIGINAL_FREEZE_SHA:raise ValueError('Original frozen UC loader/population changed')
    if _C is None:
        _C=imp(CONFIRM,'uc_nd_original_confirmation');_H,_RT=_C.load_helpers()
        sys.path.insert(0,str(RUNTIME));sys.path.insert(0,str(FOLLOWUP))
        import uc_adapters as data
        import train_uc_baselines as nn
        import neural_diving_baseline as nd
        import adapter
        import dimension_diagnostics
        _DATA,_NN,_ND=data,nn,nd
    return _C,_H,_RT,_NN,_DATA,_ND
def clean(value):return enable()[4].clean(value)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(clean(value),indent=2,allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw:raise FileExistsError(path)
        return
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if path.exists():raise FileExistsError(path)
    os.rename(temporary,path)
def costs_from_features(case,features):
    f=np.asarray(features,dtype=float);return np.asarray(case.metadata['nd_cost_offset'])+f@np.asarray(case.metadata['nd_cost_coefficients']).T
def load_case(seed):
    if seed not in SEEDS:raise ValueError('Only the fixed three UC seeds are permitted')
    c,h,rt,nn,data,nd=enable();case,_,_,offset,coeff,files=c.load_case(seed,h,rt)
    if case.n_train!=32 or case.model.n_variables!=192 or int(case.affine['rank'])!=EXPECTED_K[seed]:raise ValueError('Fixed UC cohort changed')
    if not np.array_equal(case.commitment_indices,np.arange(48)) or not np.array_equal(case.model.integrality,np.r_[np.ones(144),np.zeros(48)]):raise ValueError('Mixed UC integrality/common-u scope changed')
    if case.train_features.shape!=(32,12) or coeff.shape!=(192,12):raise ValueError('UC feature layout changed')
    case.metadata['baseline_seed']=int.from_bytes(hashlib.sha256((case.case_id+'|nd-uc-supplement-v1').encode()).digest()[:4],'little')%(2**31-1)
    case.metadata['nd_cost_offset']=offset;case.metadata['nd_cost_coefficients']=coeff
    for module in (data,rt['adapter'] if 'adapter' in rt else sys.modules['adapter'],rt['portable_adapter']):module.costs_from_features=costs_from_features
    if not np.allclose(costs_from_features(case,case.train_features),case.train_costs,rtol=1e-14,atol=1e-10):raise ValueError('Explicit affine cost map differs from frozen training costs')
    return case,files
def guard(case,files):
    _,h,rt,nn,data,nd=enable()
    sources={str(p.resolve()):sha(p) for p in [Path(__file__),CONFIRM,Path(nn.__file__),Path(data.__file__),Path(nd.__file__),RUNTIME/'nd_root_lp_features.py']}
    for name in ('adapter','portable_adapter','dimension_diagnostics'):
        module=sys.modules.get(name)
        if module is not None:sources[str(Path(module.__file__).resolve())]=sha(module.__file__)
    return dict(case_id=case.case_id,training_fingerprint=case.dataset_fingerprint,training_files=files,
        source_sha256=sources,original_global_freeze_sha256=ORIGINAL_FREEZE_SHA,n_train=32,common_u_D=48,K=int(case.affine['rank']),
        full_D=192,integer_D=144,continuous_D=48,cost_map='explicit offset + features @ coefficients.T',training_only=True,
        existing_bank_predates_supplement=True,no_existing_confirmation_outcome_used_for_selection=True)
def validate(seed):
    case,files=load_case(seed);folder=ROOT/'models'/case.case_id;marker=read(folder/'complete.json')
    if marker['guard']!=guard(case,files):raise ValueError('ND sidecar source/training binding changed')
    for name,digest in marker['artifact_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('ND checkpoint artifact changed')
    return case,folder,marker
def train(seed):
    case,files=load_case(seed);c,h,rt,nn,data,nd=enable();folder=ROOT/'models'/case.case_id
    if folder.exists():return validate(seed)[2]
    partial=folder.with_name(folder.name+'.partial-'+uuid.uuid4().hex);partial.mkdir(parents=True)
    g=guard(case,files);write(partial/'start.json',dict(guard=g,runtime=nn.runtime_metadata(),bank_read=False))
    print(json.dumps(dict(event='full_nd_fit_started',seed=seed,case_id=case.case_id,K=int(case.affine['rank']),n_train=32,complete_original_schedule=True)),flush=True)
    begin=time.perf_counter();cpu=time.process_time()
    try:
        model=nn.fit_nd_complete(case,device='cpu');fit=time.perf_counter()-begin;fit_cpu=time.process_time()-cpu
        nn.save_nd_complete(model,partial/'complete-predictor.pt')
        write(partial/'metadata.json',dict(method='Neural Diving',metadata=model.training_metadata,fit_seconds=fit,fit_cpu_seconds=fit_cpu,
            original_training_label_generation_excluded=True,bank_read=False,fit_scope='Both original widths, full sanity/selection and all32 refit',runtime=nn.runtime_metadata()))
        if guard(case,files)!=g:raise ValueError('Sources or original UC artifacts changed during training')
        artifacts={p.name:sha(p) for p in partial.iterdir() if p.is_file()}
        marker=dict(guard=g,artifact_sha256=artifacts,fit_seconds=fit,fit_cpu_seconds=fit_cpu,sanity_gate_passed=bool(model.training_metadata['sanity_gate']['passed']))
        write(partial/'complete.json',marker);os.rename(partial,folder);validate(seed)
        print(json.dumps(dict(event='full_nd_fit_complete',seed=seed,fit_seconds=fit,selected_width=model.config.hidden_size,parameter_count=model.training_metadata['parameter_count'])),flush=True)
        return marker
    except Exception as error:
        write(partial/'failure.json',dict(error=repr(error),elapsed_seconds=time.perf_counter()-begin,retained=True,bank_read=False));raise
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('smoke','train','validate'));p.add_argument('--seed',type=int,required=True,choices=SEEDS);a=p.parse_args()
    if a.command=='train':train(a.seed)
    elif a.command=='validate':case,folder,marker=validate(a.seed);print(json.dumps(dict(status='READY',seed=a.seed,fit_seconds=marker['fit_seconds'])))
    else:
        case,files=load_case(a.seed);print(json.dumps(dict(status='TRAINING_ONLY_LOAD_PASS',seed=a.seed,n=case.n_train,K=int(case.affine['rank']),rows=case.model.A.shape[0],D=case.model.n_variables,guard=guard(case,files))))
if __name__=='__main__':main()
