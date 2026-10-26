"""Fixed all-three-seed 2000-update control, decided from training error only.

This module never reads the existing confirmation bank. Preserve original ND
models, reuse all32 certified labels, reset Adam, publish only update2000.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('uc_nd_fixed_control_training',ROOT/'train_uc_nd.py')
train=importlib.util.module_from_spec(spec);sys.modules[spec.name]=train;spec.loader.exec_module(train)
def validate(seed):
    case,original,marker=train.validate(seed);folder=ROOT/'continued-models'/case.case_id;complete=train.read(folder/'complete.json')
    if complete['guard']!=dict(original_guard=marker['guard'],original_complete_sha256=train.sha(original/'complete.json'),original_weights_sha256=train.sha(original/'complete-predictor.pt'),control_source_sha256=train.sha(__file__),steps=2000):raise ValueError('Fixed UC continuation binding changed')
    if complete['optimizer_steps']!=2000:raise ValueError('Incomplete fixed control')
    for name,digest in complete['artifact_sha256'].items():
        if train.sha(folder/name)!=digest:raise ValueError('Continued model artifact changed')
    return case,folder,complete
def run(seed):
    case,original,marker=train.validate(seed);_,_,_,nn,data,nd=train.enable();import torch
    final=ROOT/'continued-models'/case.case_id
    if final.exists():return validate(seed)[2]
    partial=final.with_name(final.name+'.partial-'+uuid.uuid4().hex);partial.mkdir(parents=True)
    guard=dict(original_guard=marker['guard'],original_complete_sha256=train.sha(original/'complete.json'),original_weights_sha256=train.sha(original/'complete-predictor.pt'),control_source_sha256=train.sha(__file__),steps=2000)
    train.write(partial/'start.json',dict(guard=guard,training_only=True,bank_read=False,decision='All three seeds fixed before any ND confirmation prediction'))
    model=nn.load_nd_complete(case,original/'complete-predictor.pt');config=model.config;nd._setup(config)
    begin=time.perf_counter();cpu=time.process_time()
    try:
        costs,pools,pool_metadata=nd._prepare_pools(case,data.singleton_pools(case),model.graph.integer_indices)
        model.graph.precompute_training_root_lp(costs);indices=np.arange(32)
        optimizer=nd._optimizer(model.full_predictor,config)
        if optimizer.state or config.batch_size!=16 or config.gradient_clip!=5.:raise ValueError('Original optimizer/batch conventions changed')
        batch_seed=int.from_bytes(hashlib.sha256((case.case_id+'|UC-fixed-extra2000').encode()).digest()[:4],'little')
        rng=np.random.default_rng(batch_seed);order=np.empty(0,dtype=int);cursor=0;steps=[]
        trajectory=[dict(step=0,diagnostics=nd._full_diagnostics(model.full_predictor,costs,pools,indices))]
        for step in range(1,2001):
            if cursor>=len(order):order=rng.permutation(indices);cursor=0
            chosen=order[cursor:cursor+16];cursor+=len(chosen);lr=.001 if step<=1000 else .0001
            for group in optimizer.param_groups:group['lr']=lr
            model.full_predictor.train();optimizer.zero_grad(set_to_none=True)
            logits,_=model.full_predictor(costs[chosen]);loss=nd._full_loss(logits,chosen,pools)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training NLL')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.full_predictor.parameters(),5.)
            if not bool(torch.isfinite(norm)):raise ValueError('Nonfinite training gradient')
            optimizer.step();steps.append(dict(step=step,batch_indices=chosen.tolist(),learning_rate=lr,NLL=float(loss.detach())))
            if step%100==0:
                diagnostic=nd._full_diagnostics(model.full_predictor,costs,pools,indices);trajectory.append(dict(step=step,diagnostics=diagnostic))
                print(json.dumps(dict(event='UC_fixed_continuation',seed=seed,step=step,diagnostics=diagnostic)),flush=True)
        model.full_predictor.eval();metadata=dict(config=asdict(config),all_training_count=32,optimizer_steps=2000,final_checkpoint_step=2000,
            original_selected_width=config.hidden_size,parameter_count=model.training_metadata['parameter_count'],optimizer_state_reset=True,
            batch_seed=batch_seed,batching='original shuffled32 labels, batches16/16',learning_rate_schedule='1..1000:.001;1001..2000:.0001',
            trajectory=trajectory,updates=steps,pool_metadata=pool_metadata,training_only=True,bank_read=False,selection='fixed final update2000 only',
            wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu)
        model.training_metadata=dict(model.training_metadata,fixed_UC_extra_optimization_budget=metadata)
        nn.save_nd_complete(model,partial/'continued-predictor.pt');train.write(partial/'metadata.json',metadata)
        if train.sha(original/'complete.json')!=guard['original_complete_sha256'] or train.sha(original/'complete-predictor.pt')!=guard['original_weights_sha256']:raise ValueError('Original checkpoint changed')
        artifacts={p.name:train.sha(p) for p in partial.iterdir() if p.is_file()}
        complete=dict(guard=guard,artifact_sha256=artifacts,optimizer_steps=2000,training_only=True,bank_read=False)
        train.write(partial/'complete.json',complete);os.rename(partial,final);validate(seed)
        print(json.dumps(dict(event='UC_fixed_control_complete',seed=seed,wall_seconds=metadata['wall_seconds'],diagnostics=trajectory[-1]['diagnostics'])),flush=True)
        return complete
    except Exception as error:
        train.write(partial/'failure.json',dict(error=repr(error),retained=True,bank_read=False));raise
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('train','validate'));p.add_argument('--seed',required=True,type=int,choices=train.SEEDS);a=p.parse_args()
    if a.command=='train':run(a.seed)
    else:validate(a.seed);print(json.dumps(dict(status='FIXED2000_READY',seed=a.seed)))
if __name__=='__main__':main()
