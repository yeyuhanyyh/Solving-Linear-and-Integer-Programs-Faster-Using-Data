"""Explicit fresh fitting on frozen grouped-knapsack labels; no test access."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,sys,time,pickle,platform,json,gc
import common

def sync(device):
    if device=='cuda':
        import torch
        torch.cuda.synchronize()

def fit_case(case,method,out,device,repetitions):
    import numpy as np,torch
    import svm_mlopt_components as components
    import train_uc_baselines as nn
    from threadpoolctl import threadpool_limits
    folder=out/f'seed{case.seed}'/method
    if (folder/'complete.json').exists():
        done=common.read(folder/'complete.json')
        for name,h in done['artifacts'].items():assert common.sha(folder/name)==h
        print(json.dumps(dict(event='already_complete',seed=case.seed,method=method)),flush=True);return
    if folder.exists():raise FileExistsError('Preserve incomplete prior fit: '+str(folder))
    folder.mkdir(parents=True)
    common.write(folder/'start.json',dict(seed=case.seed,method=method,n_train=36,K=32,device=device,source_manifest_sha256=common.sha(common.ROOT/'training_manifest.json')))
    fits=[];first=None;full_first=None
    count=repetitions if method in ('ours','svm','mlopt') else 1
    for repetition in range(count):
        gc.collect()
        with threadpool_limits(limits=1):
            sync(device);begin=time.perf_counter();cpu=time.process_time()
            if method=='ours':
                model=common.fit_exact(case);metadata=model.diagnostics
                sync(device);headline=time.perf_counter()-begin
                assert model.K==32
            elif method=='svm':
                import strict_svm
                model,metadata=strict_svm.fit(case.train_features,case.train_solutions,periods=1,units=96,seed=0)
                headline=metadata['fit_seconds']
            elif method=='mlopt':
                model,metadata=components.fit_mlopt(case,device='cpu')
                headline=time.perf_counter()-begin
            elif method=='neural_diving':
                import neural_instrumentation as instrument
                model=instrument.fit_nd_complete(case,device=device)
                metadata=model.training_metadata
                headline=metadata['fit_alignment']['headline_fit_seconds']
            elif method=='preference':
                import neural_instrumentation as instrument
                model=instrument.fit_preference(case,device=device,output_dir=folder)
                metadata=model.metadata
                headline=metadata['fit_alignment']['headline_fit_seconds']
            else:raise ValueError(method)
        sync(device);wall=time.perf_counter()-begin;cpu_elapsed=time.process_time()-cpu
        fit=dict(repetition=repetition,headline_fit_seconds=headline,full_fit_call_seconds=wall,cpu_seconds=cpu_elapsed,
                 model_metadata=metadata,shared_optimum_generation_included=False,test_solve_included=False,export_included=False)
        fits.append(fit)
        if repetition==0:first=model;full_first=metadata
        print(json.dumps(dict(event='fit_done',method=method,seed=case.seed,repetition=repetition,seconds=headline)),flush=True)
    # Every final serialization occurs after the above fit clocks stop.
    sync(device);save_start=time.perf_counter()
    if method in ('ours','svm','mlopt'):
        with (folder/'model.pkl').open('xb') as f:pickle.dump(first,f)
    elif method=='neural_diving':nn.save_nd_complete(first,folder/'complete-predictor.pt')
    else:first.save(folder)
    sync(device);export=time.perf_counter()-save_start
    common.write(folder/'fit.json',dict(seed=case.seed,n_train=36,D=96,K=32,method=method,device=device,
        repetitions=fits,mean_fit_seconds=float(np.mean([x['headline_fit_seconds'] for x in fits])),
        sample_sd_fit_seconds=float(np.std([x['headline_fit_seconds'] for x in fits],ddof=1)) if len(fits)>1 else None,
        final_export_seconds=export,original_training_fingerprint=case.dataset_fingerprint,
        primary_model_repetition=0,selection_and_checkpoint_uses_test=False))
    files={str(p.relative_to(folder)).replace('\\','/'):common.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    common.write(folder/'complete.json',dict(status='COMPLETE',seed=case.seed,method=method,n_train=36,D=96,K=32,
        device=device,artifacts=files,fit_headline_seconds=float(np.mean([x['headline_fit_seconds'] for x in fits]))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=('ours','svm','mlopt','neural_diving','preference'),required=True)
    p.add_argument('--seeds',nargs='+',type=int,choices=(1009,1013,1019),default=[1009,1013,1019])
    p.add_argument('--device',choices=('cpu','cuda'),required=True)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert a.repetitions>=1
    if any(m in ('ours','svm','mlopt') for m in a.methods) and a.device!='cpu':raise ValueError('All Ours/SVM/MLOPT fits use CPU')
    if any(m in ('neural_diving','preference') for m in a.methods) and a.device!='cuda':raise ValueError('The planned neural primary fits require explicit CUDA')
    common.verify();common.enable()
    import numpy,scipy,sklearn,sympy,highspy,torch
    import torch.optim,torch._dynamo
    from sympy import Matrix
    from sympy.polys.matrices import DomainMatrix
    import strict_svm,exact_affine_common,svm_mlopt_components,train_uc_baselines,preference_baseline,neural_diving_baseline
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:assert torch.get_num_interop_threads()==1
    # Match the shared MLOPT CPU worker's generic import/optimizer prewarm.
    # This is not fitted on any experimental training example.
    warm_net=torch.nn.Linear(2,2)
    warm_optimizer=torch.optim.Adam(warm_net.parameters(),lr=1e-3)
    warm_optimizer.zero_grad()
    warm_net(torch.ones((2,2))).sum().backward()
    warm_optimizer.step()
    del warm_net,warm_optimizer
    if a.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested; no CPU fallback')
        warm=torch.zeros(8,device='cuda');warm.add_(1);torch.cuda.synchronize();del warm
        affinity=None
    else:affinity=common.pin_cpu()
    a.out.mkdir(parents=True,exist_ok=True)
    runtime=dict(device=a.device,cpu_affinity=affinity,threads=1,python=sys.version,platform=platform.platform(),
        numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,
        sympy=sympy.__version__,highs=highspy.Highs().version(),cuda_name=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
        test_data_opened=False,source_manifest_sha256=common.sha(common.ROOT/'training_manifest.json'))
    manifestpath=a.out/('runtime-'+a.device+'.json')
    if not manifestpath.exists():common.write(manifestpath,runtime)
    for seed in a.seeds:
        case=common.load_case(seed)
        for method in a.methods:fit_case(case,method,a.out,a.device,a.repetitions)

if __name__=='__main__':main()
