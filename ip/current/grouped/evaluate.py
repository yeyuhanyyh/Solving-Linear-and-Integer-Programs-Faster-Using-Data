"""Local quality replay on the same360 held-out grouped-knapsack costs."""
from __future__ import annotations
from pathlib import Path
import argparse,pickle,time,json,math
import common

def load_model(case,method,root):
    folder=root/f'seed{case.seed}'/method
    done=common.read(folder/'complete.json')
    for name,h in done['artifacts'].items():assert common.sha(folder/name)==h,(method,name)
    if method in ('ours','svm','mlopt'):
        if method=='mlopt':
            import svm_mlopt_components
            svm_mlopt_components._mlopt_runtime()  # Register original trusted pickle module.
        with (folder/'model.pkl').open('rb') as f:return pickle.load(f)
    import train_uc_baselines as nn,adapter
    if method.startswith('neural_diving'):return nn.load_nd_complete(case,folder/'complete-predictor.pt')
    import preference_baseline
    return preference_baseline.load_fitted(adapter.training_case(case),folder)

def evaluate(case,method,model,out,budget):
    import numpy as np,adapter
    import train_uc_baselines as nn,svm_mlopt_components as components
    bank=common.load_test(case.seed)
    assert bank['features'].shape==(120,32) and bank['costs'].shape==(120,96)
    assert np.allclose(np.asarray(case.metadata['cost_offset'])+bank['features']@np.asarray(case.metadata['cost_feature_coefficients']).T,bank['costs'],rtol=0,atol=1e-12)
    rows=[];directory=out/f'seed{case.seed}'/method;directory.mkdir(parents=True,exist_ok=True)
    for i,(feature,cost,reference) in enumerate(zip(bank['features'],bank['costs'],bank['objectives'])):
        path=directory/f'test{i:03d}.json'
        if path.exists():rows.append(common.read(path));continue
        begin=time.perf_counter();fixed=None;raw=None;diagnostic={};actual=32
        if method=='ours':
            expanded,objective=model.extend(case.model,cost)
            raw=adapter.solve(expanded,objective,seed=case.seed+i,time_limit=budget)
            x=None if raw.x is None else raw.x[:96]
        elif method.startswith('neural_diving'):
            import neural_diving_baseline as nd
            record=nd.solve_matched_cost(model,case,cost,(case.seed+20261025+i*4099)%(2**31-1),K=32,total_time_limit=budget,solve_fn=adapter.solve)
            x=record.pop('x',None);fixed=record.get('best_fixed');diagnostic=record
        else:
            if method=='svm':
                eqs,diagnostic=model.propose(feature,target_K=32)
                assert all(e.j is None for e in eqs)
                fixed={int(e.i):int(e.rhs) for e in eqs}
            elif method=='mlopt':fixed,diagnostic=components.propose_mlopt(model,case,cost,feature)
            elif method=='preference':fixed,diagnostic=nn.propose_li_native_anchor(model,case,cost,feature,budget)
            else:raise ValueError(method)
            remaining=budget-(time.perf_counter()-begin)
            if fixed is not None and remaining>0:
                actual=96-len(fixed);assert actual==32
                raw=adapter.solve(case.model,cost,fixed=fixed,seed=case.seed+i,time_limit=remaining)
                x=raw.x
            else:x=None;actual=None
        original=bool(x is not None and adapter.validate_solution(case.model,x)['valid'])
        if method=='ours':
            restriction=bool(original and adapter.validate_solution(model.prepared_model,raw.x)['valid'])
        else:
            if fixed is not None:actual=96-len(fixed)
            restriction=bool(original and fixed is not None and actual==32 and all(abs(x[int(j)]-v)<=1e-6 for j,v in fixed.items()))
        rawvalue=None if not original else float(cost@x);elapsed=time.perf_counter()-begin
        valid=bool(original and restriction and elapsed<=budget)
        value=rawvalue if valid else None
        tol=adapter.recovery_tolerance(float(reference))
        if value is not None:assert value>=float(reference)-tol
        row=dict(seed=case.seed,method=method,index=i,n_train=36,D=96,target_K=32,actual_K=actual,
            valid=valid,original_feasible=original,restriction_valid=restriction,within_time_budget=elapsed<=budget,
            recovery=bool(valid and abs(value-float(reference))<=tol),reference=float(reference),objective=value,
            anchor=float(cost@case.train_solutions[0]),objective_ratio=None if value is None else value/float(reference),
            solution=x,fixed=fixed,elapsed_seconds=elapsed,budget_seconds=budget,diagnostic=diagnostic,
            test_cost_sha256=common.arrays_hash(feature,cost))
        common.write(path,row);rows.append(common.clean(row))
    valid=[r for r in rows if r['valid']]
    summary=dict(seed=case.seed,method=method,n_train=36,D=96,K=32,count=120,recovery_percent=100*sum(r['recovery'] for r in rows)/120,
        mean_objective_ratio=sum(r['objective_ratio'] for r in valid)/len(valid) if valid else None,valid_count=len(valid),
        relative_objective_ratio=math.fsum(r['anchor']-r['objective'] for r in rows)/math.fsum(r['anchor']-r['reference'] for r in rows) if len(valid)==120 else None)
    target=directory/'summary.json'
    if not target.exists():common.write(target,summary)
    return rows,summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=('ours','svm','mlopt','neural_diving','preference','neural_diving_continued'),required=True)
    p.add_argument('--seeds',nargs='+',type=int,choices=(1009,1013,1019),default=[1009,1013,1019])
    p.add_argument('--cpu-fit-root',type=Path,required=True);p.add_argument('--gpu-fit-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--budget',type=float,default=30.)
    a=p.parse_args();common.verify();common.enable()
    import numpy,torch,strict_svm,exact_affine_common,svm_mlopt_components
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:assert torch.get_num_interop_threads()==1
    affinity=common.pin_cpu()
    with threadpool_limits(limits=1):
        for seed in a.seeds:
            case=common.load_case(seed)
            for method in a.methods:
                root=a.cpu_fit_root if method in ('ours','svm','mlopt') else a.gpu_fit_root
                source=root/f'seed{seed}'/method
                checkpoint='model.pkl' if method in ('ours','svm','mlopt') else ('complete-predictor.pt' if method.startswith('neural_diving') else 'preference-network.pt')
                binding=dict(seed=seed,method=method,checkpoint_sha256=common.sha(source/checkpoint),
                    fit_complete_sha256=common.sha(source/'complete.json'),training_fingerprint=case.dataset_fingerprint,
                    test_manifest_sha256=common.sha(common.ROOT/'testing_manifest.json'),budget_seconds=a.budget)
                bindingpath=a.out/f'seed{seed}'/method/'model-binding.json'
                if bindingpath.exists():assert common.read(bindingpath)==binding,'Do not reuse evaluations for changed model or tests'
                else:common.write(bindingpath,binding)
                model=load_model(case,method,root)
                _,summary=evaluate(case,method,model,a.out,a.budget)
                print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
