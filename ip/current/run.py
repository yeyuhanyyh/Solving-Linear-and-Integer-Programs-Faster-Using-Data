"""Portable fitting and evaluation for the current three IP settings.

Use a new process for each problem/method because the frozen implementations
use the same module names. No training happens on import or during verify.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, importlib.util, json, os, pickle, sys, time

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
METHODS=('full','ours','svm','mlopt','neural-diving','preference-learning')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def write(path,obj):
    import numpy as np
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,(np.ndarray,np.generic)):return clean(v.tolist())
        if isinstance(v,Path):return str(v)
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    with path.open('x',encoding='utf-8') as f:json.dump(clean(obj),f,indent=2,allow_nan=False)

def setup(args):
    import numpy as np, torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # The guard checks every source and input before any pickle is opened.
    from verify import check_manifest
    check_manifest()
    sys.path[:0]=[str(ROOT/'source'),str(ROOT/'source/strict')]
    if args.problem=='grouped':
        gc=module('current_grouped_common',ROOT/'grouped/common.py')
        sys.modules['common']=gc
        gc.enable();case=gc.load_case(args.seed);val=None
        import adapter
        rt=SimpleNamespace(adapter=adapter,grouped=gc)
    else:
        import mlopt_common as c
        rt=c.setup('uc' if args.problem=='uc' else 'correlated')
        condition='case14-n512' if args.problem=='uc' else f'knapsack-{args.seed}-n{args.n}'
        case,val,spec=c.load_condition(condition,rt)
    if args.method in ('neural-diving','preference-learning'):
        if args.problem=='uc':
            sys.path.insert(0,str(ROOT/'source/uc/baselines/neural'))
            import neural_api
            rt.neural=neural_api
            if args.method=='preference-learning':
                import tie_api
                rt.tie=tie_api
            else:
                import nd_gate_training
                rt.nd_gate=nd_gate_training
        elif args.problem=='correlated':
            rt.neural=module('current_correlated_neural',ROOT/'source/correlated/neural/baselines.py')
            rt.repair=module('current_correlated_repair',ROOT/'source/correlated/repaired_nd.py')
        else:
            sys.path.insert(0,str(ROOT/'grouped'))
            import neural_instrumentation,train_uc_baselines
            rt.neural=neural_instrumentation;rt.group_neural=train_uc_baselines
    return case,val,rt

def fit(args,case,val,rt,output):
    import numpy as np,torch
    # Match the recorded clocks: library initialization precedes fitting.
    import exact_affine_common,strict_svm
    from sympy import Matrix
    from sympy.polys.matrices import DomainMatrix
    if args.method=='mlopt':
        dummy=torch.nn.Linear(2,2);opt=torch.optim.Adam(dummy.parameters())
        opt.zero_grad();dummy(torch.ones((2,2))).sum().backward();opt.step()
        del dummy,opt
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Select --device cpu explicitly for a CPU rerun.')
    if args.method in ('ours','svm','mlopt') and args.device!='cpu':
        raise ValueError('Current Ours/SVM/MLOPT fitting uses CPU')
    if args.device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    method=args.method.replace('-','_')
    if method=='ours':
        import exact_affine_common
        model=exact_affine_common.fit_exact(case);seconds=time.perf_counter()-start
        if model.K!=case.affine['rank']:raise ValueError('Learned rank differs from the frozen setting')
        details=model.diagnostics
    elif method=='svm':
        import strict_svm
        labels=np.rint(case.train_solutions[:,case.commitment_indices]).astype(int)
        model,details=strict_svm.fit(case.train_features,labels,
            periods=36 if args.problem=='uc' else 1,
            units=5 if args.problem=='uc' else len(case.commitment_indices),
            seed=42001 if args.problem=='uc' else 0)
        seconds=details['fit_seconds']
    elif method=='mlopt':
        if args.problem=='uc':
            model=rt.api.fit(case,val,config={'seed':42001,'device':'cpu'});details=model.manifest
        elif args.problem=='correlated':
            model,details=rt.api.fit(case.model.A.toarray()[0],case.model.row_upper[0],case.train_features,
                case.train_solutions,case_id=case.case_id,device='cpu',seed=case.seed)
        else:
            import svm_mlopt_components
            model,details=svm_mlopt_components.fit_mlopt(case,device='cpu')
        seconds=time.perf_counter()-start
    elif args.problem=='uc':
        model=(rt.tie.fit_tie(rt.neural,case,val,output/'network',args.device)
            if method=='preference_learning' else rt.nd_gate.fit_nd_with_gate(rt.neural,case,val,output/'network',args.device))
        details=model.metadata['raw_training_metadata']
        if method=='neural_diving':
            excluded=sum(r['seconds'] for trial in details.get('width_candidates',[])
                for r in trial.get('matched_validation',{}).get('per_cost',[]))
            seconds=details['epoch_selection_seconds']+details['final_refit_seconds']-excluded
        else:
            seconds=details['fit_seconds']-details.get('development_decode_statistics',{}).get('seconds',0)-details.get('final_decode_statistics',{}).get('seconds',0)
    elif args.problem=='correlated':
        if method=='neural_diving':
            model,details=rt.repair.fit(case,output/'network',ROOT/'source/correlated',device=args.device)
            seconds=details['epoch_selection_seconds']+details['final_refit_seconds']
        else:
            model,details=rt.neural.fit_neural(method,case,output/'network',device=args.device)
            seconds=details['fit_seconds']
    else:
        if method=='neural_diving':
            model=rt.neural.fit_nd_complete(case,device=args.device);details=model.training_metadata
        else:
            model=rt.neural.fit_preference(case,device=args.device,output_dir=output/'network');details=model.metadata
        seconds=details['fit_alignment']['headline_fit_seconds']
        if method=='neural_diving':
            (output/'network').mkdir(exist_ok=True)
            rt.group_neural.save_nd_complete(model,output/'network/complete-predictor.pt')
        else:model.save(output/'network')
    if method in ('ours','svm','mlopt'):
        with (output/'model.pkl').open('xb') as f:pickle.dump(model,f)
    write(output/'fit.json',dict(problem=args.problem,method=args.method,n=case.n_train,seed=args.seed,
        K=case.affine['rank'],device=args.device,learning_fit_seconds=seconds,
        shared_training_IP_solves_included=False,details=details))
    return model

def load(args,case,rt):
    path=args.checkpoint.resolve()
    info=json.loads((path/'fit.json').read_text(encoding='utf-8'))
    for name,want in [('problem',args.problem),('method',args.method),('n',case.n_train),('seed',args.seed),('K',case.affine['rank'])]:
        if info[name]!=want:raise ValueError('Checkpoint setting differs: '+name)
    if args.method in ('ours','svm','mlopt'):
        if args.method=='ours':import exact_affine_common
        elif args.method=='svm':import strict_svm
        elif args.problem=='grouped':
            import svm_mlopt_components
            svm_mlopt_components._mlopt_runtime()
        with (path/'model.pkl').open('rb') as f:return pickle.load(f)
    method=args.method.replace('-','_')
    if args.problem=='uc':
        return rt.tie.load_tie(rt.neural,case,path/'network') if method=='preference_learning' else rt.neural.load(case,method,path/'network')
    if args.problem=='correlated':
        if method=='neural_diving':return rt.repair.load(case,path/'network',ROOT/'source/correlated')
        return rt.neural.load_neural(method,case,path/'network')
    if method=='neural_diving':return rt.group_neural.load_nd_complete(case,path/'network/complete-predictor.pt')
    import preference_baseline,adapter
    return preference_baseline.load_fitted(adapter.training_case(case),path/'network')

def tests(args,rt):
    import numpy as np
    def npz(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    if args.problem=='uc':
        bank=npz(ROOT/'testing/uc-confirmation.npz')
        return bank['features'],bank['costs'],bank['objectives']
    if args.problem=='grouped':
        bank=rt.grouped.load_test(args.seed)
        return bank['features'],bank['costs'],bank['objectives']
    bank=npz(ROOT/f'testing/random-{args.seed}.npz')
    refs=npz(ROOT/f'testing/random-{args.seed}-references.npz')
    return bank['profits'].astype(float),-bank['profits'].astype(float),-refs['objectives'].astype(float)

def evaluate(args,case,rt,model,output):
    import numpy as np,scipy.sparse as sp
    adapter=rt.adapter;K=int(case.affine['rank']);common=case.commitment_indices
    budget=20. if args.problem=='correlated' else 30.
    features,costs,refs=tests(args,rt)
    for i,(feature,cost,ref) in enumerate(zip(features[:args.limit],costs[:args.limit],refs[:args.limit])):
        start=time.perf_counter();fixed=None;diagnostic={}
        if args.method=='full':
            answer=adapter.solve(case.model,cost,time_limit=budget,seed=0);x=answer.x
        elif args.method=='ours':
            extended,objective=model.extend(case.model,cost)
            answer=adapter.solve(extended,objective,time_limit=budget,seed=0)
            x=None if answer.x is None else answer.x[:len(cost)]
        elif args.method=='svm':
            equations,diagnostic=model.propose(feature,target_K=K)
            ri=[];ci=[];data=[];rhs=[]
            for j,e in enumerate(equations):
                ri.append(j);ci.append(int(common[e.i]));data.append(1.)
                if e.j is not None:ri.append(j);ci.append(int(common[e.j]));data.append(-1.)
                rhs.append(float(e.rhs))
            matrix=sp.csr_matrix((data,(ri,ci)),shape=(len(equations),len(cost)))
            restricted=adapter.LinearModel(sp.vstack([case.model.A,matrix],format='csr'),
                np.r_[case.model.row_lower,rhs],np.r_[case.model.row_upper,rhs],
                case.model.col_lower,case.model.col_upper,case.model.integrality)
            answer=adapter.solve(restricted,cost,time_limit=max(.001,budget-(time.perf_counter()-start)),seed=0);x=answer.x
        elif args.problem=='uc' and args.method in ('mlopt','preference-learning'):
            answer=rt.matched.solve_matched_candidates(model,case,cost,feature,
                'mlopt' if args.method=='mlopt' else 'preference_learning',target_K=K,time_limit=budget,seed=0,
                solve_restriction=lambda c,**kw:adapter.solve(case.model,c,**kw),
                validate_original=lambda x,c:dict(adapter.validate_solution(case.model,x),objective_recomputed=float(c@x)))
            x=answer['x'];fixed=answer['fixed'];diagnostic=answer['diagnostic']
        elif args.problem=='uc':
            answer=model.solve(cost,feature,target_K=K,time_limit=budget,seed=0)
            x=answer['x'];fixed=answer['fixed'];diagnostic=answer.get('diagnostic',{})
        else:
            if args.method=='mlopt':
                if args.problem=='correlated':fixed,diagnostic=rt.api.propose_fixed(model,feature,K)
                else:
                    import svm_mlopt_components
                    fixed,diagnostic=svm_mlopt_components.propose_mlopt(model,case,cost,feature)
            elif args.problem=='correlated':
                fixed,diagnostic=rt.neural.predict_fixed(args.method.replace('-','_'),model,case,cost,feature,K=K,total_time_limit=budget)
            elif args.method=='neural-diving':
                import neural_diving_baseline
                answer=neural_diving_baseline.solve_matched_cost(model,case,cost,(case.seed+20261025+i*4099)%(2**31-1),K=K,total_time_limit=budget,solve_fn=adapter.solve)
                x=answer.pop('x',None);fixed=answer.get('best_fixed');diagnostic=answer
            else:
                fixed,diagnostic=rt.group_neural.propose_li_native_anchor(model,case,cost,feature,budget)
            if not (args.problem=='grouped' and args.method=='neural-diving'):
                remaining=budget-(time.perf_counter()-start)
                if fixed is None or remaining<=0:x=None
                else:
                    if len(fixed)!=len(common)-K:raise ValueError('Unmatched dimension')
                    answer=adapter.solve(case.model,cost,fixed=fixed,time_limit=remaining,seed=0);x=answer.x
        elapsed=time.perf_counter()-start
        valid=x is not None and adapter.validate_solution(case.model,x)['valid'] and elapsed<=budget
        value=None if not valid else float(cost@x)
        write(output/f'test{i:03d}.json',dict(index=i,valid=bool(valid),K=K,reference=float(ref),
            anchor=float(cost@case.train_solutions[0]),value=value,
            recovery=bool(valid and abs(value-ref)<=1e-6+1e-9*abs(ref)),seconds=elapsed,solution=x,
            fixed=fixed,diagnostic=diagnostic,objective_orientation='minimization; knapsack profits are negated'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('verify','check-data','train','evaluate','smoke'))
    p.add_argument('--problem',choices=('uc','grouped','correlated'),default='uc')
    p.add_argument('--method',choices=METHODS,default='ours')
    p.add_argument('--seed',type=int)
    p.add_argument('--n',type=int)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--limit',type=int,default=1)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.action=='verify':
        from verify import verify
        print(json.dumps(verify(args.output),indent=2));return
    args.seed=args.seed if args.seed is not None else {'uc':86001,'grouped':1009,'correlated':41001}[args.problem]
    args.n=args.n if args.n is not None else {'uc':512,'grouped':36,'correlated':32}[args.problem]
    if args.problem=='uc' and (args.seed!=86001 or args.n!=512):p.error('UC uses seed86001 and n512')
    if args.problem=='grouped' and (args.seed not in (1009,1013,1019) or args.n!=36):p.error('Grouped uses seeds1009/1013/1019 and n36')
    if args.problem=='correlated' and (args.seed not in (41001,41002,41003) or args.n not in (32,64,128,256)):p.error('Invalid correlated setting')
    if args.limit<1:p.error('--limit must be positive')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        case,val,rt=setup(args)
        if args.action=='check-data':
            print(json.dumps(dict(problem=args.problem,n=case.n_train,D=len(case.commitment_indices),K=case.affine['rank'],
                variables=case.model.A.shape[1],constraints=case.model.A.shape[0])));return
        if args.output is None:p.error('Choose a new --output directory')
        output=args.output.resolve()
        if output.is_relative_to(ROOT):p.error('Outputs must be outside the immutable current/ archive')
        output.mkdir(parents=True,exist_ok=False)
        if args.method=='full':model=None
        elif args.action in ('train','smoke'):model=fit(args,case,val,rt,output)
        else:
            if args.checkpoint is None:p.error('Use --checkpoint with a freshly trained run directory')
            model=load(args,case,rt)
        if args.action in ('evaluate','smoke'):evaluate(args,case,rt,model,output)
        print(json.dumps(dict(status='COMPLETE',output=str(output),action=args.action)))

if __name__=='__main__':main()
