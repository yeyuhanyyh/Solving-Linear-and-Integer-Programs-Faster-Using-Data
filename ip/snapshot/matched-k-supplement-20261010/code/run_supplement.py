"""Frozen, training-only supplemental learners followed by fresh serial tests."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse, hashlib, json, pickle, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v

def write_new(path,v):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(clean(v),f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())

def stamp(): return datetime.now(timezone.utc).isoformat()

def check_deadline(deadline):
    if deadline and datetime.now(timezone.utc).timestamp()+31 >= deadline:
        raise TimeoutError('Global compute cutoff reached; retained records are partial, not complete.')

def source_binding():
    paths=[HERE/'protocol.json']+sorted(p for p in (HERE/'code').glob('*.py') if not p.name.startswith('test_'))
    return {str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in paths}

def checked_runtime(common,protocol):
    from importlib.metadata import version
    runtime=common.runtime()
    aliases={'scikit-learn':'sklearn','highspy':'highs'}
    for package,expected in protocol['runtime_versions'].items():
        actual=(version(package) if package=='threadpoolctl' else runtime['versions'][aliases.get(package,package)])
        if str(actual).split('+',1)[0]!=expected:
            raise RuntimeError(f'Frozen runtime mismatch for {package}: expected {expected}, got {actual}')
    return runtime

def setup(parent):
    parent=Path(parent).resolve()
    sys.path.insert(0,str(parent/'code')); sys.path.insert(0,str(parent/'code/controls'))
    import adapter, runner_common, run_benchmark
    if runner_common.ROOT.resolve()!=parent or adapter.EXPERIMENT_ROOT.resolve()!=parent:
        raise RuntimeError('Parent import resolved outside the exact staged root')
    protocol=read(HERE/'protocol.json')
    checked_runtime(runner_common,protocol)
    frozen=runner_common.freeze(create=False)
    if frozen['fingerprint']!=protocol['parent_freeze_fingerprint']:
        raise RuntimeError('Original parent source/configuration differs from the approved freeze')
    return parent,adapter,runner_common,run_benchmark,protocol,frozen

def train(args):
    parent,adapter,common,bench,protocol,frozen=setup(args.parent)
    from svm_match import fit_coordinate_svm
    ids=adapter.list_case_ids()
    if args.cases: ids=args.cases
    bindings=source_binding(); model_hashes={}; parent_model_markers={}
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    for cid in ids:
        check_deadline(args.deadline)
        case=adapter.load_case(cid)
        # Verify all five parent checkpoint bindings without changing them.
        for key in common.MODEL_KEYS:
            if common.checkpoint(case,key,frozen) is None: raise RuntimeError('Parent checkpoint missing')
            rel=f'models/{cid}/{key}/complete.json'
            parent_model_markers[rel]=sha(parent/rel)
        folder=out/'models'/cid; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists(): raise RuntimeError('Refusing to overwrite a retained supplemental model')
        model,metadata=fit_coordinate_svm(case)
        with (folder/'coordinate-svm.pkl').open('xb') as f: pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
        write_new(folder/'metadata.json',metadata)
        marker={'case':cid,'parent_freeze':frozen['fingerprint'],'source_binding':bindings,
                'train_count':120,'test_information_used':False,'completed_utc':stamp(),
                'artifacts':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
        write_new(folder/'complete.json',marker)
        model_hashes[cid]={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in folder.iterdir() if p.is_file()}
        print(json.dumps({'event':'coordinate_svm_complete','case':cid,'utc':stamp()},allow_nan=False),flush=True)
    freeze={'version':protocol['version'],'parent_freeze':frozen['fingerprint'],
            'parent_training_freeze_sha256':sha(parent/'models/training-freeze.json'),
            'parent_model_marker_sha256':parent_model_markers,
            'sources':bindings,'cases':ids,'models':model_hashes,'completed_utc':stamp(),
            'test_accessed':False,'runtime':checked_runtime(common,protocol)}
    write_new(out/'supplement-freeze.json',freeze)

def validate_freeze(out,parent,protocol,frozen):
    f=read(out/'supplement-freeze.json')
    if f['parent_freeze']!=frozen['fingerprint'] or f['sources']!=source_binding() or f['test_accessed']:
        raise RuntimeError('Supplemental source or parent freeze changed after fitting')
    if sha(parent/'models/training-freeze.json')!=f['parent_training_freeze_sha256']:
        raise RuntimeError('Parent training freeze changed')
    expected_markers={f'models/{cid}/{key}/complete.json' for cid in f['cases'] for key in ('ours','svm_native','mlopt_native','neural_diving','li_bounds_residual')}
    if set(f['parent_model_marker_sha256'])!=expected_markers: raise RuntimeError('Parent model bindings incomplete')
    for rel,expected in f['parent_model_marker_sha256'].items():
        if sha(parent/rel)!=expected: raise RuntimeError('Original fitted model marker changed')
        marker=read(parent/rel)
        for name,value in marker['artifact_sha256'].items():
            if sha((parent/rel).parent/name)!=value: raise RuntimeError('Original model artifact changed')
    for files in f['models'].values():
        for rel,expected in files.items():
            p=(out/rel).resolve()
            if not p.is_relative_to(out) or sha(p)!=expected: raise RuntimeError('Supplement model artifact changed')
    if set(f['cases'])!=set(f['models']): raise RuntimeError('Supplement model cohort incomplete')
    return f

def new_method(method,models,coordinate,case,cost,features,solver_seed,adapter,protocol):
    import experiment_core
    from dimension_diagnostics import affine_equalities_metrics,assert_matched_fixings
    from svm_match import propose_capped_equalities,propose_coordinate_fixings
    from strategy_match import propose_strategy_fixings
    start=time.perf_counter(); eq=[]; fixed=None
    if method=='svm_rank_capped':
        eq,diagnostic=propose_capped_equalities(models['svm_native'],case,features)
    elif method=='svm_coordinate_matched':
        fixed,diagnostic=propose_coordinate_fixings(coordinate,case,features)
    elif method in ('mlopt_guided_matched','li_guided_matched'):
        key='mlopt_native' if method=='mlopt_guided_matched' else 'li_bounds_residual'
        fixed,diagnostic=propose_strategy_fixings(models[key],case,features,'mlopt' if method=='mlopt_guided_matched' else 'li')
    else: raise ValueError('Unknown supplemental method')
    prediction=time.perf_counter()-start
    if fixed is None:
        raw,x=experiment_core.solve(case.original_base,cost,seed=solver_seed,time_limit=max(1e-6,30-prediction),equalities=eq)
        if x is None: residual=None
        else: residual=max((abs(float(e.as_row(len(case.commitment_indices))[0]@np.asarray(x)[case.commitment_indices])-e.rhs) for e in eq),default=0.)
        restriction_valid=x is not None and residual<=adapter.FEASIBILITY_TOLERANCE
    else:
        solved=adapter.solve(case.model,cost,fixed=fixed,seed=solver_seed,time_limit=max(1e-6,30-prediction))
        x=solved.x; raw=solved.to_dict(include_solution=False)
        residual=None if x is None else max((abs(float(x[j])-v) for j,v in fixed.items()),default=0.)
        restriction_valid=x is not None and residual<=adapter.FEASIBILITY_TOLERANCE
    original=adapter.validate_solution(case.model,x) if x is not None else {'valid':False}
    valid=bool(raw.get('valid',False) and original['valid'] and restriction_valid)
    seconds=time.perf_counter()-start; valid=valid and seconds<=30
    physics=None
    if case.family=='uc' and x is not None:
        from uc_model import validate_uc_solution
        physics=validate_uc_solution(case.original_base,x)
        if bool(physics['valid'])!=bool(original['valid']): raise RuntimeError('Independent UC physics disagrees with original constraints')
    # Required coordinate/rank selection is inside the timer. Independent recheck is outside.
    if fixed is None:
        pairs=[e.as_row(len(case.commitment_indices)) for e in eq]
        dimensions=affine_equalities_metrics(np.asarray([r for r,b in pairs],float).reshape(-1,len(case.commitment_indices)),np.asarray([b for r,b in pairs],float))
        if dimensions['affine_dimension']!=diagnostic['actual_common_dimension']: raise RuntimeError('Independent SVM rank mismatch')
    else: dimensions=assert_matched_fixings(case,fixed)
    return clean({'method':method,'seconds':seconds,'prediction_hint_seconds':prediction,
        'raw_method_result':raw,'solution':x,'objective':None if x is None else float(cost@x),
        'physics':physics,
        'original_feasible':original['valid'],'valid':valid,'original_check':original,
        'restriction_check':{'valid':restriction_valid,'residual':residual,'dimension_diagnostics':dimensions},
        'equalities':[{'kind':e.kind,'i':e.i,'j':e.j,'rhs':e.rhs} for e in eq],
        'fixed':fixed,'supplement_diagnostic':diagnostic,'method_wall_budget_exceeded':seconds>30,
        'solved_model_certified':bool(raw.get('certified',False) and restriction_valid),
        'original_global_solver_certified':False,'full_fallback_used':False})

def benchmark(args):
    parent,adapter,common,bench,protocol,frozen=setup(args.parent)
    out=Path(args.out).resolve(); sf=validate_freeze(out,parent,protocol,frozen)
    if len(sf['cases'])!=18 or len(set(sf['cases']))!=18 or set(sf['cases'])!=set(adapter.list_case_ids()):
        raise RuntimeError('Formal supplement requires the exact 18 fitted cases')
    runtime=checked_runtime(common,protocol)
    write_new(out/'benchmark-started.json',{'utc':stamp(),'runtime':runtime,'freeze_sha256':sha(out/'supplement-freeze.json')})
    methods=protocol['methods']; total=0
    for cid in sf['cases']:
        check_deadline(args.deadline)
        case=adapter.load_case(cid)
        models,markers=bench.load_models(case,list(common.METHODS),frozen)
        with (out/'models'/cid/'coordinate-svm.pkl').open('rb') as f: coordinate=pickle.load(f)
        # Parent cohort is frozen unchanged, and new learners were verified above.
        costs,features,guard=adapter.draw_confirmation_costs(case,count=protocol['confirmation_count'],stream=protocol['fresh_test_stream'])
        folder=out/'results'/cid; folder.mkdir(parents=True,exist_ok=True)
        write_new(folder/'guard.json',{'sampler':guard,'new_freeze_sha256':sha(out/'supplement-freeze.json')})
        with (folder/'costs.npz').open('xb') as f: np.savez_compressed(f,costs=costs,features=features)
        references=[]
        import experiment_core
        with (folder/'full-references.jsonl').open('x',encoding='utf-8') as refs:
            for i,cost in enumerate(costs):
                check_deadline(args.deadline)
                raw,x=experiment_core.solve(case.original_base,cost,seed=case.seed+550000+i,time_limit=30)
                record=clean({'test_index':i,'result':raw,'solution':x,'utc':stamp()})
                refs.write(json.dumps(record,allow_nan=False)+'\n');refs.flush();os.fsync(refs.fileno())
                if not raw['certified'] or not raw['valid'] or x is None: raise RuntimeError('Fresh Full reference unproved; preserved raw failure')
                references.append(float(raw['objective']))
        def execute(method,cost,feature,i):
            check_deadline(args.deadline)
            if method in common.METHODS:
                return bench.run_method(case,method,models,cost,feature,case.seed+i,case.seed+104729*(i+1))
            return new_method(method,models,coordinate,case,cost,feature,case.seed+i,adapter,protocol)
        # Equal training-only warm-up; no test-based tuning.
        warm=[execute(m,case.train_costs[0],case.train_features[0],-1) for m in methods]
        write_new(folder/'warm.json',{'train_only':True,'records':warm})
        rng=np.random.default_rng(np.random.SeedSequence([case.seed,20261011,31007]))
        with (folder/'trials.jsonl').open('x',encoding='utf-8') as f:
            for i,(cost,feature,reference) in enumerate(zip(costs,features,references)):
                order=[str(x) for x in rng.permutation(methods)]
                for pos,method in enumerate(order):
                    check_deadline(args.deadline)
                    row=execute(method,cost,feature,i); tol=adapter.recovery_tolerance(reference)
                    recovered=bool(row['valid'] and row['objective'] is not None and abs(row['objective']-reference)<=tol)
                    row.update(case=cid,family=case.family,amplitude=case.amplitude,seed=case.seed,test_index=i,
                               common_ambient_dimension=len(case.commitment_indices),target_common_K=int(case.affine['rank']),
                               reference_objective=reference,recovery_tolerance=tol,quality_recovered=recovered,
                               objective_error=None if row['objective'] is None else row['objective']-reference,
                               parent_freeze=frozen['fingerprint'],new_freeze_sha256=sha(out/'supplement-freeze.json'),
                               method_order=order,method_order_position=pos,completed_utc=stamp())
                    f.write(json.dumps(clean(row),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());total+=1
                    if method=='full' and not (recovered and row['original_global_solver_certified']): raise RuntimeError('Timed Full differs from fresh certified reference')
                if (i+1)%10==0: print(json.dumps({'event':'test_progress','case':cid,'objectives':i+1,'total_records':total,'utc':stamp()}),flush=True)
        write_new(folder/'complete.json',{'case':cid,'records':len(costs)*len(methods),'methods':methods,'utc':stamp()})
    write_new(out/'benchmark-complete.json',{'cases':sf['cases'],'methods':methods,'records':total,'expected':18*40*len(methods),'utc':stamp(),'all_cases_complete':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','train','benchmark']);p.add_argument('--parent',required=True);p.add_argument('--out',required=True);p.add_argument('--cases',nargs='+');p.add_argument('--deadline-utc')
    args=p.parse_args();args.deadline=None if not args.deadline_utc else datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00')).timestamp()
    if args.phase=='preflight':
        parent,adapter,common,bench,protocol,frozen=setup(args.parent)
        if len(adapter.list_case_ids())!=18: raise RuntimeError('Wrong parent cohort')
        print(json.dumps({'parent_freeze':frozen['fingerprint'],'new_source_binding':source_binding(),'cases':adapter.list_case_ids(),'runtime':checked_runtime(common,protocol)},allow_nan=False))
    else:
        parent,adapter,common,bench,protocol,frozen=setup(args.parent)
        if bench.active_offline_processes(): raise RuntimeError('Independent experimental computation is active on the benchmark host')
        from threadpoolctl import threadpool_limits
        with common.PhaseLock('matched-k-supplement-'+args.phase), threadpool_limits(limits=1):
            if args.phase=='train': train(args)
            else: benchmark(args)

if __name__=='__main__': main()
