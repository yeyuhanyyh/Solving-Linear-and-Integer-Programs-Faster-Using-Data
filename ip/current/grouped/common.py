"""Portable loader for the unchanged grouped-knapsack training cohort."""
from __future__ import annotations
from pathlib import Path
import os,sys,json,hashlib,importlib.util
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not __import__('math').isfinite(x):return None
    return x
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(clean(x),f,indent=2,allow_nan=False)
def arrays_hash(*arrays):
    import numpy as np
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def verify():
    m=read(ROOT/'training_manifest.json')
    for rel,h in m['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    for case in m['cases'].values():
        for rel,h in case['files'].items():assert sha(ROOT/rel)==h,rel
    return m
def enable():
    for rel in ('source','source/extra','source/strict','source/generic/shared','source/generic/runtime'):
        value=str(ROOT/rel)
        if value not in sys.path:sys.path.insert(0,value)
    import uc_adapters,adapter,portable_adapter
    def signed_costs(case,features):
        import numpy as np
        return np.asarray(case.metadata['cost_offset'],float)+np.asarray(features,float)@np.asarray(case.metadata['cost_feature_coefficients'],float).T
    for m in (uc_adapters,adapter,portable_adapter):m.costs_from_features=signed_costs
    return adapter
def npz(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def load_case(seed,data_root=None):
    import numpy as np
    from scipy import sparse
    root=ROOT if data_root is None else Path(data_root)
    manifest=read(root/'training_manifest.json');spec=manifest['cases'][str(seed)]
    for rel,h in spec['files'].items():assert sha(root/rel)==h,rel
    folder=root/f'training/seed{seed}';z=npz(folder/'training.npz');meta=read(folder/'metadata.json')
    adapter=enable();D=96;n=36
    A=sparse.csr_matrix((z['A_data'],z['A_indices'],z['A_indptr']),shape=tuple(z['A_shape']))
    model=adapter.LinearModel(A=A,**{k:z[k] for k in ('row_lower','row_upper','col_lower','col_upper','integrality')})
    assert A.shape[1]==D and z['train_solutions'].shape==(n,D) and z['train_features'].shape==(n,32)
    assert np.array_equal(model.integrality,np.ones(D,int))
    assert np.array_equal(model.col_lower,np.zeros(D)) and np.array_equal(model.col_upper,np.ones(D))
    common=np.arange(D);physical=dict(meta.get('metadata',{}))
    offset=np.asarray(z.get('cost_offset',physical.get('cost_offset')),float)
    coeff=np.asarray(z.get('cost_feature_coefficients',physical.get('cost_feature_coefficients')),float)
    assert np.allclose(offset+z['train_features']@coeff.T,z['train_costs'],rtol=1e-14,atol=1e-12)
    affine=read(folder/'affine.json');assert affine['rank']==32
    physical.update(cost_offset=offset.tolist(),cost_feature_coefficients=coeff.tolist(),n_categories=32,
                    baseline_seed=spec['baseline_seed'],n_times=1,n_units=96,
                    source_case_directory=str(folder),training_metadata=meta,
                    selection_manifest_sha256=sha(root/'training_manifest.json'))
    base=dict(A=A,row_lower=model.row_lower,row_upper=model.row_upper,col_lower=model.col_lower,col_upper=model.col_upper,
              integrality=model.integrality,base_cost=offset,binary_indices=common,n_times=1,n_units=D,metadata=physical)
    case=adapter.FrozenCase(case_id=spec['case_id'],family='knapsack',seed=int(seed),amplitude=0.,model=model,
        train_features=z['train_features'],train_costs=z['train_costs'],train_solutions=z['train_solutions'],
        train_objectives=z['train_objectives'],commitment_indices=common,affine=affine,original_base=base,metadata=physical,
        source_hashes={'training.npz':sha(folder/'training.npz'),'metadata.json':sha(folder/'metadata.json')},
        dataset_fingerprint=arrays_hash(*(z[k] for k in ('train_features','train_costs','train_solutions','train_objectives'))),source_directory=folder)
    for cost,x,value in zip(case.train_costs,case.train_solutions,case.train_objectives):
        assert adapter.validate_solution(model,x)['valid']
        assert abs(float(cost@x)-value)<=adapter.recovery_tolerance(value)
    return case
def load_test(seed):
    manifest=read(ROOT/'testing_manifest.json')
    for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
    return npz(ROOT/f'testing/seed{seed}/confirmation.npz')
def fit_exact(case):
    enable();import exact_affine_common
    return exact_affine_common.fit_exact(case)
def pin_cpu():
    enable();import exact_affine_common
    return exact_affine_common.pin_one_cpu()
