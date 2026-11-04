from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,os,sys
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,Path):return str(v)
    if hasattr(v,'tolist'):return clean(v.tolist())
    if isinstance(v,float) and not __import__('math').isfinite(v):return None
    return v
def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(clean(o),f,indent=2,allow_nan=False);f.write('\n')
def arrays_hash(*a):
    import numpy as np
    h=hashlib.sha256()
    for x in a:
        x=np.ascontiguousarray(x);h.update(str(x.shape).encode());h.update(str(x.dtype).encode());h.update(x.tobytes())
    return h.hexdigest()
def verify():
    m=read(ROOT/'source-and-input-lock.json')
    for name,digest in m['files'].items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and sha(p)==digest,name
    return sha(ROOT/'source-and-input-lock.json')
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def setup(family):
    import numpy as np, scipy,torch,torch.optim,torch._dynamo
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    if family=='uc':
        sys.path[:0]=[str(ROOT/'source/uc/framework'),str(ROOT/'source/uc')]
        import adapter
        api=module('aligned_uc_mlopt_api',ROOT/'source/uc/baselines/mlopt/api.py')
        matched=module('aligned_uc_matched_candidates',ROOT/'source/uc/matched_candidates.py')
        return SimpleNamespace(adapter=adapter,api=api,matched=matched)
    sys.path.insert(0,str(ROOT/'source/correlated'))
    # The portable solver bridge is the same certified model adapter as UC.
    sys.path.insert(0,str(ROOT/'source/uc/framework'))
    import adapter
    from strategy import mlopt_adapter
    return SimpleNamespace(adapter=adapter,api=mlopt_adapter)
def npz(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def load_condition(condition,rt):
    import numpy as np,scipy.sparse as sp
    spec=read(ROOT/'data/index.json')['conditions'][condition];n=spec['n_train'];a=npz(ROOT/spec['bank'])
    if spec['family']=='case14':
        m=npz(ROOT/'data/case14-model.npz')
        model=rt.adapter.LinearModel(sp.csr_matrix((m['A_data'],m['A_indices'],m['A_indptr']),shape=tuple(m['A_shape'])),
          m['row_lower'],m['row_upper'],m['col_lower'],m['col_upper'],m['integrality'],tuple(m['variable_names']),tuple(m['row_names']))
        features=a['features'];costs=a['costs'];solutions=a['solutions'];objectives=a['objectives'];common=a['common']
    else:
        model=rt.adapter.LinearModel(sp.csr_matrix(a['weights'][None,:].astype(float)),np.array([-np.inf]),np.array([float(a['capacity'])]),np.zeros(100),np.ones(100),np.ones(100,int))
        features=a['train_profits'][:n].astype(float);costs=-features;solutions=a['train_solutions'][:n].astype(float);objectives=-a['train_objectives'][:n].astype(float);common=np.arange(100)
    case=SimpleNamespace(model=model,train_features=features,train_costs=costs,train_solutions=solutions,train_objectives=objectives,
      commitment_indices=common,affine={'rank':int(spec['K'])},n_train=n,family='ucjl_static' if spec['family']=='case14' else 'knapsack',case_id=condition,
      seed=86001 if spec['family']=='case14' else spec['seed'],metadata={'family':'ucjl_static' if spec['family']=='case14' else 'knapsack','baseline_seed':42001 if spec['family']=='case14' else spec['seed'],'n_times':36 if spec['family']=='case14' else 1,'n_units':5 if spec['family']=='case14' else 100},
      dataset_fingerprint=arrays_hash(features,costs,solutions,objectives),spec={})
    val=None
    if spec['family']=='case14':
        v=npz(ROOT/'data/case14-validation.npz');val=SimpleNamespace(**vars(case));val.case_id=condition+'-validation';val.n_train=64
        for field,key in [('train_features','features'),('train_costs','costs'),('train_solutions','solutions'),('train_objectives','objectives')]:setattr(val,field,v[key])
        val.dataset_fingerprint=arrays_hash(*(getattr(val,k) for k in ['train_features','train_costs','train_solutions','train_objectives']))
    return case,val,spec
def pin_cpu(index):
    if hasattr(os,'sched_getaffinity'):
        assert index in os.sched_getaffinity(0);os.sched_setaffinity(0,{index});return [index]
    import ctypes
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
    k.GetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t));k.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
    process=k.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t();assert k.GetProcessAffinityMask(process,ctypes.byref(allowed),ctypes.byref(system))
    assert allowed.value&(1<<index);assert k.SetProcessAffinityMask(process,1<<index);return [index]
def model_hash(model,family):
    m=model.model if family=='uc' else model
    return arrays_hash(m.mean,m.scale,m.selected,*[x for pair in m.weights for x in pair])
