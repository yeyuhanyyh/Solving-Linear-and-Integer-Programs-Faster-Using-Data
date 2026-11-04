from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import os,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
 os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,Path):return str(x)
 if hasattr(x,'tolist'):return clean(x.tolist())
 if isinstance(x,float) and not __import__('math').isfinite(x):return None
 return x
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(clean(x),f,indent=2,allow_nan=False)
def array_hash(*arrays):
 import numpy as np
 d=hashlib.sha256()
 for a in arrays:
  a=np.ascontiguousarray(a);d.update(str(a.shape).encode());d.update(str(a.dtype).encode());d.update(a.tobytes())
 return d.hexdigest()
def npz(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pin_one_cpu():
 if hasattr(os,'sched_getaffinity'):
  allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]});return [allowed[0]]
 if os.name=='nt':
  import ctypes
  kernel=ctypes.WinDLL('kernel32',use_last_error=True)
  kernel.GetCurrentProcess.restype=ctypes.c_void_p
  kernel.GetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t))
  kernel.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
  current=kernel.GetCurrentProcess();pm=ctypes.c_size_t();sm=ctypes.c_size_t()
  if not kernel.GetProcessAffinityMask(current,ctypes.byref(pm),ctypes.byref(sm)):raise ctypes.WinError(ctypes.get_last_error())
  requested=2
  bit=1<<requested
  if not pm.value&bit:raise RuntimeError('The declared CPU2 is unavailable; do not silently change the timing allocation')
  if not kernel.SetProcessAffinityMask(current,bit):raise ctypes.WinError(ctypes.get_last_error())
  return [bit.bit_length()-1]
 raise RuntimeError('Cannot pin a CPU on this platform')
def verify():
 m=read(ROOT/'manifest.json')
 for rel,h in m['files'].items():
  p=(ROOT/rel).resolve()
  if not p.is_relative_to(ROOT) or sha(p)!=h:raise ValueError('Changed bundle member: '+rel)
 return m

@dataclass
class ExactAffine:
 anchor:object
 basis:object
 common:object
 diagnostics:dict
 @property
 def K(self):return int(self.basis.shape[0])
 def extend(self,model,cost):
  import numpy as np,scipy.sparse as sp
  import adapter
  n=len(cost);k=self.K;d=len(self.common)
  if hasattr(self,'prepared_model'):
   if n!=self.original_variable_count:raise ValueError('Objective dimension changed')
   return self.prepared_model,np.r_[cost,np.zeros(k)]
  left=sp.csr_matrix((np.ones(d),(np.arange(d),self.common)),shape=(d,n))
  rows=sp.hstack((left,-self.basis.T),format='csr')
  A=sp.vstack((sp.hstack((model.A,sp.csr_matrix((model.A.shape[0],k))),format='csr'),rows),format='csr')
  augmented=adapter.LinearModel(A,np.r_[model.row_lower,self.anchor],np.r_[model.row_upper,self.anchor],
   np.r_[model.col_lower,np.full(k,-np.inf)],np.r_[model.col_upper,np.full(k,np.inf)],
   np.r_[model.integrality,np.zeros(k,int)])
  return augmented,np.r_[cost,np.zeros(k)]
 def save(self,p):
  import numpy as np
  m=self.prepared_model
  np.savez_compressed(p,anchor=self.anchor,basis_data=self.basis.data,basis_indices=self.basis.indices,
    basis_indptr=self.basis.indptr,basis_shape=self.basis.shape,common=self.common,
    A_data=m.A.data,A_indices=m.A.indices,A_indptr=m.A.indptr,A_shape=m.A.shape,
    row_lower=m.row_lower,row_upper=m.row_upper,col_lower=m.col_lower,col_upper=m.col_upper,integrality=m.integrality)

def fit_exact(case):
 """One implementation for both families. No numerical rank cutoff."""
 import numpy as np,scipy.sparse as sp
 from sympy import Matrix
 from sympy.polys.matrices import DomainMatrix
 start=time.perf_counter()
 common=np.asarray(case.commitment_indices,dtype=np.int64)
 raw=np.asarray(case.train_solutions)[:,common]
 if np.max(np.abs(raw-np.rint(raw)),initial=0)>1e-6:raise ValueError('Noninteger labels')
 points=np.rint(raw).astype(np.int64);anchor=points[0].copy()
 differences=np.unique(points-anchor,axis=0);differences=differences[np.any(differences!=0,axis=1)]
 compact=np.unique(differences,axis=1) if len(differences) else np.empty((0,0),int)
 compact=compact[:,np.any(compact!=0,axis=0)]
 if len(differences):
  _,pivots=DomainMatrix.from_Matrix(Matrix(compact.T.tolist())).rref()
  basis=sp.csr_matrix(differences[np.asarray(pivots,int)],dtype=np.int64)
 else:pivots=();basis=sp.csr_matrix((0,len(common)),dtype=np.int64)
 result=ExactAffine(anchor,basis,common,dict(rank=basis.shape[0],input_shape=points.shape,
  distinct_nonzero_difference_rows=len(differences),rank_input_shape=compact.T.shape,
  basis_rows=list(map(int,pivots)),basis_nnz=int(basis.nnz),exact_rank=True,
  implementation='shared-domainmatrix-integer-basis-v2'))
 prepared=time.perf_counter()
 result.prepared_model,_=result.extend(case.model,np.zeros(case.model.A.shape[1]))
 result.original_variable_count=case.model.A.shape[1]
 result.diagnostics.update(prepared_model_shape=result.prepared_model.A.shape,prepared_model_nnz=result.prepared_model.A.nnz,
   model_preparation_seconds=time.perf_counter()-prepared,internal_seconds=time.perf_counter()-start)
 return result

def load_condition(condition,mode='uc'):
 import numpy as np,scipy.sparse as sp
 from types import SimpleNamespace
 spec=read(ROOT/'data/index.json')['conditions'][condition]
 arrays=npz(ROOT/spec['bank']);d=spec['n_train']
 if spec['family']=='case14':
  model_data=npz(ROOT/'data/case14-model.npz')
  model=SimpleNamespace(A=sp.csr_matrix((model_data['A_data'],model_data['A_indices'],model_data['A_indptr']),shape=tuple(model_data['A_shape'])),
    **{k:model_data[k] for k in ('row_lower','row_upper','col_lower','col_upper','integrality','variable_names','row_names')})
  features=arrays['features'];costs=arrays['costs'];solutions=arrays['solutions'];objectives=arrays['objectives'];common=arrays['common']
 else:
  model=SimpleNamespace(A=sp.csr_matrix(arrays['weights'][None,:].astype(float)),row_lower=np.array([-np.inf]),
   row_upper=np.array([float(arrays['capacity'])]),col_lower=np.zeros(100),col_upper=np.ones(100),integrality=np.ones(100,int),
   variable_names=(),row_names=())
  features=arrays['train_profits'][:d].astype(float);costs=-features;solutions=arrays['train_solutions'][:d].astype(float)
  objectives=-arrays['train_objectives'][:d].astype(float);common=np.arange(100)
 if mode=='uc':
  import adapter
  model=adapter.LinearModel(model.A,model.row_lower,model.row_upper,model.col_lower,model.col_upper,model.integrality,
    tuple(model.variable_names),tuple(model.row_names))
 elif mode=='knap-neural':
  from portable_adapter import LinearModel
  model=LinearModel(model.A,model.row_lower,model.row_upper,model.col_lower,model.col_upper,model.integrality)
 case=SimpleNamespace(model=model,train_features=features,train_costs=costs,train_solutions=solutions,train_objectives=objectives,
  commitment_indices=common,affine={'rank':int(spec['K'])},n_train=d,family='ucjl_static' if spec['family']=='case14' else 'knapsack',
  case_id=condition,seed=86001 if spec['family']=='case14' else spec['seed'],
  metadata={'family':'ucjl_static' if spec['family']=='case14' else 'knapsack','baseline_seed':42001 if spec['family']=='case14' else spec['seed'],
            'n_times':36 if spec['family']=='case14' else 1,'n_units':5 if spec['family']=='case14' else 100},
  dataset_fingerprint=array_hash(features,costs,solutions,objectives),spec={})
 if spec['family']=='case14':
  v=npz(ROOT/'data/case14-validation.npz')
  validation=SimpleNamespace(**vars(case));validation.case_id=condition+'-validation';validation.n_train=64
  for field,key in [('train_features','features'),('train_costs','costs'),('train_solutions','solutions'),('train_objectives','objectives')]:setattr(validation,field,v[key])
  validation.dataset_fingerprint=array_hash(*(getattr(validation,k) for k in ('train_features','train_costs','train_solutions','train_objectives')))
 else:validation=None
 return case,validation,spec
