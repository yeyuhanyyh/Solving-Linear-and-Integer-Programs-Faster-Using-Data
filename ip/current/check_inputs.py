"""Check training feasibility, objective arithmetic, and exact learned ranks.

This does not regenerate optimality certificates or run optimization/training.
"""
from pathlib import Path
import json,os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.sparse import csr_matrix
from sympy import Matrix
from sympy.polys.matrices import DomainMatrix
from verify import check_manifest
ROOT=Path(__file__).resolve().parent
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def check(m,costs,points,values,common,K):
    A=csr_matrix((m['A_data'],m['A_indices'],m['A_indptr']),shape=tuple(m['A_shape']))
    assert points.shape==costs.shape and points.shape[1]==A.shape[1]
    assert np.isfinite(points).all() and np.isfinite(costs).all()
    ax=(A@points.T).T;ii=np.asarray(m['integrality'])!=0
    for residual in (m['row_lower']-ax,ax-m['row_upper'],m['col_lower']-points,points-m['col_upper'],np.abs(points[:,ii]-np.rint(points[:,ii]))):
        assert np.max(residual,initial=0)<=1e-6
    assert np.all(np.abs(np.sum(costs*points,axis=1)-values)<=1e-6+1e-9*np.abs(values))
    pts=np.rint(points[:,common]).astype(np.int64)
    delta=np.unique(pts-pts[0],axis=0)
    _,pivots=DomainMatrix.from_Matrix(Matrix(delta.tolist())).rref()
    assert len(pivots)==K,(len(pivots),K)
    return dict(training_rows=len(points),D=len(common),K=K)
def main():
    check_manifest();out={}
    m=load(ROOT/'data/case14-model.npz');b=load(ROOT/'data/case14-train.npz')
    out['uc']=check(m,b['costs'],b['solutions'],b['objectives'],b['common'],19)
    for seed in (1009,1013,1019):
        b=load(ROOT/f'grouped/training/seed{seed}/training.npz')
        out[f'grouped-{seed}']=check(b,b['train_costs'],b['train_solutions'],b['train_objectives'],np.arange(96),32)
        assert np.allclose(b['cost_offset']+b['train_features']@b['cost_feature_coefficients'].T,b['train_costs'],atol=1e-12,rtol=1e-14)
    settings=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))['tables']['5']
    for i,seed in enumerate(settings['seeds']):
        b=load(ROOT/f'data/knapsack-{seed}.npz');w=b['weights'];cap=b['capacity']
        assert len(w)==100 and np.all((w>=1)&(w<=1000)) and cap==np.floor(w.sum()/2)
        assert np.isin(b['train_profits']-w-100,[-2,-1,0,1,2]).all()
        A=csr_matrix(w[None,:])
        m=dict(A_data=A.data,A_indices=A.indices,A_indptr=A.indptr,A_shape=A.shape,row_lower=np.array([-np.inf]),row_upper=np.array([cap]),col_lower=np.zeros(100),col_upper=np.ones(100),integrality=np.ones(100))
        for n in settings['n']:
            out[f'correlated-{seed}-n{n}']=check(m,-b['train_profits'][:n],b['train_solutions'][:n],-b['train_objectives'][:n],np.arange(100),settings['K'][str(n)][i])
    print(json.dumps(dict(status='PASS',conditions=out),indent=2))
if __name__=='__main__':main()
