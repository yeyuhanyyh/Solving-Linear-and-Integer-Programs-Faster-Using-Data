"""Shared, single-thread HiGHS runner and exact affine restrictions."""
from __future__ import annotations
from time import perf_counter
import math
import numpy as np
import scipy.sparse as sparse
import sympy as sp
import highspy


def make_cover_base(seed, n_sets=80, n_elements=50, density=.15):
    rng = np.random.default_rng(seed)
    a = (rng.random((n_elements, n_sets)) < density).astype(np.int64)
    # Feasibility safeguards depend only on the constraint generator.
    for i in range(n_elements):
        if a[i].sum() < 3:
            a[i, rng.choice(n_sets, 3, replace=False)] = 1
    for j in range(n_sets):
        if not a[:, j].any():
            a[rng.integers(n_elements), j] = 1
    return dict(A=sparse.csr_matrix(a, dtype=float),
                row_lower=np.ones(n_elements), row_upper=np.full(n_elements, np.inf),
                col_lower=np.zeros(n_sets), col_upper=np.ones(n_sets),
                integrality=np.ones(n_sets, dtype=np.int32), base_cost=np.ones(n_sets),
                binary_indices=np.arange(n_sets), n_times=1, n_units=n_sets,
                metadata=dict(family='set_cover', seed=int(seed), n_sets=n_sets,
                              n_elements=n_elements, density=density))


def cost_for_cover(base, rng, amplitude):
    c = 1.0 + amplitude * rng.uniform(-1.0, 1.0, len(base['base_cost']))
    return c, c.copy()


def validate_solution(base, x, atol=1e-6):
    x = np.asarray(x, dtype=float)
    if x.shape != (len(base['col_lower']),) or not np.all(np.isfinite(x)):
        return dict(valid=False, original_violation=float('inf'), integrality_residual=float('inf'))
    ax = base['A'] @ x
    violation = max(float(np.max(base['row_lower'] - ax, initial=0)),
                    float(np.max(ax - base['row_upper'], initial=0)),
                    float(np.max(base['col_lower'] - x, initial=0)),
                    float(np.max(x - base['col_upper'], initial=0)))
    integer = np.asarray(base['integrality'], dtype=bool)
    int_res = float(np.max(np.abs(x[integer] - np.rint(x[integer])), initial=0))
    return dict(valid=bool(violation <= atol and int_res <= atol),
                original_violation=violation, integrality_residual=int_res)


def fit_affine(y):
    """First training solution is anchor; compression count is rank + 1."""
    started = perf_counter()
    y = np.asarray(y)
    if y.ndim != 2 or not np.all((y == 0) | (y == 1)):
        raise ValueError('Expected a matrix of exact binary training solutions')
    anchor = y[0].astype(np.int64)
    differences = sp.Matrix((y[1:] - anchor).T.tolist())
    # Exact RREF uses SymPy's ZZ/QQ DomainMatrix backend. Matrix.columnspace()
    # instead uses unnormalized generic echelon elimination, whose intermediate
    # integers can grow catastrophically on nearly full-rank binary data.
    # Both return the same leftmost independent columns, over the rationals.
    _, pivots = differences.rref()
    columns = [differences.col(i) for i in pivots]
    basis = sp.Matrix.hstack(*columns) if columns else sp.zeros(y.shape[1], 0)
    rank = basis.cols
    rows = []
    for v in basis.T.nullspace():
        multiple = math.lcm(*(int(q.q) for q in v))
        row = [int(q * multiple) for q in v]
        gcd = math.gcd(*row)
        row = [q // gcd for q in row]
        if next(q for q in row if q) < 0:
            row = [-q for q in row]
        rows.append(row)
    normals = np.array(rows, dtype=object).reshape(-1, y.shape[1])
    assert np.all((y.astype(object) - anchor) @ normals.T == 0)
    norms = np.array([max(abs(q) for q in row) for row in rows], dtype=float)
    numerical = np.array(rows, dtype=float).reshape(-1, y.shape[1])
    if len(rows):
        numerical /= norms[:, None]
    nz = np.abs(numerical[numerical != 0])
    representation = 'normals' if not len(nz) or nz.min() >= 1e-7 else 'lifted'
    return dict(anchor=anchor, basis=np.array(basis.tolist(), dtype=float).reshape(y.shape[1], rank),
                normals_exact=normals, normals=numerical, rank=rank,
                representation=representation, fit_seconds=perf_counter()-started,
                compressed_sample_count=rank+1)


def exact_slice_contains(affine, binary):
    x = np.asarray(binary, dtype=np.int64)
    return bool(np.all(affine['normals_exact'] @ (x.astype(object)-affine['anchor']) == 0))


def solve(base, cost, *, equalities=None, affine=None, warm_start=None,
          time_limit=30., seed=0):
    started = perf_counter()
    d = len(cost)
    indices = np.asarray(base['binary_indices'], dtype=int)
    a = base['A'].tocsr()
    rl = np.asarray(base['row_lower'], dtype=float).copy()
    ru = np.asarray(base['row_upper'], dtype=float).copy()
    cl = np.asarray(base['col_lower'], dtype=float).copy()
    cu = np.asarray(base['col_upper'], dtype=float).copy()
    obj = np.asarray(cost, dtype=float).copy()
    integrality = np.asarray(base['integrality'], dtype=np.int32).copy()
    added_rows = []
    added_rhs = []
    if equalities:
        for eq in equalities:
            row, rhs = eq.as_row(len(indices))
            nz = np.flatnonzero(row)
            added_rows.append(sparse.csr_matrix((row[nz], (np.zeros(len(nz)), indices[nz])),shape=(1,d)))
            added_rhs.append(float(rhs))
    if affine is not None and affine['rank'] < len(indices):
        if affine['representation'] == 'normals':
            n = affine['normals']
            rr, cc = np.nonzero(n)
            added_rows.append(sparse.csr_matrix((n[rr,cc], (rr,indices[cc])),shape=(len(n),d)))
            added_rhs.extend(n @ affine['anchor'])
        else:
            r = affine['rank']
            a = sparse.hstack([a, sparse.csr_matrix((a.shape[0],r))], format='csr')
            added_rows = [sparse.hstack([row,sparse.csr_matrix((row.shape[0],r))],format='csr') for row in added_rows]
            ix = sparse.csr_matrix((np.ones(len(indices)),(np.arange(len(indices)),indices)),shape=(len(indices),d))
            added_rows.append(sparse.hstack([ix,-sparse.csr_matrix(affine['basis'])],format='csr'))
            added_rhs.extend(affine['anchor'])
            cl=np.r_[cl,np.full(r,-np.inf)];cu=np.r_[cu,np.full(r,np.inf)]
            obj=np.r_[obj,np.zeros(r)];integrality=np.r_[integrality,np.zeros(r,dtype=np.int32)]
    count_added = sum(x.shape[0] for x in added_rows)
    added_nnz = sum(x.nnz for x in added_rows)
    if added_rows:
        a = sparse.vstack([a,*added_rows],format='csr')
        rl=np.r_[rl,added_rhs];ru=np.r_[ru,added_rhs]
    lp=highspy.HighsLp()
    lp.num_col_=len(obj);lp.num_row_=a.shape[0]
    lp.col_cost_=obj;lp.col_lower_=cl;lp.col_upper_=cu
    lp.row_lower_=rl;lp.row_upper_=ru
    lp.integrality_=[highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous for i in integrality]
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_=a.indptr.astype(np.int32)
    lp.a_matrix_.index_=a.indices.astype(np.int32)
    lp.a_matrix_.value_=a.data.astype(float)
    h=highspy.Highs()
    for key,val in dict(output_flag=False,threads=1,random_seed=int(seed),time_limit=float(time_limit),
                        mip_rel_gap=0.,mip_abs_gap=1e-8,primal_feasibility_tolerance=1e-7,
                        mip_feasibility_tolerance=1e-7).items():
        if h.setOptionValue(key,val) != highspy.HighsStatus.kOk:
            raise RuntimeError(f'HiGHS option rejected {key}={val}')
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError('HiGHS rejected model')
    if warm_start is not None:
        if h.setSolution(d,np.arange(d,dtype=np.int32),np.asarray(warm_start,dtype=float)) != highspy.HighsStatus.kOk:
            raise RuntimeError('HiGHS rejected warm start')
    build_seconds=perf_counter()-started
    run_started=perf_counter();run_status=h.run();run_seconds=perf_counter()-run_started
    if run_status == highspy.HighsStatus.kError:
        raise RuntimeError('HiGHS run returned an error')
    status=h.getModelStatus();info=h.getInfo();solution=h.getSolution()
    out=dict(status=h.modelStatusToString(status),optimal=status==highspy.HighsModelStatus.kOptimal,
             objective=None,dual_bound=float(info.mip_dual_bound),gap=float(info.mip_gap),
             nodes=int(info.mip_node_count),simplex_iterations=int(info.simplex_iteration_count),
             added_equalities=count_added,added_nonzeros=added_nnz,total_variables=len(obj),
             build_seconds=build_seconds,solve_seconds=run_seconds,solver_internal_seconds=h.getRunTime(),
             valid=False,slice_valid=None)
    x=None
    if solution.value_valid:
        x=np.asarray(solution.col_value[:d],dtype=float)
        out.update(validate_solution(base,x))
        out['objective']=float(np.dot(cost,x))
        if affine is not None:
            out['slice_valid']=exact_slice_contains(affine,np.rint(x[indices]))
            out['valid'] = out['valid'] and out['slice_valid']
        if equalities:
            b=np.rint(x[indices]).astype(np.int64)
            out['equality_valid']=all(int(eq.as_row(len(indices))[0]@b)==int(eq.rhs) for eq in equalities)
            out['valid'] = out['valid'] and out['equality_valid']
    out['certified']=bool(out['optimal'] and out['valid'] and out['objective'] is not None and
                          np.isfinite(out['dual_bound']) and
                          abs(out['objective']-out['dual_bound']) <= 1e-6+1e-9*abs(out['objective']))
    out['online_seconds']=perf_counter()-started
    return out,x


if __name__=='__main__':
    import json
    for size in [(64,40),(80,50),(100,60)]:
        b=make_cover_base(777,*size)
        for amp in [.02,.2,.8]:
            c,_=cost_for_cover(b,np.random.default_rng(991),amp)
            res,x=solve(b,c,time_limit=15)
            print(json.dumps(dict(size=size,amplitude=amp,**res)),flush=True)
