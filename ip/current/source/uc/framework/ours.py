"""Exact training affine span with integer-coefficient extended formulation.

Only commitment u is compressed. All original variables and constraints remain.
The additional alpha variables are continuous and unrestricted. K is the exact
rational rank of training commitment differences, not total MILP dimension.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter, process_time
import numpy as np
import scipy.sparse as sp
import adapter
from cases import array_hash, model_hash


@dataclass
class AffineCompression:
    anchor: np.ndarray
    basis: sp.csr_matrix
    common: np.ndarray
    manifest: dict

    @property
    def K(self): return self.basis.shape[0]

    def extend(self, model, cost):
        n, k = model.n_variables, self.K
        # Keep the original u binary domains; alpha carries no integer claims.
        left = sp.csr_matrix((np.ones(len(self.common)),(np.arange(len(self.common)),self.common)),
                             shape=(len(self.common),n))
        equality = sp.hstack([left,-sp.csr_matrix(self.basis.T)],format="csr")
        a = sp.vstack([sp.hstack([model.A,sp.csr_matrix((model.A.shape[0],k))]),equality],format="csr")
        augmented = adapter.LinearModel(a,np.r_[model.row_lower,self.anchor],np.r_[model.row_upper,self.anchor],
            np.r_[model.col_lower,np.full(k,-np.inf)],np.r_[model.col_upper,np.full(k,np.inf)],
            np.r_[model.integrality,np.zeros(k,dtype=np.int32)],
            tuple(model.variable_names)+tuple(f"affine_alpha[{j}]" for j in range(k)),
            tuple(model.row_names)+tuple(f"training_affine_u[{j}]" for j in range(len(self.common))))
        return augmented,np.r_[np.asarray(cost),np.zeros(k)]

    def save(self,path):
        import json
        np.savez_compressed(path,anchor=self.anchor,basis_data=self.basis.data,
                            basis_indices=self.basis.indices,basis_indptr=self.basis.indptr,
                            basis_shape=np.asarray(self.basis.shape),common=self.common,
                            manifest=json.dumps(self.manifest,sort_keys=True))

    @classmethod
    def load(cls,path):
        import json
        with np.load(path,allow_pickle=False) as z:
            basis=sp.csr_matrix((z["basis_data"],z["basis_indices"],z["basis_indptr"]),shape=tuple(z["basis_shape"]))
            return cls(z["anchor"].copy(),basis,z["common"].copy(),json.loads(str(z["manifest"])))


def fit(case):
    started, cpu = perf_counter(), process_time()
    common = np.asarray(case.commitment_indices,dtype=np.int64)
    raw = np.asarray(case.train_solutions)[:,common]
    if np.max(np.abs(raw-np.rint(raw)),initial=0.) > 1e-6:
        raise ValueError("Training commitments are not integral")
    points = np.rint(raw).astype(np.int64)
    anchor = points[0]
    differences = np.unique(points-anchor,axis=0)
    differences = differences[np.any(differences != 0,axis=1)]
    if not len(differences):
        basis = np.empty((0,len(common)),dtype=np.int64)
        pivots = ()
    else:
        # Duplicate columns and zero columns do not affect row independence.
        compact = np.unique(differences,axis=1)
        compact = compact[:,np.any(compact != 0,axis=0)]
        from sympy import Matrix
        from sympy.polys.matrices import DomainMatrix
        domain = DomainMatrix.from_Matrix(Matrix(compact.T.tolist()))
        _, pivots = domain.rref()
        basis = differences[np.asarray(pivots,dtype=int)]
    basis=sp.csr_matrix(basis,dtype=np.int64)
    rank=basis.shape[0]
    manifest = dict(method="Ours",implementation="exact-integer-training-affine-span-v1",
        training_label_count=case.n_train,training_array_sha256=case.dataset_fingerprint,
        physical_model_sha256=model_hash(case.model),commitment_indices=common.tolist(),
        common_D=len(common),K=rank,rank_certificate="exact rational DomainMatrix RREF",
        rank_numeric_tolerance=None,rank_truncated=False,basis_difference_rows=list(map(int,pivots)),
        affine_data_sha256=array_hash(anchor,basis.data,basis.indices,basis.indptr,np.asarray(basis.shape),common),
        basis_storage="CSR integer matrix",basis_nnz=int(basis.nnz),fit_seconds=perf_counter()-started,
        fit_cpu_seconds=process_time()-cpu,training_label_collection_included=False,
        preserved_original_binary_variables=int(np.count_nonzero(case.model.integrality)),
        preserved_original_continuous_variables=int(np.count_nonzero(case.model.integrality==0)),
        original_total_variables=case.model.n_variables,original_constraint_nnz=int(case.model.A.nnz),
        augmented_total_variables=case.model.n_variables+rank,
        augmented_constraint_nnz=int(case.model.A.nnz+len(common)+basis.nnz),
        added_continuous_variables=rank,added_equalities=len(common),
        dimension_scope="affine dimension of commitment space only before original constraints",
        test_information_used=False,validation_information_used=False)
    return AffineCompression(anchor,basis,common,manifest)


def query(compression,case,cost,*,seed=0,time_limit=30.,solve_fn=adapter.solve):
    started=perf_counter()
    augmented,extended_cost=compression.extend(case.model,cost)
    prep_seconds=perf_counter()-started
    result=solve_fn(augmented,extended_cost,seed=seed,time_limit=max(0.001,time_limit-prep_seconds))
    if result.x is not None:
        alpha=result.x[case.model.n_variables:].copy()
        x=result.x[:case.model.n_variables].copy()
        affine_residual=float(np.max(np.abs(x[compression.common]-compression.anchor-compression.basis.T@alpha),initial=0.))
        check=adapter.validate_solution(case.model,x)
        result.x=x;result.objective=float(np.asarray(cost)@x)
        result.valid=bool(result.valid and check["valid"] and affine_residual<=adapter.FEASIBILITY_TOLERANCE)
        result.certified=bool(result.certified and result.valid)
        result.original_violation=check["original_violation"]
        result.integrality_residual=check["integrality_residual"]
        result.metadata.update(affine_alpha=alpha.tolist(),affine_return_residual=affine_residual)
    result.metadata.update(method="Ours",target_K=compression.K,common_D=len(compression.common),
        original_global_optimality_certified=False,certificate_scope="affine_restricted_model",
        affine_build_seconds=prep_seconds)
    result.seconds=perf_counter()-started
    result.build_seconds+=prep_seconds
    return result
