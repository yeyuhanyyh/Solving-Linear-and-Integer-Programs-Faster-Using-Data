"""Sparse equality-KKT decoder for the fixed-feasible-set, P=0 specialization.

Avoids dense rectangular QR on the full published UC model. It factors C.T C
without a ridge. Singular/unstable systems fail explicitly; no candidate is
replaced by an incumbent. This equality decoder does not provide a signed
inequality certificate. The separate Li wrapper disables its shortcut and uses
the original reduced-inequality LP path instead.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import mlopt_baseline as source


class SparseStrategyBank(source.StrategyBank):
    def __init__(self,*args,**kwargs):
        self._normal_matrices={}; self._normal_factors={}
        super().__init__(*args,**kwargs)
        self.diagnostics["decoder_implementation"]="sparse unregularized normal-equation KKT"
        self.diagnostics["normal_equations_square_condition_number"]=True
        self.diagnostics["singular_decoder_fallback"]=False

    def _build_decoder(self,strategy):
        R,b,rhs=self._equations(strategy)
        strategy.equations,strategy.equations_rhs=R,b
        nc=len(self.continuous_indices)
        x=np.zeros(self.n_variables); x[self.integer_indices]=strategy.integer_values
        if nc:
            C=sp.csc_matrix(R[:,self.continuous_indices])
            if C.shape[0]<nc:
                strategy.decoder_status="rank_deficient"; return
            Q=sp.csc_matrix(C.T@C)
            try:
                factor=splu(Q)
                pivots=np.abs(factor.U.diagonal())
                if np.min(pivots,initial=np.inf)<=self.rank_tolerance*max(1.,np.max(pivots,initial=0.)):
                    strategy.decoder_status="unstable_normal_equations"; return
                self._normal_matrices[strategy.id]=Q
                self._normal_factors[strategy.id]=factor
                strategy.rhs=np.asarray(C.T@rhs).reshape(-1)
                x[self.continuous_indices]=factor.solve(strategy.rhs)
                strategy.decoder_rank=nc
            except (RuntimeError,ValueError):
                strategy.decoder_status="singular_normal_equations"; return
        strategy.equality_residual=float(np.max(np.abs(R@x-b),initial=0.))
        check=self.check_original(x)
        strategy.original_violation=check["violation"]
        strategy.integer_residual=check["integer_residual"]
        strategy.factor=None; strategy.basis_matrix=None; strategy.selected_equation_rows=None
        if strategy.equality_residual>self.feasibility_tolerance:
            strategy.decoder_status="inconsistent_active_equalities"
        elif not check["valid"]:
            strategy.decoder_status="original_constraint_violation"
        else:
            strategy.x=x;strategy.decoder_status="decoded"

    def decode(self,strategy_index,cost):
        s=self.strategies[int(strategy_index)]; c=np.asarray(cost,dtype=float)
        if c.shape!=(self.n_variables,) or not np.isfinite(c).all(): raise ValueError("Invalid cost")
        if s.x is None:
            return dict(x=None,objective=float("inf"),valid=False,status=s.decoder_status,
                        violation=float("inf"),integer_residual=float("inf"),strategy_id=s.id)
        x=np.zeros(self.n_variables);x[self.integer_indices]=s.integer_values
        stationarity=0.
        if len(self.continuous_indices):
            factor=self._normal_factors[s.id]
            C=s.equations[:,self.continuous_indices]
            x[self.continuous_indices]=factor.solve(s.rhs)
            dual=-C@factor.solve(c[self.continuous_indices])
            stationarity=float(np.max(np.abs(C.T@dual+c[self.continuous_indices]),initial=0.))
        active=float(np.max(np.abs(s.equations@x-s.equations_rhs),initial=0.))
        check=self.check_original(x)
        st_tol=self.feasibility_tolerance*(1+np.max(np.abs(c),initial=0.))
        valid=bool(check["valid"] and active<=self.feasibility_tolerance and stationarity<=st_tol)
        return dict(x=x,objective=float(c@x),valid=valid,status="decoded" if valid else "sparse_kkt_residual_failure",
            violation=check["violation"],integer_residual=check["integer_residual"],
            strategy_id=s.id,active_equality_residual=active,stationarity_residual=stationarity,
            stationarity_tolerance=float(st_tol))

    def __getstate__(self):
        state=dict(self.__dict__);state["_normal_factors"]={};return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        self._normal_factors={j:splu(Q) for j,Q in self._normal_matrices.items()}

