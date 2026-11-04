"""Algebraically equivalent memory optimization and native LP decoding.

The feature-linear map is factored before broadcasting. Same trainable tensors,
token activations, attention and loss; floating-point reduction order differs.
The optional equality-KKT shortcut is disabled: Li solves its actual reduced
inequality LP, retaining the prior port's permanent variable-bound convention.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
import preference_baseline as li
import mlopt_baseline


class FactorizedPreferenceNetwork(li.PreferenceNetwork):
    def forward(self,parameters):
        first=self.input_embedding[0]
        p=parameters.shape[1]
        cost_part=F.linear(parameters,first.weight[:,:p],first.bias)
        strategy_part=F.linear(self.strategies,first.weight[:,p:],None)
        tokens=cost_part[:,None,:]+strategy_part[None,:,:]
        for layer in self.input_embedding[1:]: tokens=layer(tokens)
        for layer in self.layers: tokens=layer(tokens)
        return self.output(tokens).squeeze(-1)


class NativeInequalityExtraction(mlopt_baseline.StrategyBank):
    def _build_decoder(self,strategy):
        rows,rhs,_=self._equations(strategy)
        strategy.equations,strategy.equations_rhs=rows,rhs
        strategy.factor=None
        strategy.basis_matrix=None
        strategy.selected_equation_rows=None
        strategy.x=None
        strategy.decoder_status="native_inequality_lp_required"

    def decode(self,strategy_index,cost):
        # Returning no equality shortcut is intentional. The inherited Li
        # decoder immediately solves its true reduced inequality LP.
        return dict(x=None,objective=float("inf"),valid=False,
                    status="native_inequality_lp_required",strategy_id=int(strategy_index),
                    active_equality_residual=float("inf"),stationarity_residual=float("inf"))


class NativeLPStrategyBank(li.InequalityStrategyBank):
    def __init__(self,case,config):
        from adapter import LinearModel
        self.extracted=NativeInequalityExtraction(case,active_tolerance=config.active_tolerance,
            feasibility_tolerance=config.feasibility_tolerance,rank_tolerance=config.rank_tolerance)
        self.case,self.config=case,config
        self.decode_statistics={"calls":0,"certified_kkt_fast_paths":0,"reduced_lp_solves":0,
                                "failed_reduced_lp_solves":0,"seconds":0.}
        self.reduced_models=[]
        for strategy in self.strategies:
            rows=sp.vstack((self.E,self.G[strategy.active_rows]),format="csr")
            lower=np.concatenate((self.beq,np.full(len(strategy.active_rows),-np.inf)))
            upper=np.concatenate((self.beq,self.h[strategy.active_rows]))
            column_lower=np.asarray(case.model.col_lower,dtype=float).copy()
            column_upper=np.asarray(case.model.col_upper,dtype=float).copy()
            column_lower[self.integer_indices]=strategy.integer_values
            column_upper[self.integer_indices]=strategy.integer_values
            self.reduced_models.append(LinearModel(rows,lower,upper,column_lower,column_upper,np.zeros(self.n_variables,dtype=int)))


def install():
    li.PreferenceNetwork=FactorizedPreferenceNetwork
    li.InequalityStrategyBank=NativeLPStrategyBank
