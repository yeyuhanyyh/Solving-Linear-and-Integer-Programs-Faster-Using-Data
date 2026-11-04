"""Small integration tests; no realistic UC optimization or cloud calls."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/"framework"))
sys.path.insert(0,str(ROOT))
import adapter
import neural_api as api
import neural_support as support
import neural_diving_baseline as nd
import preference_baseline as li
import torch


def make_case(costs,case_id):
    # Three binary decisions and one continuous variable. The first two
    # decisions cover one unit; remaining variables have positive costs.
    model=adapter.LinearModel(sp.csr_matrix([[1.,1.,0.,0.]]),np.array([1.]),np.array([np.inf]),
        np.zeros(4),np.ones(4),np.array([1,1,1,0]),("u0","u1","v","p"),("cover",))
    costs=np.asarray(costs,dtype=float)
    x=np.zeros_like(costs);x[np.arange(len(costs)),np.argmin(costs[:,:2],axis=1)]=1.
    return SimpleNamespace(model=model,case_id=case_id,family="uc",seed=19,n_train=len(costs),
        train_features=costs[:,:2].copy(),train_costs=costs,train_solutions=x,
        train_objectives=np.einsum("ij,ij->i",costs,x),commitment_indices=np.array([0,1]),
        affine={"rank":1},metadata={"baseline_seed":19},source_hashes={},dataset_fingerprint="toy",
        original_base={},source_directory=ROOT)


class NeuralAdapterTests(unittest.TestCase):
    def setUp(self):
        self.train=make_case([[1+i*.01,3.,2.,2.] for i in range(8)],"train")
        self.validation=make_case([[4.,1.2,2.,2.],[5.,1.4,2.,2.]],"validation")

    def test_reject_inconsistent_physics_and_labels(self):
        support.validate_training_pair(self.train,self.validation)
        bad=make_case([[4.,1.2,2.,2.]],"bad")
        bad.train_solutions[0]=0
        with self.assertRaises(ValueError): support.validate_training_pair(self.train,bad)

    def test_factorized_preference_forward_and_gradient_equivalence(self):
        from li_scaling import FactorizedPreferenceNetwork
        original=FactorizedPreferenceNetwork.__bases__[0]
        config=li.PreferenceConfig(hidden_dim=16,attention_layers=2)
        rng=np.random.default_rng(123)
        strategies=rng.normal(size=(5,17))
        reference=original(3,strategies,config).double()
        optimized=FactorizedPreferenceNetwork(3,strategies,config).double()
        optimized.load_state_dict(reference.state_dict())
        a=torch.tensor(rng.normal(size=(4,3)),dtype=torch.float64,requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        actual,expected=optimized(b),reference(a)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
        actual.square().sum().backward();expected.square().sum().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-10)
        for (name,left),(other,right) in zip(reference.named_parameters(),optimized.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(left.grad,right.grad,rtol=1e-9,atol=1e-10)

    def test_late_nd_validation_return_is_ineligible(self):
        from neural_timing import checked_return
        late=checked_return(dict(valid=True,objective=1.,seconds=.9,status="Optimal"),1.01,1.)
        self.assertFalse(late["valid"])
        self.assertTrue(late["raw_valid_before_return_budget"])
        self.assertEqual(late["objective"],1.)

    def test_li_external_validation_never_enters_strategy_bank(self):
        from li_training import ExternalValidationPreference
        config=li.PreferenceConfig(seed=19,epochs=2,learning_rates=(1e-4,),lambda1_values=(.8,),
            lambda2_values=(.1,),top_k_values=(1,),hidden_dim=16,attention_layers=1)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            fitted=api.fit(self.train,self.validation,"preference_learning",temp,config=dict(vars(config)))
            self.assertEqual(fitted.model.metadata["training_count"],8)
            self.assertEqual(fitted.model.metadata["external_validation_count"],2)
            self.assertFalse(fitted.model.metadata["validation_labels_in_final_fit"])
            self.assertEqual(fitted.model.bank.decode_statistics["certified_kkt_fast_paths"],0)
            self.assertGreater(fitted.model.bank.decode_statistics["reduced_lp_solves"],0)
            self.assertTrue(all(s.integer_values[0]==1 and s.integer_values[1]==0 for s in fitted.model.bank.strategies))
            # A changed cost makes a formerly active continuous bound inactive.
            # Li must solve its inequality LP, not keep the training equality.
            decoded=fitted.model.bank.decode(0,np.array([1.,3.,2.,-1.]))
            self.assertEqual(decoded["decoder"],"reduced_inequality_lp")
            self.assertAlmostEqual(decoded["x"][3],1.)
            self.assertAlmostEqual(decoded["objective"],0.)
            proposed=fitted.predict_fixed(self.validation.train_costs[0],self.validation.train_features[0],target_K=1)
            self.assertEqual(len(proposed["fixed"]),1)
            self.assertTrue(set(proposed["fixed"])<=set(self.train.commitment_indices))
            # A framework-owned outer marker cannot collide with neural load.
            (Path(temp)/"complete.json").write_text('{"guard":"outer framework"}')
            restored=api.load(self.train,"preference_learning",temp)
            self.assertEqual(restored.predict_fixed(self.validation.train_costs[0],self.validation.train_features[0],target_K=1)["fixed"],proposed["fixed"])

    def test_nd_external_split_and_final_refit_indices(self):
        from nd_training import fit_nd_complete
        config=nd.NeuralDivingConfig(seed=19,width_candidates=(32,),warmup_epochs=1,
            use_root_lp_features=False,validation_solver_count=2,sanity_steps=1)
        seen=[]
        def select(graph,cfg,costs,pools,fit_indices,val_indices):
            np.testing.assert_array_equal(fit_indices,np.arange(8))
            np.testing.assert_array_equal(val_indices,np.arange(8,10))
            np.testing.assert_array_equal(costs[:8],self.train.train_costs)
            np.testing.assert_array_equal(costs[8:],self.validation.train_costs)
            seen.append("selection")
            return nd._state(nd._new_network(graph,cfg,cfg.seed+110003)),1,{"best_validation_full_nll":1.}
        original_epoch=nd._train_epoch
        def train_epoch(network,optimizer,costs,pools,indices,rng,cfg):
            np.testing.assert_array_equal(indices,np.arange(8));seen.append("refit")
            return original_epoch(network,optimizer,costs,pools,indices,rng,cfg)
        with patch.object(nd,"_sanity_gate",side_effect=AssertionError("Formal fits must not run the1200-step QA loop")),\
             patch.object(nd,"_warmup_selection",side_effect=select),\
             patch.object(nd,"_train_epoch",side_effect=train_epoch):
            model=fit_nd_complete(self.train,self.validation,config=config)
        self.assertEqual(seen,["selection","refit"])
        self.assertAlmostEqual(model.graph.cost_scale,np.sqrt(np.mean(self.train.train_costs**2)))
        self.assertEqual(model.training_metadata["train_cost_count"],8)
        self.assertFalse(model.training_metadata["validation_labels_in_final_fit"])
        learner=api.NeuralLearner(self.train,"neural_diving",model,{})
        for K in (0,1,2):
            prediction=learner.predict_fixed(self.validation.train_costs[0],target_K=K)
            self.assertEqual(prediction["actual_K"],K)
            self.assertEqual(len(prediction["fixed"]),2-K)


if __name__=="__main__": unittest.main()
