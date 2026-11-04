"""Small solver-backed correctness checks; not a UC performance experiment."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"framework"),str(ROOT)]
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import adapter
import ours
from cases import make_case
from scalable_strategy_bank import SparseStrategyBank
import mlopt_baseline as dense


def fixture():
    # Two binary choices with one continuous dispatch equality.
    model=adapter.LinearModel(sp.csr_matrix([[1,1,0,0,0],[0,0,1,1,0],[-1,0,-2,0,1]]),
        np.array([1.,1.,0.]),np.array([1.,1.,0.]),np.zeros(5),np.array([1,1,1,1,3.]),
        np.array([1,1,1,1,0]),tuple([f"ug[g{i//2},{i%2+1}]" for i in range(4)]+["p"]),
        ("choice0","choice1","dispatch"))
    costs=np.array([[0,2,2,0,.01],[2,0,0,2,.01],[0,3,3,0,.02],[3,0,0,3,.02]],dtype=float)
    results=[adapter.solve(model,c,time_limit=5.) for c in costs]
    assert all(r.certified and r.valid for r in results)
    data=dict(features=costs[:,:4],costs=costs,solutions=np.array([r.x for r in results]),
        objectives=np.array([r.objective for r in results]),certified=np.ones(4,dtype=bool),
        dual_bounds=np.array([r.dual_bound for r in results]))
    return make_case(model,data,dict(case_id="contract",seed=1),np.arange(4))


class ContractTests(unittest.TestCase):
    def test_exact_affine_and_auxiliary_domains(self):
        case=fixture();fit=ours.fit(case)
        self.assertEqual(fit.K,1)
        extended,cost=fit.extend(case.model,case.train_costs[0])
        self.assertEqual(extended.n_variables,6)
        np.testing.assert_array_equal(extended.integrality[:5],case.model.integrality)
        self.assertEqual(extended.integrality[-1],0)
        for c,objective in zip(case.train_costs,case.train_objectives):
            result=ours.query(fit,case,c,time_limit=5.)
            self.assertTrue(result.valid and result.certified)
            self.assertAlmostEqual(result.objective,objective,places=8)
        # A fresh objective whose global optimizer lies outside the learned line.
        c=np.array([0.,4.,0.,4.,.01])
        full=adapter.solve(case.model,c,time_limit=5.)
        restricted=ours.query(fit,case,c,time_limit=5.)
        self.assertGreater(restricted.objective,full.objective)

    def test_sparse_decoder_matches_dense_qr(self):
        case=fixture();old=dense.StrategyBank(case);new=SparseStrategyBank(case)
        self.assertEqual(len(old.strategies),len(new.strategies))
        for j in range(len(old.strategies)):
            for c in case.train_costs:
                a,b=old.decode(j,c),new.decode(j,c)
                self.assertTrue(a["valid"] and b["valid"])
                np.testing.assert_allclose(a["x"],b["x"],atol=1e-9)
                self.assertAlmostEqual(a["objective"],b["objective"],places=8)
        # Public square-basis factor is absent, so Li cannot use its old shortcut.
        self.assertTrue(all(s.factor is None for s in new.strategies))

    def test_reject_unclosed_label_gap(self):
        case=fixture()
        data=dict(features=case.train_features,costs=case.train_costs,solutions=case.train_solutions,
            objectives=case.train_objectives,certified=np.ones(4,bool),dual_bounds=case.train_objectives-1.)
        with self.assertRaises(ValueError):make_case(case.model,data,{},case.commitment_indices)

    def test_baseline_real_fit_and_matched_scope(self):
        case=fixture();case.affine={"rank":1}
        costs=case.train_costs.copy();costs[:,0]+=.123;costs[:,3]+=.231
        results=[adapter.solve(case.model,c,time_limit=5.) for c in costs]
        data=dict(features=costs[:,:4],costs=costs,solutions=np.array([r.x for r in results]),
            objectives=np.array([r.objective for r in results]),certified=np.array([r.certified for r in results]),
            dual_bounds=np.array([r.dual_bound for r in results]))
        validation=make_case(case.model,data,dict(case_id="contract_validation"),case.commitment_indices)
        from baselines.svm.wrapper import fit as fit_svm
        svm=fit_svm(case,config={"mode":"matched"},validation_case=validation)
        proposal=svm.predict_restriction(costs[0],features=costs[0,:4],target_K=1)
        self.assertEqual(len(proposal["fixed"]),3)
        self.assertTrue(set(proposal["fixed"])<=set(case.commitment_indices))
        from baselines.mlopt.api import fit as fit_mlopt
        with patch.object(dense,"_train_network",wraps=dense._train_network) as train_network:
            mlopt=fit_mlopt(case,validation,config=dict(hidden_candidates=((4,),),learning_rates=(.01,),
                epoch_candidates=(1,),device="cpu",n_best=2))
        train_network.assert_called_once()
        self.assertFalse(mlopt.manifest["single_class_fast_path"])
        proposal=mlopt.predict_restriction(costs[0],features=costs[0,:4],target_K=1)
        self.assertEqual(len(proposal["fixed"]),3)
        self.assertEqual(mlopt.manifest["training_label_count"],4)
        self.assertFalse(mlopt.manifest["validation_labels_used_in_final_bank"])
        import pickle
        restored=pickle.loads(pickle.dumps(mlopt))
        again=restored.predict_restriction(costs[0],features=costs[0,:4],target_K=1)
        self.assertEqual(proposal["fixed"],again["fixed"])

    def test_mlopt_single_class_skips_network_preserves_predictions(self):
        original=fixture()
        def subset(indices,name):
            data=dict(features=original.train_features[indices],costs=original.train_costs[indices],
                solutions=original.train_solutions[indices],objectives=original.train_objectives[indices],
                certified=np.ones(len(indices),bool),dual_bounds=original.train_objectives[indices])
            return make_case(original.model,data,dict(case_id=name),original.commitment_indices,target_K=0)
        case=subset([0,2],"single_train");validation=subset([1,3],"single_validation")
        from baselines.mlopt.api import fit as fit_mlopt,Baseline
        with patch.object(dense,"_train_network",side_effect=AssertionError("One class needs no gradient training")):
            fitted=fit_mlopt(case,validation,config=dict(device="cpu"))
        self.assertTrue(fitted.manifest["single_class_fast_path"])
        self.assertEqual(fitted.manifest["classifier_gradient_epochs"],0)
        self.assertEqual(len(fitted.manifest["validation_trials"]),1)
        self.assertEqual(fitted.manifest["validation_trials"][0]["valid"],validation.n_train)
        m=fitted.model
        # Any finite scalar logit has the same one-class posterior and ranking.
        other_weights=[(np.ones_like(m.weights[0][0])*7,np.array([-3.],dtype=np.float32))]
        reference=Baseline(dense.MLOPTModel(m.bank,m.selected,other_weights,m.mean,m.scale,m.config),case)
        for features,cost in zip(validation.train_features,validation.train_costs):
            left=fitted.predict_native(cost,features);right=reference.predict_native(cost,features)
            np.testing.assert_array_equal(left["x"],right["x"])
            self.assertEqual(left["objective"],right["objective"])
            for K in (0,2,4):
                self.assertEqual(fitted.predict_restriction(cost,features,K)["fixed"],
                    reference.predict_restriction(cost,features,K)["fixed"])


if __name__=="__main__":unittest.main()
