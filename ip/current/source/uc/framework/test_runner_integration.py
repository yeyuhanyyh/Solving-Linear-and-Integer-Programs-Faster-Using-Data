"""Regression for nested checkpoint markers, using tiny real fits only."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest
import numpy as np
import test_contract
import adapter
import run
from cases import make_case


class RunnerCheckpointTests(unittest.TestCase):
    def test_outer_neural_checkpoint_fit_load(self):
        case=test_contract.fixture()
        factors=np.arange(1.,1.8,.1)[:8]
        train_data=dict(features=factors[:,None]*case.train_features[0],
            costs=factors[:,None]*case.train_costs[0],
            solutions=np.repeat(case.train_solutions[[0]],8,axis=0),
            objectives=factors*case.train_objectives[0],
            certified=np.ones(8,bool),dual_bounds=factors*case.train_objectives[0])
        case=make_case(case.model,train_data,dict(case_id="runner_train",seed=1),case.commitment_indices)
        costs=case.train_costs.copy();costs[:,0]+=.123;costs[:,3]+=.231
        results=[adapter.solve(case.model,c,time_limit=5.) for c in costs]
        data=dict(features=costs[:,:4],costs=costs,solutions=np.array([r.x for r in results]),
            objectives=np.array([r.objective for r in results]),certified=np.array([r.certified for r in results]),
            dual_bounds=np.array([r.dual_bound for r in results]))
        validation=make_case(case.model,data,dict(case_id="runner_validation"),case.commitment_indices)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root=Path(temporary)
            config=root/"config.json"
            config.write_text(json.dumps(dict(epochs=1,learning_rates=[1e-4],lambda1_values=[.8],
                lambda2_values=[.1],top_k_values=[1],hidden_dim=16,attention_layers=1)))
            args=SimpleNamespace(out=str(root/"run"),n=8,validation_n=8,method="ours",config=None,device="cpu")
            with patch.object(run,"load_training",return_value=(case,validation)):
                run.fit(args)
                args.method="preference_learning";args.config=str(config)
                run.fit(args)
            folder,_=run.verify_checkpoint(args,case,validation)
            self.assertTrue((folder/"complete.json").exists())
            self.assertTrue((folder/"neural-complete.json").exists())
            learner=run.load_neural_api().load(case,args.method,folder)
            proposed=learner.predict_fixed(costs[0],costs[0,:4],target_K=0,time_limit=5.)
            self.assertEqual(len(proposed["fixed"]),4)


if __name__=="__main__":unittest.main()
