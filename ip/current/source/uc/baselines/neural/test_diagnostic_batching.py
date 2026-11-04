"""Evaluation batching equivalence and full-path memory-probe regressions."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parents[1]/"framework"))
import neural_support
neural_support.enable_runtime()
import neural_diving_baseline as nd
from test_neural_adapters import make_case


def unbatched_reference(network,costs,pools,indices):
    network.eval()
    with torch.no_grad():
        logits,_=network(costs[indices])
        nll=float(nd._full_loss(logits,indices,pools))
        targets=torch.stack([(pools[int(i)]["weights"][:,None]*pools[int(i)]["labels"]).sum(0)
                             for i in indices]).to(logits)
        errors=torch.where(torch.sigmoid(logits)>=.5,1-targets,targets)
        losses=F.binary_cross_entropy_with_logits(logits,targets,reduction="none")
        blocks={name:{"full_nll_per_integer":float(losses[:,lo:hi].mean()),
                      "expected_map_bit_error":float(errors[:,lo:hi].mean())}
                for name,(lo,hi) in network.graph.diagnostic_blocks.items()}
    floor=nd._entropy_floor(indices,pools)
    return dict(full_nll_per_integer=nll,marginal_entropy_floor=floor,excess_nll=nll-floor,
                expected_map_bit_error=float(errors.mean()),blocks=blocks)


class DiagnosticBatchingTests(unittest.TestCase):
    def test_uneven_chunks_preserve_weighted_metrics_and_parameters(self):
        case=make_case([[1.+i*.01,3.,2.,2.] for i in range(11)],"batch-equivalence")
        config=nd.NeuralDivingConfig(batch_size=3,use_root_lp_features=False)
        nd._setup(config)
        graph=nd.FixedBipartiteGraph(case.model,case.train_costs,True,"cpu",False,5.,19)
        network=nd._new_network(graph,config,19)
        rng=np.random.default_rng(9)
        pools=[]
        for i in range(11):
            labels=torch.tensor(rng.integers(0,2,size=(1+i%3,len(graph.integer_indices))),dtype=torch.float32)
            weights=torch.tensor(rng.uniform(.1,1,size=len(labels)),dtype=torch.float64)
            weights/=weights.sum()
            pools.append(dict(labels=labels,weights=weights))
        indices=np.array([10,2,6,0,8,9,1,3,5,4,7])
        expected=unbatched_reference(network,case.train_costs,pools,indices)
        before={name:p.detach().clone() for name,p in network.named_parameters()}
        for p in network.parameters():p.grad=torch.ones_like(p)
        observed=[]
        hook=network.register_forward_pre_hook(lambda module,args:observed.append(len(args[0])))
        try:actual=nd._full_diagnostics(network,case.train_costs,pools,indices)
        finally:hook.remove()
        self.assertEqual(observed,[3,3,3,2])
        def compare(a,b):
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                if isinstance(a[key],dict):compare(a[key],b[key])
                else:self.assertAlmostEqual(a[key],b[key],delta=2e-7)
        compare(actual,expected)
        for name,p in network.named_parameters():
            torch.testing.assert_close(p,before[name],rtol=0,atol=0)
            torch.testing.assert_close(p.grad,torch.ones_like(p),rtol=0,atol=0)

    def test_resource_probe_covers_validation_size_and_full_training_size(self):
        import profile_resources
        case=make_case([[1.+i*.001,3.,2.,2.] for i in range(67)],"probe-complete-paths")
        result=profile_resources.profile_case(case,device="cpu",batch_size=16,widths=(32,))
        row=result["results"][0]
        self.assertTrue(row["success"],row.get("error"))
        self.assertTrue(row["full_training_root_lp_cache_precomputed"])
        self.assertTrue(row["diagnostics_after_optimizer_step"])
        self.assertTrue(row["optimizer_state_and_gradients_retained_for_diagnostics"])
        self.assertEqual([r["training_count"] for r in row["diagnostics"]],[64,67])
        for check in row["diagnostics"]:
            self.assertLessEqual(check["max_forward_batch_size"],16)
            self.assertEqual(sum(check["forward_batch_sizes"]),check["training_count"])


if __name__=="__main__":unittest.main()
