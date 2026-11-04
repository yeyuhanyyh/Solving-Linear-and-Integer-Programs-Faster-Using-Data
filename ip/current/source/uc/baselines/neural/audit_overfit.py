"""Optional one-time ND implementation QA, explicitly outside learning time.

This 1200-step eight-instance check was a prior local diagnostic, not a Nair
algorithm requirement. Formal fit never calls it and never reads its outcome.
No reusable prediction checkpoint is saved here.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/"framework"))
sys.path.insert(0,str(ROOT))
import neural_support as support
support.enable_runtime()
import neural_diving_baseline as nd
import adapter
from neural_timing import install
install()


def audit(case,*,device="cuda",widths=(32,64)):
    records=[]
    for width in widths:
        config=nd.NeuralDivingConfig(seed=int(case.metadata.get("baseline_seed",case.seed)),device=device,
            hidden_size=width,width_candidates=(width,))
        nd._setup(config)
        start=time.perf_counter()
        graph=nd.FixedBipartiteGraph(case.model,case.train_costs,config.normalize_constraint_rows,
            device,config.use_root_lp_features,config.root_lp_time_limit,config.seed)
        graph.precompute_training_root_lp(case.train_costs[:config.sanity_train_count])
        costs,pools,_=nd._prepare_pools(case,support.singleton_pools(case),graph.integer_indices)
        diagnostic=nd._sanity_gate(graph,config,case,costs,pools,np.arange(case.n_train),
            np.asarray(case.train_objectives),adapter.solve)
        records.append(dict(width=width,qa_seconds=time.perf_counter()-start,config=asdict(config),diagnostic=diagnostic))
    return dict(scope="Optional implementation QA only, not a formal fitting prerequisite",
        charged_to_learning_time=False,usable_checkpoint_saved=False,
        training_hash=support.training_hash(case),records=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--n",type=int,default=256)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--widths",type=int,nargs="+",default=[32,64])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve previous QA")
    from cases import load_bank_split
    case=load_bank_split(args.bank,"train",n=args.n,target_K=0)
    support.write_json(args.output,audit(case,device=args.device,widths=args.widths))


if __name__=="__main__": main()
