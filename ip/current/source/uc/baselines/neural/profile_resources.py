"""Gradient and full diagnostic resource probes on certified training only.

This is not fitting, validation or test evaluation. It saves no usable network.
Run on the intended training host before freezing a memory-feasible batch size.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/"framework"))
sys.path.insert(0,str(ROOT))
import neural_support as support
support.enable_runtime()
import torch
from torch.nn import functional as F
import neural_diving_baseline as nd


def profile_case(case,*,device="cuda",batch_size=16,widths=(32,64),root_lp_time_limit=5.):
    if batch_size<1 or batch_size>case.n_train: raise ValueError("Batch must fit supplied training bank")
    if device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    results=[]
    for width in widths:
        config=nd.NeuralDivingConfig(seed=int(case.metadata.get("baseline_seed",case.seed)),device=device,
            hidden_size=int(width),width_candidates=(int(width),),batch_size=batch_size,
            root_lp_time_limit=root_lp_time_limit)
        nd._setup(config)
        row={"config":asdict(config),"training_hash":support.training_hash(case),
             "probe_training_indices":list(range(batch_size)),"saved_fitted_model":False}
        graph=network=optimizer=logits=acceptance=target=loss=None
        try:
            if device=="cuda": torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            graph=nd.FixedBipartiteGraph(case.model,case.train_costs,config.normalize_constraint_rows,
                device,config.use_root_lp_features,config.root_lp_time_limit,config.seed)
            root=graph.precompute_training_root_lp(case.train_costs)
            row["graph_and_root_feature_seconds"]=time.perf_counter()-start
            row["root_features"]=root
            row["full_training_root_lp_cache_precomputed"]=True
            network=nd._new_network(graph,config,config.seed)
            optimizer=nd._optimizer(network,config)
            nd._sync(device);start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits,acceptance=network(case.train_costs[:batch_size])
            target=torch.as_tensor(case.train_solutions[:batch_size,graph.integer_indices],dtype=torch.float32,device=device)
            loss=F.binary_cross_entropy_with_logits(logits,target,reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),config.gradient_clip)
            optimizer.step()
            nd._sync(device)
            row.update(one_gradient_step_seconds=time.perf_counter()-start,
                singleton_full_assignment_nll=float(loss.detach().cpu()),
                parameter_count=sum(p.numel() for p in network.parameters()),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None,
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device=="cuda" else None)
            # Formal diagnostics run after a step while Adam state and parameter
            # gradients remain live. They must cover both validation-size and
            # full-training-size calls, using training data only for this probe.
            del logits,acceptance,target,loss
            logits=acceptance=target=loss=None
            costs,pools,_=nd._prepare_pools(case,support.singleton_pools(case),graph.integer_indices)
            diagnostics=[]
            for count in sorted({min(64,case.n_train),case.n_train}):
                observed=[]
                hook=network.register_forward_pre_hook(lambda module,args: observed.append(len(args[0])))
                nd._sync(device);start=time.perf_counter()
                try:
                    values=nd._full_diagnostics(network,costs,pools,np.arange(count,dtype=int))
                finally:
                    hook.remove()
                nd._sync(device)
                if (not np.isfinite(values["full_nll_per_integer"]) or not observed
                        or max(observed)>batch_size or sum(observed)!=count):
                    raise RuntimeError("Full ND diagnostics did not use bounded complete evaluation batches")
                diagnostics.append(dict(training_count=count,max_forward_batch_size=max(observed),
                    forward_batch_sizes=observed,success=True,seconds=time.perf_counter()-start,
                    finite_diagnostics=True,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None,
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device=="cuda" else None))
            row.update(success=True,diagnostics=diagnostics,diagnostics_after_optimizer_step=True,
                optimizer_state_and_gradients_retained_for_diagnostics=True,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None,
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device=="cuda" else None)
        except Exception as error:
            row.update(success=False,error=repr(error),traceback=traceback.format_exc())
        finally:
            del graph,network,optimizer,logits,acceptance,target,loss
            gc.collect()
            if device=="cuda": torch.cuda.empty_cache()
        results.append(row)
        print(json.dumps({key:row.get(key) for key in ("success","one_gradient_step_seconds","parameter_count","peak_gpu_allocated_bytes","error")}),flush=True)
    return dict(probe="one training gradient batch and complete bounded diagnostic paths, no fit or test result",results=results,
        device=device,gpu_name=torch.cuda.get_device_name(0) if device=="cuda" else None,
        torch_version=torch.__version__,training_count=case.n_train)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--n",type=int,default=256)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--widths",type=int,nargs="+",default=[32,64])
    parser.add_argument("--root-lp-time-limit",type=float,default=5.)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve prior resource probes")
    from cases import load_bank_split
    case=load_bank_split(args.bank,"train",n=args.n,target_K=0)
    result=profile_case(case,device=args.device,batch_size=args.batch_size,widths=args.widths,
        root_lp_time_limit=args.root_lp_time_limit)
    support.write_json(args.output,result)


if __name__=="__main__": main()
