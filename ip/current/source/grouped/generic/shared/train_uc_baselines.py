"""Resumable Table 5/6 supplemental ND-complete and Li training/evaluation.

No mode starts by default. `--validate-only` checks frozen training data without
optimization; `--fit` trains selected case/method checkpoints; `--evaluate`
opens the archived 80-cost confirmation stream only after checkpoint validation.
This file does not launch cloud instances and never edits manuscript files.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid
import numpy as np
import uc_adapters as data

METHODS=("neural_diving","li")


def utc(): return datetime.now(timezone.utc).isoformat()


def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def event(**fields): print(json.dumps(data.clean(dict(utc=utc(),**fields))),flush=True)


def runtime_metadata():
    import scipy,torch,highspy,threadpoolctl
    return dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
        numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,highs=highspy.Highs().version(),
        cuda_available=torch.cuda.is_available(),cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        torch_threads=torch.get_num_threads(),threadpools=threadpoolctl.threadpool_info(),
        shared_device_contention_not_measured=True)


def fit_nd_complete(case, device="cpu"):
    """Reuse validated GCN/NLL/gates without training five unused mask models.

    Select width and NLL early-stopping epoch within the training prefix. As
    inference is the common-K complete predictor, width validation evaluates
    that same rule at archived K, rather than a native five-mask ensemble.
    """
    import torch
    import neural_diving_baseline as nd
    config=nd.NeuralDivingConfig(seed=int(case.metadata["baseline_seed"]),device=device)
    nd._setup(config)
    begin=time.perf_counter()
    costs=np.asarray(case.train_costs,dtype=float)
    perm=np.random.default_rng(config.seed).permutation(case.n_train)
    n_val=max(1,int(round(case.n_train*config.validation_fraction)))
    validation_indices=np.sort(perm[:n_val]); training_indices=np.sort(perm[n_val:])
    feature_start=time.perf_counter()
    graph=nd.FixedBipartiteGraph(case.model,costs[training_indices],config.normalize_constraint_rows,
        config.device,config.use_root_lp_features,config.root_lp_time_limit,config.seed)
    root_lp_metadata=graph.precompute_training_root_lp(costs)
    feature_seconds=time.perf_counter()-feature_start
    costs,pools,pool_metadata=nd._prepare_pools(case,data.singleton_pools(case),graph.integer_indices)
    reference=np.asarray(case.train_objectives,dtype=float)
    validation_solver=np.sort(np.random.default_rng(config.seed+130003).choice(validation_indices,
        min(config.validation_solver_count,n_val),replace=False))
    import adapter
    candidates=[]; chosen=None; best_score=None; selection_start=time.perf_counter()
    for width in config.width_candidates:
        candidate_config=replace(config,hidden_size=int(width))
        gate=nd._sanity_gate(graph,candidate_config,case,costs,pools,training_indices,reference,adapter.solve)
        candidate=dict(width=int(width),sanity_gate=gate,eligible=False)
        if not gate["passed"]:
            candidate["reason"]="original training-only tiny-fit/integration gate failed"
            candidates.append(candidate); continue
        state,epochs,history=nd._warmup_selection(graph,candidate_config,costs,pools,training_indices,validation_indices)
        predictor=nd._new_network(graph,candidate_config,config.seed+110003)
        predictor.load_state_dict(state); predictor.eval()
        shell=nd.NeuralDivingModel(graph,[],predictor,candidate_config,{})
        records=[nd.solve_matched_cost(shell,case,costs[int(i)],config.seed+300001+int(i)*4099,
            K=int(case.affine["rank"]),total_time_limit=config.final_validation_time_limit,solve_fn=adapter.solve)
            for i in validation_solver]
        quality=nd._quality(records,reference,validation_solver)
        # No test costs and no speed metric select architecture. The same
        # fixed train-only heldout cases and K are used for both widths.
        gaps=[max(0.,(float(r["objective"])-reference[int(i)])/max(abs(reference[int(i)]),1e-12)) for i,r in zip(validation_solver,records)
              if r.get("valid") and r.get("objective") is not None]
        mean_gap=float(np.mean(gaps)) if gaps else 1e6
        score=(quality["recovered_count"],quality["valid_count"],-mean_gap,
               -history["best_validation_full_nll"],-int(width))
        candidate.update(eligible=True,warmup_epochs=int(epochs),warmup_history=history,
                         matched_validation=quality,validation_mean_gap_valid=mean_gap,selection_score=list(score))
        candidates.append(candidate)
        if chosen is None or score>best_score: chosen=candidate;best_score=score
        del shell,predictor
        if device=="cuda": torch.cuda.empty_cache()
    if chosen is None:
        raise nd.TrainingSanityError("No complete-predictor width passed original sanity gate",
            {"passed":False,"width_candidates":candidates})
    selection_seconds=time.perf_counter()-selection_start
    chosen_config=replace(config,hidden_size=chosen["width"])
    training_lp_cache=dict(graph.training_lp_cache)
    graph=nd.FixedBipartiteGraph(case.model,costs,config.normalize_constraint_rows,config.device,
        config.use_root_lp_features,config.root_lp_time_limit,config.seed)
    graph.training_lp_cache=training_lp_cache
    final_indices=np.arange(case.n_train,dtype=int)
    refit_start=time.perf_counter()
    predictor=nd._new_network(graph,chosen_config,config.seed+110003)
    optimizer=nd._optimizer(predictor,chosen_config);rng=np.random.default_rng(config.seed+110003)
    losses=[]
    for epoch in range(1,chosen["warmup_epochs"]+1):
        losses.append(nd._train_epoch(predictor,optimizer,costs,pools,final_indices,rng,chosen_config))
        if epoch==1 or epoch%config.log_every==0:
            event(event="nd_complete_refit",case=case.case_id,width=chosen["width"],epoch=epoch,nll=losses[-1])
    synchronize(device); refit_seconds=time.perf_counter()-refit_start
    predictor.eval(); diagnostics=nd._full_diagnostics(predictor,costs,pools,final_indices)
    metadata=dict(method="Neural Diving common-K complete predictor",implementation="nd-complete-only-frozen-uc-v1",
        source_module_sha256=data.sha(Path(nd.__file__)),paper=nd.PAPER_URL,case_id=case.case_id,config=asdict(chosen_config),
        train_cost_count=case.n_train,train_cost_digest=nd._array_digest(costs),matrix_digest=graph.matrix_digest,
        common_coordinate_indices=nd._common_indices(case).tolist(),target_K=int(case.affine["rank"]),
        integer_variable_count=len(graph.integer_indices),continuous_variable_count=graph.n_variables-len(graph.integer_indices),
        train_indices=training_indices.tolist(),validation_indices=validation_indices.tolist(),
        solver_validation_indices=validation_solver.tolist(),final_fit_indices=final_indices.tolist(),
        validation_selection="matched-K recovery, valid count, smaller valid objective gap, NLL, smaller width; training prefix only",
        feature_version="formulation13-plus-certified-root-lp15-v1",root_lp_training=root_lp_metadata,
        feature_preparation_seconds=feature_seconds,epoch_selection_seconds=selection_seconds,final_refit_seconds=refit_seconds,
        training_seconds=time.perf_counter()-begin,selected_width=chosen["width"],selected_warmup_epochs=chosen["warmup_epochs"],
        selected_coverage_epochs=[],width_candidates=candidates,full_refit_losses=losses,
        final_full_predictor_diagnostics=diagnostics,parameter_count=sum(p.numel() for p in predictor.parameters()),
        sanity_gate={"passed":True,"width_candidates":[c["sanity_gate"] for c in candidates],"scope":"training_only"},
        training_label_budget=case.n_train,label_collection_seconds=0.,conditional_lp_completion_seconds=0.,
        complete_assignment_labels=case.n_train,all_labels_original_certified_optima=True,
        original_training_label_generation_cost_excluded=True,test_information_used=False,
        adaptation="Singleton certified-optimum pools, complete predictor only, no unused selective-mask training. Original GCN, root features, optimizer and sanity schedule retained.",
        **pool_metadata)
    return nd.NeuralDivingModel(graph,[],predictor,chosen_config,metadata)


def save_nd_complete(model,path):
    import torch
    import neural_diving_baseline as nd
    torch.save(dict(format_version="uc-nd-complete-v1",config=asdict(model.config),metadata=model.training_metadata,
                    full_predictor_state=nd._state(model.full_predictor)),path)


def load_nd_complete(case,path):
    import torch
    import neural_diving_baseline as nd
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload["format_version"]!="uc-nd-complete-v1": raise ValueError("Wrong checkpoint format")
    config=nd.NeuralDivingConfig(**payload["config"]);metadata=payload["metadata"]
    if not metadata["sanity_gate"]["passed"]: raise ValueError("Unready checkpoint")
    if nd._array_digest(case.train_costs)!=metadata["train_cost_digest"]: raise ValueError("Training checkpoint mismatch")
    graph=nd.FixedBipartiteGraph(case.model,case.train_costs,config.normalize_constraint_rows,"cpu",
        config.use_root_lp_features,config.root_lp_time_limit,config.seed)
    if graph.matrix_digest!=metadata["matrix_digest"]: raise ValueError("Physical checkpoint mismatch")
    predictor=nd._new_network(graph,replace(config,device="cpu"),config.seed)
    predictor.load_state_dict(payload["full_predictor_state"]);predictor.eval()
    return nd.NeuralDivingModel(graph,[],predictor,replace(config,device="cpu"),metadata)


def model_guard(case,method,bundle):
    manifest=data.read(Path(bundle)/"manifest.json")
    return dict(case_id=case.case_id,method=method,training_array_hash=case.dataset_fingerprint,
        target_K=int(case.affine["rank"]),runtime_sha256={key:value for key,value in manifest["file_sha256"].items() if key.startswith("runtime/")},
        adapter_sha256=data.sha(Path(data.__file__)),runner_sha256=data.sha(Path(__file__)))


def checkpoint_dir(out,case,method): return Path(out)/"models"/case.case_id/method


def validate_checkpoint(out,case,method,bundle):
    folder=checkpoint_dir(out,case,method);marker=data.read(folder/"complete.json")
    if marker["guard"]!=model_guard(case,method,bundle): raise ValueError(f"Checkpoint guard changed: {folder}")
    for name,expected in marker["artifact_sha256"].items():
        if data.sha(folder/name)!=expected: raise ValueError(f"Checkpoint artifact changed: {folder/name}")
    return folder,marker


def train_one(case,method,out,bundle,device):
    import torch
    final=checkpoint_dir(out,case,method)
    if final.exists():
        _,marker=validate_checkpoint(out,case,method,bundle)
        event(event="fit_already_complete",case=case.case_id,method=method,fit_seconds=marker["fit_seconds"]);return
    partial=final.with_name(method+".partial-"+uuid.uuid4().hex);partial.mkdir(parents=True)
    guard=model_guard(case,method,bundle);data.write_json(partial/"start.json",dict(utc=utc(),guard=guard,runtime=runtime_metadata()))
    try:
        synchronize(device);begin=time.perf_counter();cpu_begin=time.process_time()
        if method=="neural_diving":
            model=fit_nd_complete(case,device=device);metadata=model.training_metadata
            synchronize(device);fit_seconds=time.perf_counter()-begin;cpu_seconds=time.process_time()-cpu_begin
            save_nd_complete(model,partial/"complete-predictor.pt")
        elif method=="li":
            import preference_baseline as li
            import adapter
            config=li.PreferenceConfig(seed=int(case.metadata["baseline_seed"]),device=device,top_k_values=(1,3,5,10,30))
            class ExpandedCandidatePreference(li.PreferenceBaseline):
                # The development SetCover count is available here. Clipping
                # and deduplicating changes no network/epoch/optimizer setting.
                def _train_network(self, development_case, bank, pairs, *args, **kwargs):
                    effective=tuple(dict.fromkeys(min(k,len(pairs["selected"])) for k in (1,3,5,10,30)))
                    self.config=replace(self.config,top_k_values=effective)
                    return super()._train_network(development_case,bank,pairs,*args,**kwargs)
            model=ExpandedCandidatePreference(adapter.training_case(case),config=config).fit(output_dir=partial)
            metadata=model.metadata
            metadata["requested_top_k_grid"]=[1,3,5,10,30]
            metadata["effective_top_k_grid"]=list(model.config.top_k_values)
            metadata["top_k_clipping_scope"]="development SetCover size; duplicate effective candidate counts removed"
            metadata["matched_inference_adaptation"]="Native selected top_k and Eq21 winner strategy, then common-coordinate weighted agreement fixes D-K"
            model.save(partial)
            synchronize(device);fit_seconds=time.perf_counter()-begin;cpu_seconds=time.process_time()-cpu_begin
            if not metadata.get("sanity_gate",{}).get("passed"): raise ValueError("Li readiness gate failed")
        else: raise ValueError(method)
        if not metadata.get("sanity_gate",{}).get("passed"): raise ValueError("Training readiness gate failed")
        data.write_json(partial/"metadata.json",dict(method=method,metadata=metadata,fit_seconds=fit_seconds,
            fit_cpu_seconds=cpu_seconds,original_training_label_generation_excluded=True,
            fit_stage_includes="training-only selection, pair generation/root-LP features, final refit and readiness checks",
            test_information_used=False,runtime=runtime_metadata(),utc=utc()))
        artifacts={str(path.relative_to(partial)).replace("\\","/"):data.sha(path) for path in partial.rglob("*") if path.is_file()}
        data.write_json(partial/"complete.json",dict(guard=guard,artifact_sha256=artifacts,fit_seconds=fit_seconds,
            fit_cpu_seconds=cpu_seconds,sanity_gate_passed=True,training_label_count=case.n_train,utc=utc()))
        os.rename(partial,final)
        event(event="fit_complete",case=case.case_id,method=method,fit_seconds=fit_seconds,cpu_seconds=cpu_seconds)
        del model
    except Exception as error:
        data.write_json(partial/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),utc=utc()))
        raise
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def propose_li_native_anchor(model,case,cost,features,budget):
    """Preserve Li's trained top-k/Eq.(21) winner before common-K adaptation.

    The winning historical integer pattern is feasible in this fixed-feasibility
    family, even when Li's relaxed active-row decoder is itself infeasible. The
    added coordinate restriction uses that stored pattern, never a test optimum.
    No assignment is proposed when native selection exhausts its shared budget.
    """
    from strategy_match import _native_scores,_standardized_features
    from dimension_diagnostics import common_coordinate_indices,target_common_dimension,assert_matched_fixings
    native=model.solve(cost,features,total_time_limit=float(budget))
    native_x=native.pop("x",None)
    diagnostic=dict(implementation="li-native-eq21-anchor-common-k-v1",native_selection=native,
        anchor_selection="original top_k and Eq21 minimum normalized infeasibility, stable predicted-order ties",
        old_table34_top1_score_anchor_used=False,original_algorithm=False,test_optimum_used=False,
        native_decoded_original_feasible=bool(native.get("valid")),
        native_decoded_objective=None if native_x is None else float(np.asarray(cost)@native_x),
        no_full_fallback=True,no_warm_start=True)
    sid=native.get("strategy_id")
    if sid is None or native.get("time_limit_exhausted",False):
        diagnostic["anchor_available"]=False
        return None,diagnostic
    common=common_coordinate_indices(case);K=target_common_dimension(case)
    selected=np.asarray(model.selected,dtype=int)
    positions=np.flatnonzero(selected==int(sid))
    if len(positions)!=1: raise ValueError("Native Li winner is not a unique retained strategy")
    scores=_native_scores(model,_standardized_features(model,features),"li")
    if scores.shape!=selected.shape: raise ValueError("Li retained scores mismatch")
    integer=np.asarray(model.bank.integer_indices,dtype=int)
    locations={int(j):i for i,j in enumerate(integer)}
    common_positions=np.asarray([locations[int(j)] for j in common],dtype=int)
    patterns=np.asarray([np.rint(model.bank.strategies[int(s)].integer_values).astype(int)[common_positions]
                         for s in selected])
    anchor=patterns[int(positions[0])]
    weights=np.exp(scores-np.max(scores));weights/=weights.sum()
    agreement=np.sum(weights[:,None]*(patterns==anchor),axis=0)
    order=np.lexsort((common,-agreement));chosen=order[:len(common)-K]
    fixed={int(common[i]):int(anchor[i]) for i in chosen}
    dimensions=assert_matched_fixings(case,fixed,target_K=K)
    diagnostic.update(anchor_available=True,anchor_strategy_id=int(sid),anchor_retained_position=int(positions[0]),
        retained_strategy_ids=selected.tolist(),all_retained_strategy_scores=scores.tolist(),
        confidence_softmax_temperature=1.,weighted_agreement=agreement.tolist(),confidence_weights=weights.tolist(),
        common_scope=common.tolist(),fixed_original_columns=common[chosen].tolist(),
        historical_integer_pattern_used=True,selected_strategy_decoder_rows_retained=False,
        dimension_diagnostics=dimensions)
    return fixed,diagnostic


def evaluate_one(case,method,out,bundle,budget):
    import torch
    import adapter
    from dimension_diagnostics import assert_matched_fixings
    folder,marker=validate_checkpoint(out,case,method,bundle)
    result_dir=Path(out)/"results"/case.case_id/method
    result_dir.mkdir(parents=True,exist_ok=True)
    guard=dict(checkpoint_sha256=data.sha(folder/"complete.json"),time_limit=float(budget),
        target_K=int(case.affine["rank"]),old_test_stream=True)
    if (result_dir/"complete.json").exists():
        if data.read(result_dir/"complete.json")["guard"]!=guard: raise ValueError("Evaluation guard changed")
        event(event="evaluation_already_complete",case=case.case_id,method=method);return
    if method=="neural_diving": model=load_nd_complete(case,folder/"complete-predictor.pt")
    else:
        import preference_baseline as li
        model=li.load_fitted(adapter.training_case(case),folder)
    test=data.load_confirmation(case,bundle)
    # One immutable file per test index: safe resume, no duplicate denominator.
    for i,(cost,features,reference) in enumerate(zip(test["costs"],test["features"],test["objectives"])):
        path=result_dir/f"test{i:03d}.json"
        cost_hash=data.array_hash(cost,features)
        if path.exists():
            old=data.read(path)
            if old["guard"]!=guard or old["cost_hash"]!=cost_hash: raise ValueError("Old test checkpoint changed")
            continue
        seed=(case.seed+20261025+int(i)*4099)%(2**31-1)
        begin=time.perf_counter()
        if method=="neural_diving":
            import neural_diving_baseline as nd
            record=nd.solve_matched_cost(model,case,cost,seed,K=int(case.affine["rank"]),
                total_time_limit=float(budget),solve_fn=adapter.solve)
            fixed=record.get("best_fixed")
        else:
            fixed,diagnostic=propose_li_native_anchor(model,case,cost,features,budget)
            remaining=float(budget)-(time.perf_counter()-begin)
            if fixed is None or remaining<=0:
                record=dict(x=None,objective=None,valid=False,certified=False,
                    status="no_native_strategy_anchor" if fixed is None else "prediction_budget_exhausted",diagnostic=diagnostic)
            else:
                solved=adapter.solve(case.model,cost,fixed=fixed,seed=seed,time_limit=remaining)
                record=solved.to_dict(include_solution=False);record["x"]=solved.x;record["diagnostic"]=diagnostic
        x=record.pop("x",None)
        original_feasible=bool(record.get("valid") and x is not None and adapter.validate_solution(case.model,x)["valid"])
        restriction_valid=bool(x is not None and fixed is not None and
            all(abs(float(x[int(j)])-float(value))<=1e-6 for j,value in fixed.items()))
        raw_objective=float(cost@x) if original_feasible else None
        seconds=time.perf_counter()-begin
        elapsed_overrun=seconds>float(budget)
        valid=bool(original_feasible and restriction_valid and not elapsed_overrun)
        objective=raw_objective if valid else None
        if fixed is not None: dimensions=assert_matched_fixings(case,fixed,target_K=int(case.affine["rank"]))
        else: dimensions=dict(target_K=int(case.affine["rank"]),no_returned_assignment=True)
        tol=adapter.recovery_tolerance(float(reference))
        if objective is not None and objective<float(reference)-tol:
            raise ValueError("Restricted objective beats archived certified optimum beyond tolerance")
        # Preserve late results as diagnostics only. Match archived methods:
        # a late objective enters neither recovery nor the objective-ratio mean.
        recovered=bool(valid and abs(objective-float(reference))<=tol)
        row=dict(guard=guard,case_id=case.case_id,method=method,test_index=i,cost_hash=cost_hash,
            training_count=case.n_train,physical_seed=case.seed,K=int(case.affine["rank"]),
            valid=valid,recovered=recovered,objective=objective,reference_objective=float(reference),
            raw_original_feasible=original_feasible,restriction_valid=restriction_valid,raw_objective=raw_objective,
            objective_ratio=objective/float(reference) if objective is not None else None,
            actual_dimension_diagnostics=dimensions,seconds=seconds,time_limit_exceeded=elapsed_overrun,
            record=record,solution=None if x is None else np.asarray(x),fixed=fixed,utc=utc())
        data.write_json(path,row)
    rows=[data.read(result_dir/f"test{i:03d}.json") for i in range(80)]
    ratios=[row["objective_ratio"] for row in rows if row["objective_ratio"] is not None]
    summary=dict(count=80,recovered_count=sum(row["recovered"] for row in rows),
        recovery_percent=100*sum(row["recovered"] for row in rows)/80,
        objective_ratio_mean_valid=float(np.mean(ratios)) if ratios else None,
        ratio_denominator=len(ratios),mean_seconds=float(np.mean([row["seconds"] for row in rows])),
        late_count=sum(row["time_limit_exceeded"] for row in rows),
        K=int(case.affine["rank"]),fit_seconds=marker["fit_seconds"])
    data.write_json(result_dir/"complete.json",dict(guard=guard,summary=summary,
        test_record_sha256={f"test{i:03d}.json":data.sha(result_dir/f"test{i:03d}.json") for i in range(80)},utc=utc()))
    event(event="evaluation_complete",case=case.case_id,method=method,summary=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=data.DEFAULT_BUNDLE)
    parser.add_argument("--out",type=Path,default=data.ROOT/"uc_results")
    parser.add_argument("--cases",nargs="+",choices=data.list_case_ids())
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--budget",type=float,default=30.)
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--fit",action="store_true")
    parser.add_argument("--evaluate",action="store_true")
    args=parser.parse_args()
    if not any((args.validate_only,args.fit,args.evaluate)): parser.error("Choose an explicit operation")
    if args.validate_only and (args.fit or args.evaluate): parser.error("Validation mode cannot start training/evaluation")
    data.enable_runtime(args.bundle)
    import torch
    torch.set_num_threads(1)
    cases=args.cases or data.list_case_ids()
    for cid in cases:
        case=data.load_case(cid,args.bundle)
        event(event="training_data_verified",case=cid,n=case.n_train,K=int(case.affine["rank"]),
            hash=case.dataset_fingerprint,confirmation_accessed=False)
        if args.validate_only: continue
        for method in args.methods:
            if args.fit: train_one(case,method,args.out,args.bundle,args.device)
            if args.evaluate: evaluate_one(case,method,args.out,args.bundle,args.budget)


if __name__=="__main__": main()
