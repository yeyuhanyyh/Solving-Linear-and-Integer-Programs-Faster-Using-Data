"""Audited complete-predictor ND fit with separate external validation.

Derived mechanically from the frozen October25 runner. Learner math unchanged.
"""
from dataclasses import asdict, replace
from pathlib import Path
import json
import time
import numpy as np
import neural_support as data
data.enable_runtime()
from neural_timing import install as _install_nd_deadline_check
_install_nd_deadline_check()

def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()

def event(**fields):
    print(json.dumps(data.clean(fields)),flush=True)

def fit_nd_complete(case, validation_case, device="cpu", config=None):
    """Reuse validated GCN/NLL/gates without training five unused mask models.

    Select width and NLL early-stopping epoch on the separate validation bank. As
    inference is the common-K complete predictor, width validation evaluates
    that same rule at archived K, rather than a native five-mask ensemble.
    """
    import torch
    import neural_diving_baseline as nd
    config=config or nd.NeuralDivingConfig(seed=int(case.metadata["baseline_seed"]),device=device)
    device=config.device
    data.validate_training_pair(case,validation_case)
    nd._setup(config)
    begin=time.perf_counter()
    selection_case=data.concatenate_for_selection(case,validation_case)
    costs=np.asarray(selection_case.train_costs,dtype=float)
    n_val=validation_case.n_train
    training_indices=np.arange(case.n_train,dtype=int)
    validation_indices=np.arange(case.n_train,case.n_train+n_val,dtype=int)
    feature_start=time.perf_counter()
    graph=nd.FixedBipartiteGraph(case.model,costs[training_indices],config.normalize_constraint_rows,
        config.device,config.use_root_lp_features,config.root_lp_time_limit,config.seed)
    root_lp_metadata=graph.precompute_training_root_lp(costs)
    feature_seconds=time.perf_counter()-feature_start
    costs,pools,pool_metadata=nd._prepare_pools(selection_case,data.singleton_pools(selection_case),graph.integer_indices)
    reference=np.asarray(selection_case.train_objectives,dtype=float)
    validation_solver=np.sort(np.random.default_rng(config.seed+130003).choice(validation_indices,
        min(config.validation_solver_count,n_val),replace=False))
    import adapter
    candidates=[]; chosen=None; best_score=None; selection_start=time.perf_counter()
    for width in config.width_candidates:
        candidate_config=replace(config,hidden_size=int(width))
        gate={"passed":True,"scope":"actual_training_numerical_checks_only","extra_overfit_steps":0}
        candidate=dict(width=int(width),sanity_gate=gate,eligible=False)
        if not gate["passed"]:
            candidate["reason"]="original training-only tiny-fit/integration gate failed"
            candidates.append(candidate); continue
        state,epochs,history=nd._warmup_selection(graph,candidate_config,costs,pools,training_indices,validation_indices)
        if state is None or epochs<1 or not np.isfinite(history["best_validation_full_nll"]):
            raise FloatingPointError("Actual ND training did not return a finite selected state")
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
    graph=nd.FixedBipartiteGraph(case.model,costs[training_indices],config.normalize_constraint_rows,config.device,
        config.use_root_lp_features,config.root_lp_time_limit,config.seed)
    graph.training_lp_cache=training_lp_cache
    final_indices=np.arange(case.n_train,dtype=int)
    refit_start=time.perf_counter()
    predictor=nd._new_network(graph,chosen_config,config.seed+110003)
    initial_refit_diagnostics=nd._full_diagnostics(predictor,costs,pools,final_indices)
    optimizer=nd._optimizer(predictor,chosen_config);rng=np.random.default_rng(config.seed+110003)
    losses=[]
    for epoch in range(1,chosen["warmup_epochs"]+1):
        losses.append(nd._train_epoch(predictor,optimizer,costs,pools,final_indices,rng,chosen_config))
        if epoch==1 or epoch%config.log_every==0:
            event(event="nd_complete_refit",case=case.case_id,width=chosen["width"],epoch=epoch,nll=losses[-1])
    synchronize(device); refit_seconds=time.perf_counter()-refit_start
    predictor.eval(); diagnostics=nd._full_diagnostics(predictor,costs,pools,final_indices)
    if not np.isfinite(diagnostics["full_nll_per_integer"]):
        raise FloatingPointError("Final ND predictor has nonfinite training loss")
    metadata=dict(method="Neural Diving common-K complete predictor",implementation="nd-complete-external-validation-uc-v1",
        source_module_sha256=data.sha(Path(nd.__file__)),paper=nd.PAPER_URL,case_id=case.case_id,config=asdict(chosen_config),
        train_cost_count=case.n_train,train_cost_digest=nd._array_digest(np.asarray(case.train_costs,dtype=float)),matrix_digest=graph.matrix_digest,
        common_coordinate_indices=nd._common_indices(case).tolist(),target_K=int(case.affine["rank"]),
        integer_variable_count=len(graph.integer_indices),continuous_variable_count=graph.n_variables-len(graph.integer_indices),
        train_indices=training_indices.tolist(),validation_indices=validation_indices.tolist(),
        solver_validation_indices=validation_solver.tolist(),final_fit_indices=final_indices.tolist(),
        validation_selection="matched-K recovery, valid count, smaller valid objective gap, NLL, smaller width; external validation only",
        external_validation_count=n_val, external_validation_hash=data.training_hash(validation_case),
        validation_labels_in_final_fit=False,
        feature_version="formulation13-plus-certified-root-lp15-v1",root_lp_training=root_lp_metadata,
        feature_preparation_seconds=feature_seconds,epoch_selection_seconds=selection_seconds,final_refit_seconds=refit_seconds,
        training_seconds=time.perf_counter()-begin,selected_width=chosen["width"],selected_warmup_epochs=chosen["warmup_epochs"],
        selected_coverage_epochs=[],width_candidates=candidates,full_refit_losses=losses,
        initial_full_predictor_diagnostics=initial_refit_diagnostics,
        final_full_predictor_diagnostics=diagnostics,
        actual_training_loss_decreased=bool(diagnostics["full_nll_per_integer"]<=initial_refit_diagnostics["full_nll_per_integer"]),
        diagnostic_forward_batch_size=chosen_config.batch_size,
        diagnostic_aggregation="All costs, original pool weights, cost-count-weighted chunk means; unchanged entropy floor",
        optional_overfit_qa_run=False,optional_overfit_qa_seconds=0.,parameter_count=sum(p.numel() for p in predictor.parameters()),
        sanity_gate={"passed":True,"width_candidates":[c["sanity_gate"] for c in candidates],"scope":"training_only"},
        training_label_budget=case.n_train,label_collection_seconds=0.,conditional_lp_completion_seconds=0.,
        complete_assignment_labels=case.n_train,all_labels_original_certified_optima=True,
        original_training_label_generation_cost_excluded=True,test_information_used=False,
        adaptation="Singleton certified-optimum pools, complete predictor only, no unused selective-mask training. Original GCN, root features and optimizer retained. The optional overfit diagnostic is excluded from formal fitting and its learning clock. Separate external validation replaces the internal split; final refit uses only the N training rows.",
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
