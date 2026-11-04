"""Stable external-validation fit/predict/solve API for audited UC baselines.

Framework must make its adapter, dimension_diagnostics and mlopt_baseline
modules importable. This module never generates labels or opens test banks.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import neural_support as data
data.enable_runtime()
import neural_diving_baseline as nd
import preference_baseline as li
from li_scaling import install as _install_scalable_li
_install_scalable_li()

METHODS=("neural_diving","preference_learning")


def propose_li_native_anchor(model,case,cost,features,budget,target_K):
    """Same Oct25 native top-k/Eq21 anchor, then weighted-agreement fixings."""
    from strategy_match import _native_scores,_standardized_features
    from dimension_diagnostics import common_coordinate_indices,assert_matched_fixings
    native=model.solve(cost,features,total_time_limit=float(budget))
    native_x=native.pop("x",None)
    diagnostic=dict(implementation="li-native-eq21-anchor-common-k-v1",native_selection=native,
        original_algorithm=False,anchor_selection="Native top-k and Eq21 normalized infeasibility; stable predicted-order ties",
        native_decoded_original_feasible=bool(native.get("valid")),
        native_decoded_objective=None if native_x is None else float(np.asarray(cost)@native_x),
        no_full_fallback=True,no_warm_start=True,test_optimum_used=False)
    sid=native.get("strategy_id")
    if sid is None or native.get("time_limit_exhausted",False):
        diagnostic["anchor_available"]=False
        return None,diagnostic
    common=common_coordinate_indices(case);K=int(target_K)
    selected=np.asarray(model.selected,dtype=int)
    positions=np.flatnonzero(selected==int(sid))
    if len(positions)!=1: raise ValueError("Native winner is not a unique retained strategy")
    scores=_native_scores(model,_standardized_features(model,features),"li")
    if scores.shape!=selected.shape: raise ValueError("Retained scores mismatch")
    locations={int(j):i for i,j in enumerate(model.bank.integer_indices)}
    common_positions=np.asarray([locations[int(j)] for j in common],dtype=int)
    patterns=np.asarray([np.rint(model.bank.strategies[int(s)].integer_values).astype(int)[common_positions] for s in selected])
    anchor=patterns[int(positions[0])]
    weights=np.exp(scores-np.max(scores));weights/=weights.sum()
    agreement=np.sum(weights[:,None]*(patterns==anchor),axis=0)
    chosen=np.lexsort((common,-agreement))[:len(common)-K]
    fixed={int(common[i]):int(anchor[i]) for i in chosen}
    dimensions=assert_matched_fixings(case,fixed,target_K=K)
    diagnostic.update(anchor_available=True,anchor_strategy_id=int(sid),weighted_agreement=agreement.tolist(),
        confidence_softmax_temperature=1.,historical_integer_pattern_used=True,
        selected_strategy_decoder_rows_retained=False,dimension_diagnostics=dimensions)
    return fixed,diagnostic


class NeuralLearner:
    def __init__(self,case,method,model,metadata):
        self.case,self.method,self.model,self.metadata=case,method,model,metadata

    def predict_fixed(self,cost,features=None,*,target_K=None,time_limit=30.,seed=None):
        """Return global-column fixings plus complete prediction timing.

        ND prediction includes a fresh certified root LP. Li prediction includes
        native candidate decoding. Neither method gets a free query cache.
        """
        requested=self.case.affine["rank"] if target_K is None else target_K
        K=int(requested)
        if K!=requested: raise ValueError("K must be an integer")
        if not 0<=K<=len(self.case.commitment_indices): raise ValueError("Invalid K")
        start=time.perf_counter()
        if time_limit<=0: raise ValueError("Positive prediction budget required")
        if self.method=="neural_diving":
            from nd_root_lp_features import RootLPFeatureError
            try:
                root=self.model.graph.begin_online_root_lp(np.asarray(cost),time_limit)
                result=self.model.matched_partial_assignment(self.case,np.asarray(cost),K)
                fixed=result.pop("fixed")
                diagnostic=dict(result,root_lp_metadata=root,root_lp_seconds=root["seconds"],original_algorithm=False)
            except RootLPFeatureError as error:
                fixed=None;diagnostic=dict(status="root_lp_feature_failure",root_lp_metadata=error.metadata)
            finally:
                self.model.graph.end_online_root_lp()
        else:
            if features is None: raise ValueError("Li requires the original cost-parameter feature vector")
            fixed,diagnostic=propose_li_native_anchor(self.model,self.case,np.asarray(cost),np.asarray(features),time_limit,K)
        elapsed=time.perf_counter()-start
        if elapsed>time_limit:
            diagnostic.update(status="prediction_budget_exhausted",late_fixed_assignment=fixed)
            fixed=None
        return dict(fixed=fixed,prediction_seconds=elapsed,diagnostic=diagnostic,target_K=K,
                    actual_K=None if fixed is None else len(self.case.commitment_indices)-len(fixed))

    def solve(self,cost,features=None,*,mode="matched",target_K=None,time_limit=30.,seed=None):
        import adapter
        seed=int(self.case.seed if seed is None else seed)
        if mode=="native":
            if self.method=="neural_diving":
                raise ValueError("Complete-only ND checkpoint does not contain native acceptance heads")
            return self.model.solve(cost,features,total_time_limit=time_limit)
        if mode!="matched": raise ValueError("mode must be native or matched")
        start=time.perf_counter()
        predicted=self.predict_fixed(cost,features,target_K=target_K,time_limit=time_limit,seed=seed)
        remaining=time_limit-(time.perf_counter()-start)
        if predicted["fixed"] is None or remaining<=0:
            return dict(predicted,x=None,objective=None,valid=False,status="prediction_failed_or_budget_exhausted",
                        seconds=time.perf_counter()-start,solver_seconds=0.,method=self.method)
        solved=adapter.solve(self.case.model,np.asarray(cost),fixed=predicted["fixed"],seed=seed,time_limit=remaining)
        row=solved.to_dict(include_solution=False)
        row.update(predicted,x=solved.x,seconds=time.perf_counter()-start,solver_seconds=solved.solve_seconds,method=self.method)
        row["within_return_budget"]=row["seconds"]<=time_limit
        row["online_seconds"]=row["seconds"]
        row["raw_restricted_feasible"]=row["valid"]
        row["valid"]=bool(row["valid"] and row["within_return_budget"])
        row["globally_certified"]=False
        row["certificate_scope"]="restricted_model"
        return row

    forecast=solve


def fit(case,validation_case,method,output_dir,*,device="cpu",config=None):
    if method not in METHODS: raise ValueError(method)
    if isinstance(config,dict):
        config=dict(config)
        config.setdefault("seed",int(case.metadata["baseline_seed"]))
        config.setdefault("device",device)
        config=(nd.NeuralDivingConfig(**config) if method=="neural_diving" else li.PreferenceConfig(**config))
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
    if any(p.is_file() for p in destination.rglob("*")):
        raise FileExistsError("Timed formal fitting requires a fresh output directory. Preserve prior attempts separately; use load for a completed fit.")
    started=time.perf_counter()
    guard=dict(method=method,training_hash=data.training_hash(case),validation_hash=data.training_hash(validation_case),
        source_sha256=data.implementation_fingerprint(),device=device,
        configuration=None if config is None else asdict(config))
    guard_path=destination/"fit_guard.json"
    if guard_path.exists() and json.loads(guard_path.read_text())!=data.clean(guard):
        raise ValueError("An incomplete fit has different source/data/configuration; use a new output directory")
    data.write_json(guard_path,guard)
    data.validate_training_pair(case,validation_case)
    if method=="neural_diving":
        from nd_training import fit_nd_complete,save_nd_complete
        model=fit_nd_complete(case,validation_case,device=device,config=config)
        model.to("cpu")
        save_nd_complete(model,destination/"complete-predictor.pt")
        raw=model.training_metadata
    else:
        from li_training import ExternalValidationPreference
        config=config or li.PreferenceConfig(seed=int(case.metadata["baseline_seed"]),device=device,top_k_values=(1,3,5,10,30))
        model=ExternalValidationPreference(case,validation_case,config=config).fit(output_dir=destination)
        raw=model.metadata
    wall=time.perf_counter()-started
    learning=float(raw["training_seconds"] if method=="neural_diving" else raw["fit_seconds"])
    metadata=dict(method=method,fit_wall_seconds=wall,learning_time_seconds=learning,training_count=case.n_train,
        external_validation_count=validation_case.n_train,training_hash=data.training_hash(case),
        validation_hash=data.training_hash(validation_case),target_K=int(case.affine["rank"]),
        learning_time_excludes_shared_label_generation=True,
        learning_time_includes="Feature/pair preparation, validation model selection, final fitting and readiness checks",
        fit_wall_time_additionally_includes="Wrapper validation, deployment transfer and checkpoint preparation",
        original_algorithm=False,matched_dimension_rule="Exactly D-K common commitment coordinates are fixed",
        matched_inference_is_explicit_adaptation=True,raw_training_metadata=raw)
    data.write_json(destination/"metadata.json",metadata)
    artifacts={p.name:data.sha(p) for p in destination.iterdir() if p.is_file()}
    data.write_json(destination/"neural-complete.json",dict(training_hash=metadata["training_hash"],method=method,
        source_sha256=guard["source_sha256"],artifact_sha256=artifacts))
    return NeuralLearner(case,method,model,metadata)


def load(case,method,output_dir):
    destination=Path(output_dir)
    complete=json.loads((destination/"neural-complete.json").read_text())
    if complete["method"]!=method or complete["training_hash"]!=data.training_hash(case): raise ValueError("Checkpoint binding differs")
    if complete["source_sha256"]!=data.implementation_fingerprint(): raise ValueError("Checkpoint implementation changed")
    for name,digest in complete["artifact_sha256"].items():
        if data.sha(destination/name)!=digest: raise ValueError("Checkpoint artifact changed")
    if method=="neural_diving":
        from nd_training import load_nd_complete
        model=load_nd_complete(case,destination/"complete-predictor.pt")
    elif method=="preference_learning": model=li.load_fitted(case,destination)
    else: raise ValueError(method)
    metadata=json.loads((destination/"metadata.json").read_text())
    return NeuralLearner(case,method,model,metadata)
