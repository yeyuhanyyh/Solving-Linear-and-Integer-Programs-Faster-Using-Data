"""Offline CPU controls and GPU paper models on training-only exported data.

This process never generates, reads, or solves confirmation test costs. Native
and matched-dimension Neural Diving share one trained predictor checkpoint.
"""
from __future__ import annotations
import argparse, os, pickle, sys, time, traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import adapter
from runner_common import (ROOT, METHODS, MODEL_KEYS, MODEL_FOR_METHOD, read, dump, clean, utc,
    event, selected, runtime, freeze, checkpoint, model_guard, artifact_hashes, PhaseLock, ensure_cpu)
import experiment_core
from xqa_baseline import XQAAffineBaseline


def pool_event(**values):
    if values["completed"]%10==0 or values["completed"]==1:
        event(event="conditional_pool_progress", **values)
        dump(ROOT/"logs"/"training-status.json",dict(phase="conditional_pool_completion",**values,utc=utc()))


def gate_passed(key: str, metadata: dict) -> bool:
    if key not in ("neural_diving","li_bounds_residual"):
        return True
    gate=metadata.get("sanity_gate")
    if not isinstance(gate,dict) or gate.get("passed") is not True:
        raise RuntimeError(f"{key} cannot enter confirmation without an explicit passed train/validation-only sanity gate: {gate}")
    return True


def train_case(case: adapter.FrozenCase, model_keys: list[str], frozen: dict, protocol: dict) -> None:
    traincase=adapter.training_case(case)
    for key in model_keys:
        existing=checkpoint(case,key,frozen,required=False)
        if existing is not None:
            event(event="model_already_complete",case=case.case_id,model=key,fit_seconds=existing["fit_seconds"])
            continue
        final=ROOT/"models"/case.case_id/key
        if final.exists():
            raise RuntimeError(f"Unguarded existing model cannot be overwritten: {final}")
        temporary=final.with_name(key+f".partial-{os.getpid()}")
        if temporary.exists():
            raise RuntimeError(f"Inspect interrupted partial fit before resuming: {temporary}")
        temporary.mkdir(parents=True)
        event(event="fit_start",case=case.case_id,model=key,training_costs=120)
        dump(ROOT/"logs"/"training-status.json",dict(phase="fit",case=case.case_id,model=key,utc=utc()))
        config=frozen["resolved_configs"][case.case_id][key]
        extra_pool_seconds=0.
        conditional_seconds=0.
        pools=None
        if key=="neural_diving":
            pools=adapter.complete_training_pools(case, per_lp_time_limit=float(protocol.get("conditional_pool_lp_time_limit",10.)),
                                                 progress=pool_event)
            extra_pool_seconds=sum(float(p["collection_seconds"]) for p in pools)
            conditional_seconds=sum(float(p["extra_conditional_lp_seconds"]) for p in pools)
            dump(temporary/"pool-summary.json",dict(training_costs=120,source_callback_pool_seconds=extra_pool_seconds,
                conditional_lp_completion_seconds=conditional_seconds,per_cost_sizes=[len(p["solutions"]) for p in pools],
                source_pool_sha256=[p["exported_pool_sha256"] for p in pools],
                completed_pool_sha256=[p["completed_pool_sha256"] for p in pools],
                conditional_lp_completion_certified=all(p["conditional_lp_completion_certified"] for p in pools)))
        begin=time.perf_counter()
        metadata={}
        model=None
        if key=="ours":
            binary=np.rint(case.train_solutions[:,case.commitment_indices]).astype(np.int64)
            affine=experiment_core.fit_affine(binary)
            for name in ("rank","representation","compressed_sample_count"):
                if affine[name]!=case.affine[name]:
                    raise RuntimeError(f"Ours refit differs from frozen Appendix C.3 in {name}")
            for name in ("anchor","basis","normals","normals_exact"):
                if not np.array_equal(affine[name],case.affine[name]):
                    raise RuntimeError(f"Ours exact refit differs in {name}")
            with (temporary/"affine.pkl").open("wb") as stream:
                pickle.dump(affine,stream,protocol=pickle.HIGHEST_PROTOCOL)
            metadata=dict(rank=affine["rank"],ambient_common_dimension=len(case.commitment_indices),
                representation=affine["representation"],compressed_sample_count=affine["compressed_sample_count"],
                singular_values=np.linalg.svd(affine["basis"],compute_uv=False))
        elif key=="svm_native":
            binary=np.rint(case.train_solutions[:,case.commitment_indices]).astype(np.int64)
            model=XQAAffineBaseline(random_state=case.seed).fit(case.train_features,binary,
                        n_times=case.original_base["n_times"],n_units=case.original_base["n_units"])
            with (temporary/"svm.pkl").open("wb") as stream:
                pickle.dump(model,stream,protocol=pickle.HIGHEST_PROTOCOL)
            metadata=dict(training_stats=model.training_stats,candidate_records=model.candidate_records(),
                          random_state=case.seed,estimator="unchanged original native SVM baseline")
        elif key=="mlopt_native":
            import mlopt_baseline
            model=mlopt_baseline.fit(traincase,config=config)
            ensure_cpu(model)
            model.save(temporary/"mlopt.pkl")
            metadata=model.metadata
        elif key=="neural_diving":
            import neural_diving_baseline
            model=neural_diving_baseline.fit(traincase,pools,config=config,validation=None,solve_fn=adapter.solve)
            ensure_cpu(model)
            model.save(temporary/"neural-diving.pt")
            metadata=model.training_metadata
        elif key=="li_bounds_residual":
            from preference_baseline import fit_case,PreferenceConfig
            model=fit_case(traincase,output_dir=temporary,config=PreferenceConfig(**config))
            ensure_cpu(model)
            metadata=model.metadata
        else:
            raise ValueError(key)
        seconds=time.perf_counter()-begin
        passed=gate_passed(key,metadata)
        dump(temporary/"metadata.json",dict(metadata=metadata,fit_seconds=seconds,
            source_pool_collection_seconds=extra_pool_seconds,conditional_lp_completion_seconds=conditional_seconds,
            cached_original_training_labels=120,offline_only=True,confirmation_data_accessed=False))
        marker=dict(guard=model_guard(case,key,frozen),artifact_sha256=artifact_hashes(temporary),
            fit_seconds=seconds,source_pool_collection_seconds=extra_pool_seconds,
            conditional_lp_completion_seconds=conditional_seconds,sanity_gate_passed=passed,utc=utc())
        dump(temporary/"complete.json",marker)
        os.replace(temporary,final)
        checkpoint(case,key,frozen)
        event(event="fit_complete",case=case.case_id,model=key,fit_seconds=seconds,
              conditional_lp_seconds=conditional_seconds,sanity_gate_passed=passed)
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",nargs="+")
    parser.add_argument("--methods",nargs="+",choices=METHODS)
    parser.add_argument("--device",choices=("cuda","cpu","auto"))
    parser.add_argument("--allow-cpu-training",action="store_true",help="Explicitly permit substantial CPU model training")
    parser.add_argument("--validate-only",action="store_true",help="Check data/interfaces without freezing, fitting, or drawing tests")
    args=parser.parse_args()
    cases=selected(args.cases,adapter.list_case_ids())
    methods=selected(args.methods,METHODS)
    keys=[key for key in MODEL_KEYS if key in {MODEL_FOR_METHOD[m] for m in methods if m!="full"}]
    protocol=read(ROOT/"protocol.json")
    if args.validate_only:
        from runner_common import resolved_configs
        device=args.device or protocol.get("device","cuda")
        adapter.dataset_manifest(verify=True)
        for cid in cases:
            adapter.load_case(cid,verify=True)
            resolved_configs(cid,protocol,device)
        event(event="portable_training_inputs_verified",cases=cases,model_keys=keys,freeze_created=False,solves=0,fits=0)
        return
    if (ROOT/"results"/"confirmation-model-freeze.json").exists():
        raise RuntimeError("Confirmation cohort is frozen; no further fitting or model selection is permitted in this run")
    import torch
    chosen_device=args.device or protocol.get("device","cuda")
    if chosen_device=="auto":
        chosen_device="cuda" if torch.cuda.is_available() else "cpu"
    uses_neural=bool(set(keys)&{"mlopt_native","neural_diving","li_bounds_residual"})
    if uses_neural and chosen_device=="cpu" and not args.allow_cpu_training:
        raise RuntimeError("Heavy CPU training needs --allow-cpu-training; this package defaults to authorized remote CUDA fitting")
    if uses_neural and chosen_device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no GPU is available; no fit or test started")
    frozen=freeze(device=chosen_device)
    with PhaseLock("offline_training"):
        if (ROOT/"results"/"confirmation-model-freeze.json").exists():
            raise RuntimeError("Confirmation cohort became frozen before training acquired the phase lock; no fit is permitted")
        dump(ROOT/"logs"/"training-runtime.json",runtime())
        with threadpool_limits(limits=1):
            for cid in cases:
                train_case(adapter.load_case(cid,verify=True),keys,frozen,protocol)
        markers={cid:{key:object_hash for key in keys
                    if (object_hash:=checkpoint(adapter.load_case(cid,verify=False),key,frozen)) is not None} for cid in cases}
        dump(ROOT/"models"/"requested-training-complete.json",dict(cases=cases,model_keys=keys,
            freeze_fingerprint=frozen["fingerprint"],all_models_complete=True,confirmation_data_generated=False,
            markers=markers,utc=utc()))
        event(event="offline_training_complete",cases=cases,model_keys=keys)


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        dump(ROOT/"logs"/"training-blocked.json",dict(utc=utc(),exception=traceback.format_exc(),
             diagnostics=getattr(exc,"diagnostics",getattr(exc,"metadata",None)),no_confirmation_started=True))
        traceback.print_exc()
        raise
