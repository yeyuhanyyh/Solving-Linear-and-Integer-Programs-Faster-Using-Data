"""Training-only contracts and source checks shared by neural adapters."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enable_runtime():
    manifest=json.loads((ROOT/"source_provenance.json").read_text())
    for name,digest in manifest["vendored_byte_identical_files"].items():
        if sha(ROOT/"vendor"/name)!=digest: raise ValueError(f"Vendored source changed: {name}")
    for name,record in manifest.get("vendored_compatibility_patches",{}).items():
        if sha(ROOT/"vendor"/name)!=record["sha256"]:
            raise ValueError(f"Patched vendor source changed: {name}")
    path=str(ROOT/"vendor")
    if path not in sys.path: sys.path.insert(0,path)


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+"\n",encoding="utf-8")


def training_hash(case):
    digest=hashlib.sha256()
    for name in ("train_features","train_costs","train_solutions","train_objectives"):
        a=np.ascontiguousarray(getattr(case,name))
        digest.update(str(a.shape).encode());digest.update(str(a.dtype).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def implementation_fingerprint():
    import adapter,dimension_diagnostics,mlopt_baseline
    local=("neural_api.py","neural_support.py","nd_training.py","li_training.py","li_scaling.py","neural_timing.py","source_provenance.json")
    hashes={name:sha(ROOT/name) for name in local}
    for module in (adapter,dimension_diagnostics,mlopt_baseline):
        hashes["framework/"+module.__name__+".py"]=sha(module.__file__)
    return hashes


def validate_training_pair(case,validation_case):
    """Checks only explicitly supplied training/validation objects, never tests."""
    if case is validation_case: raise ValueError("A separate validation bank is required")
    import adapter
    from preference_baseline import _model_hash
    if _model_hash(case)!=_model_hash(validation_case): raise ValueError("Training/validation feasible sets differ")
    if not np.array_equal(case.commitment_indices,validation_case.commitment_indices):
        raise ValueError("Training/validation commitment layout differs")
    if not 0<=int(case.affine["rank"])<=len(case.commitment_indices): raise ValueError("Invalid target K")
    for current in (case,validation_case):
        if current.n_train<1: raise ValueError("An empty bank is not valid")
        f,c,x,v=[np.asarray(getattr(current,name)) for name in
                 ("train_features","train_costs","train_solutions","train_objectives")]
        if f.ndim!=2 or len(f)!=current.n_train or c.shape!=x.shape or c.shape!=(current.n_train,current.model.n_variables) or v.shape!=(current.n_train,):
            raise ValueError("Malformed training/validation arrays")
        if not all(np.isfinite(a).all() for a in (f,c,x,v)): raise ValueError("Nonfinite bank")
        for cost,solution,value in zip(c,x,v):
            if not adapter.validate_solution(current.model,solution)["valid"]:
                raise ValueError("A supplied training/validation label violates its full formulation")
            if abs(float(cost@solution)-float(value))>adapter.recovery_tolerance(value):
                raise ValueError("Training/validation objective does not match its full solution")
    if case.train_features.shape[1]!=validation_case.train_features.shape[1]: raise ValueError("Feature schema differs")


def concatenate_for_selection(case,validation_case):
    # A private selection-only view. Final indices remain0..N-1 and graph cost
    # normalization is fitted on those training rows only.
    out=SimpleNamespace(**vars(case))
    for name in ("train_features","train_costs","train_solutions","train_objectives"):
        setattr(out,name,np.concatenate((getattr(case,name),getattr(validation_case,name)),axis=0))
    out.n_train=len(out.train_costs)
    return out


def singleton_pools(case):
    return [dict(train_index=i,solutions=x[None,:].copy(),objectives=np.asarray([value]),
        source_tags=["shared_certified_training_optimum"],conditional_lp_completion_certified=True,
        collection_seconds=0.,extra_conditional_lp_seconds=0.,
        source="Globally certified original optimum also certifies its conditional LP optimum")
        for i,(x,value) in enumerate(zip(case.train_solutions,case.train_objectives))]


def memory_estimate(n_variables,n_constraints,nnz,training_count,validation_count,width=64,batch_size=16,layers=6):
    nodes=n_variables+n_constraints
    retained_concat=4*batch_size*nodes*sum(28+width*i for i in range(layers+1))
    root_cache=4*15*nodes*(training_count+validation_count)
    adjacency=(2*nnz+nodes)*20
    return dict(node_count=nodes,width=width,batch_size=batch_size,
        retained_concatenated_tensor_gib=retained_concat/2**30,
        host_root_feature_cache_gib=root_cache/2**30,sparse_adjacency_gib=adjacency/2**30,
        interpretation="Static tensor sizes, not measured peak GPU memory. Autograd, layer temporaries and workspace add to this estimate. Root-LP cache resides on host.",
        full_fit_runtime_estimated=False)
