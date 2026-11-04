"""Shared bank contract: fitting opens training/validation files only."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
import scipy.sparse as sp
import adapter


def array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def model_hash(model):
    a = sp.csr_matrix(model.A)
    return array_hash(a.data, a.indices, a.indptr, np.asarray(a.shape), model.row_lower,
        model.row_upper, model.col_lower, model.col_upper, model.integrality)


def as_linear_model(model):
    if isinstance(model, adapter.LinearModel):
        return model
    def get(*names):
        for name in names:
            if isinstance(model, dict) and name in model: return model[name]
            if hasattr(model, name): return getattr(model, name)
        raise AttributeError(names)
    return adapter.LinearModel(get("A", "matrix"), get("row_lower"), get("row_upper"),
        get("col_lower"), get("col_upper"), get("integrality"),
        tuple(get("variable_names", "var_names")), tuple(get("row_names")))


def make_case(model, data, metadata, commitment_indices, *, n=None, split="train", target_K=None):
    """No test data are accepted by a learner; all supplied labels must certify."""
    model = as_linear_model(model)
    full_count=len(data["costs"])
    for key in ("features","costs","solutions","objectives","certified","dual_bounds"):
        raw_array=np.asarray(data[key])
        if raw_array.ndim==0 or raw_array.shape[0]!=full_count:
            raise ValueError(f"Full label-bank lengths differ: {key}")
    if "solver_status" in data and np.asarray(data["solver_status"]).shape!=(full_count,):
        raise ValueError("Full solver-status array has wrong shape")
    count = full_count if n is None else int(n)
    if count < 1 or count > len(data["costs"]): raise ValueError("Invalid label prefix")
    arrays = {key: np.asarray(data[key])[:count].copy()
              for key in ("features", "costs", "solutions", "objectives")}
    if arrays["features"].ndim!=2 or arrays["features"].shape[0]!=count:
        raise ValueError("Feature array has wrong shape")
    if arrays["objectives"].shape!=(count,):raise ValueError("Objective array has wrong shape")
    if arrays["costs"].shape!=(count,model.n_variables) or arrays["solutions"].shape!=(count,model.n_variables):
        raise ValueError("Full cost/solution array has wrong shape")
    certified = np.asarray(data["certified"], dtype=bool)[:count]
    if certified.shape != (count,) or not certified.all():
        raise ValueError("Uncertified incumbents cannot enter the optimal-label bank")
    if "solver_status" in data:
        statuses=np.asarray(data["solver_status"]).astype(str)[:count]
        if statuses.shape!=(count,):raise ValueError("Solver status array has wrong shape")
        if any(s.strip().lower() not in ("optimal","koptimal","highsmodelstatus.koptimal") for s in statuses):
            raise ValueError("A purported optimal label has a nonoptimal solver status")
    if arrays["costs"].shape != arrays["solutions"].shape or arrays["costs"].shape[1] != model.n_variables:
        raise ValueError("Full original cost and solution arrays are required")
    if any(not np.isfinite(a).all() for a in arrays.values()): raise ValueError("Nonfinite label data")
    lower = np.asarray(data["dual_bounds"], dtype=float)[:count]
    if lower.shape!=(count,):raise ValueError("Dual bound array has wrong shape")
    for i, (c, x, objective, bound) in enumerate(zip(arrays["costs"], arrays["solutions"], arrays["objectives"], lower)):
        tol = adapter.recovery_tolerance(objective)
        if not adapter.validate_solution(model, x)["valid"]: raise ValueError(f"Invalid label {i}")
        if abs(float(c @ x)-objective) > tol: raise ValueError(f"Objective mismatch {i}")
        if not np.isfinite(bound) or not -tol <= objective-bound <= tol:
            raise ValueError(f"Unclosed original primal/dual gap {i}")
    raw_common=np.asarray(commitment_indices)
    common = np.asarray(commitment_indices, dtype=np.int64)
    if (common.ndim!=1 or len(common)==0 or not np.array_equal(raw_common,common)
        or np.any(common<0) or np.any(common>=model.n_variables)):
        raise ValueError("Commitment indices are not valid original column indices")
    if len(np.unique(common)) != len(common) or np.any(model.integrality[common] == 0):
        raise ValueError("Common space must contain distinct original commitment columns")
    prefixes=("is_on[",) if metadata.get("family")=="ucjl_static" else ("ug[","u[")
    def commitment_name(j):
        name=model.variable_names[j]
        if metadata.get("family")=="ucjl_static":name=name.removeprefix(f"x_{int(j)+1}__")
        return name
    if model.variable_names and any(not commitment_name(j).startswith(prefixes) for j in common):
        raise ValueError("Common indices include a noncommitment variable")
    meta = dict(metadata, split=split, n_train=count, all_training_labels_certified=True,
                model_sha256=model_hash(model), common_D=len(common), test_information_used=False)
    meta.setdefault("baseline_seed",int.from_bytes(hashlib.sha256(
        f"{metadata.get('case_id','uc')}|n{count}|formal-fit-v1".encode()).digest()[:4],"little")%(2**31-1))
    fingerprint = array_hash(*(arrays[k] for k in ("features", "costs", "solutions", "objectives")))
    identity=f"{metadata.get('case_id','uc')}_rep{metadata.get('replicate',0)}_seed{metadata.get('seed',0)}"
    return adapter.FrozenCase(case_id=f"{identity}_{split}_n{count}",
        family=str(metadata.get("family","uc")), seed=int(metadata.get("seed",0)), amplitude=0., model=model,
        train_features=arrays["features"], train_costs=arrays["costs"],
        train_solutions=arrays["solutions"], train_objectives=arrays["objectives"],
        commitment_indices=common, affine={} if target_K is None else {"rank":int(target_K)},
        original_base={}, metadata=meta, source_hashes={}, dataset_fingerprint=fingerprint,
        source_directory=Path(metadata.get("source_directory", ".")))


def load_bank_split(folder, split, *, n=None, target_K=None):
    """Explicit split loader. A fitting invocation never needs to open test.npz."""
    if split not in ("train", "validation", "test"): raise ValueError(split)
    root = Path(folder)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8-sig"))
    for name in ("model.npz", f"{split}.npz"):
        expected = manifest.get("file_sha256", {}).get(name)
        if expected is None or file_hash(root/name) != expected:
            raise ValueError(f"Missing or mismatched sealed file: {name}")
    with np.load(root/"model.npz", allow_pickle=False) as z:
        shape = tuple(z["A_shape"].astype(int))
        a = sp.csr_matrix((z["A_data"],z["A_indices"],z["A_indptr"]),shape=shape)
        model = adapter.LinearModel(a,z["row_lower"],z["row_upper"],z["col_lower"],
            z["col_upper"],z["integrality"],tuple(z["variable_names"].tolist()),
            tuple(z["row_names"].tolist()))
        common = z["commitment_indices"].copy()
    with np.load(root/f"{split}.npz",allow_pickle=False) as z:
        data = {key:z[key].copy() for key in z.files}
    case = make_case(model,data,dict(manifest.get("metadata",{}),source_directory=str(root)),
                     common,n=n,split=split,target_K=target_K)
    case.source_hashes = {name:manifest["file_sha256"][name] for name in ("model.npz",f"{split}.npz")}
    case.source_hashes["manifest.json"]=file_hash(root/"manifest.json")
    return case


def assert_same_physics(train, validation):
    if model_hash(train.model) != model_hash(validation.model): raise ValueError("Feasible sets differ")
    if not np.array_equal(train.commitment_indices, validation.commitment_indices):
        raise ValueError("Commitment layouts differ")
    if set(map(bytes,np.ascontiguousarray(train.train_costs))) & set(map(bytes,np.ascontiguousarray(validation.train_costs))):
        raise ValueError("Duplicate objective vectors across fitting and validation")

