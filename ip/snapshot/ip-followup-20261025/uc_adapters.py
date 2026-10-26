"""Frozen Table 5/6 UC data and common-K Neural Diving/Li experiment adapter.

Training loaders never read confirmation costs or outcomes. Old experiment
outputs are input-only. Run --export to create a portable source/data bundle;
exporting uses NumPy only and performs no optimization or network training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT.parent
DEFAULT_BUNDLE = ROOT / "uc_bundle_frozen"
SEEDS = (401, 503, 607)
SCENARIOS = {
    "original": dict(scenario_id="log_normal_sigma1_first2_three_time_blocks", sigma=1.0,
        demand_scale=1.0, counts=(48, 64, 96),
        source="natural-distribution-pilot-20261012/covered-geometry-followup",
        base_run="cloud-results/discovery-run",
        test_run="confirmation-preparation/cloud-results/extracted/confirmation-run"),
    "half": dict(scenario_id="demand0.5_sigma1.5", sigma=1.5,
        demand_scale=0.5, counts=(32, 48, 64), source="low-load-wide-cost-pilot-20261013",
        base_run="cloud-results/final-backups/run",
        test_run="cloud-results/final-backups/confirmation-run"),
}
RUNTIME_FILES = (
    "portable_adapter.py", "adapter.py", "dimension_diagnostics.py",
    "neural_diving_baseline.py", "nd_root_lp_features.py", "preference_baseline.py",
    "mlopt_baseline.py",
)


def clean(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw: raise FileExistsError(f"Refusing to replace changed artifact: {path}")
        return
    # Create exclusively: incomplete writes never appear under a completion marker.
    with path.open("xb") as stream: stream.write(raw)


def write_json(path: Path, data: Any) -> None:
    publish(Path(path), (json.dumps(clean(data), ensure_ascii=True, indent=2, allow_nan=False)+"\n").encode())


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.shape).encode()); digest.update(str(value.dtype).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    # NPZ's ZIP timestamps otherwise change between identical exports.
    import io, zipfile
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO(); np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(key+".npy", date_time=(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())
    return target.getvalue()


def group_id(law: str, seed: int) -> str:
    return f"uc_seed{seed}_{SCENARIOS[law]['scenario_id']}"


def list_case_ids() -> list[str]:
    return [f"{group_id(law, seed)}_n{n}" for law, spec in SCENARIOS.items()
            for seed in SEEDS for n in spec["counts"]]


def export_bundle(bundle: Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    """Export exact frozen training prefixes and old test identities, no solves.

    Full reference objective/solutions remain under confirmation/, which fitting
    never opens. Each training prefix is its own file (no access to n96 labels
    in an n32/n48 fitting process). All source outputs are left untouched.
    """
    bundle = Path(bundle).resolve()
    hashes: dict[str, str] = {}; sources: dict[str, str] = {}; cases = []
    def store(name: str, raw: bytes, source: Path | None = None):
        publish(bundle/name, raw); hashes[name] = sha(bundle/name)
        if source is not None: sources[str(source.relative_to(EXPERIMENT_ROOT)).replace("\\", "/")] = sha(source)
    for filename in RUNTIME_FILES:
        src = EXPERIMENT_ROOT/"code"/filename; store("runtime/"+filename, src.read_bytes(), src)
    src = EXPERIMENT_ROOT/"matched-k-supplement-20261010/code/strategy_match.py"
    store("runtime/strategy_match.py", src.read_bytes(), src)
    # This exact physical generator was used to create the Table 5 costs/data.
    src = EXPERIMENT_ROOT/SCENARIOS["original"]["source"]/"code/uc_model.py"
    store("runtime/uc_model.py", src.read_bytes(), src)
    for law, spec in SCENARIOS.items():
        source_root = EXPERIMENT_ROOT/spec["source"]
        for seed in SEEDS:
            gid = group_id(law, seed)
            train_dir = source_root/spec["base_run"]/"training"/gid
            test_dir = source_root/spec["test_run"]/"results"/gid
            source_npz = train_dir/"training.npz"
            with np.load(source_npz, allow_pickle=False) as stream:
                training = {key: stream[key].copy() for key in ("features", "costs", "solutions", "objectives")}
            with np.load(test_dir/"test-costs.npz", allow_pickle=False) as stream:
                test = {key: stream[key].copy() for key in ("features", "costs")}
            if test["costs"].shape != (80,576): raise ValueError(f"Unexpected old test identity: {gid}")
            full = []
            for line in (test_dir/"trials.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["method"] == "full": full.append(row)
            # One full reference may be stored per n; deduplicate by cost index.
            index_key = next((key for key in ("test_index", "sample_index", "test_id", "query_index")
                              if full and key in full[0]), None)
            if index_key is None: raise ValueError(f"Missing explicit frozen test index: {gid}")
            by_index: dict[int, dict] = {}
            for row in full:
                i = int(row[index_key])
                if i in by_index and abs(float(row["objective"])-float(by_index[i]["objective"])) > 1e-5:
                    raise ValueError("Inconsistent repeated full reference")
                by_index[i] = row
            if set(by_index) != set(range(80)): raise ValueError(f"Missing full reference: {gid}")
            reference = np.asarray([by_index[i]["objective"] for i in range(80)], dtype=float)
            for i in range(80):
                row = by_index[i]
                if not row.get("valid") or not row.get("original_global_solver_certified", row.get("certified", row.get("optimal", False))):
                    raise ValueError(f"Uncertified old reference at {gid}/{i}")
                if "solution" in row and not np.isclose(test["costs"][i]@np.asarray(row["solution"]), reference[i], rtol=1e-9, atol=1e-6):
                    raise ValueError("Frozen full cost/reference mismatch")
            confirmation_name = f"confirmation/{gid}.npz"
            store(confirmation_name, deterministic_npz(dict(test, objectives=reference)), test_dir/"test-costs.npz")
            sources[str((test_dir/"trials.jsonl").relative_to(EXPERIMENT_ROOT)).replace("\\","/")] = sha(test_dir/"trials.jsonl")
            for n in spec["counts"]:
                cid = f"{gid}_n{n}"; metadata_path = train_dir/f"n{n}/metadata.json"; old_meta = read(metadata_path)
                rank_path=train_dir/"geometry.json"
                rank_evidence=next(row for row in read(rank_path)["prefixes"] if row["training_count"]==n)
                if not rank_evidence["exact_rational_rank"] or int(rank_evidence["exact_K"])!=int(old_meta["target_K"]):
                    raise ValueError("Archived exact-rational rank evidence mismatch")
                subset = {key: values[:n].copy() for key, values in training.items()}
                if subset["costs"].shape != (n,576) or subset["features"].shape != (n,6): raise ValueError(cid)
                train_name = f"training/{cid}.npz"; meta_name = f"training/{cid}.json"
                store(train_name, deterministic_npz(subset), source_npz)
                meta = dict(case_id=cid, group_id=gid, law=law, seed=seed, sigma=spec["sigma"],
                    demand_scale=spec["demand_scale"], n_train=n, target_K=int(old_meta["target_K"]),
                    common_D=144, training_array_hash=array_hash(*[subset[key] for key in ("features","costs","solutions","objectives")]),
                    source_training_sha256=sha(source_npz), source_metadata_sha256=sha(metadata_path),
                    archived_exact_rank_evidence=rank_evidence, source_geometry_sha256=sha(rank_path),
                    source_training_case_metadata=str(metadata_path.relative_to(EXPERIMENT_ROOT)).replace("\\","/"),
                    baseline_seed=int.from_bytes(hashlib.sha256((cid+"|nd-li-followup-20261025").encode()).digest()[:4],"little") % (2**31-1),
                    feature_layout="first2_three_time_blocks", objective="minimize original positive total UC cost",
                    training_labels="same certified optimal training prefix as archived Ours", test_information_used=False)
                store(meta_name, (json.dumps(meta,indent=2,sort_keys=True)+"\n").encode(), metadata_path)
                cases.append(dict(case_id=cid, train=train_name, metadata=meta_name, confirmation=confirmation_name,
                                  n_train=n, K=meta["target_K"], law=law, seed=seed))
    manifest = dict(schema="uc-nd-li-supplement-frozen-v1", cases=cases,
                    file_sha256=hashes, source_sha256=sources,
                    data_bytes=sum((bundle/name).stat().st_size for name in hashes if name.startswith(("training/","confirmation/"))),
                    runtime_bytes=sum((bundle/name).stat().st_size for name in hashes if name.startswith("runtime/")),
                    fits_expected=36, test_records_expected=2880,
                    training_access="Only training/<case>.npz and .json; fit never opens confirmation/",
                    archived_test_reuse=True, old_outputs_modified=False)
    write_json(bundle/"manifest.json", manifest)
    return manifest


def enable_runtime(bundle: Path = DEFAULT_BUNDLE) -> None:
    bundle=Path(bundle).resolve()
    manifest=read(bundle/"manifest.json")
    for name,expected in manifest["file_sha256"].items():
        if name.startswith("runtime/") and sha(bundle/name)!=expected:
            raise ValueError(f"Frozen runtime source changed: {name}")
    path = str(bundle/"runtime")
    if path not in sys.path: sys.path.insert(0,path)


def load_case(case_id: str, bundle: Path = DEFAULT_BUNDLE):
    """Training-only loader; validates every original model constraint."""
    bundle = Path(bundle); manifest = read(bundle/"manifest.json")
    item = next(row for row in manifest["cases"] if row["case_id"] == case_id)
    for key in ("train","metadata"):
        if sha(bundle/item[key]) != manifest["file_sha256"][item[key]]: raise ValueError("Frozen training hash changed")
    meta = read(bundle/item["metadata"])
    with np.load(bundle/item["train"],allow_pickle=False) as stream:
        data = {key: stream[key].copy() for key in stream.files}
    if array_hash(*[data[key] for key in ("features","costs","solutions","objectives")]) != meta["training_array_hash"]:
        raise ValueError("Frozen training array identity changed")
    enable_runtime(bundle)
    from uc_model import make_uc_base
    import adapter
    base = make_uc_base(meta["seed"],n_units=6,n_times=24)
    if meta["demand_scale"] != 1:
        # The archived low-load modification: only the 24 balance RHSs change.
        rows=np.asarray([base["metadata"]["row_names"].index(f"balance[{t}]") for t in range(24)],dtype=int)
        base["row_lower"][rows] *= meta["demand_scale"]; base["row_upper"][rows] *= meta["demand_scale"]
        base["metadata"]["demand"] = np.asarray(base["metadata"]["demand"])*meta["demand_scale"]
    names=tuple(f"{kind}[{t},{g}]" for kind in ("u","v","w","p") for t in range(24) for g in range(6))
    model=adapter.LinearModel(A=base["A"],row_lower=base["row_lower"],row_upper=base["row_upper"],
        col_lower=base["col_lower"],col_upper=base["col_upper"],integrality=base["integrality"],
        variable_names=names,row_names=tuple(base["metadata"]["row_names"]))
    case=adapter.FrozenCase(case_id=case_id,family="uc",seed=meta["seed"],amplitude=0.,model=model,
        train_features=data["features"],train_costs=data["costs"],train_solutions=data["solutions"],
        train_objectives=data["objectives"],commitment_indices=np.asarray(base["binary_indices"],dtype=int),
        affine={"rank":meta["target_K"]},original_base=base,metadata=dict(base["metadata"],**meta),
        source_hashes={"training":manifest["file_sha256"][item["train"]],"metadata":manifest["file_sha256"][item["metadata"]]},
        dataset_fingerprint=meta["training_array_hash"],source_directory=bundle/"training")
    reproduced=costs_from_features(case,case.train_features)
    if not np.allclose(reproduced,case.train_costs,rtol=1e-14,atol=1e-12): raise ValueError("Archived block cost map changed")
    for cost,x,objective in zip(case.train_costs,case.train_solutions,case.train_objectives):
        if not adapter.validate_solution(model,x)["valid"]: raise ValueError("Archived optimizer violates reconstructed physics")
        if abs(float(cost@x)-objective)>adapter.recovery_tolerance(objective): raise ValueError("Archived optimizer objective mismatch")
    return case


def costs_from_features(case: Any, features: np.ndarray) -> np.ndarray:
    features=np.asarray(features,dtype=float); scalar=features.ndim==1
    if scalar: features=features[None,:]
    rho=np.ones((len(features),24,6)); rho[:,:,:2]=features.reshape(-1,3,2).repeat(8,axis=1)
    value=(rho[:,None,:,:]*np.asarray(case.original_base["base_cost"]).reshape(1,4,24,6)).reshape(-1,576)
    return value[0] if scalar else value


def singleton_pools(case: Any) -> list[dict[str,Any]]:
    return [dict(train_index=i,solutions=x[None,:].copy(),objectives=np.asarray([objective]),
            source_tags=["cached_certified_optimum"],conditional_lp_completion_certified=True,
            collection_seconds=0.,extra_conditional_lp_seconds=0.,
            source="archived globally certified optimum also certifies its conditional LP optimum")
            for i,(x,objective) in enumerate(zip(case.train_solutions,case.train_objectives))]


def load_confirmation(case: Any, bundle: Path = DEFAULT_BUNDLE) -> dict[str,np.ndarray]:
    """Separate evaluation-only loader, to call after checkpoint completion."""
    bundle=Path(bundle); manifest=read(bundle/"manifest.json")
    item=next(row for row in manifest["cases"] if row["case_id"]==case.case_id)
    if sha(bundle/item["confirmation"]) != manifest["file_sha256"][item["confirmation"]]: raise ValueError("Frozen test identity changed")
    with np.load(bundle/item["confirmation"],allow_pickle=False) as stream:
        test={key:stream[key].copy() for key in stream.files}
    if not np.allclose(costs_from_features(case,test["features"]),test["costs"],rtol=1e-14,atol=1e-12):
        raise ValueError("Frozen test cost map changed")
    return test


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export",action="store_true")
    parser.add_argument("--bundle",type=Path,default=DEFAULT_BUNDLE)
    args=parser.parse_args()
    if not args.export: parser.error("Use --export; this command does not train or solve")
    result=export_bundle(args.bundle)
    print(json.dumps({key:result[key] for key in ("schema","fits_expected","test_records_expected","data_bytes","runtime_bytes")}))
