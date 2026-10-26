"""Seal the geometry-selected, training-only six-case Uniform cohort.

Adds new transfer artifacts; never rewrites the immutable producer outputs.
No neural training, prediction, development-bank loading or optimization.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parent
SCREEN = ROOT / "screen-uniform-gross"
OUT = ROOT / "primary-cohort-20261025-v2"
CORE = ROOT.parents[2] / "code" / "controls" / "experiment_core.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    if OUT.exists(): raise FileExistsError("Immutable primary transfer already exists")
    OUT.mkdir()
    now = datetime.now(timezone.utc).isoformat()
    summary = json.loads((SCREEN / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE"
    spec = importlib.util.spec_from_file_location("coupled_readonly_exact_core", CORE)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    protocol = dict(created_utc=now,
        selection_basis="Root preselected G16n37 and G20n45, every physical seed, from exact rank and original LP fractionality; no neural result used",
        settings=[dict(G=16, n=37), dict(G=20, n=45)], seeds=[1009,1013,1019],
        fitting_inputs="training-only files; no development or confirmation arrays read",
        original_local_formulation="One original capacity row wA*A+wB*B+wC*C <= wA+wB per group; no strong linking rows",
        gross_profit="0.5*G+sum((xi_g-0.5)*C_g), xi IID Uniform[0.1,2.5]; no base subtraction",
        source_sha256={str(Path(__file__).resolve()):sha(__file__), str(CORE):sha(CORE),
                       str(SCREEN / "summary.json"):sha(SCREEN / "summary.json"),
                       str(SCREEN / "protocol.json"):sha(SCREEN / "protocol.json")})
    write_new(OUT / "selection-protocol.json", protocol)
    cases=[]; bindings={}
    for G,n in [(16,37),(20,45)]:
        for seed in [1009,1013,1019]:
            row = next(r for r in summary["rows"] if r["G"]==G and r["n_train"]==n and r["physical_seed"]==seed)
            assert row["training_K"] == row["global_K"] == G
            folder = Path(row["source"]["path"])
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            assert sha(folder/"training.npz")==row["source"]["training_npz_sha256"]
            assert sha(folder/"metadata.json")==row["source"]["metadata_sha256"]
            with np.load(folder / "training.npz", allow_pickle=False) as data:
                y=data["train_solutions"].copy()
                assert y.shape==(n,3*G)
                assert np.all((y==0)|(y==1))
                y=y.astype(np.int64)
                t=perf_counter(); cpu=process_time(); affine=core.fit_affine(y)
                wall=perf_counter()-t; cpu=process_time()-cpu
                assert affine["rank"]==G
                # Empty and unit independent sets certify the entire optimizer hull.
                cert=np.ones((G+1,3*G),dtype=np.int64); cert[:,2::3]=0
                for j in range(G): cert[j+1,3*j:3*j+3]=[0,0,1]
                assert np.all((cert.astype(object)-affine["anchor"]) @ affine["normals_exact"].T == 0)
            write_new(folder / "affine.json", affine)
            rawpath=SCREEN/"references"/f"random{G}_b0.5_seed{seed}"/"raw-reference-checks.json"
            assert sha(rawpath)==row["raw_reference_checks_sha256"]
            raw=json.loads(rawpath.read_text(encoding="utf-8"))
            train=[r for r in raw if r["split"]=="training" and r["split_index"]<n]
            assert len(train)==n and all(r["ip"]["certified"] for r in train)
            certificate=dict(case_id=meta["case_id"],training_only=True,count=n,
                source_raw_reference_sha256=sha(rawpath),
                original_label_generation_wall_seconds=None,
                timing_note="Producer did not record wall/CPU elapsed; no timing reconstructed",
                original_IP_reference_and_all_IS_oracle_agree=True,records=train)
            certpath=OUT/"training-certificates"/(meta["case_id"]+".json")
            write_new(certpath,certificate)
            files={name:sha(folder/name) for name in ["training.npz","metadata.json","affine.json"]}
            cases.append(dict(case_id=meta["case_id"], directory=str(folder.resolve()),files=files,
                D=3*G, rows=row["original_rows"], feature_dimension=G,n_train=n,K=G,
                family="knapsack",case_schema="coupled-bundle-uniform-gross-v1",local_formulation="capacity",
                physical_seed=seed,development_count=48,global_optimizer_hull_K=G,training_covers_global_hull=True,
                exact_global_containment_certificate="Empty plus G singleton independent-set optimizer vectors lie exactly in the learned affine slice",
                affine_fit_wall_seconds=wall,affine_fit_cpu_seconds=cpu,
                original_label_generation_seconds=None,
                training_reference_certificate=str(certpath.resolve()),training_reference_certificate_sha256=sha(certpath)))
            for name,digest in files.items(): bindings[str((folder/name).resolve())]=digest
            bindings[str(certpath.resolve())]=sha(certpath)
    seal=dict(schema="coupled-bundle-uniform-primary-source-data-seal-v1",created_utc=now,
        cohort_selection_frozen_before_any_neural_baseline_fit=True,training_only=True,
        confirmation_included=False,case_count=6,files=bindings,
        source_hashes=protocol["source_sha256"],selection_protocol_sha256=sha(OUT/"selection-protocol.json"),
        original_generation_timings_missing=True,
        note="Old screen artifacts preserved. New timed reference recertification may be recorded separately; it is not original label-generation time.")
    write_new(OUT/"source-data-seal.json",seal)
    manifest=dict(schema="generic-binary-primary-cohort-v2",created_utc=now,
        frozen_before_any_fit=True,frozen_before_any_fit_scope="Neural/learned-baseline fits; exact-affine producer fits are recorded separately",
        training_only=True,confirmation_included=False,
        source_data_seal_sha256=sha(OUT/"source-data-seal.json"),source_data_seal=str((OUT/"source-data-seal.json").resolve()),cases=cases)
    write_new(OUT/"training-only-manifest.json",manifest)
    print(json.dumps(dict(manifest=str(OUT/"training-only-manifest.json"),manifest_sha256=sha(OUT/"training-only-manifest.json"),seal_sha256=sha(OUT/"source-data-seal.json"),cases=len(cases))))


if __name__ == "__main__": main()
