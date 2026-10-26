"""Seal all three predeclared G32 cases, without changing the six-case cohort."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from time import perf_counter, process_time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location("sealed_transfer_utilities",ROOT/"transfer_primary_uniform.py")
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
SCREEN=HERE/"run-v1"
OUT=HERE/"primary-cohort-v1"
CORE=util.CORE


def main():
    if OUT.exists():raise FileExistsError("Immutable scale cohort already exists")
    OUT.mkdir()
    now=datetime.now(timezone.utc).isoformat()
    summary=json.loads((SCREEN/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="COMPLETE" and len(summary["rows"])==3
    for path,digest in summary["source_sha256"].items():
        assert util.sha(path)==digest,path
    spec=importlib.util.spec_from_file_location("sealed_scale_exact_core",CORE)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    protocol=dict(created_utc=now,selection_basis="Root preselected all three G32n69 scale cases based on exact rank and original LP gaps, before G32 neural fitting",
        G=32,n=69,seeds=[1009,1013,1019],development_count=48,
        no_primary_six_case_file_changed=True,no_neural_fit_or_prediction_by_transfer=True,
        fitting_inputs="Training-only arrays; no development or confirmation arrays read",
        source_sha256={str(Path(__file__).resolve()):util.sha(__file__),str(ROOT/"transfer_primary_uniform.py"):util.sha(ROOT/"transfer_primary_uniform.py"),
            str(CORE):util.sha(CORE),str(SCREEN/"summary.json"):util.sha(SCREEN/"summary.json"),str(SCREEN/"protocol.json"):util.sha(SCREEN/"protocol.json")})
    util.write_new(OUT/"selection-protocol.json",protocol)
    cases=[];bindings={}
    for seed in [1009,1013,1019]:
        row=next(r for r in summary["rows"] if r["seed"]==seed)
        assert row["training_K"]==row["exact_global_K"]==32
        folder=Path(row["source"]["path"])
        meta=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        assert util.sha(folder/"training.npz")==row["source"]["training_npz_sha256"]
        assert util.sha(folder/"metadata.json")==row["source"]["metadata_sha256"]
        with np.load(folder/"training.npz",allow_pickle=False) as data:
            y=data["train_solutions"].copy()
            assert y.shape==(69,96) and np.all((y==0)|(y==1))
            started=perf_counter();cpu=process_time();affine=core.fit_affine(y.astype(np.int64))
            wall=perf_counter()-started;cpu=process_time()-cpu
            assert affine["rank"]==32
            witness=np.ones((33,96),dtype=np.int64);witness[:,2::3]=0
            for j in range(32):witness[j+1,3*j:3*j+3]=[0,0,1]
            assert np.all((witness.astype(object)-affine["anchor"]) @ affine["normals_exact"].T==0)
        util.write_new(folder/"affine.json",affine)
        rawpath=SCREEN/"references"/f"random32_b0.5_seed{seed}"/"raw-reference-checks.json"
        assert util.sha(rawpath)==row["raw_reference_sha256"]
        raw=json.loads(rawpath.read_text(encoding="utf-8"))
        train=[r for r in raw if r["split"]=="training"]
        assert len(train)==69 and all(r["ip"]["certified"] for r in train)
        certificate=dict(case_id=meta["case_id"],training_only=True,count=69,
            source_raw_reference_sha256=util.sha(rawpath),original_IP_and_all_IS_oracle_agree=True,
            original_IP_label_certification_wall_seconds=sum(r["ip"]["outer_wall_seconds"] for r in train),
            original_IP_label_certification_cpu_seconds=sum(r["ip"]["outer_cpu_seconds"] for r in train),
            original_total_label_generation_wall_seconds=None,
            timing_note="Recorded original-IP recertification wall/CPU includes build/presolve/solve/validation; enumeration/oracle total generation time was not recorded; concurrent environment, not speedup evidence",
            records=train)
        certpath=OUT/"training-certificates"/(meta["case_id"]+".json")
        util.write_new(certpath,certificate)
        files={name:util.sha(folder/name) for name in ["training.npz","metadata.json","affine.json"]}
        cases.append(dict(case_id=meta["case_id"],directory=str(folder.resolve()),files=files,
            D=96,rows=row["original_rows"],feature_dimension=32,n_train=69,K=32,
            family="knapsack",case_schema="coupled-bundle-uniform-gross-v1",local_formulation="capacity",
            physical_seed=seed,development_count=48,global_optimizer_hull_K=32,training_covers_global_hull=True,
            exact_global_containment_certificate="Empty and all singleton IS optimizer vectors lie exactly in the learned affine slice",
            affine_fit_wall_seconds=wall,affine_fit_cpu_seconds=cpu,
            original_label_generation_seconds=None,
            training_reference_certificate=str(certpath.resolve()),training_reference_certificate_sha256=util.sha(certpath)))
        for name,digest in files.items():bindings[str((folder/name).resolve())]=digest
        bindings[str(certpath.resolve())]=util.sha(certpath)
    util.write_new(OUT/"source-data-seal.json",dict(schema="coupled-bundle-uniform-scale-source-data-seal-v1",created_utc=now,
        training_only=True,confirmation_included=False,case_count=3,
        cohort_selection_frozen_before_any_neural_baseline_fit=True,files=bindings,
        source_hashes=protocol["source_sha256"],selection_protocol_sha256=util.sha(OUT/"selection-protocol.json")))
    util.write_new(OUT/"training-only-manifest.json",dict(schema="generic-binary-primary-cohort-v2",created_utc=now,
        frozen_before_any_fit=True,frozen_before_any_fit_scope="Neural/learned-baseline fits; exact-affine producer fit is recorded separately",
        training_only=True,confirmation_included=False,
        source_data_seal_sha256=util.sha(OUT/"source-data-seal.json"),source_data_seal=str((OUT/"source-data-seal.json").resolve()),cases=cases))
    print(json.dumps(dict(manifest=str(OUT/"training-only-manifest.json"),manifest_sha256=util.sha(OUT/"training-only-manifest.json"),
        seal_sha256=util.sha(OUT/"source-data-seal.json"),cases=3)))


if __name__=="__main__":main()
