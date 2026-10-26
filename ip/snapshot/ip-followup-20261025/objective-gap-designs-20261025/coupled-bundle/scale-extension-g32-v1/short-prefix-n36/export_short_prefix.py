"""Derive the same first-36 training prefix for every predeclared G32 seed.

The prefix was selected after exploring 69 pre-existing labels, using only exact
training ranks. This does not claim that the exploration required only36 labels.
No optimization, neural training, prediction or random sampling is performed.
"""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
from time import perf_counter, process_time
import numpy as np

HERE=Path(__file__).resolve().parent
SCALE=HERE.parent
ROOT=SCALE.parent
spec=importlib.util.spec_from_file_location("sealed_short_prefix_utilities",ROOT/"transfer_primary_uniform.py")
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
SOURCE_MANIFEST=SCALE/"primary-cohort-v1"/"training-only-manifest.json"
PREFIX_AUDIT=ROOT/"training-prefix-exact-audit-20261025"/"summary.json"
OUT=HERE/"sealed-cohort-v1"
N=36


def main():
    if OUT.exists():raise FileExistsError("Immutable short-prefix cohort already exists")
    OUT.mkdir(parents=True)
    source=json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    assert len(source["cases"])==3
    audit=json.loads(PREFIX_AUDIT.read_text(encoding="utf-8"))
    group=next(r for r in audit["grouped"] if r["G"]==32)
    assert group["minimum_prefixes"]==[35,33,36] and group["common_minimum_prefix"]==N
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),G=32,n_train=N,seeds=[1009,1013,1019],
        selection_basis="All-three-seed exact training-prefix rank; no development/neural/fresh result used",
        exploration_cost_disclosure="Prefix length was chosen after collecting69 labels per seed. New model fitting uses only36; the exploratory label cost is not erased",
        unchanged_ancestor_cohort=True,no_optimization_or_neural_fit_or_new_sampling=True,
        paired_development="Same existing48 physical-case costs as n69; attached separately and excluded from fitting",
        paired_confirmation="Future n36/n69 evaluation must share the same120 new physical-case costs, after all12 models freeze",
        source_sha256={str(Path(__file__).resolve()):util.sha(__file__),str(SOURCE_MANIFEST):util.sha(SOURCE_MANIFEST),
                       str(PREFIX_AUDIT):util.sha(PREFIX_AUDIT),str(util.CORE):util.sha(util.CORE),
                       str(ROOT/"transfer_primary_uniform.py"):util.sha(ROOT/"transfer_primary_uniform.py")})
    util.write_new(OUT/"selection-protocol.json",protocol)
    spec=importlib.util.spec_from_file_location("readonly_short_prefix_core",util.CORE)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    cases=[];bindings={};attachments=[]
    for original in sorted(source["cases"],key=lambda r:r["physical_seed"]):
        seed=original["physical_seed"];old=Path(original["directory"])
        for name,digest in original["files"].items():assert util.sha(old/name)==digest
        original_meta=json.loads((old/"metadata.json").read_text(encoding="utf-8"))
        assert original_meta["n_train"]==69
        cid=original["case_id"].removesuffix("n69")+"n36"
        folder=HERE/"training-candidates"/f"random32_b0.5_seed{seed}"/"n36"
        folder.mkdir(parents=True,exist_ok=False)
        with np.load(old/"training.npz",allow_pickle=False) as ancestor:
            arrays={key:ancestor[key].copy() for key in ancestor.files}
        subset_keys=["train_features","train_costs","train_solutions","train_objectives"]
        for key in subset_keys:
            assert len(arrays[key])==69
            arrays[key]=arrays[key][:N].copy()
        np.savez_compressed(folder/"training.npz",**arrays)
        with np.load(folder/"training.npz",allow_pickle=False) as saved,np.load(old/"training.npz",allow_pickle=False) as ancestor:
            assert set(saved.files)==set(ancestor.files)
            for key in saved.files:
                expected=ancestor[key][:N] if key in subset_keys else ancestor[key]
                assert saved[key].dtype==expected.dtype and np.array_equal(saved[key],expected),key
        meta=deepcopy(original_meta)
        meta.update(case_id=cid,n_train=N,K=32,training_npz_sha256=util.sha(folder/"training.npz"))
        meta["prefix_selection_provenance"]=dict(ancestor_case_id=original["case_id"],ancestor_training_count=69,
            fitting_prefix_count=N,ancestor_training_npz_sha256=original["files"]["training.npz"],
            prefix_audit_sha256=util.sha(PREFIX_AUDIT),all_seeds_retained=True,
            exploration_labels_already_collected_per_seed=69,
            selection_used_only_training_rank=True,no_development_or_neural_or_confirmation_used=True)
        util.write_new(folder/"metadata.json",meta)
        y=arrays["train_solutions"]
        assert y.shape==(36,96) and np.all((y==0)|(y==1))
        wall=perf_counter();cpu=process_time();affine=core.fit_affine(y.astype(np.int64))
        wall=perf_counter()-wall;cpu=process_time()-cpu
        assert affine["rank"]==32
        witness=np.ones((33,96),dtype=np.int64);witness[:,2::3]=0
        for j in range(32):witness[j+1,3*j:3*j+3]=[0,0,1]
        assert np.all((witness.astype(object)-affine["anchor"])@affine["normals_exact"].T==0)
        util.write_new(folder/"affine.json",affine)
        oldcert=Path(original["training_reference_certificate"])
        assert util.sha(oldcert)==original["training_reference_certificate_sha256"]
        certificate=json.loads(oldcert.read_text(encoding="utf-8"))
        records=certificate["records"][:N]
        assert len(records)==N and all(r["split"]=="training" and r["split_index"]==i and r["ip"]["certified"] for i,r in enumerate(records))
        newcert=dict(case_id=cid,training_only=True,count=N,ancestor_training_certificate_sha256=util.sha(oldcert),
            original_IP_and_independent_oracle_agree=True,records=records,
            original_IP_label_certification_wall_seconds=sum(r["ip"]["outer_wall_seconds"] for r in records),
            original_IP_label_certification_cpu_seconds=sum(r["ip"]["outer_cpu_seconds"] for r in records),
            original_total_label_generation_wall_seconds=None,
            timing_note="Reused first36 original certificates; whole exploratory cohort contained69 labels. Timings are not isolated")
        certpath=OUT/"training-certificates"/(cid+".json")
        util.write_new(certpath,newcert)
        files={name:util.sha(folder/name) for name in ["training.npz","metadata.json","affine.json"]}
        entry={key:value for key,value in original.items() if key not in ["case_id","directory","files","n_train","affine_fit_wall_seconds","affine_fit_cpu_seconds","training_reference_certificate","training_reference_certificate_sha256"]}
        entry.update(case_id=cid,directory=str(folder.resolve()),files=files,n_train=N,
            affine_fit_wall_seconds=wall,affine_fit_cpu_seconds=cpu,
            training_reference_certificate=str(certpath.resolve()),training_reference_certificate_sha256=util.sha(certpath),
            ancestor_case_id=original["case_id"],exploration_labels_already_collected=69,
            prefix_selection_used_all_three_training_rank_audits=True)
        cases.append(entry)
        for name,digest in files.items():bindings[str((folder/name).resolve())]=digest
        bindings[str(certpath.resolve())]=util.sha(certpath)
        bank=SCALE/"run-v1"/"references"/f"random32_b0.5_seed{seed}"/"exploratory-development-reference.npz"
        destination=folder/"development.npz"
        shutil.copyfile(bank,destination)
        assert util.sha(bank)==util.sha(destination)
        attachments.append(dict(case_id=cid,ancestor_case_id=original["case_id"],source=str(bank.resolve()),
                                destination=str(destination.resolve()),sha256=util.sha(bank),count=48,
                                purpose="Existing paired development attachment; never a fitting input"))
    util.write_new(OUT/"source-data-seal.json",dict(schema="coupled-bundle-uniform-short-prefix-source-data-seal-v1",
        created_utc=protocol["created_utc"],training_only=True,confirmation_included=False,case_count=3,
        files=bindings,source_hashes=protocol["source_sha256"],selection_protocol_sha256=util.sha(OUT/"selection-protocol.json")))
    util.write_new(OUT/"training-only-manifest.json",dict(schema="generic-binary-primary-cohort-v2",created_utc=protocol["created_utc"],
        frozen_before_any_fit=True,frozen_before_any_fit_scope="Neural baseline fits; exact affine producer fit recorded separately",
        training_only=True,confirmation_included=False,
        source_data_seal=str((OUT/"source-data-seal.json").resolve()),source_data_seal_sha256=util.sha(OUT/"source-data-seal.json"),cases=cases))
    util.write_new(OUT/"paired-development-attachments.json",dict(training_input=False,confirmation=False,attachments=attachments))
    print(json.dumps(dict(manifest=str(OUT/"training-only-manifest.json"),manifest_sha256=util.sha(OUT/"training-only-manifest.json"),
        seal_sha256=util.sha(OUT/"source-data-seal.json"),paired_development_receipt_sha256=util.sha(OUT/"paired-development-attachments.json"))))


if __name__=="__main__":main()
