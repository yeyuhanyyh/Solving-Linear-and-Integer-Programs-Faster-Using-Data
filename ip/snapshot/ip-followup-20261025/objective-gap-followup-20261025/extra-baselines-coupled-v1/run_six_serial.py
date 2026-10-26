"""One-worker fixed six-case SVM/MLOPT queue. Root controls when it may start."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os
import subprocess
import sys
import coupled_svm_mlopt_driver as driver

ROOT=Path(__file__).resolve().parent
FOLLOWUP=ROOT.parents[1]
G32=FOLLOWUP/"objective-gap-designs-20261025/coupled-bundle/scale-extension-g32-v1/short-prefix-n36/sealed-cohort-v1/training-only-manifest.json"
G16=FOLLOWUP/"objective-gap-designs-20261025/coupled-bundle/primary-cohort-20261025-v2/training-only-manifest.json"


def main():
    sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"results")
    parser.add_argument("--logs",type=Path,default=ROOT/"fit-logs")
    parser.add_argument("--cpu-slot-released-receipt",type=Path,required=True,
        help="Root-authorized prior UC completion record, verified before any fit")
    args=parser.parse_args()
    if not args.cpu_slot_released_receipt.is_file():raise ValueError("Missing prior UC completion evidence")
    jobs=[]
    for manifest,G,n in ((G32,32,36),(G16,16,37)):
        selection=driver.generic.validate_selection(manifest)
        entries=[entry for entry in selection["cases"] if int(entry["D"])==3*G and int(entry["n_train"])==n]
        if len(entries)!=3 or {int(e["physical_seed"]) for e in entries}!={1009,1013,1019}:
            raise ValueError("Retain all three physical seeds at the fixed prefix")
        for entry in entries:
            driver.selected_entry(manifest,entry["directory"])
            jobs.append((manifest,entry))
    args.logs.mkdir(parents=True,exist_ok=True)
    guard=dict(worker_count=1,maximum_total_workers_root_managed=4,
        prior_completion_receipt=str(args.cpu_slot_released_receipt.resolve()),
        prior_completion_sha256=driver.generic.sha(args.cpu_slot_released_receipt),
        driver_sha256=driver.generic.sha(ROOT/"coupled_svm_mlopt_driver.py"),
        scheduler_sha256=driver.generic.sha(__file__),cases=[e["case_id"] for _,e in jobs],
        manifests={str(m):driver.generic.sha(m) for m in (G32,G16)},
        start_utc=datetime.now(timezone.utc).isoformat(),development_accessed=False,confirmation_accessed=False)
    driver.generic.write_new(args.logs/"start.json",guard)
    rows=[]
    env=os.environ.copy();env["PYTHONIOENCODING"]="utf-8"
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):env[name]="1"
    for manifest,entry in jobs:
        log=args.logs/(entry["case_id"]+".log")
        argv=[sys.executable,"-u",str(ROOT/"coupled_svm_mlopt_driver.py"),"--manifest",str(manifest),
            "--case-dir",entry["directory"],"--out",str(args.out),"--methods","svm","mlopt","--device","cpu","--fit"]
        with log.open("x",encoding="utf-8") as stream:
            result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=env)
        row=dict(case_id=entry["case_id"],exit_code=result.returncode,argv=argv,
            log_sha256=driver.generic.sha(log),completed_utc=datetime.now(timezone.utc).isoformat())
        driver.generic.write_new(log.with_suffix(".json"),row);rows.append(row)
        print(driver.generic.json.dumps(row),flush=True)
        if result.returncode:raise RuntimeError("Stop queue on the first unresolved fit error")
    artifacts={str(p.relative_to(args.out)).replace("\\","/"):driver.generic.sha(p)
        for p in sorted(args.out.glob("models/*/*/*")) if p.is_file()}
    driver.generic.write_new(args.logs/"complete.json",dict(guard=guard,results=rows,
        all_exit_zero=True,all_models_artifact_sha256=artifacts,
        frozen_models_utc=datetime.now(timezone.utc).isoformat(),
        existing_fresh_stream_predates_these_fits=True,
        no_claim_of_prefreezing_for_prior_fresh_stream=True,
        development_accessed=False,confirmation_accessed=False))


if __name__=="__main__":main()
