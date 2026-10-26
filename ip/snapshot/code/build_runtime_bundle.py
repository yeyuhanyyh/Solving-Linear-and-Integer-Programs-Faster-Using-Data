"""Build a synthetic-only runtime ZIP without local sources or credentials."""
from pathlib import Path
import argparse, hashlib, json, os, zipfile
ROOT=Path(__file__).resolve().parents[1]
RUNTIME_CODE=("adapter.py","portable_adapter.py","dimension_diagnostics.py",
    "mlopt_baseline.py","neural_diving_baseline.py","nd_root_lp_features.py","preference_baseline.py",
    "runner_common.py","run_training.py","run_benchmark.py","summarize_results.py",
    "run_pipeline.py","gpu_smoke.py","verify_package.py",
    "controls/experiment_core.py","controls/xqa_baseline.py","controls/uc_model.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"cloud"/"synthetic-ip-baselines-v2.zip")
    args=parser.parse_args()
    files=[ROOT/"code"/name for name in RUNTIME_CODE]
    if not all(p.is_file() for p in files):
        raise RuntimeError("A required runtime code member is missing")
    files += [p for p in (ROOT/"data").rglob("*") if p.is_file()]
    files += [p for p in (ROOT/"docs").rglob("*.md") if p.is_file()]
    files += [ROOT/"protocol.json",ROOT/"requirements-cpu.txt",ROOT/"run_linux.sh"]
    if any(p.suffix.lower() in (".tex",".bib",".pdf",".pem",".key") for p in files):
        raise RuntimeError("A non-runtime or credential artifact entered the explicit file allowlist")
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
    manifest=dict(kind="synthetic_training_only_runtime",file_sha256=hashes,
        old_test_contents=False,manuscript=False,provider_credentials=False,
        expected_cases=18,training_costs_per_case=120,confirmation_generation="only after fitted-model cohort freeze",
        fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_name(args.output.name+".writing")
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted((p.relative_to(ROOT).as_posix(),p) for p in set(files)):
            if sha(path)!=hashes[name]:
                raise RuntimeError(f"Runtime source changed while packaging: {name}")
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            entry.create_system=3
            entry.compress_type=zipfile.ZIP_DEFLATED
            entry.external_attr=(0o755 if name=="run_linux.sh" else 0o600)<<16
            archive.writestr(entry,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
        archive.writestr("bundle-manifest.json",json.dumps(manifest,indent=2,ensure_ascii=True).encode())
    os.replace(temporary,args.output)
    sidecar=args.output.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(dict(manifest,zip_sha256=sha(args.output),zip_bytes=args.output.stat().st_size),indent=2,ensure_ascii=True),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output),zip_sha256=sha(args.output),bytes=args.output.stat().st_size,members=len(files)+1),ensure_ascii=True))


if __name__=="__main__":
    main()
