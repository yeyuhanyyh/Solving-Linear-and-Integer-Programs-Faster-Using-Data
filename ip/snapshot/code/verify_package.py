"""No-fit/no-test verification of the portable package and interfaces."""
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
import adapter


def main():
    # This is executable inference code and must travel with the ND model.
    root_features=ROOT/"code"/"nd_root_lp_features.py"
    if not root_features.is_file():
        raise RuntimeError("Required Neural Diving root-LP runtime dependency is missing")
    bundle=ROOT/"bundle-manifest.json"
    if bundle.exists():
        import hashlib
        manifest=json.loads(bundle.read_text(encoding="utf-8"))
        if "code/nd_root_lp_features.py" not in manifest["file_sha256"]:
            raise RuntimeError("Bundle does not hash the Neural Diving root-LP runtime dependency")
        for name,expected in manifest["file_sha256"].items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f"Runtime member changed: {name}")
    dataset=adapter.dataset_manifest(verify=True)
    report=[]
    for cid in adapter.list_case_ids():
        case=adapter.load_case(cid,verify=True)
        if any("test" in name for name in vars(case)):
            raise RuntimeError("A fitting case contains a test attribute")
        report.append(dict(case=cid,variables=case.model.n_variables,integer_variables=len(case.model.integer_indices),
            training_costs=case.n_train,common_dimension=len(case.commitment_indices),Ours_K=case.affine["rank"]))
    print(json.dumps(dict(status="portable_data_verified",dataset_fingerprint=dataset["dataset_fingerprint"],cases=report,
                         fitting_calls=0,solver_calls=0,confirmation_draws=0),ensure_ascii=True),flush=True)


if __name__=="__main__":
    main()
