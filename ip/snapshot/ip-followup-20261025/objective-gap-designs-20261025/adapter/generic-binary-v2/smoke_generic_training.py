"""Training-only loader, original oracle, feature and resume-integrity smoke.

No model fitting, development bank, fresh cost generation or confirmation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import uuid
import numpy as np
import generic_binary_pilot as d


def payload(row):
    row=d.clean(row)
    row["record_payload_sha256"]=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    d.enable_runtime();d.bind_signed_cost_map()
    import torch
    import adapter
    import neural_diving_baseline as nd
    torch.set_num_threads(1)
    manifest=d.validate_selection(args.manifest);rows=[]
    for entry in manifest["cases"]:
        case=d.load_case(entry["directory"],args.manifest)
        cost=case.train_costs[0];feature=case.train_features[0];reference=float(case.train_objectives[0])
        solved=adapter.solve(case.model,cost,seed=case.seed,time_limit=30.)
        assert solved.certified and solved.valid
        assert abs(solved.objective-reference)<=adapter.recovery_tolerance(reference)
        assert np.allclose(d.signed_costs_from_features(case,case.train_features),case.train_costs,rtol=1e-14,atol=1e-12)
        config=nd.NeuralDivingConfig(seed=int(case.metadata["baseline_seed"]),device="cpu")
        graph=nd.FixedBipartiteGraph(case.model,case.train_costs,normalize_rows=config.normalize_constraint_rows,
            device="cpu",use_root_lp_features=True,root_lp_time_limit=5.,root_lp_seed=case.seed)
        shape=list(graph.features(cost).shape)
        assert shape==[1,case.model.n_variables+case.model.A.shape[0],28]
        guard=dict(training=d.pilot_guard(case,"ours"),development_bank_sha256="training-only-toy",budget=30.)
        original=payload(dict(guard=guard,case_id=case.case_id,method="ours",index=0,n_train=case.n_train,
            K=int(case.affine["rank"]),actual_K=int(case.affine["rank"]),D=case.model.n_variables,
            cost=cost,feature=feature,budget_seconds=30.,original_feasible=True,restriction_valid=True,
            timely=True,valid=True,objective=reference,reference_objective=reference,profit_ratio=1.,
            recovered=True,raw_objective=reference,seconds=0.,solution=case.train_solutions[0],fixed=None,
            experiment_phase="development",confirmation=False))
        d.verify_development_record(original,case,"ours",0,cost,reference,guard,30.,feature)
        poisoned={key:value for key,value in original.items() if key!="record_payload_sha256"}
        poisoned["profit_ratio"]=.8;poisoned=payload(poisoned)
        try:d.verify_development_record(poisoned,case,"ours",0,cost,reference,guard,30.,feature)
        except ValueError:pass
        else:raise AssertionError("A recomputed-checksum false ratio was accepted")
        rows.append(dict(case_id=case.case_id,n=case.n_train,D=case.model.n_variables,K=int(case.affine["rank"]),
            rows=case.model.A.shape[0],feature_dimension=case.train_features.shape[1],training_cost_index=0,
            original_IP_certified=True,explicit_signed_affine_map_passed=True,ND_feature_shape=shape,
            recomputed_checksum_false_ratio_rejected=True,
            primary_schema_assertor_evidence=case.metadata["primary_schema_assertor_evidence"]))
    folder=d.ROOT/("io-smoke-"+uuid.uuid4().hex)
    if folder.resolve().parent!=d.ROOT.resolve():raise ValueError("I/O smoke is outside the intended workspace")
    folder.mkdir();toy=folder/"toy.json"
    try:
        d.write_new(toy,dict(value="keep"));before=toy.read_bytes();d.write_new(toy,dict(value="keep"))
        try:d.write_new(toy,dict(value="must not replace"))
        except FileExistsError:pass
        else:raise AssertionError("Existing artifact was overwritten")
        assert toy.read_bytes()==before and not list(folder.glob("*.partial-*"))
    finally:
        if toy.exists():toy.unlink()
        folder.rmdir()
    receipt=dict(status="PASS",driver_sha256=d.sha(Path(d.__file__)),protocol_sha256=d.sha(d.ROOT/"GENERIC_BINARY_PROTOCOL.md"),
        smoke_source_sha256=d.sha(Path(__file__)),manifest_sha256=d.sha(args.manifest),cases=rows,
        models_trained=[],development_opened=False,confirmation_read=False,new_costs_sampled=False,
        runtime_scientific_sources_modified=False,publication_no_overwrite_test_passed=True)
    d.write_new(args.receipt,receipt)
    print(json.dumps(receipt))


if __name__=="__main__":main()
