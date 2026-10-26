"""Fixed extra optimization budget, using only the 36 frozen training costs.

Original checkpoints/sources are read-only. No development/fresh bank loader is
called. There is no validation selection: update 2000 is the only final model.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parent
DESIGNS=ROOT.parents[1]
FOLLOWUP=DESIGNS.parent
PROTOCOL=ROOT/"control-protocol.json"
SOURCE_SEAL=ROOT/"control-source-seal.json"
FROZEN_FRESH=DESIGNS/"coupled-bundle/confirmation-v1/confirmation_runner.py"
FROZEN_FRESH_SHA="d5805a5f0b214ce7ce078f06046403334878f97f8afdfe2b9890ebd9d514ec51"


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def enable():
    if sha(FROZEN_FRESH)!=FROZEN_FRESH_SHA:raise ValueError("Frozen original source/role validator changed")
    spec=importlib.util.spec_from_file_location("training_only_original_role_validator",FROZEN_FRESH)
    shared=importlib.util.module_from_spec(spec);sys.modules[spec.name]=shared;spec.loader.exec_module(shared)
    return shared,shared.enable()


def state_digest(network):
    digest=hashlib.sha256()
    for name,value in sorted(network.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def protocol():
    p=read(PROTOCOL)
    if p.get("optimizer_steps")!=2000 or p.get("batch_size")!=16 or p.get("gradient_clip")!=5. or p.get("learning_rates")!=[.001,.0001] or p.get("learning_rate_boundary_step")!=1000:
        raise ValueError("The pre-specified extra optimization budget changed")
    if p.get("physical_seeds")!=[1009,1013,1019] or p.get("n_train")!=36 or p.get("G")!=32:
        raise ValueError("The fixed all-three-seed control cohort changed")
    return p


def originals(case_id=None):
    p=protocol();shared,d=enable();manifest=d.validate_selection(p["manifest_path"])
    entries=[entry for entry in manifest["cases"] if case_id is None or entry["case_id"]==case_id]
    if case_id is not None and len(entries)!=1:raise ValueError("Unknown continuation case")
    result={}
    for entry in entries:
        case=d.load_case(entry["directory"],p["manifest_path"])
        if case.n_train!=36 or case.train_features.shape[1]!=32 or case.seed not in p["physical_seeds"]:raise ValueError("Wrong continuation training data")
        role=shared._semantic_roles(case,dict(model_root=p["original_model_root"],G=32))
        files={name:expected for method in role.values() for name,expected in method["files"].items()}
        files.update({str((Path(entry["directory"])/name).resolve()):expected for name,expected in entry["files"].items()})
        result[case.case_id]=dict(case=case,original_files=files,original_roles=role)
    if case_id is None and (len(result)!=3 or {value["case"].seed for value in result.values()}!=set(p["physical_seeds"])):
        raise ValueError("All three continuation seeds are required")
    return result


def freeze_source():
    if SOURCE_SEAL.exists():return validate_source()
    p=protocol();_,d=enable();roles=originals()
    files={str(path.resolve()):sha(path) for path in (Path(__file__),PROTOCOL,FROZEN_FRESH)}
    for source_seal in p["original_source_seals"]:
        files[str(Path(source_seal).resolve())]=sha(source_seal)
        for name,expected in read(source_seal)["source_sha256"].items():
            if sha(name)!=expected:raise ValueError("Original scientific source changed")
            files[name]=expected
    for value in roles.values():files.update(value["original_files"])
    result=dict(schema="fixed-ND-extra-budget-source-freeze-v1",frozen_before_continuation=True,source_sha256=files,
        case_ids=sorted(roles),original_weights_untouched=True,training_only=True,steps=2000,
        optimizer_state_reset=True,validation_or_test_selection=False)
    d.write_new(SOURCE_SEAL,result);return result


def validate_source():
    s=read(SOURCE_SEAL)
    if s.get("schema")!="fixed-ND-extra-budget-source-freeze-v1" or s.get("frozen_before_continuation") is not True:raise ValueError("Need a continuation source/original-weight freeze")
    for name,expected in s["source_sha256"].items():
        if sha(name)!=expected:raise ValueError("Frozen continuation/original artifact changed: "+name)
    return s


def validate_final(case_id,out):
    validate_source();p=protocol();record=originals(case_id)[case_id]
    folder=Path(out)/case_id;marker=read(folder/"complete.json")
    expected=dict(case_id=case_id,control_source_seal_sha256=sha(SOURCE_SEAL),protocol_sha256=sha(PROTOCOL),original_files=record["original_files"])
    if marker.get("guard")!=expected or marker.get("optimizer_steps")!=2000 or marker.get("final_checkpoint_step")!=2000:
        raise ValueError("Continued final budget/source guard differs")
    for name,digest in marker["artifact_sha256"].items():
        if sha(folder/name)!=digest:raise ValueError("Continued final artifact changed")
    return folder,marker


def run(case_id,out,smoke=False):
    validate_source();p=protocol();shared,d=enable()
    import torch
    import neural_diving_baseline as nd
    import train_uc_baselines as nn
    import uc_adapters as data
    selected=originals(case_id)[case_id];case=selected["case"]
    final=Path(out)/case_id
    if final.exists():
        validate_final(case_id,out);return dict(status="already_complete",case_id=case_id)
    original_folder,_=nn.validate_checkpoint(p["original_model_root"],case,"neural_diving",d.ROOT)
    model=nn.load_nd_complete(case,original_folder/"complete-predictor.pt")
    config=model.config;nd._setup(config)
    if config.batch_size!=16 or config.gradient_clip!=5. or config.learning_rate!=.001:
        raise ValueError("Original selected model optimizer convention changed")
    initial_state=state_digest(model.full_predictor)
    parameters={name:list(value.shape) for name,value in model.full_predictor.state_dict().items()}
    begin=time.perf_counter();cpu_begin=time.process_time()
    costs,pools,pool_metadata=nd._prepare_pools(case,data.singleton_pools(case),model.graph.integer_indices)
    root_features=model.graph.precompute_training_root_lp(costs)
    preparation_seconds=time.perf_counter()-begin
    indices=np.arange(case.n_train,dtype=np.int64)
    optimizer=nd._optimizer(model.full_predictor,config)
    if optimizer.state:raise ValueError("Control must start with a fresh, empty Adam state")
    batch_seed=int.from_bytes(hashlib.sha256((case_id+"|fixed-extra-optimization-budget-v1").encode()).digest()[:4],"little")
    rng=np.random.default_rng(batch_seed)
    order=np.empty(0,dtype=int);cursor=0;epochs_started=0
    trajectory=[dict(step=0,learning_rate=.001,diagnostics=nd._full_diagnostics(model.full_predictor,costs,pools,indices))]
    updates=[];steps=1 if smoke else 2000
    # Every epoch keeps the original 16/16/4 permutation batching convention.
    for step in range(1,steps+1):
        if cursor>=len(order):order=rng.permutation(indices);cursor=0;epochs_started+=1
        chosen=order[cursor:cursor+16];cursor+=len(chosen)
        lr=.001 if step<=1000 else .0001
        for group in optimizer.param_groups:group["lr"]=lr
        model.full_predictor.train();optimizer.zero_grad(set_to_none=True)
        logits,_=model.full_predictor(costs[chosen]);loss=nd._full_loss(logits,chosen,pools)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite original full-predictor NLL")
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.full_predictor.parameters(),5.)
        if not bool(torch.isfinite(gradient)):raise FloatingPointError("Nonfinite continuation gradient")
        optimizer.step()
        updates.append(dict(step=step,learning_rate=lr,batch_indices=chosen.tolist(),batch_count=len(chosen),
            batch_NLL_before_update=float(loss.detach()),unclipped_gradient_norm=float(gradient)))
        if step%100==0 or step==steps:
            diagnostics=nd._full_diagnostics(model.full_predictor,costs,pools,indices)
            trajectory.append(dict(step=step,learning_rate=lr,diagnostics=diagnostics))
            print(json.dumps(dict(event="continuation_training_only",case_id=case_id,step=step,diagnostics=diagnostics)),flush=True)
    model.full_predictor.eval()
    if parameters!={name:list(value.shape) for name,value in model.full_predictor.state_dict().items()}:raise ValueError("Selected architecture changed")
    for name,expected in selected["original_files"].items():
        if sha(name)!=expected:raise ValueError("An original source/model/training artifact was modified")
    if smoke:return dict(status="SMOKE_PASS_NO_MODEL_PUBLISHED",case_id=case_id,updates=1,training_only=True,original_files_unchanged=True,
        original_width=config.hidden_size,parameter_shapes_unchanged=True,initial_state_digest=initial_state,source_seal_sha256=sha(SOURCE_SEAL))
    partial=final.with_name(final.name+".partial-"+uuid.uuid4().hex);partial.mkdir(parents=True,exist_ok=False)
    guard=dict(case_id=case_id,control_source_seal_sha256=sha(SOURCE_SEAL),protocol_sha256=sha(PROTOCOL),original_files=selected["original_files"])
    metadata=dict(original_training_metadata=model.training_metadata,control="fixed extra optimization budget",config=asdict(config),
        optimizer_state_reset=True,initial_Adam_state_empty=True,optimizer_steps=2000,final_checkpoint_step=2000,
        all_training_count=36,train_cost_digest=nd._array_digest(costs),batch_seed=batch_seed,batching="original per-epoch permutation, batches16/16/4",
        learning_rate_schedule=[dict(steps=[1,1000],learning_rate=.001),dict(steps=[1001,2000],learning_rate=.0001)],
        original_selected_width=config.hidden_size,initial_state_digest=initial_state,final_state_digest=state_digest(model.full_predictor),
        preparation_seconds=preparation_seconds,total_wall_seconds=time.perf_counter()-begin,total_CPU_seconds=time.process_time()-cpu_begin,
        epochs_started=epochs_started,final_batch_cursor=cursor,pool_metadata=pool_metadata,root_features=root_features,
        trajectory=trajectory,updates=updates,training_only=True,validation_or_test_read=False,checkpoint_selected_by="fixed final update2000 only",
        resume_policy="No mid-training resume; failures restart from the same original weights and prescribed batch seed")
    model.training_metadata=dict(model.training_metadata,fixed_extra_optimization_budget=metadata)
    nn.save_nd_complete(model,partial/"continued-predictor.pt")
    d.write_new(partial/"metadata.json",metadata)
    artifacts={str(path.relative_to(partial)).replace("\\","/"):sha(path) for path in partial.rglob("*") if path.is_file()}
    d.write_new(partial/"complete.json",dict(guard=guard,artifact_sha256=artifacts,optimizer_steps=2000,final_checkpoint_step=2000,training_only=True))
    os.rename(partial,final);validate_final(case_id,out)
    return dict(status="COMPLETE",case_id=case_id,optimizer_steps=2000,original_width=config.hidden_size,
        initial_train_diagnostics=trajectory[0]["diagnostics"],final_train_diagnostics=trajectory[-1]["diagnostics"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","validate","smoke","continue"))
    parser.add_argument("--case-id")
    parser.add_argument("--out",type=Path,default=ROOT/"continued-models")
    args=parser.parse_args()
    if args.command=="freeze":result=freeze_source()
    elif args.command=="validate":result=validate_source()
    else:
        if not args.case_id:parser.error("Supply one fixed --case-id")
        result=run(args.case_id,args.out,smoke=args.command=="smoke")
    print(json.dumps(enable()[1].clean(result),allow_nan=False),flush=True)


if __name__=="__main__":main()
