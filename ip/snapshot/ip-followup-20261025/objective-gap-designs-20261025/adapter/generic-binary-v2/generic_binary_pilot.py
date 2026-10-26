"""Sealed generic binary-IP complete-ND/Ours exploratory adapter.

Arbitrary original binary matrices and signed affine cost maps are supported.
The selected cohort, shape, global K and training source hashes must be bound
by a training-only manifest before any model fit. No optimizer equations are
used as ND features. Existing frozen adapters and runtimes remain unchanged.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parent
FOLLOWUP=ROOT.parents[2]
RUNTIME=ROOT/"runtime"
SHARED=ROOT/"shared"


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def publish_bytes_new(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise FileExistsError(path)
        return
    temporary=path.with_name(path.name+".partial-"+uuid.uuid4().hex)
    with temporary.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        if path.exists():raise FileExistsError(path)
        if os.name=="nt":os.rename(temporary,path)
        else:os.link(temporary,path)
    finally:
        if temporary.exists():
            if temporary.parent.resolve()!=path.parent.resolve() or not temporary.name.startswith(path.name+".partial-") or not temporary.is_file():
                raise ValueError("Refusing to remove a non-owned publication temporary")
            temporary.unlink()


def write_new(path,value):
    raw=(json.dumps(clean(value),indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    publish_bytes_new(path,raw)


def utc():return datetime.now(timezone.utc).isoformat()


def initialize_runtime():
    """Copy audited sources into this new pilot directory, never alter originals."""
    source=FOLLOWUP/"uc_bundle_frozen";manifest=read(source/"manifest.json");hashes={}
    def store(name,raw):
        path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=raw:raise FileExistsError(path)
        if not path.exists():path.write_bytes(raw)
        hashes[name]=sha(path)
    for name,expected in manifest["file_sha256"].items():
        if not name.startswith("runtime/"):continue
        if sha(source/name)!=expected:raise ValueError("Original frozen runtime changed")
        store(name,(source/name).read_bytes())
    for name in ("uc_adapters.py","train_uc_baselines.py"):
        store("shared/"+name,(FOLLOWUP/name).read_bytes())
    core_source=FOLLOWUP.parent/"code/controls/experiment_core.py"
    store("runtime/experiment_core.py",core_source.read_bytes())
    write_new(ROOT/"runtime-seal.json",dict(schema="generic-binary-ND-runtime-v2",file_sha256=hashes,
        unchanged_source_uc_seal=sha(FOLLOWUP/"uc-source-data-seal.json")))


def enable_runtime():
    seal=read(ROOT/"runtime-seal.json")
    for name,expected in seal["file_sha256"].items():
        if sha(ROOT/name)!=expected:raise ValueError(f"Pilot runtime changed: {name}")
    for path in (SHARED,RUNTIME):
        if str(path) not in sys.path:sys.path.insert(0,str(path))


def signed_costs_from_features(case,features):
    values=np.asarray(features,dtype=float)
    return np.asarray(case.metadata["cost_offset"],dtype=float)+values@np.asarray(case.metadata["cost_feature_coefficients"],dtype=float).T


def bind_signed_cost_map():
    import uc_adapters,adapter,portable_adapter
    for module in (uc_adapters,adapter,portable_adapter):
        module.costs_from_features=signed_costs_from_features
    # Checkpoint JSON serialization stays original; only atomic publication
    # uses Windows no-overwrite rename rather than any hard-link operation.
    uc_adapters.publish=publish_bytes_new


def array_hash(*arrays):
    digest=hashlib.sha256()
    for value in arrays:
        value=np.ascontiguousarray(value);digest.update(str(value.shape).encode());digest.update(str(value.dtype).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def validate_selection(path):
    manifest=read(path)
    if manifest.get("schema")!="generic-binary-primary-cohort-v2" or manifest.get("frozen_before_any_fit") is not True:
        raise ValueError("Need the frozen training-only primary cohort manifest")
    if manifest.get("training_only") is not True or manifest.get("confirmation_included") is not False:
        raise ValueError("The cohort manifest must not include development/confirmation arrays")
    entries=manifest.get("cases",[])
    if not entries or len({entry["case_id"] for entry in entries})!=len(entries):raise ValueError("Missing or duplicate selected cases")
    for entry in entries:
        directory=Path(entry["directory"])
        if not directory.is_absolute():raise ValueError("Use absolute selected source directories")
        if not {"training.npz","metadata.json"}<=set(entry["files"]):raise ValueError("Need source training and metadata hashes")
        if not set(entry["files"])<={"training.npz","metadata.json","affine.json"}:raise ValueError("No development banks may be accessed through the training cohort")
        for name,expected in entry["files"].items():
            if sha(directory/name)!=expected:raise ValueError("Selected primary training source changed: "+str(directory/name))
        if int(entry["K"])<0 or int(entry["K"])>int(entry["D"]):raise ValueError("Invalid selected K")
    return manifest


def assert_selected_case_schema(entry,meta,data,matrix,offset,coefficient):
    """Validate a declared physical schema; never add an optimizer equation."""
    schema=entry.get("case_schema","generic_binary")
    if schema=="generic_binary":return dict(case_schema=schema)
    if schema!="coupled-bundle-uniform-gross-v1":raise ValueError("Unknown frozen case schema")
    G=data["train_features"].shape[1];D=data["train_costs"].shape[1]
    if D!=3*G or int(entry["K"])!=G:raise ValueError("Coupled-bundle physical/global geometry declaration changed")
    expected_offset=np.tile([-.25,-.25,0.],G);expected_coeff=np.zeros((D,G))
    expected_coeff[3*np.arange(G)+2,np.arange(G)]=-1.
    if not np.array_equal(offset,expected_offset) or not np.array_equal(coefficient,expected_coeff):
        raise ValueError("Original A/B=.25,C=xi gross-profit coefficients changed")
    if not np.all((data["train_features"]>=.1)&(data["train_features"]<=2.5)):
        raise ValueError("Primary uniform gross-value feature support changed")
    if not np.all(np.isneginf(data["row_lower"])):raise ValueError("Unexpected coupled-bundle row lower bound")
    pair_rows=set();capacity_rows=set();edges=set()
    for row in range(matrix.shape[0]):
        begin,end=matrix.indptr[row:row+2];columns=matrix.indices[begin:end];values=matrix.data[begin:end]
        if len(columns)==2 and np.all(values==1.) and data["row_upper"][row]==1.:
            groups=columns//3;roles=columns%3
            if np.all(roles==2) and groups[0]!=groups[1]:
                edge=tuple(sorted(map(int,groups)))
                if edge in edges:raise ValueError("Duplicate global conflict edge")
                edges.add(edge)
            elif groups[0]==groups[1] and set(map(int,roles)) in ({0,2},{1,2}):
                key=(int(groups[0]),min(map(int,roles)))
                if key in pair_rows:raise ValueError("Duplicate local strong packing row")
                pair_rows.add(key)
            else:raise ValueError("Unexpected local/global packing row")
        elif len(columns)==3 and len(set(map(int,columns//3)))==1 and set(map(int,columns%3))=={0,1,2}:
            group=int(columns[0]//3);weights={int(column%3):float(value) for column,value in zip(columns,values)}
            capacity=float(data["row_upper"][row])
            if group in capacity_rows or not (weights[0]>0 and weights[1]>0 and weights[0]+weights[1]==capacity and max(weights[0],weights[1])<weights[2]<capacity):
                raise ValueError("Local substitute-capacity physical structure changed")
            capacity_rows.add(group)
        else:raise ValueError("Unrecognized original coupled-bundle constraint row")
    if capacity_rows==set(range(G)) and not pair_rows:local="capacity"
    elif pair_rows=={(group,role) for group in range(G) for role in (0,1)} and not capacity_rows:local="pairwise_hull"
    else:raise ValueError("Missing or mixed local bundle formulation")
    if not edges:raise ValueError("Expected genuinely coupled C-variable conflict graph")
    if entry.get("local_formulation",local)!=local:raise ValueError("Selected local formulation changed")
    return dict(case_schema=schema,local_formulation=local,local_row_count=G if local=="capacity" else 2*G,global_conflict_edges=len(edges))


def load_case(directory,manifest_path):
    """Accept the existing export schema and preserve all original matrix rows."""
    directory=Path(directory);meta=read(directory/"metadata.json")
    selection=validate_selection(manifest_path)
    entries=[entry for entry in selection["cases"] if Path(entry["directory"]).resolve()==directory.resolve()]
    if len(entries)!=1:raise ValueError("Case is outside the frozen primary cohort")
    entry=entries[0]
    if meta["case_id"]!=entry["case_id"]:raise ValueError("Selected case identity changed")
    if meta.get("development_costs_exported",False):raise ValueError("Training metadata admits development outcomes")
    with np.load(directory/"training.npz",allow_pickle=False) as stream:
        data={key:stream[key].copy() for key in stream.files}
    if meta.get("training_npz_sha256",sha(directory/"training.npz"))!=sha(directory/"training.npz"):
        raise ValueError("Source training hash changed")
    import adapter
    import experiment_core as core
    from scipy import sparse
    matrix=sparse.csr_matrix((data["A_data"],data["A_indices"],data["A_indptr"]),shape=tuple(data["A_shape"]))
    model=adapter.LinearModel(A=matrix,**{key:data[key] for key in ("row_lower","row_upper","col_lower","col_upper","integrality")})
    n,D=data["train_costs"].shape;q=data["train_features"].shape[1]
    if not np.array_equal(model.integrality,np.ones(D,dtype=int)):raise ValueError("This pilot requires binary variables only")
    if not (np.array_equal(model.col_lower,np.zeros(D)) and np.array_equal(model.col_upper,np.ones(D))):raise ValueError("Expected binary bounds [0,1]")
    if data["train_solutions"].shape!=(n,D) or len(data["train_objectives"])!=n:raise ValueError("Training prefix shape mismatch")
    physical=dict(meta.get("metadata",{}))
    offset=np.asarray(data.get("cost_offset",physical.get("cost_offset")),dtype=float)
    coefficient=np.asarray(data.get("cost_feature_coefficients",physical.get("cost_feature_coefficients")),dtype=float)
    if offset.shape!=(D,) or coefficient.shape!=(D,q):raise ValueError("Need explicit D-vector offset and D-by-q affine cost map")
    schema_evidence=assert_selected_case_schema(entry,meta,data,matrix,offset,coefficient)
    for key,value in dict(D=D,n_train=n,rows=matrix.shape[0],feature_dimension=q).items():
        if int(entry[key])!=value:raise ValueError("Selected case dimension changed: "+key)
    if not np.isfinite(data["train_features"]).all():raise ValueError("Nonfinite original cost features")
    if not np.allclose(offset+data["train_features"]@coefficient.T,data["train_costs"],rtol=1e-14,atol=1e-12):raise ValueError("Signed affine training cost map changed")
    if not np.all(np.isfinite(data["train_costs"])) or not np.all(data["train_objectives"]<0):
        raise ValueError("Need finite signed original costs and strictly positive optimal net benefits")
    integer=np.rint(data["train_solutions"]).astype(np.int64)
    if not np.allclose(integer,data["train_solutions"],rtol=0,atol=1e-6):raise ValueError("Noninteger training label")
    affine=core.fit_affine(integer)
    declared=meta.get("K")
    if declared is not None and int(declared)!=affine["rank"]:raise ValueError("Exact rational prefix rank differs from supplied K")
    if int(entry["K"])!=affine["rank"]:raise ValueError("Selected training prefix has not covered its declared global reachable-optimizer K")
    if "global_optimizer_hull_K" in meta and int(meta["global_optimizer_hull_K"])!=int(entry["K"]):raise ValueError("Global optimizer geometry evidence changed")
    cid=meta["case_id"];seed=int(meta["seed"])
    physical.update(cost_offset=offset.tolist(),cost_feature_coefficients=coefficient.tolist(),n_categories=q,
        baseline_seed=int(meta.get("baseline_seed",int.from_bytes(hashlib.sha256((cid+"|generic-binary-v2-20261025").encode()).digest()[:4],"little")%(2**31-1))),
        problem_name=meta.get("problem_name",physical.get("problem_name","generic binary project selection")),
        full_constraint_matrix_shape=list(matrix.shape),exact_rank_from_same_training_prefix=True,
        training_metadata=meta,source_case_directory=str(directory.resolve()),
        primary_schema_assertor_evidence=schema_evidence,
        selection_manifest_sha256=sha(manifest_path),selection_manifest_path=str(Path(manifest_path).resolve()))
    common=np.asarray(data.get("commitment_indices",data.get("binary_indices",np.arange(D))),dtype=int)
    if not np.array_equal(common,np.arange(D)):raise ValueError("Common scope must contain all original binary variables")
    base=dict(A=matrix,row_lower=model.row_lower,row_upper=model.row_upper,col_lower=model.col_lower,col_upper=model.col_upper,
        integrality=model.integrality,base_cost=offset,binary_indices=common,n_times=1,n_units=D,metadata=physical)
    case=adapter.FrozenCase(case_id=cid,family=str(meta.get("family",entry.get("family","generic_binary"))),seed=seed,amplitude=float(meta.get("amplitude",0.)),model=model,
        train_features=data["train_features"],train_costs=data["train_costs"],train_solutions=data["train_solutions"],
        train_objectives=data["train_objectives"],commitment_indices=common,affine=affine,original_base=base,metadata=physical,
        source_hashes={"training.npz":sha(directory/"training.npz"),"metadata.json":sha(directory/"metadata.json")},
        dataset_fingerprint=array_hash(*[data[key] for key in ("train_features","train_costs","train_solutions","train_objectives")]),source_directory=directory)
    for cost,x,value in zip(case.train_costs,case.train_solutions,case.train_objectives):
        if not adapter.validate_solution(model,x)["valid"]:raise ValueError("Archived training solution violates original model")
        if abs(float(cost@x)-value)>adapter.recovery_tolerance(value):raise ValueError("Archived training objective mismatch")
    return case


def pilot_guard(case,method,bundle=None):
    return dict(case_id=case.case_id,method=method,training_array_hash=case.dataset_fingerprint,
        source_hashes=case.source_hashes,target_K=int(case.affine["rank"]),
        pilot_runner_sha256=sha(Path(__file__)),runtime_seal_sha256=sha(ROOT/"runtime-seal.json"),
        pilot_protocol_sha256=sha(ROOT/"GENERIC_BINARY_PROTOCOL.md"),
        selection_manifest_sha256=case.metadata["selection_manifest_sha256"],
        all_shared_source_sha256=read(ROOT/"runtime-seal.json")["file_sha256"],
        cost_map="signed affine; offset + coefficients @ feature",development_used_in_training=False,
        new_confirmation_created=False)


def verify_training_oracles(case,out):
    """Independent original binary-IP certificates before singleton ND fitting."""
    import adapter
    folder=Path(out)/"training-certificates"/case.case_id;folder.mkdir(parents=True,exist_ok=True)
    guard=pilot_guard(case,"training_oracle")
    if (folder/"complete.json").exists():
        result=read(folder/"complete.json")
        if result["guard"]!=guard:raise ValueError("Training certificate guard changed")
        return result
    started=time.perf_counter();rows=[]
    for i,(cost,reference) in enumerate(zip(case.train_costs,case.train_objectives)):
        path=folder/f"train{i:03d}.json"
        if path.exists():
            row=read(path)
            if row["guard"]!=guard:raise ValueError("Training oracle source changed")
        else:
            solved=adapter.solve(case.model,cost,seed=case.seed+i,time_limit=30.)
            if not (solved.valid and solved.certified and solved.objective is not None):raise ValueError(f"Training oracle not globally certified: {i}")
            if abs(float(solved.objective)-float(reference))>adapter.recovery_tolerance(reference):raise ValueError("Source label is not an optimal original IP solution")
            row=dict(guard=guard,index=i,source_objective=float(reference),oracle=solved.to_dict(include_solution=False))
            write_new(path,row)
        rows.append(row)
    result=dict(guard=guard,certified_label_count=len(rows),extra_oracle_verification_seconds=time.perf_counter()-started,
        original_label_generation_not_included=True,all_labels_globally_certified=True)
    write_new(folder/"complete.json",result);return result


def load_development(case):
    path=case.source_directory/"development.npz"
    with np.load(path,allow_pickle=False) as stream:
        raw={key:stream[key].copy() for key in stream.files}
    arrays={key:raw.get("dev_"+key,raw.get(key)) for key in ("features","costs","solutions","objectives")}
    if any(value is None for value in arrays.values()):raise ValueError("Need a separately fixed development bank")
    offset=np.asarray(case.metadata["cost_offset"]);coefficient=np.asarray(case.metadata["cost_feature_coefficients"])
    if not np.allclose(offset+arrays["features"]@coefficient.T,arrays["costs"],rtol=1e-14,atol=1e-12):raise ValueError("Development signed affine cost map mismatch")
    if not np.all(arrays["objectives"]<0):raise ValueError("Expected negative optimal profit references")
    import adapter
    if arrays["costs"].shape!=arrays["solutions"].shape or len(arrays["objectives"])!=len(arrays["costs"]):
        raise ValueError("Development bank shapes differ")
    for cost,x,objective in zip(arrays["costs"],arrays["solutions"],arrays["objectives"]):
        if not adapter.validate_solution(case.model,x)["valid"]:raise ValueError("Invalid development reference solution")
        if abs(float(cost@x)-objective)>adapter.recovery_tolerance(objective):raise ValueError("Development reference objective mismatch")
    return arrays,sha(path)


def verify_development_record(row,case,method,index,cost,reference,guard,budget,feature):
    import adapter
    import experiment_core as core
    if row.get("guard")!=guard or row.get("index")!=index or row.get("method")!=method:
        raise ValueError("Development record identity/source guard changed")
    if row.get("case_id")!=case.case_id or row.get("budget_seconds")!=float(budget) or not np.array_equal(np.asarray(row.get("cost")),cost):
        raise ValueError("Development original cost/context changed")
    if not np.array_equal(np.asarray(row.get("feature")),feature) or not np.allclose(signed_costs_from_features(case,row.get("feature")),cost,rtol=1e-14,atol=1e-12):
        raise ValueError("Development original feature/cost map changed")
    if row.get("n_train")!=case.n_train or row.get("K")!=int(case.affine["rank"]) or row.get("D")!=case.model.n_variables or row.get("confirmation") is not False:
        raise ValueError("Development population metadata changed")
    stored=row.get("record_payload_sha256")
    payload={key:value for key,value in row.items() if key!="record_payload_sha256"}
    current=hashlib.sha256(json.dumps(clean(payload),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if stored!=current:raise ValueError("Development payload checksum changed")
    x=None if row.get("solution") is None else np.asarray(row["solution"],dtype=float)
    K=int(case.affine["rank"]);fixed=row.get("fixed")
    original=bool(x is not None and adapter.validate_solution(case.model,x)["valid"])
    if method=="ours":
        if fixed is not None:raise ValueError("Ours must use its learned affine space")
        actual=K;restriction=bool(original and core.exact_slice_contains(case.affine,np.rint(x).astype(int)))
    else:
        actual=None if fixed is None else case.model.n_variables-len(fixed)
        if fixed is not None:
            if not isinstance(fixed,dict) or actual!=K:raise ValueError("Common K/fixings changed")
            normalized={}
            for key,value in fixed.items():
                column=int(key)
                if str(column)!=str(key) or column in normalized or not 0<=column<case.model.n_variables or value not in (0,1):
                    raise ValueError("Malformed original binary fixing")
                normalized[column]=value
            restriction=bool(original and all(abs(x[j]-value)<=1e-6 for j,value in normalized.items()))
        else:restriction=False
    seconds=float(row["seconds"])
    if not np.isfinite(seconds) or seconds<0:raise ValueError("Invalid elapsed time")
    timely=seconds<=budget;valid=bool(original and restriction and timely and actual==K)
    raw=float(cost@x) if original else None;objective=raw if valid else None
    ratio=objective/float(reference) if valid else None
    recovered=bool(valid and abs(objective-reference)<=adapter.recovery_tolerance(reference))
    for key,value in dict(actual_K=actual,original_feasible=original,restriction_valid=restriction,timely=timely,valid=valid,recovered=recovered).items():
        if row.get(key)!=value:raise ValueError("Development derived status changed: "+key)
    for key,value in dict(objective=objective,raw_objective=raw,reference_objective=float(reference),profit_ratio=ratio).items():
        saved=row.get(key)
        if (saved is None)!=(value is None) or (value is not None and abs(float(saved)-value)>1e-10*max(1.,abs(value))):
            raise ValueError("Development derived value changed: "+key)
    if ratio is not None and ratio>1.+adapter.recovery_tolerance(reference)/abs(reference):
        raise ValueError("A restricted result beats the original certified optimizer")


def certify_development(case,bank,out,bank_hash):
    import adapter
    folder=Path(out)/"development-certificates"/case.case_id;folder.mkdir(parents=True,exist_ok=True)
    guard=dict(training=pilot_guard(case,"development_oracle"),development_bank_sha256=bank_hash)
    for i,(cost,reference) in enumerate(zip(bank["costs"],bank["objectives"])):
        path=folder/f"dev{i:03d}.json"
        if path.exists():
            if read(path)["guard"]!=guard:raise ValueError("Development certificate source changed")
            continue
        solved=adapter.solve(case.model,cost,seed=case.seed+i,time_limit=30.)
        if not (solved.valid and solved.certified and solved.objective is not None):raise ValueError("Original development IP reference was not certified")
        if abs(float(solved.objective)-reference)>adapter.recovery_tolerance(reference):raise ValueError("Source development reference is not globally optimal")
        write_new(path,dict(guard=guard,index=i,reference=float(reference),oracle=solved.to_dict(include_solution=False)))


def lp_diagnostics(case,costs,split,out,bank_hash):
    from nd_root_lp_features import RootLPFeatureExtractor
    extractor=RootLPFeatureExtractor(case.model,case.seed,5.)
    rows=[];folder=Path(out)/"lp-diagnostics"/case.case_id/split;folder.mkdir(parents=True,exist_ok=True)
    guard=dict(training=pilot_guard(case,"lp_diagnostics"),bank_hash=bank_hash,split=split)
    for i,cost in enumerate(costs):
        path=folder/f"cost{i:03d}.json"
        if path.exists():
            row=read(path)
            if row["guard"]!=guard:raise ValueError("LP diagnostics source changed")
        else:
            features,metadata=extractor.extract(cost)
            fractional=features[:case.model.n_variables,1]
            row=dict(guard=guard,index=i,lp_objective=float(metadata["primal_objective"]),
                fractional_variable_count=int((fractional>1e-6).sum()),max_fractionality=float(fractional.max(initial=0.)),
                mean_fractionality=float(fractional.mean()),root_lp_metadata=metadata)
            write_new(path,row)
        rows.append(row)
    return dict(split=split,count=len(rows),fractional_optimum_count=sum(row["fractional_variable_count"]>0 for row in rows),
        mean_fractional_variables=float(np.mean([row["fractional_variable_count"] for row in rows])),
        mean_fractionality=float(np.mean([row["mean_fractionality"] for row in rows])))


def evaluate_development(case,out,methods,budget):
    import adapter
    import experiment_core as core
    import train_uc_baselines as nn
    import neural_diving_baseline as nd
    # Fail closed: complete training guard precedes all development-bank access.
    nn.validate_checkpoint(out,case,"neural_diving",ROOT)
    bank,bank_hash=load_development(case)
    certify_development(case,bank,out,bank_hash)
    model=None
    if "neural_diving" in methods:
        folder,_=nn.validate_checkpoint(out,case,"neural_diving",ROOT)
        model=nn.load_nd_complete(case,folder/"complete-predictor.pt")
    K=int(case.affine["rank"]);results=[]
    for method in methods:
        folder=Path(out)/"development"/case.case_id/method;folder.mkdir(parents=True,exist_ok=True)
        guard=dict(training=pilot_guard(case,method),development_bank_sha256=bank_hash,budget=float(budget))
        rows=[]
        for i,(cost,reference) in enumerate(zip(bank["costs"],bank["objectives"])):
            path=folder/f"dev{i:03d}.json"
            if path.exists():
                row=read(path)
                verify_development_record(row,case,method,i,cost,reference,guard,budget,bank["features"][i])
                rows.append(row);continue
            begin=time.perf_counter()
            if method=="ours":
                raw,x=core.solve(case.original_base,cost,affine=case.affine,time_limit=budget,seed=case.seed+i)
                fixed=None
            elif method=="neural_diving":
                raw=nd.solve_matched_cost(model,case,cost,seed=case.seed+i,K=K,total_time_limit=budget,solve_fn=adapter.solve)
                x=raw.pop("x",None);fixed=raw.get("best_fixed")
            else:raise ValueError(method)
            seconds=time.perf_counter()-begin
            original_valid=bool(x is not None and adapter.validate_solution(case.model,x)["valid"])
            actual_K=K if method=="ours" else (case.model.n_variables-len(fixed) if fixed is not None else None)
            if fixed is not None and actual_K!=K:raise ValueError("ND returned a different common-coordinate K")
            restriction_valid=bool(original_valid and (core.exact_slice_contains(case.affine,np.rint(x[case.commitment_indices]).astype(int)) if method=="ours"
                else fixed is not None and all(abs(x[int(j)]-v)<=1e-6 for j,v in fixed.items())))
            raw_objective=float(cost@x) if original_valid else None
            timely=seconds<=budget
            valid=bool(original_valid and restriction_valid and timely and actual_K==K)
            objective=raw_objective if valid else None
            ratio=objective/float(reference) if objective is not None else None
            if ratio is not None and ratio>1.+adapter.recovery_tolerance(reference)/abs(reference):
                raise ValueError("Restricted result profit exceeds original certified optimum")
            row=dict(guard=guard,case_id=case.case_id,method=method,index=i,n_train=case.n_train,K=K,actual_K=actual_K,D=case.model.n_variables,
                cost=cost,feature=bank["features"][i],budget_seconds=float(budget),
                original_feasible=original_valid,restriction_valid=restriction_valid,timely=timely,valid=valid,
                objective=objective,reference_objective=float(reference),profit_ratio=ratio,
                recovered=bool(valid and abs(objective-reference)<=adapter.recovery_tolerance(reference)),
                raw_objective=raw_objective,seconds=seconds,raw=raw,solution=x,fixed=fixed,
                experiment_phase="development",confirmation=False)
            row=clean(row)
            row["record_payload_sha256"]=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            verify_development_record(row,case,method,i,cost,reference,guard,budget,bank["features"][i])
            write_new(path,row);rows.append(row)
        ratios=[row["profit_ratio"] for row in rows if row["profit_ratio"] is not None]
        summary=dict(case_id=case.case_id,method=method,phase="development",n_train=case.n_train,K=K,D=case.model.n_variables,
            count=len(rows),recovery_count=sum(row["recovered"] for row in rows),
            recovery_percent=100.*sum(row["recovered"] for row in rows)/len(rows),
            profit_ratio_mean_valid=float(np.mean(ratios)) if ratios else None,ratio_denominator=len(ratios),
            negative_returned_profit_count=sum(value<0 for value in ratios),
            no_claim_of_confirmation=True)
        complete=dict(guard=guard,summary=summary,test_record_sha256={f"dev{i:03d}.json":sha(folder/f"dev{i:03d}.json") for i in range(len(rows))})
        write_new(folder/"complete.json",complete);results.append(summary)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-runtime",action="store_true")
    parser.add_argument("--case-dir",type=Path,nargs="+")
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"pilot-results")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--budget",type=float,default=30.)
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--fit",action="store_true")
    parser.add_argument("--evaluate-development",action="store_true")
    parser.add_argument("--lp-diagnostics",choices=("training","development","both"))
    parser.add_argument("--methods",nargs="+",choices=("ours","neural_diving"),default=["ours","neural_diving"])
    args=parser.parse_args()
    if args.initialize_runtime:initialize_runtime()
    if not args.case_dir:
        if args.initialize_runtime:return
        parser.error("Supply fixed --case-dir input directories")
    if not any((args.validate_only,args.fit,args.evaluate_development,args.lp_diagnostics)):parser.error("Select an explicit operation")
    if args.manifest is None:parser.error("Supply the frozen training-only --manifest")
    enable_runtime();bind_signed_cost_map()
    import torch
    import train_uc_baselines as nn
    torch.set_num_threads(1);nn.model_guard=pilot_guard
    for directory in args.case_dir:
        case=load_case(directory,args.manifest)
        print(json.dumps(dict(event="fixed_case_verified",case_id=case.case_id,n=case.n_train,K=case.affine["rank"],
            D=case.model.n_variables,constraints=case.model.A.shape[0],confirmation=False)),flush=True)
        if args.validate_only:continue
        if args.fit:
            certify=verify_training_oracles(case,args.out)
            receipt_path=args.out/"training-case-receipts"/(case.case_id+".json")
            if receipt_path.exists():
                if read(receipt_path)["training_guard"]!=pilot_guard(case,"case"):raise ValueError("Training-case receipt guard changed")
            else:
                write_new(receipt_path,dict(training_guard=pilot_guard(case,"case"),
                    exact_affine=case.affine,original_labels_certified=certify,source=case.source_hashes))
            if "neural_diving" in args.methods:nn.train_one(case,"neural_diving",args.out,ROOT,args.device)
        if args.lp_diagnostics:
            diagnostics=[]
            if args.lp_diagnostics in ("training","both"):
                diagnostics.append(lp_diagnostics(case,case.train_costs,"training",args.out,case.dataset_fingerprint))
            if args.lp_diagnostics in ("development","both"):
                nn.validate_checkpoint(args.out,case,"neural_diving",ROOT)
                bank,bank_hash=load_development(case);diagnostics.append(lp_diagnostics(case,bank["costs"],"development",args.out,bank_hash))
            print(json.dumps(dict(event="lp_diagnostics",case_id=case.case_id,splits=diagnostics)),flush=True)
        if args.evaluate_development:
            results=evaluate_development(case,args.out,args.methods,args.budget)
            print(json.dumps(dict(event="development_results",case_id=case.case_id,results=results)),flush=True)


if __name__=="__main__":main()
