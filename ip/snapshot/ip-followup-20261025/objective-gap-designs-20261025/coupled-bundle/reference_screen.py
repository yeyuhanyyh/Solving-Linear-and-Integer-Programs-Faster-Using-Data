"""Reference-only screen of capacity bundles with coupled upgrade conflicts.

Never loads or trains a learned baseline. All objectives are original gross
profits, including real base-project returns. Small/large stages have separate
immutable protocols and output folders. No existing experiment is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import highspy
import numpy as np
from scipy import sparse
import sympy as sp


ROOT = Path(__file__).resolve().parent
SEEDS = [1009, 1013, 1019]
SIGMAS = [.5, 1.0]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def rng(label):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little"))


def graph(family, seed):
    if family.startswith("cycle"):
        length = int(family[5:].split("x")[0]); G = length * 2
        edges = {tuple(sorted((offset+i, offset+(i+1) % length)))
                 for offset in (0, length) for i in range(length)}
    else:
        G = int(family[6:])
        edges = {tuple(sorted((i, (i+1) % G))) for i in range(G)} | {(0, 2)}
        # Preserve the initially fixed G12 graph streams exactly. New larger
        # candidates receive their own declared G-specific graph streams.
        label = (f"coupled-bundle-randomgraph-v1|{seed}" if G == 12 else
                 f"coupled-bundle-randomgraph-v1|G{G}|{seed}")
        random = rng(label)
        for i in range(G):
            for j in range(i+1, G):
                if (i, j) not in edges and random.random() < .15:
                    edges.add((i, j))
    return G, sorted(edges)


def maximal_independent_sets(G, edges):
    original = [0] * G
    for i, j in edges:
        original[i] |= 1 << j; original[j] |= 1 << i
    all_bits = (1 << G)-1
    complement = [all_bits & ~(original[i] | (1 << i)) for i in range(G)]
    found = []

    def visit(chosen, possible, excluded):
        if not possible and not excluded:
            found.append(chosen); return
        union = possible | excluded
        vertices = [i for i in range(G) if union & (1 << i)]
        pivot = max(vertices, key=lambda i: (possible & complement[i]).bit_count()) if vertices else 0
        candidates = possible & ~complement[pivot]
        while candidates:
            bit = candidates & -candidates; vertex = bit.bit_length()-1
            visit(chosen | bit, possible & complement[vertex], excluded & complement[vertex])
            possible &= ~bit; excluded |= bit; candidates &= ~bit

    visit(0, all_bits, 0)
    masks = sorted(found)
    rows = np.asarray([[(mask >> i) & 1 for i in range(G)] for mask in masks], dtype=np.int64)
    if len(set(masks)) != len(masks) or len(rows) == 0:
        raise ValueError("Invalid maximal-independent-set enumeration")
    for mask in masks:
        if any((mask & (1 << i)) and (mask & original[i]) for i in range(G)):
            raise ValueError("Enumerated set is not independent")
        if any(not (mask & (1 << i)) and not (mask & original[i]) for i in range(G)):
            raise ValueError("Enumerated set is not maximal")
    return rows, masks


def exact_rank(points):
    points = np.asarray(points, dtype=np.int64)
    return int(sp.Matrix((points[1:]-points[0]).tolist()).rank()) if len(points) > 1 else 0


def full_solutions(independent):
    x = np.empty((len(independent), 3*independent.shape[1]), dtype=float)
    x[:, 0::3] = 1-independent
    x[:, 1::3] = 1-independent
    x[:, 2::3] = independent
    return x


def physical(family, seed):
    G, edges = graph(family, seed)
    random = rng(f"coupled-bundle-physical-v1|{family}|{seed}")
    a = random.integers(10, 21, G); b = random.integers(10, 21, G)
    capacity = a+b
    c = np.asarray([random.integers(max(x,y)+1, x+y) for x,y in zip(a,b)], dtype=np.int64)
    if not np.all((c > np.maximum(a,b)) & (c < capacity)):
        raise ValueError("C must be incompatible with either base project")
    A = sparse.lil_matrix((G+len(edges), 3*G), dtype=float)
    for i in range(G):
        A[i, 3*i:3*i+3] = [a[i],b[i],c[i]]
    for row, (i,j) in enumerate(edges, start=G):
        A[row,3*i+2] = A[row,3*j+2] = 1
    return dict(G=G, edges=edges, weight_A=a, weight_B=b, weight_C=c,
                capacity=capacity, A=A.tocsr(),
                row_upper=np.r_[capacity, np.ones(len(edges))],
                row_lower=np.full(G+len(edges), -np.inf))


def cost_vectors(premiums, base_return):
    premiums = np.asarray(premiums, dtype=float)
    count, G = premiums.shape
    costs = np.empty((count, 3*G), dtype=float)
    costs[:,0::3] = costs[:,1::3] = -base_return/2
    costs[:,2::3] = -base_return-premiums
    return costs


def reference_optima(premiums, sets, base_return):
    scores = premiums @ sets.T
    selected = np.argmax(scores, axis=1)
    chosen = sets[selected]
    x = full_solutions(chosen)
    profit = premiums.shape[1]*base_return + scores[np.arange(len(premiums)), selected]
    return x, -profit, selected


def solve_reference(model, cost, integer, seed):
    A = model["A"]
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = cost
    lp.col_lower_, lp.col_upper_ = np.zeros(A.shape[1]), np.ones(A.shape[1])
    lp.row_lower_, lp.row_upper_ = model["row_lower"], model["row_upper"]
    lp.integrality_ = [highspy.HighsVarType.kInteger if integer else highspy.HighsVarType.kContinuous] * A.shape[1]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data
    h = highspy.Highs()
    for key, value in dict(output_flag=False, threads=1, random_seed=seed, time_limit=30.,
                           mip_rel_gap=0., mip_abs_gap=1e-9,
                           primal_feasibility_tolerance=1e-8, mip_feasibility_tolerance=1e-8).items():
        if h.setOptionValue(key,value) != highspy.HighsStatus.kOk:
            raise ValueError(f"Rejected reference solver setting {key}")
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise ValueError("Reference model rejected")
    h.presolve()
    presolved = h.getPresolvedLp()
    pre = dict(status=str(h.getModelPresolveStatus()), variables=presolved.num_col_,
               rows=presolved.num_row_, nonzeros=len(presolved.a_matrix_.value_),
               integer_variables=sum(value != highspy.HighsVarType.kContinuous for value in presolved.integrality_))
    h.run()
    status = h.getModelStatus()
    solution = h.getSolution()
    if status != highspy.HighsModelStatus.kOptimal or not solution.value_valid:
        return dict(certified=False,status=str(status),presolved=pre), None
    x = np.asarray(solution.col_value,dtype=float)
    activity = A @ x
    violation = max(float(np.max(model["row_lower"]-activity,initial=0)),
                    float(np.max(activity-model["row_upper"],initial=0)),
                    float(np.max(-x,initial=0)),float(np.max(x-1,initial=0)))
    intres = float(np.max(np.abs(x-np.rint(x)),initial=0))
    if violation > 1e-6 or (integer and intres > 1e-6):
        raise ValueError("Certified reference violates original model")
    objective = float(cost @ x)
    if abs(objective-h.getObjectiveValue()) > 1e-7+1e-10*abs(objective):
        raise ValueError("Reference original objective mismatch")
    info = h.getInfo()
    return dict(certified=True,status=str(status),objective=objective,
                original_violation=violation,integrality_residual=intres,presolved=pre,
                nodes=int(info.mip_node_count) if integer else None,
                mip_gap=float(info.mip_gap) if integer else None), x


def item_report(family, seed, model, sets, global_K):
    return dict(family=family, physical_seed=seed, G=model["G"], D=3*model["G"],
                edge_count=len(model["edges"]), edges=model["edges"],
                globally_reachable_optimizer_count=len(sets), global_K=global_K,
                maximal_IS_cardinalities=sorted(set(int(x) for x in sets.sum(axis=1))),
                n_prefixes=[global_K+8,2*global_K+5],
                weight_A=model["weight_A"],weight_B=model["weight_B"],
                weight_C=model["weight_C"],capacity=model["capacity"])


def export_training(folder, cid, family, seed, sigma, base_return, model, premiums, x, objectives, K, global_K, n):
    folder.mkdir(parents=True,exist_ok=False)
    A, G = model["A"],model["G"]
    offset = np.full(3*G,-base_return/2); offset[2::3] = -base_return
    coefficients = np.zeros((3*G,G));coefficients[np.arange(G)*3+2,np.arange(G)] = -1
    costs = cost_vectors(premiums[:n],base_return)
    if not np.all(costs < 0) or not np.allclose(costs, offset+premiums[:n]@coefficients.T,rtol=1e-14,atol=1e-12):
        raise ValueError("Signed affine gross-profit map is inconsistent")
    path = folder / "training.npz"
    np.savez_compressed(path,A_data=A.data,A_indices=A.indices,A_indptr=A.indptr,A_shape=np.asarray(A.shape),
                        row_lower=model["row_lower"],row_upper=model["row_upper"],
                        col_lower=np.zeros(3*G),col_upper=np.ones(3*G),integrality=np.ones(3*G,dtype=np.int32),
                        train_features=premiums[:n],train_costs=costs,train_solutions=x[:n],train_objectives=objectives[:n],
                        cost_offset=offset,cost_feature_coefficients=coefficients,
                        commitment_indices=np.arange(3*G),binary_indices=np.arange(3*G))
    metadata = dict(case_id=cid,family="knapsack",problem_name="capacity bundles with conflicting upgrades",
                    seed=seed,n_train=n,D=3*G,K=K,G=G,
                    training_only=True,development_costs_exported=False,
                    training_npz_sha256=sha(path),cost_map_instruction="cost=cost_offset+cost_feature_coefficients@feature",
                    metadata=dict(graph_family=family,sigma=sigma,base_return=base_return,
                                  edges=model["edges"],weights_A=model["weight_A"],weights_B=model["weight_B"],
                                  weights_C=model["weight_C"],block_capacities=model["capacity"],
                                  cost_offset=offset,cost_feature_coefficients=coefficients,
                                  n_categories=G,global_optimizer_hull_K=global_K,
                                  actual_training_optimizer_hull_K=K,
                                  same_train_test_law="independent lognormal premiums exp(sigma Z), no clipping or mean shift",
                                  objective="original gross total profit G*b+sum(v_g*C_g); never subtract the real fallback return",
                                  original_LP_has_no_added_optimizer_equalities=True))
    write_new(folder/"metadata.json",metadata)
    return dict(training_npz_sha256=sha(path),metadata_sha256=sha(folder/"metadata.json"))


def one_setting(out, family, seed, sigma, base_return, model, sets, global_K, dev_count):
    G = model["G"]; prefixes = sorted(set([global_K+8,2*global_K+5])); maximum = max(prefixes)
    train_z = rng(f"coupled-bundle-training-standard-normal-v1|{family}|{seed}").standard_normal((maximum,G))
    dev_z = rng(f"coupled-bundle-development-standard-normal-v1|{family}|{seed}").standard_normal((dev_count,G))
    train, dev = np.exp(sigma*train_z),np.exp(sigma*dev_z)
    train_x,train_v,train_selected = reference_optima(train,sets,base_return)
    dev_x,dev_v,dev_selected = reference_optima(dev,sets,base_return)
    costs = cost_vectors(np.vstack([train,dev]),base_return)
    truths = np.r_[train_v,dev_v]
    key = f"{family}_sigma{sigma:g}_b{base_return:g}_seed{seed}"
    records=[]
    for i,(cost,expected) in enumerate(zip(costs,truths)):
        ip,ip_x = solve_reference(model,cost,True,seed+i)
        lp,lp_x = solve_reference(model,cost,False,seed+i)
        if not ip["certified"] or not lp["certified"]:
            raise ValueError(f"Uncertified reference for {key} index{i}")
        error = abs(ip["objective"]-expected)
        if error > 1e-7+1e-9*abs(expected):
            raise ValueError(f"HiGHS/enumeration mismatch {key} index{i}")
        relation = max(float(np.max(np.abs(lp_x[0::3]+lp_x[2::3]-1),initial=0)),
                       float(np.max(np.abs(lp_x[1::3]+lp_x[2::3]-1),initial=0)))
        ip_relation = max(float(np.max(np.abs(ip_x[0::3]+ip_x[2::3]-1),initial=0)),
                          float(np.max(np.abs(ip_x[1::3]+ip_x[2::3]-1),initial=0)))
        if ip_relation>1e-6:
            raise ValueError("IP optimizer violates proved bundle relation")
        fractional = int(np.count_nonzero(np.abs(lp_x-np.rint(lp_x))>1e-7))
        records.append(dict(index=i,split="training" if i<maximum else "exploratory_development_reference",
                            split_index=i if i<maximum else i-maximum,
                            enumeration_objective=expected,reference_profit=-expected,
                            ip=ip,lp=lp,enumeration_IP_absolute_error=error,
                            lp_fractional_coordinates=fractional,LP_fractional=bool(fractional),
                            LP_optimizer_relation_residual=relation,
                            LP_violates_optimizer_affine_relations=relation>1e-7,
                            IP_optimizer_relation_residual=ip_relation,
                            LP_relative_integrality_gap=(expected-lp["objective"])/abs(expected),
                            IP_profit_over_LP_profit_upper_bound=expected/lp["objective"],
                            gross_fallback_fraction=G*base_return/(-expected)))
    folder = out / "references" / key
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/"exploratory-development-reference.npz",features=dev,costs=cost_vectors(dev,base_return),
                        solutions=dev_x,objectives=dev_v)
    write_new(folder/"raw-reference-checks.json",records)
    rows=[]
    for n in prefixes:
        K = exact_rank(train_x[:n])
        cid=f"coupledbundle_{key}_n{n}"
        source=export_training(out/"training-candidates"/key/f"n{n}",cid,family,seed,sigma,base_return,
                               model,train,train_x,train_v,K,global_K,n)
        subset = records[:n]+records[maximum:]
        development = records[maximum:]
        rows.append(dict(case_id=cid,graph_family=family,physical_seed=seed,sigma=sigma,base_return=base_return,
                          G=G,D=3*G,original_rows=model["A"].shape[0],original_nonzeros=model["A"].nnz,
                          global_K=global_K,training_K=K,training_covers_global_hull=K==global_K,
                          n_train=n,train_samples_over_learned_K=n/max(K,1),
                          reachable_optimizer_count=len(sets),
                          training_distinct_optimizers=len(set(map(int,train_selected[:n]))),
                          exploratory_development_count=dev_count,
                          development_unseen_optimizer_fraction=float(np.mean(~np.isin(dev_selected,train_selected[:n]))),
                          reference_IP_count=len(subset),all_IP_references_match_enumeration=True,
                          LP_fractional_fraction=float(np.mean([r["LP_fractional"] for r in development])),
                          LP_optimizer_relation_violation_fraction=float(np.mean([r["LP_violates_optimizer_affine_relations"] for r in development])),
                          LP_relative_integrality_gap_mean=float(np.mean([r["LP_relative_integrality_gap"] for r in development])),
                          IP_profit_over_LP_upper_bound_mean=float(np.mean([r["IP_profit_over_LP_profit_upper_bound"] for r in development])),
                          LP_fractional_coordinate_count_mean=float(np.mean([r["lp_fractional_coordinates"] for r in development])),
                          original_IP_presolved_vars_min=min(r["ip"]["presolved"]["variables"] for r in development),
                          original_IP_presolved_vars_max=max(r["ip"]["presolved"]["variables"] for r in development),
                          original_IP_presolved_vars_mean=float(np.mean([r["ip"]["presolved"]["variables"] for r in development])),
                          original_IP_presolved_rows_mean=float(np.mean([r["ip"]["presolved"]["rows"] for r in development])),
                          original_IP_presolve_reduced_to_empty_fraction=float(np.mean([r["ip"]["presolved"]["variables"]==0 for r in development])),
                          original_IP_nodes_mean=float(np.mean([r["ip"]["nodes"] for r in development])),
                          gross_fallback_fraction_mean=float(np.mean([r["gross_fallback_fraction"] for r in development])),
                          source_hashes=source,
                          reference_checks_sha256=sha(folder/"raw-reference-checks.json"),
                          exploratory_reference_bank_sha256=sha(folder/"exploratory-development-reference.npz")))
    return rows


def render(rows, stage, complete):
    text=[f"# Coupled capacity-bundle reference screen: {stage}","",
          f"Status: {'COMPLETE' if complete else 'PARTIAL'}. No learned baseline was run; this does not establish an objective-ratio advantage.","",
          "Each region chooses two ordinary base projects, each returning b/2, or a capacity-incompatible upgrade returning b+v. Upgrade projects have graph conflicts. All returns are positive. Every integer optimum chooses AB or C, so gross profit is G*b+sum(v*C); this real fallback is never subtracted from a ratio.","",
          "Positive premiums make reachable upgrade choices exactly the maximal independent sets. Their entire affine hull is enumerated, and training ranks are exact. The original LP has no imposed optimizer equalities. Both predeclared prefixes and all physical seeds are retained, including failures to cover the global hull.","",
          "| Graph | seed | sigma | b | D | n | trainK/globalK | #global opt | #train opt | LP fractional (%) | LP gap (%) | Presolved IP vars mean | New dev opt (%) |",
          "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['graph_family']} | {r['physical_seed']} | {r['sigma']:g} | {r['base_return']:g} | {r['D']} | {r['n_train']} | {r['training_K']}/{r['global_K']} | {r['reachable_optimizer_count']} | {r['training_distinct_optimizers']} | {100*r['LP_fractional_fraction']:.1f} | {100*r['LP_relative_integrality_gap_mean']:.2f} | {r['original_IP_presolved_vars_mean']:.1f} | {100*r['development_unseen_optimizer_fraction']:.1f} |")
    return "\n".join(text)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["small","large"],default="small")
    parser.add_argument("--include-G20",action="store_true")
    args=parser.parse_args()
    families=(["random12","cycle5x2","cycle7x2"] if args.stage=="small" else
              ["random16","random18"]+(["random20"] if args.include_G20 else []))
    bases=[.25,.5,.1,.05] if args.stage=="small" else [.25]
    dev_count=48
    protocol=dict(stage=args.stage,created_utc=now(),families=families,physical_seeds=SEEDS,sigmas=SIGMAS,
                  base_returns=bases,primary_base_return=.25,
                  training_prefix_rules=["global_exact_K+8","2*global_exact_K+5"],
                  independent_development_reference_count=dev_count,
                  weights="wA,wB IID integerUniform10..20; B=wA+wB; wC integerUniform(max(wA,wB)+1,B-1)",
                  graph="Two disjoint odd cycles, or connected cycle+triangle(0,1,2)+independent extraedges p=.15",
                  costs="profitA=profitB=b/2; profitC=b+exp(sigma Z); independent same-law train/development premiums, no clipping",
                  objective_ratio_policy="Original gross totalprofit G*b+sum(v*C); no artificial constant or fallback subtraction",
                  numerical_labels="MaximalIS enumeration using positive-premium proof; every sampled training/development value independently checked by original HiGHS MIP",
                  source_sha256=sha(Path(__file__)),HiGHS=highspy.Highs().version(),
                  no_baseline_training=True,no_existing_fresh_bank_read=True,no_confirmation_created=True)
    out=ROOT/("screen-"+args.stage)
    if out.exists():raise FileExistsError("Immutable stage output already exists; use a separately named future stage")
    out.mkdir(parents=True)
    write_new(out/"protocol.json",protocol)
    all_rows=[]; geometries=[]
    for family in families:
        for seed in SEEDS:
            model=physical(family,seed);sets,masks=maximal_independent_sets(model["G"],model["edges"])
            global_K=exact_rank(sets)
            item=item_report(family,seed,model,sets,global_K);geometries.append(item)
            write_new(out/"global-geometry"/(family+f"_seed{seed}.json"),dict(**item,maximal_IS_masks=masks,maximal_IS_vectors=sets))
    for family in families:
        for base_return in bases:
            for sigma in SIGMAS:
                for seed in SEEDS:
                    model=physical(family,seed);sets,_=maximal_independent_sets(model["G"],model["edges"])
                    geometry=next(x for x in geometries if x["family"]==family and x["physical_seed"]==seed)
                    rows=one_setting(out,family,seed,sigma,base_return,model,sets,geometry["global_K"],dev_count)
                    all_rows.extend(rows)
                    with (out/"progress.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(clean(dict(utc=now(),completed_setting=family,
                                                          seed=seed,sigma=sigma,base_return=base_return,rows=rows)),allow_nan=False)+"\n")
                    print(json.dumps(dict(event="reference_setting_complete",family=family,seed=seed,sigma=sigma,
                                          base_return=base_return,rows=len(all_rows),full_hull=[x["training_covers_global_hull"] for x in rows])),flush=True)
                    # These are explicitly in-progress new-design reports, not old artifacts.
                    (out/"summary-progress.json").write_text(json.dumps(clean(dict(status="PARTIAL",rows=all_rows,protocol_sha256=sha(out/"protocol.json"))),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write_new(out/"summary.json",dict(status="COMPLETE",stage=args.stage,created_utc=now(),
                                      protocol_sha256=sha(out/"protocol.json"),source_sha256=sha(Path(__file__)),
                                      geometries=geometries,rows=all_rows,
                                      no_baseline_training=True,no_algorithm_objective_ratio_claim=True))
    (out/"summary.md").write_text(render(all_rows,args.stage,True),encoding="utf-8")
    print(json.dumps(dict(status="COMPLETE",stage=args.stage,rows=len(all_rows),summary_sha256=sha(out/"summary.json"))))


if __name__=="__main__":main()
