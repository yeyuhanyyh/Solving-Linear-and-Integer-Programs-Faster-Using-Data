"""New reference-only law: positive Uniform upgrade gross values and real base returns.

Separate immutable stage; the previous lognormal screen and its source are
read-only. All independent sets are enumerated because upgrade premiums may be
negative. No learned baseline, model forward, cloud or old/fresh bank is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import highspy
import numpy as np


ROOT=Path(__file__).resolve().parent
HELPER=ROOT/"reference_screen.py"
spec=importlib.util.spec_from_file_location("coupled_reference_helpers_readonly",HELPER)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SEEDS=[1009,1013,1019]
FAMILIES=["random16","random18","random20"]
LOW,HIGH,B=.1,2.5,.5


def all_independent_sets(G,edges):
    neighbors=[0]*G
    for i,j in edges:neighbors[i]|=1<<j;neighbors[j]|=1<<i
    masks=[]
    def visit(chosen,available):
        if not available:masks.append(chosen);return
        bit=available&-available;i=bit.bit_length()-1
        visit(chosen,available&~bit)
        visit(chosen|bit,available&~(bit|neighbors[i]))
    visit(0,(1<<G)-1)
    masks.sort()
    if len(masks)!=len(set(masks)) or 0 not in masks or any((1<<i) not in masks for i in range(G)):
        raise ValueError("All-independent-set enumeration lacks zero/unit vectors")
    if any(any(mask&(1<<i) and mask&neighbors[i] for i in range(G)) for mask in masks):
        raise ValueError("Nonindependent oracle assignment")
    return np.asarray([[(mask>>i)&1 for i in range(G)] for mask in masks],dtype=np.int64),masks


def costs_from_gross(features):
    features=np.asarray(features,dtype=float);costs=np.full((len(features),3*features.shape[1]),-B/2)
    costs[:,2::3]=-features
    return costs


def oracle(features,sets):
    premium=np.asarray(features)-B
    scores=premium@sets.T;selected=np.argmax(scores,axis=1)
    solutions=base.full_solutions(sets[selected])
    objective=-(B*features.shape[1]+scores[np.arange(len(features)),selected])
    direct=np.einsum("ij,ij->i",costs_from_gross(features),solutions)
    if not np.allclose(objective,direct,rtol=1e-13,atol=1e-12):raise ValueError("Gross oracle objective mismatch")
    return solutions,objective,selected


def export(folder,cid,family,seed,model,features,solutions,objectives,n,K):
    G=model["G"];D=3*G;A=model["A"]
    folder.mkdir(parents=True,exist_ok=False)
    offset=np.full(D,-B/2);offset[2::3]=0
    coefficient=np.zeros((D,G));coefficient[np.arange(G)*3+2,np.arange(G)]=-1
    features=features[:n];costs=costs_from_gross(features)
    if not np.all(costs<0) or not np.allclose(costs,offset+features@coefficient.T,rtol=0,atol=1e-14):
        raise ValueError("Positive gross-profit signed affine map changed")
    path=folder/"training.npz"
    np.savez_compressed(path,A_data=A.data,A_indices=A.indices,A_indptr=A.indptr,A_shape=np.asarray(A.shape),
                        row_lower=model["row_lower"],row_upper=model["row_upper"],
                        col_lower=np.zeros(D),col_upper=np.ones(D),integrality=np.ones(D,dtype=np.int32),
                        commitment_indices=np.arange(D),binary_indices=np.arange(D),
                        train_features=features,train_costs=costs,train_solutions=solutions[:n],
                        train_objectives=objectives[:n],cost_offset=offset,cost_feature_coefficients=coefficient)
    meta=dict(case_id=cid,family="knapsack",problem_name="capacity bundles with incompatible positive-value upgrades",
              objective="maximize original positive gross project returns; solver minimizes their negatives",
              seed=seed,n_train=n,D=D,K=K,G=G,feature_dimension=G,
              global_optimizer_hull_K=G,training_only=True,development_costs_exported=False,
              training_npz_sha256=base.sha(path),
              cost_map_instruction="cost=cost_offset+cost_feature_coefficients@feature",
              metadata=dict(graph_family=family,edges=model["edges"],weights_A=model["weight_A"],
                            weights_B=model["weight_B"],weights_C=model["weight_C"],block_capacities=model["capacity"],
                            cost_offset=offset,cost_feature_coefficients=coefficient,n_categories=G,
                            base_AB_gross_return=B,upgrade_gross_value_distribution="independent Uniform[0.1,2.5]",
                            feature_name="upgrade gross value xi_g",xi_lower=LOW,xi_upper=HIGH,
                            premium_interval=[LOW-B,HIGH-B],
                            gross_totalprofit_formula="0.5*G+sum((xi_g-0.5)*C_g); real base-project return retained",
                            no_artificial_objective_shift=True,global_optimizer_hull_K=G,
                            actual_training_optimizer_hull_K=K,
                            global_hull_proof="All independent sets can be uniquely optimal; zero and every unit vector are feasible",
                            all_original_training_labels_enumerated_and_HiGHS_certified=True,
                            original_LP_has_no_added_optimizer_equalities=True))
    base.write_new(folder/"metadata.json",meta)
    return dict(path=str(folder),training_npz_sha256=base.sha(path),metadata_sha256=base.sha(folder/"metadata.json"))


def run_setting(out,family,seed):
    model=base.physical(family,seed);G=model["G"];D=3*G
    sets,masks=all_independent_sets(G,model["edges"])
    # These vectors certify exact full rank without a large floating-point SVD.
    global_K=G
    prefixes=[G+8,2*G+5];maximum=max(prefixes);dev_count=48
    train=base.rng(f"coupled-bundle-uniform-gross-training-v1|{family}|{seed}").uniform(LOW,HIGH,(maximum,G))
    dev=base.rng(f"coupled-bundle-uniform-gross-development-v1|{family}|{seed}").uniform(LOW,HIGH,(dev_count,G))
    train_x,train_v,train_selected=oracle(train,sets);dev_x,dev_v,dev_selected=oracle(dev,sets)
    costs=costs_from_gross(np.vstack([train,dev]));values=np.r_[train_v,dev_v]
    records=[]
    for i,(cost,expected) in enumerate(zip(costs,values)):
        ip,ip_x=base.solve_reference(model,cost,True,seed+i)
        lp,lp_x=base.solve_reference(model,cost,False,seed+i)
        if not ip["certified"] or not lp["certified"]:raise ValueError("Uncertified original reference")
        if abs(ip["objective"]-expected)>1e-7+1e-9*abs(expected):raise ValueError("All-IS oracle/IP reference differ")
        residual=max(float(np.max(np.abs(lp_x[0::3]+lp_x[2::3]-1),initial=0)),
                     float(np.max(np.abs(lp_x[1::3]+lp_x[2::3]-1),initial=0)))
        ipres=max(float(np.max(np.abs(ip_x[0::3]+ip_x[2::3]-1),initial=0)),
                  float(np.max(np.abs(ip_x[1::3]+ip_x[2::3]-1),initial=0)))
        if ipres>1e-6:raise ValueError("Integer optimum violates proved bundle relation")
        fractional=int(np.count_nonzero(np.abs(lp_x-np.rint(lp_x))>1e-7))
        records.append(dict(index=i,split="training" if i<maximum else "exploratory_development_reference",
                            split_index=i if i<maximum else i-maximum,
                            enumeration_objective=float(expected),reference_profit=float(-expected),
                            ip=ip,lp=lp,original_gross_objective_used=True,
                            enumeration_IP_absolute_error=abs(ip["objective"]-expected),
                            LP_fractional=bool(fractional),LP_fractional_coordinates=fractional,
                            LP_optimizer_relation_residual=residual,
                            LP_violates_optimizer_affine_relations=residual>1e-7,
                            LP_relative_integrality_gap=(expected-lp["objective"])/abs(expected),
                            IP_profit_over_LP_upper_bound=expected/lp["objective"],
                            real_gross_fallback_fraction=B*G/(-expected)))
    key=f"{family}_b0.5_seed{seed}"
    ref=out/"references"/key;ref.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(ref/"exploratory-development-reference.npz",features=dev,costs=costs_from_gross(dev),
                        solutions=dev_x,objectives=dev_v)
    base.write_new(ref/"raw-reference-checks.json",records)
    base.write_new(out/"global-geometry"/(key+".json"),dict(G=G,D=D,global_K=global_K,
                                                        all_independent_set_count=len(sets),masks=masks,
                                                        exact_rank_certificate="empty + G singleton independent sets",
                                                        edges=model["edges"]))
    development=records[maximum:];rows=[]
    for n in prefixes:
        K=base.exact_rank(train_x[:n]);cid=f"coupledbundle_uniformgross_{key}_n{n}"
        source=export(out/"training-candidates"/key/f"n{n}",cid,family,seed,model,train,train_x,train_v,n,K)
        rows.append(dict(case_id=cid,graph_family=family,physical_seed=seed,G=G,D=D,n_train=n,
                          base_return=B,upgrade_gross_value_distribution="Uniform[0.1,2.5]",
                          global_K=G,training_K=K,training_covers_global_hull=K==G,
                          original_rows=model["A"].shape[0],original_nonzeros=model["A"].nnz,
                          globally_reachable_optimizer_count=len(sets),
                          training_distinct_optimizers=len(set(map(int,train_selected[:n]))),
                          exploratory_development_count=dev_count,
                          development_unseen_optimizer_fraction=float(np.mean(~np.isin(dev_selected,train_selected[:n]))),
                          LP_fractional_fraction=float(np.mean([r["LP_fractional"] for r in development])),
                          LP_relation_violation_fraction=float(np.mean([r["LP_violates_optimizer_affine_relations"] for r in development])),
                          LP_relative_integrality_gap_mean=float(np.mean([r["LP_relative_integrality_gap"] for r in development])),
                          original_IP_presolved_vars_mean=float(np.mean([r["ip"]["presolved"]["variables"] for r in development])),
                          original_IP_presolved_vars_min=min(r["ip"]["presolved"]["variables"] for r in development),
                          original_IP_presolved_vars_max=max(r["ip"]["presolved"]["variables"] for r in development),
                          original_IP_presolve_empty_fraction=float(np.mean([r["ip"]["presolved"]["variables"]==0 for r in development])),
                          original_IP_nodes_mean=float(np.mean([r["ip"]["nodes"] for r in development])),
                          gross_fallback_fraction_mean=float(np.mean([r["real_gross_fallback_fraction"] for r in development])),
                          all_training_and_development_reference_IPs_certified=True,
                          reference_count=maximum+dev_count,source=source,
                          raw_reference_checks_sha256=base.sha(ref/"raw-reference-checks.json")))
    return rows


def main():
    out=ROOT/"screen-uniform-gross"
    if out.exists():raise FileExistsError("Immutable Uniform gross-value screen already exists")
    out.mkdir(parents=True)
    protocol=dict(stage="uniform-gross",created_utc=base.now(),families=FAMILIES,physical_seeds=SEEDS,
                  base_AB_gross_return=B,upgrade_gross_value_distribution="IID Uniform[0.1,2.5]",
                  premium_range=[-.4,2.0],prefix_rules=["G+8","2*G+5"],development_reference_count=48,
                  graph="connected cycle + triangle(0,1,2) + p0.15 independent extra edges; same fixed graph/weights as larger lognormal screen",
                  global_hull="Every independent set can be uniquely optimal by positive/negative premium signs; empty and singletons certify globalK=G",
                  all_original_item_profits_positive=True,
                  objective_ratio_rule="Original gross profit 0.5G+sum((xi-0.5)C); no base-return subtraction",
                  sampling="Independent hashed Uniform train/development streams; same law, nested prefixes, all three seeds retained",
                  source_sha256={str(Path(__file__)):base.sha(Path(__file__)),str(HELPER):base.sha(HELPER)},
                  HiGHS=highspy.Highs().version(),only_original_reference_solves=True,no_baseline_training=True,
                  no_old_or_fresh_bank_read=True,no_confirmation_created=True)
    base.write_new(out/"protocol.json",protocol)
    rows=[]
    for family in FAMILIES:
        for seed in SEEDS:
            new=run_setting(out,family,seed);rows+=new
            with (out/"progress.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(base.clean(dict(utc=base.now(),rows=new)),allow_nan=False)+"\n")
            (out/"summary-progress.json").write_text(json.dumps(base.clean(dict(status="PARTIAL",rows=rows)),indent=2,allow_nan=False)+"\n",encoding="utf-8")
            print(json.dumps(dict(event="uniform_gross_reference_complete",family=family,seed=seed,
                                  prefix_ranks=[r["training_K"] for r in new],global_K=new[0]["global_K"],
                                  reachable_optimizers=new[0]["globally_reachable_optimizer_count"])),flush=True)
    source_hashes={str(path.resolve()):base.sha(path) for path in sorted(out.rglob("*"))
                   if path.is_file() and path.name not in ["summary-progress.json","progress.jsonl"]}
    report=dict(status="COMPLETE",created_utc=base.now(),rows=rows,protocol_sha256=base.sha(out/"protocol.json"),
                no_baseline_training=True,no_algorithm_objective_ratio_advantage_claim=True,
                global_rank_proof="All independent sets, including empty/unit vectors; globalK=G",
                all_references_use_original_gross_profit=True,source_sha256=source_hashes)
    base.write_new(out/"summary.json",report)
    header=["# Uniform gross-value coupled-bundle reference screen","",
            "No learned baseline was run. All original profits are positive: base projects return0.25 each; upgrade gross values are IID Uniform[0.1,2.5]. Premiums relative to the real base bundle are allowed to be negative. Every independent set is reachable, so exact globalK=G, certified by empty and singleton assignments.","",
            "All costs are original gross-profit negatives; no base-return or artificial constant is subtracted. LP/MIP references use the original constraints. All fixed seeds and both prefixes are retained.","",
            "| Graph | seed | D | n | trainK/globalK | #global opt | #train opt | LP fractional (%) | LP gap (%) | Presolved IP vars mean | New dev opt (%) |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:header.append(f"| {r['graph_family']} | {r['physical_seed']} | {r['D']} | {r['n_train']} | {r['training_K']}/{r['global_K']} | {r['globally_reachable_optimizer_count']} | {r['training_distinct_optimizers']} | {100*r['LP_fractional_fraction']:.1f} | {100*r['LP_relative_integrality_gap_mean']:.2f} | {r['original_IP_presolved_vars_mean']:.1f} | {100*r['development_unseen_optimizer_fraction']:.1f} |")
    (out/"summary.md").write_text("\n".join(header)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="COMPLETE",rows=len(rows),summary_sha256=base.sha(out/"summary.json"))))


if __name__=="__main__":main()
