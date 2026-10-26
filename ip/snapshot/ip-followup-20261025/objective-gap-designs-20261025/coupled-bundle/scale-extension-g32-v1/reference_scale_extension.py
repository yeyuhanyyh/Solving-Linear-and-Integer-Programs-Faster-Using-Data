"""Predeclared G32, n69 reference-only scale extension of Uniform gross law.

Only new files in this directory are written. All three seeds are retained.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import highspy

HERE = Path(__file__).resolve().parent
PRODUCER = HERE.parent / "reference_screen_uniform.py"
spec=importlib.util.spec_from_file_location("sealed_uniform_gross_producer",PRODUCER)
uniform=importlib.util.module_from_spec(spec);spec.loader.exec_module(uniform)
base=uniform.base
G,N,DEV=32,69,48
SEEDS=[1009,1013,1019]
MAX_SETS=250000
MAX_ENUM_SECONDS=30.


def enumerate_bounded(edges):
    neighbors=[0]*G
    for i,j in edges:neighbors[i]|=1<<j;neighbors[j]|=1<<i
    masks=[];started=perf_counter();calls=0
    def visit(chosen,available):
        nonlocal calls
        calls+=1
        if calls%2048==0 and perf_counter()-started>MAX_ENUM_SECONDS:
            raise RuntimeError("all-independent-set enumeration exceeded predeclared 30 seconds")
        if not available:
            masks.append(chosen)
            if len(masks)>MAX_SETS:raise RuntimeError("all-independent-set count exceeded predeclared 250000 threshold")
            return
        bit=available&-available;i=bit.bit_length()-1
        visit(chosen,available&~bit)
        visit(chosen|bit,available&~(bit|neighbors[i]))
    try:visit(0,(1<<G)-1)
    except RuntimeError as error:
        return None,None,dict(available=False,reason=str(error),visited_calls=calls,
                              completed_sets_lower_bound=len(masks),wall_seconds=perf_counter()-started)
    masks.sort()
    assert len(masks)==len(set(masks)) and 0 in masks and all(1<<j in masks for j in range(G))
    sets=np.asarray([[(mask>>j)&1 for j in range(G)] for mask in masks],dtype=np.int64)
    return sets,masks,dict(available=True,count=len(masks),wall_seconds=perf_counter()-started,
                          exact_global_K=G,exact_certificate="empty and all singleton independent sets")


def timed(model,cost,integer,seed):
    wall=perf_counter();cpu=process_time()
    result,x=base.solve_reference(model,cost,integer,seed)
    result["outer_wall_seconds"]=perf_counter()-wall
    result["outer_cpu_seconds"]=process_time()-cpu
    result["timer_scope"]="cold original-model build, presolve, solve and validation; may overlap unrelated baseline training"
    return result,x


def main():
    out=HERE/"run-v1"
    if out.exists():raise FileExistsError("Immutable G32 scale extension already exists")
    out.mkdir()
    protocol=dict(created_utc=base.now(),stage="G32 reference-only scale extension",G=G,n_train=N,
        development_reference_count=DEV,physical_seeds=SEEDS,
        sampling="Same Uniform gross producer hashed train/development streams, same graph/weight generator, no graph/seed replacement",
        law="All A/B values0.25; each C gross value IID Uniform[0.1,2.5]",
        local_model="one original capacity row per three-item bundle; global graph conflicts only on C",
        graph="connected cycle plus triangle(0,1,2), independent extra edges p0.15",
        exact_global_K=G,training_rank_not_assumed=True,
        objective_metric="Unshifted original positive gross profit; no real base return subtraction",
        resources=dict(maximum_all_independent_sets=MAX_SETS,maximum_enumeration_seconds=MAX_ENUM_SECONDS,
                       reference_solver_seconds_per_IP_or_LP=30,HiGHS_threads=1),
        resource_failure_retention="All failures retained; no replacement graph, seed or law",
        predeclared_without_neural_outcomes=True,no_neural_training=True,no_neural_forward=True,
        no_old_or_fresh_bank_read=True,no_confirmation_created=True,
        source_sha256={str(Path(__file__).resolve()):base.sha(__file__),str(PRODUCER):base.sha(PRODUCER),
                       str(base.ROOT/"reference_screen.py"):base.sha(base.ROOT/"reference_screen.py")},HiGHS=highspy.Highs().version())
    base.write_new(out/"protocol.json",protocol)
    results=[]
    for seed in SEEDS:
        family="random32";model=base.physical(family,seed)
        sets,masks,availability=enumerate_bounded(model["edges"])
        row=dict(seed=seed,G=G,D=3*G,n_train=N,exact_global_K=G,
                 original_rows=model["A"].shape[0],original_nonzeros=model["A"].nnz,
                 enumeration=availability)
        if not availability["available"]:
            results.append(row);base.write_new(out/f"seed{seed}-availability.json",row);continue
        train=base.rng(f"coupled-bundle-uniform-gross-training-v1|{family}|{seed}").uniform(.1,2.5,(N,G))
        dev=base.rng(f"coupled-bundle-uniform-gross-development-v1|{family}|{seed}").uniform(.1,2.5,(DEV,G))
        tx,tv,ti=uniform.oracle(train,sets);dx,dv,di=uniform.oracle(dev,sets)
        # C coordinates parameterize the full optimizer hull injectively.
        K=base.exact_rank(tx[:,2::3])
        records=[]
        for i,(cost,expected) in enumerate(zip(uniform.costs_from_gross(np.vstack([train,dev])),np.r_[tv,dv])):
            ip,x=timed(model,cost,True,seed+i)
            lp,lx=timed(model,cost,False,seed+i)
            if not ip["certified"] or not lp["certified"]:raise ValueError("Uncertified scale reference")
            if abs(ip["objective"]-expected)>1e-7+1e-9*abs(expected):raise ValueError("Full IP disagrees with all-IS enumeration")
            residual=max(float(np.max(np.abs(lx[0::3]+lx[2::3]-1))),float(np.max(np.abs(lx[1::3]+lx[2::3]-1))))
            records.append(dict(index=i,split="training" if i<N else "exploratory_development_reference",
                split_index=i if i<N else i-N,enumeration_objective=float(expected),original_profit=-float(expected),
                ip=ip,lp=lp,LP_fractional=bool(np.any(np.abs(lx-np.rint(lx))>1e-7)),
                LP_optimizer_relation_residual=residual,LP_relative_integrality_gap=(expected-lp["objective"])/abs(expected)))
        cid=f"coupledbundle_uniformgross_random32_b0.5_seed{seed}_n69"
        folder=out/"training-candidates"/f"random32_b0.5_seed{seed}"/"n69"
        source=uniform.export(folder,cid,family,seed,model,train,tx,tv,N,K)
        ref=out/"references"/f"random32_b0.5_seed{seed}"
        ref.mkdir(parents=True)
        np.savez_compressed(ref/"exploratory-development-reference.npz",features=dev,costs=uniform.costs_from_gross(dev),solutions=dx,objectives=dv)
        base.write_new(ref/"raw-reference-checks.json",records)
        base.write_new(out/"global-geometry"/f"seed{seed}.json",dict(G=G,all_independent_set_count=len(sets),masks=masks,edges=model["edges"],global_K=G))
        devr=records[N:]
        row.update(case_id=cid,training_K=K,training_covers_global_hull=K==G,
            training_distinct_optimizers=len(set(map(int,ti))),
            development_unseen_optimizer_fraction=float(np.mean(~np.isin(di,ti))),
            LP_fractional_fraction=float(np.mean([r["LP_fractional"] for r in devr])),
            LP_relative_integrality_gap_mean=float(np.mean([r["LP_relative_integrality_gap"] for r in devr])),
            original_IP_presolved_vars_mean=float(np.mean([r["ip"]["presolved"]["variables"] for r in devr])),
            original_IP_nodes_mean=float(np.mean([r["ip"]["nodes"] for r in devr])),
            original_IP_reference_wall_seconds_mean=float(np.mean([r["ip"]["outer_wall_seconds"] for r in devr])),
            source=source,raw_reference_sha256=base.sha(ref/"raw-reference-checks.json"))
        results.append(row)
        base.write_new(out/f"seed{seed}-complete.json",row)
        print(json.dumps(dict(event="G32_reference_complete",seed=seed,K=K,global_K=G,all_independent_sets=len(sets))),flush=True)
    hashes={str(p.resolve()):base.sha(p) for p in out.rglob("*") if p.is_file()}
    summary=dict(status="COMPLETE",created_utc=base.now(),rows=results,protocol_sha256=base.sha(out/"protocol.json"),
                 source_sha256=hashes,no_neural_performance_claim=True,timing_not_isolated=True)
    base.write_new(out/"summary.json",summary)
    lines=["# Predeclared G32 reference-only extension","",
           "All three fixed physical seeds are retained. No neural method was trained or evaluated. Timings are raw concurrent reference measurements, not speedup evidence.","",
           "| Seed | allIS | trainK/globalK | LP fractional (%) | LP gap (%) | presolved vars | full wall mean (s) |",
           "|---|---:|---|---:|---:|---:|---:|"]
    for r in results:
        if not r["enumeration"]["available"]:lines.append(f"| {r['seed']} | resource limit | unavailable/32 | | | | |")
        else:lines.append(f"| {r['seed']} | {r['enumeration']['count']} | {r['training_K']}/32 | {100*r['LP_fractional_fraction']:.1f} | {100*r['LP_relative_integrality_gap_mean']:.2f} | {r['original_IP_presolved_vars_mean']:.1f} | {r['original_IP_reference_wall_seconds_mean']:.4f} |")
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="COMPLETE",summary=str(out/"summary.json"),sha256=base.sha(out/"summary.json"))))


if __name__=="__main__":main()
