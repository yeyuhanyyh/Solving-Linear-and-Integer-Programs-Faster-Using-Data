"""Lightweight synthetic interface checks; no learned case or MIP solve."""
from __future__ import annotations
import itertools
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
import neural_diving_baseline as nd
from dimension_diagnostics import assert_matched_fixings


def self_test() -> dict:
    torch.set_num_threads(1)
    torch.sparse.check_sparse_tensor_invariants.enable()
    # A one-period, two-generator miniature has the same four variable blocks
    # [u,v,w,p] and the same coupling path via generation and demand balance.
    A = sparse.csr_matrix(np.array([
        [1,0,-1,0,1,0,0,0], [0,1,0,-1,0,1,0,0],
        [-1,0,0,0,0,0,1,0], [0,-1,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,1]], dtype=float))
    model = SimpleNamespace(A=A, Acsr=A, row_lower=np.array([0,0,-np.inf,-np.inf,1.]),
        row_upper=np.array([0,0,0,0,1.]), col_lower=np.zeros(8), col_upper=np.ones(8),
        integrality=np.r_[np.ones(6),np.zeros(2)])
    costs = np.array([[1,2,.1,.2,.1,.1,3,4], [2,1,.2,.1,.1,.1,4,3]],dtype=float)
    labels = np.array([[1,0,1,0,0,0,1,0], [0,1,0,1,0,0,0,1]],dtype=float)
    case = SimpleNamespace(model=model, case_id="synthetic", family="uc",
                           commitment_indices=np.arange(2), train_costs=costs,
                           train_solutions=labels, affine={"rank":1})
    cfg = replace(nd.NeuralDivingConfig(), hidden_size=8, width_candidates=(8,),
                  epochs=1, warmup_epochs=1, sanity_steps=1, use_root_lp_features=False,
                  normalize_constraint_rows=False)
    graph = nd.FixedBipartiteGraph(model,costs)
    adjacency = graph.adjacency.to_dense().numpy()
    topology = sparse.csr_matrix((adjacency!=0).astype(int))
    hops = shortest_path(topology, directed=False, unweighted=True)
    assert hops[0,7] == 4 and hops[0,1] == 6
    assert np.array_equal(adjacency[:8,8:], A.toarray().T)
    assert torch.allclose(graph.aggregate(torch.ones(2,13,3)),
                          torch.tensor(adjacency).sum(dim=1)[None,:,None].expand(2,13,3))
    network = nd._new_network(graph,cfg,71)
    predictor = nd._new_network(graph,cfg,73)
    p,q = network(costs)
    assert p.shape == q.shape == (2,6)

    # Weighted full-label BCE at the exact marginals equals the entropy floor,
    # with degenerate coordinates handled at their limiting logits.
    pool = [{"labels":torch.tensor([[0.,1.,0.],[1.,1.,0.]]),
             "weights":torch.tensor([.75,.25],dtype=torch.float64)}]
    exact_logits = torch.tensor([[np.log(.25/.75),40.,-40.]],dtype=torch.float64)
    nll = float(nd._full_loss(exact_logits,np.array([0]),pool))
    floor = nd._entropy_floor(np.array([0]),pool)
    assert abs(nll-floor) < 1e-12
    perturbed = exact_logits.clone(); perturbed[0,0] = 0
    assert float(nd._full_loss(perturbed,np.array([0]),pool)) > floor+.01
    loss = nd._selective_loss(exact_logits,torch.zeros_like(exact_logits),
                              np.array([0]),pool,.75,32)
    assert abs(float(loss) - (float(nd._full_loss(exact_logits,np.array([0]),pool))+2.)) < 1e-10

    # Actual original-feasibility checking is retained even for the fake oracle.
    # Enumerate only this tiny synthetic model, not a solver/real benchmark.
    calls=[]
    feasible=[]
    for bits in itertools.product([0,1],repeat=6):
        for dispatch in ([1.,0.],[0.,1.]):
            x=np.r_[bits,dispatch].astype(float)
            activity=A@x
            if np.all(activity>=model.row_lower-1e-9) and np.all(activity<=model.row_upper+1e-9):
                feasible.append(x)
    def fake_solve(m,c,fixed,seed,time_limit):
        calls.append({"fixed":dict(fixed),"cap":time_limit})
        candidates=[x for x in feasible if all(x[j]==v for j,v in fixed.items())]
        x=min(candidates,key=lambda t:float(c@t)) if candidates else None
        return SimpleNamespace(valid=x is not None,x=x,certified=x is not None,seconds=0.)
    metadata={"sanity_gate":{"passed":True},"matrix_digest":graph.matrix_digest,
              "train_cost_digest":nd._array_digest(costs),"final_fit_indices":[0,1],
              "common_coordinate_indices":[0,1]}
    networks=[nd._new_network(graph,cfg,80+i) for i in range(len(cfg.coverage_targets))]
    learner=nd.NeuralDivingModel(graph,networks,predictor,cfg,metadata)
    first=learner.partial_assignments(costs[0],91)
    assert first==learner.partial_assignments(costs[0],91)
    for K in (0,1,2):
        partial=learner.matched_partial_assignment(case,costs[0],K)
        metrics=assert_matched_fixings(case,partial["fixed"],K)
        assert metrics["common_nominal_free_dimension"]==K
        assert all(j<2 for j in partial["fixed"])
    result=nd.solve_cost(learner,case,costs[0],91,30,fake_solve,"matched",1)
    assert len(result["submip_records"])==1 and result["remaining_scope_count"]==1
    assert all(j<2 for j in result["submip_records"][0]["fixed"])
    if result["valid"]:
        assert all(abs(result["x"][j]-v)<1e-9 for j,v in result["best_fixed"].items())
        assert not result["globally_certified"]
    native=nd.solve_cost(learner,case,costs[0],91,30,fake_solve)
    assert len(native["submip_records"])==5
    assert all(call["cap"] <= 30 for call in calls)
    try:
        nd.solve_cost(learner,case,costs[0],91,30,fake_solve,"matched",1.5)
        raise AssertionError("fractional K silently accepted")
    except ValueError:
        pass

    # Continuous-label energy completion is an input contract, not assumed
    # from any incumbent pool emitted by a MIP callback.
    pools=[{"train_index":i,"solutions":labels[i:i+1],
            "objectives":np.array([costs[i]@labels[i]])} for i in range(2)]
    try:
        nd._prepare_pools(case,pools,graph.integer_indices)
        raise AssertionError("uncertified UC pool accepted")
    except ValueError as error:
        assert "LP-optimal" in str(error)
    for item in pools:
        item["conditional_lp_completion_certified"]=True
    _,prepared,poolmeta=nd._prepare_pools(case,pools,graph.integer_indices)
    assert poolmeta["conditional_lp_completion_all_certified"]
    assert max(abs(nd._entropy_floor(np.array([i]),prepared)) for i in range(2))==0.
    with tempfile.TemporaryDirectory(prefix=".nd-v2-interface-", dir=Path(__file__).parent) as tmp:
        path=Path(tmp)/"model.pt"
        learner.save(path)
        restored=nd.load(case,path)
        assert restored.partial_assignments(costs[0],91)==first
        assert restored.matched_partial_assignment(case,costs[0],1)==learner.matched_partial_assignment(case,costs[0],1)
        restored.to("cpu")
        assert restored.graph.device.type=="cpu"
        # Uniform positive row scaling changes graph inputs and preserves every
        # original interval constraint. This uses no LP/MIP/training call.
        factors=np.array([10.,.25,25.,2.,100.])
        original_A=(sparse.diags(factors)@A).tocsr()
        scaled_model=SimpleNamespace(A=original_A,Acsr=original_A,
            row_lower=factors*model.row_lower,row_upper=factors*model.row_upper,
            col_lower=model.col_lower.copy(),col_upper=model.col_upper.copy(),
            integrality=model.integrality.copy())
        rowscale=np.maximum(np.asarray(abs(original_A).max(axis=1).toarray()).ravel(),1.)
        normalized_A=sparse.diags(1./rowscale)@original_A
        normalized_graph=nd.FixedBipartiteGraph(scaled_model,costs,normalize_rows=True)
        normalized_adjacency=normalized_graph.adjacency.to_dense().numpy()
        assert np.allclose(normalized_adjacency[:8,8:],normalized_A.toarray().T)
        assert np.allclose(normalized_adjacency[8:,:8],normalized_A.toarray())
        for x in np.vstack((labels,np.random.default_rng(731).uniform(-1.,2.,size=(10,8)))):
            activity=original_A@x
            activity_scaled=normalized_A@x
            assert np.array_equal(activity>=scaled_model.row_lower,
                                  activity_scaled>=scaled_model.row_lower/rowscale)
            assert np.array_equal(activity<=scaled_model.row_upper,
                                  activity_scaled<=scaled_model.row_upper/rowscale)
        normalized_case=SimpleNamespace(**{**vars(case),"model":scaled_model})
        normalized_cfg=replace(cfg,normalize_constraint_rows=True)
        normalized_predictor=nd._new_network(normalized_graph,normalized_cfg,274)
        normalized_networks=[nd._new_network(normalized_graph,normalized_cfg,280+i) for i in range(5)]
        normalized_metadata={**metadata,"matrix_digest":normalized_graph.matrix_digest}
        normalized_learner=nd.NeuralDivingModel(normalized_graph,normalized_networks,
                                               normalized_predictor,normalized_cfg,normalized_metadata)
        normalized_path=Path(tmp)/"normalized-model.pt"
        normalized_learner.save(normalized_path)
        normalized_restored=nd.load(normalized_case,normalized_path)
        assert normalized_restored.config.normalize_constraint_rows
        assert torch.allclose(normalized_restored.full_predictor(costs)[0],
                              normalized_predictor(costs)[0],atol=1e-6)
        assert np.array_equal(scaled_model.A.toarray(),original_A.toarray())
    # Training features can be cached, while every timed native/matched call
    # independently pays for fresh LP extraction and clears temporary data.
    # This provider is a counter-only mock; genuine certificates are checked
    # separately in test_nd_root_lp_features.py.
    class RecordingLP:
        def __init__(self):
            self.calls=0
            self.fail=False
        def extract(self,cost,time_limit=None):
            self.calls+=1
            if self.fail:
                from nd_root_lp_features import RootLPFeatureError
                raise RootLPFeatureError("synthetic failure",{"seconds":0.,"status":"mock_failed"})
            return np.zeros((13,15),dtype=np.float32),{"seconds":0.,"certified":True}
    lpgraph=nd.FixedBipartiteGraph(model,costs,use_root_lp_features=True)
    recorder=RecordingLP()
    lpgraph.root_lp_extractor=recorder
    cached=lpgraph.precompute_training_root_lp(costs)
    assert cached["count"]==2 and recorder.calls==2
    lpcfg=replace(cfg,use_root_lp_features=True)
    lpnetworks=[nd._new_network(lpgraph,lpcfg,180+i) for i in range(5)]
    lppredictor=nd._new_network(lpgraph,lpcfg,173)
    lplearner=nd.NeuralDivingModel(lpgraph,lpnetworks,lppredictor,lpcfg,metadata)
    lpnetworks[0](costs)
    assert recorder.calls==2
    for mode in ("native","matched","native"):
        nd.solve_cost(lplearner,case,costs[0],191,30,fake_solve,mode,K=1 if mode=="matched" else None)
        assert lpgraph.active_root_lp is None
    assert recorder.calls==5 and len(lpgraph.training_lp_cache)==2
    count_before=len(calls)
    recorder.fail=True
    failed=nd.solve_cost(lplearner,case,costs[0],191,30,fake_solve)
    assert not failed["valid"] and failed["status"]=="root_lp_feature_failure"
    assert failed["best_fixed"] is None and len(calls)==count_before
    assert lpgraph.active_root_lp is None
    return {"passed":True,"training_calls":0,"real_solver_calls":0,
            "fake_oracle_calls":len(calls),
            "checks":["six-hop cross-generator price path and raw coefficient adjacency",
                      "uniform graph row-max scaling preserves interval constraints and checkpoint predictions",
                      "weighted full-NLL entropy identity and coverage penalty",
                      "native two-Bernoulli replay and exact matched common-coordinate counts",
                      "winning restriction, local certificate scope, shared subproblem budget",
                      "mandatory certified UC conditional LP completion",
                      "CPU save/load complete predictor and coverage networks",
                      "training-only LP cache; fresh per-call LP; one shared native LP; failure has no fallback"],
            "config_defaults":asdict(nd.NeuralDivingConfig())}


if __name__ == "__main__":
    print(json.dumps(self_test(),ensure_ascii=False,indent=2))

