"""Matched-dimension adapters around the unchanged archived neural baselines.

No test fields are copied into training cases. Neural Diving trains only the
complete-assignment head used by its matched-K policy. Preference Learning uses
the original archived preference fitting procedure and the separately disclosed
strategy-guided coordinate-fixing adaptation.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
import torch
import neural_diving_baseline as nd
import preference_baseline as li
from portable_adapter import LinearModel, solve as default_solve
import adapter as vendor_adapter

IMPLEMENTATION = "asc-neural-matched-training-v2-no-debug-refit"


def _kind(kind):
    aliases = {"nd": "neural_diving", "li": "preference_learning",
               "neural_diving": "neural_diving", "preference_learning": "preference_learning"}
    if kind not in aliases:
        raise ValueError("kind must be neural_diving/nd or preference_learning/li")
    return aliases[kind]


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def training_case(case):
    """Accept an object or plain dictionary, retaining only training fields."""
    get = case.get if isinstance(case, dict) else lambda name, default=None: getattr(case, name, default)
    model = get("model")
    if isinstance(model, dict):
        model = LinearModel(**{name: model[name] for name in
            ("A", "row_lower", "row_upper", "col_lower", "col_upper", "integrality")})
    features = np.asarray(get("train_features"), dtype=np.float64)
    costs = np.asarray(get("train_costs"), dtype=np.float64)
    solutions = np.asarray(get("train_solutions"), dtype=np.float64)
    if costs.ndim != 2 or solutions.shape != costs.shape or features.shape[0] != len(costs):
        raise ValueError("Training arrays have inconsistent shapes")
    if not all(np.isfinite(a).all() for a in (features, costs, solutions)):
        raise ValueError("Training arrays must be finite")
    if np.any(np.asarray(model.integrality) == 0):
        raise ValueError("This supplement expects binary knapsack variables only")
    return SimpleNamespace(model=model, family="knapsack", case_id=str(get("case_id", "asc")),
        seed=int(get("seed", 41001)), train_features=features, train_costs=costs,
        train_solutions=solutions, affine=dict(get("affine")))


def _verify_vendor():
    manifest = json.loads((VENDOR / "source-manifest.json").read_text(encoding="utf-8-sig"))
    for member in manifest:
        if hashlib.sha256((VENDOR / member["name"]).read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError("Archived baseline source changed: " + member["name"])
    return {member["name"]: member["sha256"] for member in manifest}


def _constant_coordinate_diagnostics(prepared):
    """Training-only optimum for a cost-independent Bernoulli per coordinate."""
    marginals = []
    for pool in prepared:
        weights = pool["weights"].detach().cpu().numpy()
        labels = pool["labels"].detach().cpu().numpy()
        marginals.append(np.sum(weights[:, None] * labels, axis=0))
    probabilities = np.mean(marginals, axis=0)
    interior = (probabilities > 0) & (probabilities < 1)
    entropy = np.zeros_like(probabilities)
    p = probabilities[interior]
    entropy[interior] = -p * np.log(p) - (1-p) * np.log(1-p)
    return {"scope": "all supplied training costs only",
        "constant_per_coordinate_nll": float(np.mean(entropy)),
        "constant_per_coordinate_expected_bit_error": float(np.mean(np.minimum(probabilities, 1-probabilities))),
        "varying_label_coordinates": int(np.sum(interior)),
        "affects_hyperparameter_selection": False, "affects_readiness": False}


def _fit_nd(case, output, device, solver, overrides):
    cfg = nd.NeuralDivingConfig(seed=case.seed + 1909, device=device)
    cfg = replace(cfg, **overrides)
    nd._setup(cfg)
    started = perf_counter()
    costs = case.train_costs
    permutation = np.random.default_rng(cfg.seed).permutation(len(costs))
    count = max(1, int(round(len(costs) * cfg.validation_fraction)))
    validation = np.sort(permutation[:count])
    training = np.sort(permutation[count:])
    graph = nd.FixedBipartiteGraph(case.model, costs[training],
        cfg.normalize_constraint_rows, cfg.device, cfg.use_root_lp_features,
        cfg.root_lp_time_limit, cfg.seed)
    root_features = graph.precompute_training_root_lp(costs)
    pools = [{"train_index": i, "solutions": case.train_solutions[i:i+1],
              "objectives": np.asarray([costs[i] @ case.train_solutions[i]]),
              "collection_seconds": 0.0} for i in range(len(costs))]
    costs, prepared, pool_metadata = nd._prepare_pools(case, pools, graph.integer_indices)
    candidates, winner = [], None
    selection_started = perf_counter()
    for width in cfg.width_candidates:
        candidate_cfg = replace(cfg, hidden_size=int(width))
        state, epochs, history = nd._warmup_selection(graph, candidate_cfg, costs,
                                                      prepared, training, validation)
        if epochs < 1 or not np.isfinite(history["best_validation_full_nll"]):
            raise FloatingPointError("Complete-predictor selection produced no finite trained state")
        if not all(torch.isfinite(tensor).all().item() for tensor in state.values()):
            raise FloatingPointError("Selected complete-predictor weights are nonfinite")
        for row in history["epochs"]:
            if not np.isfinite(row["train_full_nll"]) or not np.isfinite(row["full_nll_per_integer"]):
                raise FloatingPointError("Complete-predictor training or validation loss is nonfinite")
        candidate = {"width": int(width), "eligible": True, "warmup_epochs": epochs,
                     "warmup_history": history}
        candidates.append(candidate)
        score = (history["best_validation_full_nll"], int(width))
        if winner is None or score < winner[0]:
            winner = score, candidate
    if winner is None:
        diagnostics = {"passed": False, "candidates": candidates,
                       "reason": "No finite complete-predictor candidate"}
        _json(output / "training-failure.json", diagnostics)
        raise nd.TrainingSanityError(diagnostics["reason"], diagnostics)
    selection_seconds = perf_counter() - selection_started
    chosen = winner[1]
    final_cfg = replace(cfg, hidden_size=chosen["width"])
    final_graph = nd.FixedBipartiteGraph(case.model, costs, cfg.normalize_constraint_rows,
        cfg.device, cfg.use_root_lp_features, cfg.root_lp_time_limit, cfg.seed)
    final_graph.training_lp_cache = dict(graph.training_lp_cache)
    indices = np.arange(len(costs), dtype=np.int64)
    seed = cfg.seed + 110003
    predictor = nd._new_network(final_graph, final_cfg, seed)
    optimizer = nd._optimizer(predictor, final_cfg)
    rng = np.random.default_rng(seed)
    losses = []
    refit_started = perf_counter()
    for epoch in range(1, chosen["warmup_epochs"] + 1):
        losses.append(nd._train_epoch(predictor, optimizer, costs, prepared, indices,
                                      rng, final_cfg))
        if epoch == 1 or epoch % cfg.log_every == 0:
            print(f"ND matched refit width={chosen['width']} epoch={epoch} nll={losses[-1]:.5f}", flush=True)
    predictor.eval()
    diagnostics = nd._full_diagnostics(predictor, costs, prepared, indices)
    if not all(np.isfinite(diagnostics[name]) for name in
               ("full_nll_per_integer", "marginal_entropy_floor", "excess_nll", "expected_map_bit_error")):
        raise FloatingPointError("Final complete-predictor diagnostics are nonfinite")
    if not all(torch.isfinite(parameter).all().item() for parameter in predictor.parameters()):
        raise FloatingPointError("Final complete-predictor weights are nonfinite")
    nd._sync(final_graph.device)
    refit_seconds = perf_counter() - refit_started
    metadata = {"method": "Neural Diving matched-dimension complete predictor",
        "implementation": IMPLEMENTATION, "paper": nd.PAPER_URL, "config": asdict(final_cfg),
        "training_seconds": perf_counter() - started,
        "epoch_selection_seconds": selection_seconds, "final_refit_seconds": refit_seconds,
        "training_device": device, "inference_device": "cpu", "threads": 1,
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "matrix_digest": final_graph.matrix_digest,
        "train_cost_digest": nd._array_digest(costs),
        "training_indices": training.tolist(), "validation_indices": validation.tolist(),
        "final_fit_indices": indices.tolist(), "training_count": len(costs),
        "common_coordinate_indices": nd._common_indices(case).tolist(),
        "root_lp_training": root_features, "selected_width": chosen["width"],
        "selected_warmup_epochs": chosen["warmup_epochs"], "width_candidates": candidates,
        "final_full_predictor_diagnostics": diagnostics, "final_full_refit_losses": losses,
        "sanity_gate": {"passed": True, "scope": "finite ordinary selection/refit losses and fitted parameters",
            "extra_tiny_subset_training_executed": False, "extra_debug_gradient_steps": 0,
            "strict_memorization_threshold_used": False},
        "constant_coordinate_training_diagnostics": _constant_coordinate_diagnostics(prepared),
        "selection_rule": "minimum training-heldout complete-assignment NLL, then narrower width",
        "adaptation": "Only complete predictor used by matched policy is trained. Native selective coverage heads and the additional local 1200-step-per-width debugging fit are omitted. Original graph, root-LP features, Eq11 loss, optimizer and ordinary warmup epoch grid are unchanged.",
        "debug_protocol_change": "The local v1 memorization criterion was not a Nair algorithm requirement. Its first-case finite loss reduction and 4/4 feasible/optimal integration checks are preserved separately as implementation-validation evidence, not repeated algorithm training cost. Final loss and cost-independent coordinate diagnostics are reported without a hard memorization gate.",
        "pool_adaptation": "One shared certified optimal binary solution per training cost; no extra feasible-solution pool generation.",
        "native_algorithm": False, "unused_native_coverage_heads_trained": False,
        "test_information_used": False, "learning_time_excludes_shared_training_optima": True,
        **pool_metadata}
    model = nd.NeuralDivingModel(final_graph, [], predictor, final_cfg, metadata).to("cpu")
    # Training-only root-feature cache is not needed by timed test inference.
    model.graph.training_lp_cache.clear()
    with (output / "nd-matched.pt").open("wb") as handle:
        torch.save({"format": "asc-nd-matched-only-v1", "config": asdict(final_cfg),
                    "metadata": metadata, "predictor_state": nd._state(predictor)}, handle)
    return model, metadata


def fit_neural(kind, case, output_dir, device="cpu", solve_fn=None, config_overrides=None):
    """Return (fitted_model, metadata); time covers preprocessing, selection, refit."""
    kind = _kind(kind)
    case = training_case(case)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hashes = _verify_vendor()
    torch.set_num_threads(1)
    started = perf_counter()
    with threadpool_limits(limits=1):
        if kind == "neural_diving":
            model, metadata = _fit_nd(case, output, device, solve_fn or default_solve,
                                      config_overrides or {})
        else:
            config = li.PreferenceConfig(seed=case.seed + 20261009, device=device)
            config = replace(config, **(config_overrides or {}))
            model = li.fit_case(case, output_dir=output, config=config)
            model.network.eval().cpu()
            metadata = dict(model.report)
            metadata.update({"implementation": IMPLEMENTATION,
                "matched_prediction": "Archived strategy-score weighted-agreement coordinate-fixing variant; original preference fitting and SetCover unchanged",
                "native_algorithm": False, "threads": 1,
                "learning_time_excludes_shared_training_optima": True})
    if device == "cuda":
        torch.cuda.synchronize()
    metadata["learning_seconds_wall"] = perf_counter() - started
    metadata["source_sha256"] = hashes
    metadata["parameter_count"] = sum(p.numel() for p in (
        model.full_predictor.parameters() if kind == "neural_diving" else model.network.parameters()))
    _json(output / "fit-metadata.json", metadata)
    return model, metadata


def load_neural(kind, case, output_dir):
    kind = _kind(kind)
    case = training_case(case)
    output = Path(output_dir)
    _verify_vendor()
    torch.set_num_threads(1)
    if kind == "preference_learning":
        return li.load_fitted(case, output)
    with (output / "nd-matched.pt").open("rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    if payload["format"] != "asc-nd-matched-only-v1":
        raise ValueError("Wrong matched predictor checkpoint format")
    cfg = replace(nd.NeuralDivingConfig(**payload["config"]), device="cpu")
    nd._setup(cfg)
    metadata = payload["metadata"]
    if not metadata.get("sanity_gate", {}).get("passed", False):
        raise ValueError("Checkpoint has not passed finite-fit readiness checks")
    if nd._array_digest(case.train_costs) != metadata["train_cost_digest"]:
        raise ValueError("Checkpoint training costs differ")
    graph = nd.FixedBipartiteGraph(case.model, case.train_costs, cfg.normalize_constraint_rows,
        "cpu", cfg.use_root_lp_features, cfg.root_lp_time_limit, cfg.seed)
    if graph.matrix_digest != metadata["matrix_digest"]:
        raise ValueError("Checkpoint original model differs")
    network = nd._new_network(graph, cfg, cfg.seed)
    network.load_state_dict(payload["predictor_state"])
    network.eval()
    return nd.NeuralDivingModel(graph, [], network, cfg, metadata)


def predict_native_preference(model, case, cost, features, total_time_limit=20.0):
    """Run unchanged Li Eq21 and report positive profit plus full return time.

    Eq21 minimizes normalized infeasibility among the predicted top-k and breaks
    ties by predicted reward order. It does not maximize their actual objective.
    A forward hook captures the same scores for later matched-coordinate
    confidence without performing a second network prediction. Solver wrapping
    records timings only and returns every original solver result unchanged.
    """
    started = perf_counter()
    case = training_case(case)
    torch.set_num_threads(1)
    model.network.eval().cpu()
    captured = []
    hook = model.network.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().cpu().numpy().copy()))
    original_solver = vendor_adapter.solve
    solver_records = []
    def timed_solver(*args, **kwargs):
        answer = original_solver(*args, **kwargs)
        solver_records.append({"seconds": float(answer.seconds),
                               "relax_integrality": bool(kwargs.get("relax_integrality", False))})
        return answer
    decoder_before = float(model.bank.decode_statistics.get("seconds", 0.0))
    try:
        vendor_adapter.solve = timed_solver
        with threadpool_limits(limits=1):
            raw = model.solve(np.asarray(cost, dtype=float), np.asarray(features, dtype=float),
                total_time_limit=max(0.0, total_time_limit - (perf_counter() - started)))
    finally:
        vendor_adapter.solve = original_solver
        hook.remove()
    if len(captured) != 1 or captured[0].shape != (1, len(model.selected)):
        raise RuntimeError("Native preference inference did not yield one complete strategy-score vector")
    elapsed = perf_counter() - started
    x = None if raw.get("x") is None else np.asarray(raw["x"], dtype=float)
    valid = bool(raw.get("valid", False) and x is not None and elapsed <= total_time_limit)
    objective = float(np.asarray(cost, dtype=float) @ x) if valid else None
    return {**raw, "x": None if x is None else x.tolist(),
        "solution": None if x is None else x.tolist(),
        "objective": None if objective is None else -objective,
        "original_minimization_objective": objective,
        "valid": valid, "seconds": elapsed, "total_seconds": elapsed,
        "prediction_seconds": elapsed,
        "solver_seconds": sum(record["seconds"] for record in solver_records),
        "lp_solver_seconds": sum(record["seconds"] for record in solver_records if record["relax_integrality"]),
        "ip_solver_seconds": sum(record["seconds"] for record in solver_records if not record["relax_integrality"]),
        "solver_call_count": len(solver_records),
        "decoder_seconds": float(model.bank.decode_statistics.get("seconds", 0.0)) - decoder_before,
        "all_retained_strategy_scores": captured[0][0].tolist(),
        "selection": "Li Eq21 minimum normalized infeasibility, stable predicted-reward-order ties",
        "K": 0, "K_scope": "Complete historical binary strategy; no free integer coordinate",
        "inference_device": "cpu", "test_optimum_used": False}


def _preference_native_winner_fixings(model, case, cost, features, K, total_time_limit):
    native = predict_native_preference(model, case, cost, features, total_time_limit)
    if not native["valid"] or native.get("strategy_id") is None:
        error = RuntimeError("Native preference decoder returned no timely feasible strategy; no matched-anchor fallback")
        error.diagnostics = native
        raise error
    selected = np.asarray(model.selected, dtype=int)
    scores = np.asarray(native["all_retained_strategy_scores"], dtype=float)
    positions = np.flatnonzero(selected == int(native["strategy_id"]))
    if len(positions) != 1 or scores.shape != selected.shape or not np.isfinite(scores).all():
        raise ValueError("Native preference winner or scores do not match the retained strategy bank")
    anchor_position = int(positions[0])
    integer_columns = np.asarray(model.bank.integer_indices, dtype=int)
    dimension = case.model.A.shape[1]
    if not np.array_equal(integer_columns, np.arange(dimension)):
        raise ValueError("Expected all original knapsack columns in strategy integer order")
    strategies = np.asarray([model.bank.strategies[int(i)].integer_values for i in selected], dtype=float)
    if strategies.shape != (len(selected), dimension) or not np.isin(strategies, (0, 1)).all():
        raise ValueError("Invalid retained binary strategy values")
    anchor = strategies[anchor_position].astype(int)
    witness = np.asarray(native["solution"], dtype=float)
    if not np.array_equal(witness, anchor):
        raise ValueError("Native decoded winner disagrees with its stored binary assignment")
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    confidence = np.sum(weights[:, None] * (strategies == anchor), axis=0)
    order = np.lexsort((np.arange(dimension), -confidence))
    chosen = order[:dimension - K]
    fixed = {int(column): int(anchor[column]) for column in chosen}
    return fixed, {"implementation_version": "preference-native-winner-matched-k-v2",
        "original_algorithm": False, "native_method_unchanged": True,
        "anchor_selection": native["selection"],
        "native_winner_strategy_id": int(native["strategy_id"]),
        "native_winner_retained_position": anchor_position,
        "highest_predicted_reward_strategy_id": int(selected[np.argsort(-scores, kind="stable")[0]]),
        "native_objective": native["objective"],
        "native_total_seconds": native["total_seconds"],
        "native_solver_seconds": native["solver_seconds"],
        "native_decoder_seconds": native["decoder_seconds"],
        "native_candidate_count": native["candidate_count"],
        "native_candidates": native["candidates"],
        "native_witness_solution": native["solution"],
        "all_retained_strategy_scores": scores.tolist(),
        "strategy_confidence_weights": weights.tolist(),
        "common_weighted_agreement": confidence.tolist(),
        "fixed_original_columns": chosen.tolist(),
        "coordinate_selection": "All-retained-score weighted agreement with the actual native winner; descending confidence then original column",
        "softmax_temperature": 1.0, "test_optimum_used": False,
        "full_solver_fallback": False}


def predict_fixed(kind, model, case, cost, features=None, K=None, total_time_limit=20.0):
    """Return (exact d-K integer fixings, prediction diagnostics).

    For ND a fresh unrestricted root LP is always included in elapsed time.
    Callers subtract prediction_seconds from their shared return budget before
    solving the original IP with only these coordinate fixings added.
    """
    started = perf_counter()
    kind = _kind(kind)
    case = training_case(case)
    target = case.affine["rank"] if K is None else K
    K = int(target)
    if K != target or not 0 <= K <= case.model.A.shape[1]:
        raise ValueError("K must be an integer in the original dimension range")
    case.affine = {"rank": K}
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if kind == "neural_diving":
            model.to("cpu")
            root = model.graph.begin_online_root_lp(np.asarray(cost, dtype=float),
                min(model.config.root_lp_time_limit, total_time_limit))
            try:
                proposal = model.matched_partial_assignment(case, np.asarray(cost, dtype=float), K)
            finally:
                model.graph.end_online_root_lp()
            fixed = proposal.pop("fixed")
            metadata = {**proposal, "root_lp": root, "root_lp_seconds": root.get("seconds", 0.0)}
        else:
            if features is None:
                raise ValueError("Preference prediction requires original cost features")
            fixed, metadata = _preference_native_winner_fixings(model, case,
                np.asarray(cost, dtype=float), np.asarray(features, dtype=float), K,
                max(0.0, total_time_limit - (perf_counter() - started)))
    metadata["prediction_seconds"] = perf_counter() - started
    metadata["inference_device"] = "cpu"
    metadata["target_K"] = K
    if len(fixed) != case.model.A.shape[1] - K:
        raise AssertionError("Neural baseline did not match target coordinate dimension")
    return fixed, metadata
