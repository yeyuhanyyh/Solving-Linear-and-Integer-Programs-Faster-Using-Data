"""Training repair around the frozen, manuscript-matching Neural Diving GCN.

The graph, features, loss, MAP fixing rule, widths and base warmup are imported
unchanged from the archived source. Only train-heldout selection and a bounded
continuation schedule change. No test arrays are retained by the training API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


IMPLEMENTATION = "nd-repair-v1-heldout-containment-longer-training"
CHECKPOINT_NAME = "nd-repaired.pt"
METADATA_NAME = "fit-metadata.json"


@dataclass(frozen=True)
class RepairConfig:
    first_steps: int = 1000
    second_steps: int = 1000
    checkpoint_every: int = 100
    first_learning_rate: float = 0.001
    second_learning_rate: float = 0.0001
    validation_cost_limit: int = 8

    def validate(self) -> None:
        if min(self.first_steps, self.second_steps) < 0:
            raise ValueError("Continuation steps must be nonnegative")
        if self.checkpoint_every <= 0 or self.validation_cost_limit <= 0:
            raise ValueError("Checkpoint interval and validation limit must be positive")
        if min(self.first_learning_rate, self.second_learning_rate) <= 0:
            raise ValueError("Learning rates must be positive")

    @property
    def total_steps(self) -> int:
        return self.first_steps + self.second_steps


def _archived(base_source: str | Path):
    base = Path(base_source).resolve()
    module_path = base / "neural" / "baselines.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    name = "_nd_repair_archived_" + hashlib.sha256(str(base).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    expected_vendor = (base / "neural" / "vendor").resolve()
    if Path(module.nd.__file__).resolve().parent != expected_vendor:
        raise RuntimeError("An incompatible archived Neural Diving module is already imported")
    return module


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _finite_network(network, torch) -> None:
    if not all(torch.isfinite(p).all().item() for p in network.parameters()):
        raise FloatingPointError("Neural Diving has nonfinite fitted parameters")


def _diagnostics(nd, network, costs, prepared, indices):
    import numpy as np
    result = nd._full_diagnostics(network, costs, prepared, indices)
    for key in ("full_nll_per_integer", "marginal_entropy_floor", "excess_nll",
                "expected_map_bit_error"):
        if not np.isfinite(result[key]):
            raise FloatingPointError("Nonfinite diagnostic: " + key)
    return result


def _validation_metrics(nd, network, case, costs, prepared, validation,
                        containment_indices, K: int):
    """Check containment of the supplied certified validation optimum.

    This does not solve a validation IP. With multiple optima, excluding this
    particular certified point need not exclude every optimum. Its meaning is
    recorded explicitly rather than calling this exact recovery.
    """
    import numpy as np
    import torch
    network.eval()
    with torch.no_grad():
        logits, _ = network(costs[containment_indices])
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    rows, contained = [], 0
    for local, index in enumerate(containment_indices):
        fixed = nd._map_fixed(case, probabilities[local], target_K=K)
        witness = np.rint(case.train_solutions[int(index)]).astype(np.int64)
        ok = all(witness[column] == value for column, value in fixed.items())
        contained += int(ok)
        rows.append({"train_index": int(index), "contained": bool(ok),
                     "fixed_count": len(fixed),
                     "fixed": {str(j): int(v) for j, v in fixed.items()}})
    return {"optimizer_contained_count": contained,
            "optimizer_containment_count": len(containment_indices),
            "optimizer_containment_fraction": contained / len(containment_indices),
            "optimizer_containment_rows": rows,
            "validation_diagnostics": _diagnostics(nd, network, costs, prepared, validation)}


def _candidate_key(record):
    return (-record["optimizer_contained_count"],
            record["validation_diagnostics"]["full_nll_per_integer"],
            record["continuation_steps"], record["width"])


def _continuation_step(nd, network, optimizer, costs, prepared, indices, rng,
                       config, learning_rate: float):
    import numpy as np
    import torch
    # Independent training-only mini-batches. Every optimizer call is one step.
    selected = np.asarray(rng.choice(indices, size=min(config.batch_size, len(indices)),
                                     replace=False), dtype=np.int64)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    network.train()
    prediction, _ = network(costs[selected])
    loss = nd._full_loss(prediction, selected, prepared)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite continuation loss")
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip))
    if not np.isfinite(norm):
        raise FloatingPointError("Nonfinite continuation gradient")
    optimizer.step()
    network.eval()
    return {"loss": float(loss.detach()), "gradient_norm_before_clip": norm,
            "learning_rate": float(learning_rate)}


def _checkpoint_steps(repair: RepairConfig, limit: int | None = None):
    count = repair.total_steps if limit is None else int(limit)
    steps = set(range(0, count + 1, repair.checkpoint_every))
    steps.add(count)
    if 0 <= repair.first_steps <= count:
        steps.add(repair.first_steps)
    return steps


def fit(case, output_dir: str | Path, base_source: str | Path,
        device: str = "cpu", config_overrides: dict | None = None,
        repair_overrides: dict | None = None):
    """Fit using only supplied training arrays; return (model, metadata).

    Overrides exist for bounded interface checks. Formal study callers leave
    them empty; their complete values are recorded if supplied.
    Learning wall time includes feature preparation, every selection fit,
    validation checkpoint, final refit, model construction and weight export.
    """
    started = perf_counter()
    b = _archived(base_source)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    nd = b.nd
    source_hashes = b._verify_vendor()
    clean = b.training_case(case)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / CHECKPOINT_NAME).exists() or (output / METADATA_NAME).exists():
        raise FileExistsError("Refusing to replace a previous repaired fit")
    cfg = replace(nd.NeuralDivingConfig(seed=clean.seed + 1909, device=device),
                  **(config_overrides or {}))
    repair = replace(RepairConfig(), **(repair_overrides or {}))
    repair.validate()
    nd._setup(cfg)
    K = int(clean.affine["rank"])
    if K != clean.affine["rank"] or not 0 <= K <= clean.model.A.shape[1]:
        raise ValueError("Invalid shared training-derived target K")
    costs = clean.train_costs
    if len(costs) < 3:
        raise ValueError("At least three training costs are required")
    permutation = np.random.default_rng(cfg.seed).permutation(len(costs))
    count = max(1, int(round(len(costs) * cfg.validation_fraction)))
    validation = np.sort(permutation[:count])
    training = np.sort(permutation[count:])
    # Match manuscript validation-cost cap without drawing new data.
    containment_indices = validation[:min(len(validation), repair.validation_cost_limit)]
    with threadpool_limits(limits=1):
        graph = nd.FixedBipartiteGraph(clean.model, costs[training],
            cfg.normalize_constraint_rows, cfg.device, cfg.use_root_lp_features,
            cfg.root_lp_time_limit, cfg.seed)
        root_features = graph.precompute_training_root_lp(costs)
        pools = [{"train_index": i, "solutions": clean.train_solutions[i:i+1],
                  "objectives": np.asarray([costs[i] @ clean.train_solutions[i]]),
                  "collection_seconds": 0.0} for i in range(len(costs))]
        costs, prepared, pool_metadata = nd._prepare_pools(clean, pools, graph.integer_indices)
        candidates, winner = [], None
        selection_started = perf_counter()
        checkpoints = _checkpoint_steps(repair)
        for width in cfg.width_candidates:
            candidate_cfg = replace(cfg, hidden_size=int(width))
            warm_state, warm_epochs, warm_history = nd._warmup_selection(
                graph, candidate_cfg, costs, prepared, training, validation)
            if warm_state is None or warm_epochs < 1:
                raise FloatingPointError("No finite base warmup checkpoint")
            network = nd._new_network(graph, candidate_cfg, cfg.seed + 110003)
            network.load_state_dict(warm_state)
            _finite_network(network, torch)
            optimizer = nd._optimizer(network, candidate_cfg)  # reset at continuation start
            rng = np.random.default_rng(cfg.seed + 510007)
            ledger = []
            window_losses = []
            for step in range(repair.total_steps + 1):
                last = None
                if step:
                    rate = (repair.first_learning_rate if step <= repair.first_steps
                            else repair.second_learning_rate)
                    last = _continuation_step(nd, network, optimizer, costs, prepared,
                                              training, rng, candidate_cfg, rate)
                    window_losses.append(last["loss"])
                if step not in checkpoints:
                    continue
                record = {"width": int(width), "base_warmup_epochs": int(warm_epochs),
                          "continuation_steps": step,
                          "last_training_update": last,
                          "mean_training_loss_since_checkpoint":
                              float(np.mean(window_losses)) if window_losses else None,
                          **_validation_metrics(nd, network, clean, costs, prepared,
                                                validation, containment_indices, K)}
                window_losses = []
                _finite_network(network, torch)
                ledger.append(record)
                if winner is None or _candidate_key(record) < _candidate_key(winner):
                    winner = dict(record)
                print(f"ND repair width={width} extra_step={step} "
                      f"val_contained={record['optimizer_contained_count']}/"
                      f"{len(containment_indices)} "
                      f"val_nll={record['validation_diagnostics']['full_nll_per_integer']:.6f}",
                      flush=True)
            candidates.append({"width": int(width), "warmup_history": warm_history,
                               "continuation_checkpoints": ledger})
        nd._sync(graph.device)
        selection_seconds = perf_counter() - selection_started
        if winner is None:
            raise RuntimeError("No selected Neural Diving configuration")
        chosen_cfg = replace(cfg, hidden_size=winner["width"])
        final_graph = nd.FixedBipartiteGraph(clean.model, costs, cfg.normalize_constraint_rows,
            cfg.device, cfg.use_root_lp_features, cfg.root_lp_time_limit, cfg.seed)
        final_graph.training_lp_cache = dict(graph.training_lp_cache)
        indices = np.arange(len(costs), dtype=np.int64)
        predictor = nd._new_network(final_graph, chosen_cfg, cfg.seed + 110003)
        optimizer = nd._optimizer(predictor, chosen_cfg)
        rng = np.random.default_rng(cfg.seed + 110003)
        base_losses = []
        refit_started = perf_counter()
        for epoch in range(1, winner["base_warmup_epochs"] + 1):
            base_losses.append(nd._train_epoch(predictor, optimizer, costs, prepared,
                                               indices, rng, chosen_cfg))
            if epoch == 1 or epoch % cfg.log_every == 0:
                print(f"ND repair full refit epoch={epoch} nll={base_losses[-1]:.6f}",
                      flush=True)
        refit_checkpoints = []
        best_training_state, best_training_record = None, None
        optimizer = nd._optimizer(predictor, chosen_cfg)
        continuation_rng = np.random.default_rng(cfg.seed + 510007)
        refit_steps = _checkpoint_steps(repair, winner["continuation_steps"])
        for step in range(winner["continuation_steps"] + 1):
            last = None
            if step:
                rate = (repair.first_learning_rate if step <= repair.first_steps
                        else repair.second_learning_rate)
                last = _continuation_step(nd, predictor, optimizer, costs, prepared,
                                          indices, continuation_rng, chosen_cfg, rate)
            if step not in refit_steps:
                continue
            diagnostics = _diagnostics(nd, predictor, costs, prepared, indices)
            _finite_network(predictor, torch)
            record = {"continuation_steps": step, "last_training_update": last,
                      "training_diagnostics": diagnostics}
            refit_checkpoints.append(record)
            key = (diagnostics["full_nll_per_integer"], step)
            if best_training_record is None or key < (
                    best_training_record["training_diagnostics"]["full_nll_per_integer"],
                    best_training_record["continuation_steps"]):
                best_training_state, best_training_record = nd._state(predictor), record
            print(f"ND repair full refit extra_step={step} "
                  f"train_nll={diagnostics['full_nll_per_integer']:.6f}", flush=True)
        predictor.load_state_dict(best_training_state)
        predictor.eval()
        final_diagnostics = _diagnostics(nd, predictor, costs, prepared, indices)
        nd._sync(final_graph.device)
        refit_seconds = perf_counter() - refit_started
        metadata = {
            "method": "Neural Diving matched-dimension complete predictor",
            "implementation": IMPLEMENTATION, "paper": nd.PAPER_URL,
            "config": asdict(chosen_cfg), "repair_config": asdict(repair),
            "training_device": device, "inference_device": "cpu", "threads": 1,
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "matrix_digest": final_graph.matrix_digest,
            "train_cost_digest": nd._array_digest(costs),
            "train_solution_digest": nd._array_digest(clean.train_solutions),
            "training_indices": training.tolist(), "validation_indices": validation.tolist(),
            "containment_validation_indices": containment_indices.tolist(),
            "final_fit_indices": indices.tolist(), "training_count": len(costs),
            "common_coordinate_indices": nd._common_indices(clean).tolist(),
            "target_K": K, "root_lp_training": root_features,
            "selected_width": winner["width"],
            "selected_warmup_epochs": winner["base_warmup_epochs"],
            "selected_continuation_steps": winner["continuation_steps"],
            "selected_candidate": winner, "width_candidates": candidates,
            "final_base_refit_losses": base_losses,
            "final_refit_checkpoints": refit_checkpoints,
            "final_retained_continuation_steps": best_training_record["continuation_steps"],
            "final_full_predictor_diagnostics": final_diagnostics,
            "constant_coordinate_training_diagnostics": b._constant_coordinate_diagnostics(prepared),
            "epoch_selection_seconds": selection_seconds, "final_refit_seconds": refit_seconds,
            "selection_rule": "Most certified validation optimizer witnesses contained by the actual D-K MAP fixings on at most eight fixed heldout training costs; then lower NLL on all heldout training costs, fewer continuation steps, narrower width.",
            "selection_does_not_measure_all_optima": True,
            "final_checkpoint_rule": "Among the base-end and every 100 continuation-update checkpoints through the selected budget, retain minimum full-training NLL; no test or validation information used for final checkpoint retention.",
            "final_checkpoint_implementation_change": "The archived wrapper retained the terminal full-refit model. This repaired wrapper retains the minimum full-training-NLL checkpoint within the already selected refit budget.",
            "architecture_features_loss_and_fixing_rule_unchanged": True,
            "continuation_optimizer": "Adam reset once at continuation start; 1000 updates at .001, then 1000 at .0001; uniform training-only minibatches without replacement within each minibatch.",
            "native_algorithm": False, "unused_native_coverage_heads_trained": False,
            "pool_adaptation": "One shared certified optimal solution per training cost; no additional feasible pool generation.",
            "source_sha256": source_hashes,
            "parameter_count": sum(p.numel() for p in predictor.parameters()),
            "test_information_used": False,
            "learning_time_excludes_shared_training_optima": True,
            "learning_scope": "Preprocessing, root LP features, both full warmup/continuation searches, heldout containment/NLL evaluation, final refit/checkpoint retention, model construction and exported weights. Shared labels and post-export metadata serialization excluded.",
            "source_config_overrides": config_overrides or {},
            "repair_config_overrides": repair_overrides or {},
            **pool_metadata,
        }
        model = nd.NeuralDivingModel(final_graph, [], predictor, chosen_cfg, metadata).to("cpu")
        model.graph.training_lp_cache.clear()
        with (output / CHECKPOINT_NAME).open("wb") as handle:
            torch.save({"format": IMPLEMENTATION, "config": asdict(chosen_cfg),
                        "metadata": metadata, "predictor_state": nd._state(predictor)}, handle)
        metadata["learning_seconds_wall"] = perf_counter() - started
        _json(output / METADATA_NAME, metadata)
    return model, metadata


def load(case, output_dir: str | Path, base_source: str | Path):
    """Restore the exact exported predictor, leaving its training cache empty."""
    b = _archived(base_source)
    import torch
    b._verify_vendor()
    nd = b.nd
    clean = b.training_case(case)
    output = Path(output_dir)
    with (output / CHECKPOINT_NAME).open("rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    if payload["format"] != IMPLEMENTATION:
        raise ValueError("Unrecognized repaired Neural Diving checkpoint")
    metadata = json.loads((output / METADATA_NAME).read_text(encoding="utf-8"))
    if nd._array_digest(clean.train_costs) != metadata["train_cost_digest"] or \
            nd._array_digest(clean.train_solutions) != metadata["train_solution_digest"]:
        raise ValueError("Checkpoint training arrays differ")
    cfg = replace(nd.NeuralDivingConfig(**payload["config"]), device="cpu")
    nd._setup(cfg)
    graph = nd.FixedBipartiteGraph(clean.model, clean.train_costs,
        cfg.normalize_constraint_rows, "cpu", cfg.use_root_lp_features,
        cfg.root_lp_time_limit, cfg.seed)
    if graph.matrix_digest != metadata["matrix_digest"]:
        raise ValueError("Checkpoint physical model differs")
    predictor = nd._new_network(graph, cfg, cfg.seed + 110003)
    predictor.load_state_dict(payload["predictor_state"])
    predictor.eval()
    _finite_network(predictor, torch)
    return nd.NeuralDivingModel(graph, [], predictor, cfg, metadata)


fit_neural = fit
load_neural = load
