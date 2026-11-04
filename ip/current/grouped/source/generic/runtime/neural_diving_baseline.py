"""Neural Diving v2: source-disclosed native mask and matched-K variant.

This is a local implementation of the conditional-independent binary variant,
not a claim to reproduce author code or the paper's large-scale experiments.
The graph GCN, weighted feasible labels, acceptance/coverage loss, Bernoulli
partial assignments, and sequential sub-MIP search follow the paper's core.
See ../docs/neural-diving-v2-method.md for protocol and explicit adaptations.

Only ``case.train_costs`` and supplied training pools are accessed by ``fit``.
UC dispatch variables remain continuous and are never predicted or fixed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F


PAPER_URL = "https://arxiv.org/html/2012.13349v3"
METHOD_NAME = "Neural Diving native masks (v2 paper implementation, HiGHS port)"
MATCHED_METHOD_NAME = "Neural Diving matched-K MAP-confidence variant (v2)"


@dataclass(frozen=True)
class NeuralDivingConfig:
    coverage_targets: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    hidden_size: int = 32
    width_candidates: tuple[int, ...] = (32, 64)
    graph_layers: int = 6
    epochs: int = 200
    warmup_epochs: int = 150
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    coverage_penalty: float = 32.0
    validation_fraction: float = 0.2
    patience: int = 20
    gradient_clip: float = 5.0
    seed: int = 1909
    threads: int = 1
    device: str = "cpu"
    normalize_constraint_rows: bool = True
    use_root_lp_features: bool = True
    root_lp_time_limit: float = 5.0
    validation_solver_count: int = 8
    validation_every: int = 25
    validation_time_limit: float = 3.0
    final_validation_time_limit: float = 30.0
    sanity_train_count: int = 8
    sanity_steps: int = 1200
    sanity_decay_after_steps: int = 400
    sanity_learning_rate_decay: float = 0.1
    sanity_excess_nll_limit: float = 0.03
    sanity_solver_count: int = 4
    sanity_time_limit: float = 5.0
    sanity_fixed_fraction: float = 0.1
    sanity_feasible_fraction: float = 0.5
    require_native_validation_feasible_fraction: float = 0.5
    log_every: int = 25

    def validate(self) -> None:
        if not self.coverage_targets or any(not 0 < c < 1 for c in self.coverage_targets):
            raise ValueError("coverage targets must lie strictly between zero and one")
        if min(self.hidden_size, self.graph_layers, self.epochs, self.batch_size) <= 0:
            raise ValueError("network sizes and training limits must be positive")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation fraction must lie strictly between zero and 0.5")
        if self.threads != 1:
            raise ValueError("this fixed sequential CPU protocol uses one PyTorch thread")
        if self.graph_layers < 6:
            raise ValueError("v2 requires at least six message layers for UC price information")
        if not self.width_candidates or min(self.width_candidates) <= 0:
            raise ValueError("width candidates must be positive")
        if self.validation_solver_count <= 0 or self.validation_every <= 0:
            raise ValueError("bounded validation settings must be positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if min(self.warmup_epochs, self.sanity_steps, self.log_every) <= 0:
            raise ValueError("warmup, sanity and logging limits must be positive")
        if self.sanity_decay_after_steps <= 0 or not 0 < self.sanity_learning_rate_decay < 1:
            raise ValueError("sanity decay step and factor must be positive and below one")
        if min(self.validation_time_limit, self.final_validation_time_limit,
               self.sanity_time_limit, self.root_lp_time_limit) <= 0:
            raise ValueError("solver budgets must be positive")
        if not 0 < self.sanity_fixed_fraction < 1:
            raise ValueError("sanity fixed fraction must be between zero and one")
        if any(not 0 <= value <= 1 for value in (
                self.sanity_feasible_fraction,
                self.require_native_validation_feasible_fraction)):
            raise ValueError("feasibility gate fractions must be in [0,1]")


def _model_arrays(model: Any) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray]:
    a = getattr(model, "Acsr", None)
    if a is None:
        a = getattr(model, "A", None)
    if a is None:
        raise AttributeError("LinearModel needs Acsr or A")
    return (sparse.csr_matrix(a, dtype=np.float64),
            np.asarray(model.row_lower, dtype=np.float64),
            np.asarray(model.row_upper, dtype=np.float64),
            np.asarray(model.col_lower, dtype=np.float64),
            np.asarray(model.col_upper, dtype=np.float64),
            np.asarray(model.integrality))


def _array_digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        contiguous = np.ascontiguousarray(a)
        h.update(str(contiguous.shape).encode("ascii"))
        h.update(str(contiguous.dtype).encode("ascii"))
        h.update(contiguous.tobytes())
    return h.hexdigest()


class FixedBipartiteGraph:
    """Coefficient-weighted symmetric adjacency with self loops.

    The fitted v2 default applies uniform row-max scaling to graph inputs:
    each constraint and both bounds are divided by the same positive value.
    Original LP/MIP models and root-LP feature units remain unchanged. Raw
    coefficients remain available explicitly for source/ablation checks.
    The structural node features include both sides of interval constraints.
    No variable index or training/test solution is an input feature.
    """
    feature_count = 13

    def __init__(self, model: Any, train_costs: np.ndarray,
                 normalize_rows: bool = False, device: str = "cpu",
                 use_root_lp_features: bool = False,
                 root_lp_time_limit: float = 5.0, root_lp_seed: int = 1909):
        a, rlo, rup, clo, cup, integer = _model_arrays(model)
        self.use_root_lp_features = bool(use_root_lp_features)
        self.training_lp_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        self.active_root_lp: tuple[str, np.ndarray] | None = None
        self.root_lp_extractor = None
        if self.use_root_lp_features:
            from nd_root_lp_features import RootLPFeatureExtractor, ROOT_FEATURE_NAMES
            self.root_lp_extractor = RootLPFeatureExtractor(model, root_lp_seed, root_lp_time_limit)
            self.feature_count = 13 + len(ROOT_FEATURE_NAMES)
        self.n_variables = a.shape[1]
        self.n_constraints = a.shape[0]
        self.integer_indices = np.flatnonzero(integer != 0).astype(np.int64)
        # Diagnostic labels only, never graph input features. The exported UC
        # formulation has three contiguous equal binary blocks followed by p.
        if (a.shape[1] % 4 == 0 and len(self.integer_indices) == 3*a.shape[1]//4 and
                np.array_equal(self.integer_indices, np.arange(len(self.integer_indices)))):
            d = a.shape[1]//4
            self.diagnostic_blocks = {"u": (0, d), "v": (d, 2*d), "w": (2*d, 3*d)}
        else:
            self.diagnostic_blocks = {"x": (0, len(self.integer_indices))}
        if self.integer_indices.size == 0:
            raise ValueError("Neural Diving needs integer variables")
        if not (np.all(np.isclose(clo[self.integer_indices], 0.0)) and
                np.all(np.isclose(cup[self.integer_indices], 1.0))):
            raise ValueError("this benchmark implementation supports binary integers only")
        costs = np.asarray(train_costs, dtype=np.float64)
        if costs.ndim != 2 or costs.shape[1] != self.n_variables or not np.isfinite(costs).all():
            raise ValueError("training costs have invalid shape or entries")
        self.cost_scale = max(float(np.sqrt(np.mean(costs ** 2))), 1e-8)
        row_max = np.asarray(abs(a).max(axis=1).toarray()).ravel()
        row_scale = np.maximum(row_max, 1.0) if normalize_rows else np.ones_like(row_max)
        self.normalize_rows = normalize_rows
        normalized = sparse.diags(1.0 / row_scale) @ a
        zero_v = sparse.csr_matrix((self.n_variables, self.n_variables))
        zero_c = sparse.csr_matrix((self.n_constraints, self.n_constraints))
        adjacency = sparse.bmat([[zero_v, normalized.T], [normalized, zero_c]], format="csr")
        adjacency = adjacency + sparse.eye(adjacency.shape[0], format="csr")
        coo = adjacency.tocoo()
        self.adjacency = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
            torch.from_numpy(coo.data.astype(np.float32)), coo.shape,
            check_invariants=True).coalesce()
        nodes = np.zeros((adjacency.shape[0], self.feature_count), dtype=np.float32)
        nodes[:self.n_variables, 0] = 1.0
        nodes[self.n_variables:, 1] = 1.0
        nodes[:self.n_variables, 3] = (integer != 0).astype(np.float32)
        finite_bounds = np.concatenate((clo[np.isfinite(clo)], cup[np.isfinite(cup)],
                                        (rlo / row_scale)[np.isfinite(rlo)],
                                        (rup / row_scale)[np.isfinite(rup)]))
        bound_scale = max(float(np.max(np.log1p(abs(finite_bounds)))), 1.0)

        def bound_feature(values: np.ndarray) -> np.ndarray:
            out = np.zeros_like(values, dtype=np.float64)
            good = np.isfinite(values)
            out[good] = np.sign(values[good]) * np.log1p(abs(values[good])) / bound_scale
            return out.astype(np.float32)

        nodes[:self.n_variables, 4] = bound_feature(clo)
        nodes[:self.n_variables, 5] = bound_feature(cup)
        nodes[:self.n_variables, 6] = np.isfinite(clo)
        nodes[:self.n_variables, 7] = np.isfinite(cup)
        nodes[self.n_variables:, 8] = bound_feature(rlo / row_scale)
        nodes[self.n_variables:, 9] = bound_feature(rup / row_scale)
        nodes[self.n_variables:, 10] = np.isfinite(rlo)
        nodes[self.n_variables:, 11] = np.isfinite(rup)
        degree = np.concatenate((np.diff(a.tocsc().indptr), np.diff(a.indptr)))
        nodes[:, 12] = np.log1p(degree) / max(float(np.max(np.log1p(degree))), 1.0)
        self.static_features = torch.from_numpy(nodes)
        self.integer_tensor = torch.from_numpy(self.integer_indices)
        self.matrix_digest = _array_digest(a.indptr, a.indices, a.data, rlo, rup, clo, cup, integer)
        self.to(device)

    def to(self, device: str) -> "FixedBipartiteGraph":
        self.device = torch.device(device)
        self.adjacency = self.adjacency.to(self.device)
        self.static_features = self.static_features.to(self.device)
        self.integer_tensor = self.integer_tensor.to(self.device)
        return self

    def features(self, costs: np.ndarray | torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(costs, dtype=torch.float32, device=self.device)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        if c.ndim != 2 or c.shape[1] != self.n_variables:
            raise ValueError("cost vector does not match graph variables")
        features = self.static_features.unsqueeze(0).expand(c.shape[0], -1, -1).clone()
        features[:, :self.n_variables, 2] = c / self.cost_scale
        if self.use_root_lp_features:
            original_costs = (costs.detach().cpu().numpy() if isinstance(costs, torch.Tensor)
                              else np.asarray(costs, dtype=np.float64))
            if original_costs.ndim == 1:
                original_costs = original_costs[None, :]
            dynamic = []
            for cost in original_costs:
                key = _array_digest(np.asarray(cost, dtype=np.float64))
                if self.active_root_lp is not None and key == self.active_root_lp[0]:
                    values = self.active_root_lp[1]
                elif key in self.training_lp_cache:
                    values = self.training_lp_cache[key][0]
                else:
                    # Uncached manual diagnostics may extract features, but
                    # never install a confirmation/test cache in this object.
                    values, _ = self.root_lp_extractor.extract(cost)
                dynamic.append(values)
            features[:, :, 13:] = torch.as_tensor(np.stack(dynamic), dtype=torch.float32,
                                                device=self.device)
        return features

    def precompute_training_root_lp(self, training_costs: np.ndarray) -> dict[str, Any]:
        if not self.use_root_lp_features:
            return {"enabled": False, "count": 0, "seconds": 0.0}
        records, start = [], time.perf_counter()
        for index, cost in enumerate(np.asarray(training_costs, dtype=np.float64)):
            key = _array_digest(cost)
            if key not in self.training_lp_cache:
                self.training_lp_cache[key] = self.root_lp_extractor.extract(cost)
            records.append({"training_cost_index": index, "cost_digest": key,
                            **self.training_lp_cache[key][1]})
        return {"enabled": True, "count": len(records), "seconds": time.perf_counter()-start,
                "records": records, "cache_scope": "supplied archived training costs only"}

    def begin_online_root_lp(self, cost: np.ndarray, time_limit: float) -> dict[str, Any]:
        if not self.use_root_lp_features:
            return {"enabled": False, "seconds": 0.0}
        if self.active_root_lp is not None:
            raise RuntimeError("nested online root-LP feature scope")
        # Every timed native/matched call gets its own root LP even if this cost
        # is in the offline training cache. Only its five native heads share it.
        values, metadata = self.root_lp_extractor.extract(cost, time_limit=time_limit)
        self.active_root_lp = (_array_digest(np.asarray(cost, dtype=np.float64)), values)
        return {"enabled": True, **metadata}

    def end_online_root_lp(self) -> None:
        self.active_root_lp = None

    def aggregate(self, messages: torch.Tensor) -> torch.Tensor:
        # Apply one fixed graph to a whole mini-batch without replicating edges.
        b, n, h = messages.shape
        flattened = messages.permute(1, 0, 2).reshape(n, b * h)
        result = torch.sparse.mm(self.adjacency, flattened)
        return result.reshape(n, b, h).permute(1, 0, 2)


class _CoverageGCN(nn.Module):
    """Sections 3.2-3.3 GCN and two shared integer-node output heads."""
    def __init__(self, graph: FixedBipartiteGraph, config: NeuralDivingConfig):
        super().__init__()
        self.graph = graph
        h = config.hidden_size
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        dimensions = graph.feature_count
        for _ in range(config.graph_layers):
            self.layers.append(nn.Sequential(nn.Linear(dimensions, h), nn.ReLU(), nn.Linear(h, h)))
            self.norms.append(nn.LayerNorm(h))
            dimensions += h
        self.prediction = nn.Sequential(nn.Linear(dimensions, h), nn.ReLU(), nn.Linear(h, 1))
        self.acceptance = nn.Sequential(nn.Linear(dimensions, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, costs: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.graph.features(costs)
        for layer, norm in zip(self.layers, self.norms):
            update = norm(self.graph.aggregate(layer(embedding)))
            embedding = torch.cat((embedding, update), dim=-1)
        integers = embedding[:, self.graph.integer_tensor, :]
        return self.prediction(integers).squeeze(-1), self.acceptance(integers).squeeze(-1)


def _prepare_pools(case: Any, pools: list[dict[str, Any]], integer_indices: np.ndarray
                   ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    costs = np.asarray(case.train_costs, dtype=np.float64)
    n_train, n_var = costs.shape
    by_index: dict[int, dict[str, Any]] = {}
    for order, pool in enumerate(pools):
        index = int(pool.get("train_index", order))
        if index in by_index or not 0 <= index < n_train:
            raise ValueError("training pools must have unique valid train_index entries")
        by_index[index] = pool
    if len(by_index) != n_train:
        raise ValueError("one feasible-assignment pool is required for each training cost")
    prepared = []
    pool_sizes = []
    effective_sizes = []
    a, rlo, rup, clo, cup, integer = _model_arrays(case.model)
    for i in range(n_train):
        pool = by_index[i]
        if np.any(integer == 0) and not pool.get("conditional_lp_completion_certified", False):
            raise ValueError("v2 MIP pools require certified LP-optimal continuous completion for each integer label")
        solutions = np.asarray(pool.get("solutions", pool.get("incumbent_solutions")), dtype=np.float64)
        if solutions.ndim == 1:
            solutions = solutions[None, :]
        if solutions.ndim != 2 or solutions.shape[1] != n_var or not np.isfinite(solutions).all():
            raise ValueError(f"invalid training solutions at cost {i}")
        activity = a @ solutions.T
        if not (np.all(solutions >= clo - 1e-5) and np.all(solutions <= cup + 1e-5) and
                np.all(activity >= rlo[:, None] - 1e-5) and
                np.all(activity <= rup[:, None] + 1e-5)):
            raise ValueError(f"infeasible training solution at cost {i}")
        assignments = np.rint(solutions[:, integer_indices]).astype(np.int64)
        if not np.all(np.isclose(solutions[:, integer_indices], assignments, atol=1e-5)):
            raise ValueError(f"nonintegral training label at cost {i}")
        if not np.isin(assignments, [0, 1]).all():
            raise ValueError("nonbinary training label")
        objectives = solutions @ costs[i]
        supplied = np.asarray(pool.get("objectives", objectives), dtype=np.float64)
        if supplied.shape != objectives.shape or not np.allclose(supplied, objectives, rtol=1e-6, atol=1e-6):
            raise ValueError(f"pool objective does not agree with original cost at {i}")
        # Unique integer assignments only; keep the lowest-energy full assignment.
        unique: dict[bytes, int] = {}
        for j, assignment in enumerate(assignments):
            key = assignment.tobytes()
            if key not in unique or objectives[j] < objectives[unique[key]]:
                unique[key] = j
        chosen = np.array(list(unique.values()), dtype=np.int64)
        assignments = assignments[chosen]
        objectives = objectives[chosen]
        if len(chosen) == 0:
            raise ValueError(f"empty training pool at cost {i}")
        # Paper Eq.12, unnormalized original minimization energy, stable shift.
        negative_energy = -(objectives - objectives.min())
        weights = np.exp(negative_energy)
        weights /= weights.sum()
        pool_sizes.append(len(chosen))
        effective_sizes.append(float(1.0 / np.sum(weights ** 2)))
        prepared.append({"labels": torch.as_tensor(assignments, dtype=torch.float32),
                         "weights": torch.as_tensor(weights, dtype=torch.float64),
                         "objectives": objectives})
    metadata = {"pool_sizes": pool_sizes, "effective_pool_sizes": effective_sizes,
                "pool_label_digest": _array_digest(*[
                    p["labels"].numpy() for p in prepared]),
                "energy_weighting": "softmax(-raw_original_objective), no temperature rescaling",
                "conditional_lp_completion_required": bool(np.any(integer == 0)),
                "conditional_lp_completion_all_certified": bool(all(
                    p.get("conditional_lp_completion_certified", not np.any(integer == 0)) for p in pools)),
                "extra_conditional_lp_seconds": float(sum(p.get("extra_conditional_lp_seconds", 0.0) for p in pools)),
                "collection_seconds": float(sum(p.get("collection_seconds", 0.0) for p in pools))}
    return costs, prepared, metadata


def _selective_loss(prediction_logits: torch.Tensor, acceptance_logits: torch.Tensor,
                    indices: np.ndarray, pools: list[dict[str, Any]],
                    coverage: float, penalty: float) -> torch.Tensor:
    acceptance = torch.sigmoid(acceptance_logits)
    terms = []
    for local, i in enumerate(indices):
        labels = pools[int(i)]["labels"].to(device=prediction_logits.device,
                                           dtype=prediction_logits.dtype)
        weights = pools[int(i)]["weights"].to(device=prediction_logits.device,
                                             dtype=prediction_logits.dtype)
        logits = prediction_logits[local].expand_as(labels)
        variable_losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        per_assignment = (variable_losses * acceptance[local]).sum(dim=-1) / acceptance[local].sum().clamp_min(1e-8)
        risk = torch.sum(weights * per_assignment)
        shortfall = torch.relu(coverage - acceptance[local].mean())
        terms.append(risk + penalty * shortfall ** 2)
    return torch.stack(terms).mean()


def _full_loss(logits: torch.Tensor, indices: np.ndarray,
               pools: list[dict[str, Any]]) -> torch.Tensor:
    """Eq.11 weighted full-assignment NLL, averaged per integer coordinate."""
    terms = []
    for local, index in enumerate(indices):
        pool = pools[int(index)]
        labels = pool["labels"].to(device=logits.device, dtype=logits.dtype)
        weights = pool["weights"].to(device=logits.device, dtype=logits.dtype)
        losses = F.binary_cross_entropy_with_logits(
            logits[local].expand_as(labels), labels, reduction="none").mean(dim=1)
        terms.append(torch.sum(weights * losses))
    return torch.stack(terms).mean()


def _entropy_floor(indices: np.ndarray, pools: list[dict[str, Any]]) -> float:
    terms = []
    for index in indices:
        pool = pools[int(index)]
        weights = pool["weights"] / pool["weights"].sum()
        marginals = (weights[:, None] * pool["labels"].double()).sum(dim=0)
        # Clamp only possible float64 summation roundoff outside [0,1].
        marginals = marginals.clamp(0.0, 1.0)
        # xlogy gives the exact limiting value zero at r=0 or r=1.
        entropy = -torch.special.xlogy(marginals, marginals) - torch.special.xlogy(
            1.0 - marginals, 1.0 - marginals)
        terms.append(float(entropy.mean()))
    return float(np.mean(terms))


def _state(network: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}


def _sync(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _setup(config: NeuralDivingConfig) -> None:
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training is unavailable")
    torch.set_num_threads(config.threads)
    # CUDA sparse reductions can have nondeterministic implementations. Record
    # this adaptation instead of failing late or claiming CPU-bitwise identity.
    torch.use_deterministic_algorithms(True, warn_only=config.device == "cuda")
    torch.sparse.check_sparse_tensor_invariants.enable()


def _new_network(graph: FixedBipartiteGraph, config: NeuralDivingConfig,
                 seed: int) -> _CoverageGCN:
    torch.manual_seed(seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    return _CoverageGCN(graph, config).to(graph.device)


def _optimizer(network: nn.Module, config: NeuralDivingConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(network.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)


def _train_epoch(network: _CoverageGCN, optimizer: torch.optim.Optimizer,
                 costs: np.ndarray, pools: list[dict[str, Any]], indices: np.ndarray,
                 rng: np.random.Generator, config: NeuralDivingConfig,
                 coverage: float | None = None) -> float:
    network.train()
    order = rng.permutation(indices)
    total = 0.0
    for offset in range(0, len(order), config.batch_size):
        selected = order[offset:offset + config.batch_size]
        optimizer.zero_grad(set_to_none=True)
        prediction, acceptance = network(costs[selected])
        loss = (_full_loss(prediction, selected, pools) if coverage is None else
                _selective_loss(prediction, acceptance, selected, pools,
                                coverage, config.coverage_penalty))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite Neural Diving training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
        optimizer.step()
        total += float(loss.detach()) * len(selected)
    network.eval()
    return total / len(order)


def _full_diagnostics(network: _CoverageGCN, costs: np.ndarray,
                      pools: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    network.eval()
    with torch.no_grad():
        logits, _ = network(costs[indices])
        nll = float(_full_loss(logits, indices, pools))
        targets = torch.stack([
            (pools[int(i)]["weights"][:, None] * pools[int(i)]["labels"]).sum(dim=0)
            for i in indices]).to(device=logits.device, dtype=logits.dtype)
        p = torch.sigmoid(logits)
        expected_bit_error = float(torch.where(p >= .5, 1.0 - targets, targets).mean())
        expected_losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bit_errors = torch.where(p >= .5, 1.0-targets, targets)
        blocks = {name: {"full_nll_per_integer": float(expected_losses[:, lo:hi].mean()),
                         "expected_map_bit_error": float(bit_errors[:, lo:hi].mean())}
                  for name, (lo, hi) in network.graph.diagnostic_blocks.items()}
    floor = _entropy_floor(indices, pools)
    return {"full_nll_per_integer": nll, "marginal_entropy_floor": floor,
            "excess_nll": nll - floor,
            "expected_map_bit_error": expected_bit_error, "blocks": blocks}


def _common_indices(case: Any) -> np.ndarray:
    from dimension_diagnostics import common_coordinate_indices
    return common_coordinate_indices(case)


def _map_fixed(case: Any, p: np.ndarray, target_K: int) -> dict[int, int]:
    from dimension_diagnostics import matched_map_fixings
    # This function accepts probabilities ordered on all original integer nodes.
    return matched_map_fixings(case, p, target_K=target_K)


def _draw_partial(network: _CoverageGCN, graph: FixedBipartiteGraph,
                  cost: np.ndarray, coverage: float, seed: int) -> dict[str, Any]:
    network.eval()
    with torch.no_grad():
        logits, accept_logits = network(np.asarray(cost)[None, :])
        p = torch.sigmoid(logits[0]).cpu().numpy()
        q = torch.sigmoid(accept_logits[0]).cpu().numpy()
    rng = np.random.default_rng(seed)
    values = (rng.random(len(p)) < p).astype(np.int64)
    accepted = rng.random(len(q)) < q
    return {"fixed": {int(index): int(value) for index, value in zip(
                graph.integer_indices[accepted], values[accepted])},
            "coverage_target": float(coverage),
            "accepted_fraction": float(accepted.mean()),
            "mean_acceptance_probability": float(q.mean()),
            "mean_prediction_confidence": float(np.maximum(p, 1.0-p).mean())}


def _checked_solution(case: Any, cost: np.ndarray, result: Any,
                      fixed: dict[int, int]) -> tuple[np.ndarray | None, float | None]:
    if not result.valid or result.x is None:
        return None, None
    a, rlo, rup, clo, cup, integer = _model_arrays(case.model)
    x = np.asarray(result.x, dtype=np.float64)
    if x.shape != clo.shape or not np.isfinite(x).all():
        return None, None
    activity = a @ x
    ii = np.flatnonzero(integer != 0)
    valid = bool(np.all(x >= clo-1e-5) and np.all(x <= cup+1e-5) and
                 np.all(activity >= rlo-1e-5) and np.all(activity <= rup+1e-5) and
                 np.all(np.abs(x[ii]-np.rint(x[ii])) <= 1e-5) and
                 all(abs(x[j]-value) <= 1e-6 for j, value in fixed.items()))
    return (x.copy(), float(np.asarray(cost) @ x)) if valid else (None, None)


class TrainingSanityError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass
class NeuralDivingModel:
    graph: FixedBipartiteGraph
    networks: list[_CoverageGCN]
    full_predictor: _CoverageGCN
    config: NeuralDivingConfig
    training_metadata: dict[str, Any]

    def to(self, device: str) -> "NeuralDivingModel":
        if device not in ("cpu", "cuda"):
            raise ValueError("inference device must be cpu or cuda")
        self.graph.to(device)
        self.full_predictor.to(self.graph.device)
        for network in self.networks:
            network.to(self.graph.device)
        return self

    def partial_assignments(self, cost: np.ndarray, seed: int) -> list[dict[str, Any]]:
        # Separate deterministic per-model seeds and one order seed allow exact
        # replay of individual validation masks as well as the final ensemble.
        output = [_draw_partial(network, self.graph, cost, coverage,
                                seed + 1009*index)
                  for index, (coverage, network) in enumerate(zip(
                      self.config.coverage_targets, self.networks))]
        order = np.random.default_rng(seed + 880301).permutation(len(output))
        return [output[int(i)] for i in order]

    def matched_partial_assignment(self, case: Any, cost: np.ndarray,
                                   K: int) -> dict[str, Any]:
        from dimension_diagnostics import assert_matched_fixings
        scope = _common_indices(case)
        self.full_predictor.eval()
        with torch.no_grad():
            logits, _ = self.full_predictor(np.asarray(cost)[None, :])
            p = torch.sigmoid(logits[0]).cpu().numpy()
        fixed = _map_fixed(case, p, K)
        assert_matched_fixings(case, fixed, target_K=K)
        common_positions = np.searchsorted(self.graph.integer_indices, scope)
        return {"fixed": fixed, "coverage_target": None,
                "accepted_fraction": len(fixed)/len(scope),
                "mean_acceptance_probability": None,
                "mean_prediction_confidence": float(np.maximum(
                    p[common_positions], 1.0-p[common_positions]).mean()),
                "scope_indices": scope.tolist(), "target_K": int(K),
                "remaining_scope_count": len(scope)-len(fixed),
                "selection": "MAP values; descending max(p,1-p); column-index tie break"}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            torch.save({"format_version": 3, "config": asdict(self.config),
                        "training_metadata": self.training_metadata,
                        "state_dicts": [_state(network) for network in self.networks],
                        "full_predictor_state": _state(self.full_predictor)}, handle)


def _solver(solve_fn: Callable[..., Any] | None) -> Callable[..., Any]:
    if solve_fn is None:
        from adapter import solve
        return solve
    return solve_fn


def _run_assignments(model: NeuralDivingModel, case: Any, cost: np.ndarray,
                     assignments: list[dict[str, Any]], seed: int, start: float,
                     inference_seconds: float, total_time_limit: float,
                     solve_fn: Callable[..., Any], mode: str) -> dict[str, Any]:
    best_x, best_objective, best_certified = None, np.inf, False
    best_fixed, best_submip_index = None, None
    solver_seconds, records = 0.0, []
    for index, partial in enumerate(assignments):
        remaining = total_time_limit - (time.perf_counter()-start)
        if remaining <= 0:
            break
        cap = remaining/(len(assignments)-index)
        result = solve_fn(case.model, np.asarray(cost, dtype=np.float64),
                          fixed=partial["fixed"], seed=int(seed+104729*index),
                          time_limit=cap)
        solver_seconds += float(result.seconds)
        x, objective = _checked_solution(case, cost, result, partial["fixed"])
        if x is not None and (objective < best_objective-1e-8 or (
                abs(objective-best_objective) <= 1e-8 and
                result.certified and not best_certified)):
            best_x, best_objective = x, objective
            best_certified = bool(result.certified)
            best_fixed, best_submip_index = dict(partial["fixed"]), index
        record = {key: value for key, value in partial.items() if key != "fixed"}
        record.update({"fixed": dict(partial["fixed"]), "fixed_count": len(partial["fixed"]),
                       "valid": x is not None, "objective": objective,
                       "submip_certified": bool(result.certified),
                       "solver_seconds": float(result.seconds), "time_limit_seconds": cap})
        records.append(record)
    output = {"method": METHOD_NAME if mode == "native" else MATCHED_METHOD_NAME,
              "mode": mode, "x": best_x,
              "objective": best_objective if best_x is not None else None,
              "valid": best_x is not None, "certified": best_certified,
              "globally_certified": False,
              "status": "best_submip_optimal" if best_certified else
                        "feasible" if best_x is not None else "no_feasible_assignment",
              "seconds": time.perf_counter()-start,
              "inference_seconds": inference_seconds, "solver_seconds": solver_seconds,
              "submip_records": records, "best_fixed": best_fixed,
              "best_submip_index": best_submip_index,
              "integer_count": len(model.graph.integer_indices),
              "mean_fixed_count": float(np.mean([r["fixed_count"] for r in records])) if records else 0.0,
              "seed": int(seed), "total_time_limit_seconds": total_time_limit,
              "inference_device": str(model.graph.device)}
    if mode == "matched":
        output.update({key: assignments[0][key] for key in (
            "scope_indices", "target_K", "remaining_scope_count", "selection")})
    return output


def solve_cost(model: NeuralDivingModel, case: Any, cost: np.ndarray, seed: int,
               total_time_limit: float = 30.0, solve_fn: Callable[..., Any] | None = None,
               mode: str = "native", K: int | None = None) -> dict[str, Any]:
    """Native mask ensemble or explicitly named common-coordinate K variant.

    All sub-MIPs share one solve budget. Model creation/checking overhead is
    included in the recorded elapsed time and can slightly exceed the cap.
    Local sub-MIP certificates never certify the original problem.
    """
    if total_time_limit <= 0 or mode not in ("native", "matched"):
        raise ValueError("positive budget and native/matched mode are required")
    c = np.asarray(cost, dtype=np.float64)
    if c.shape != (model.graph.n_variables,) or not np.isfinite(c).all():
        raise ValueError("invalid cost vector")
    if mode == "matched" and K is None:
        raise ValueError("matched mode requires an explicit common-coordinate K")
    if mode == "matched" and int(K) != K:
        raise ValueError("matched K must be an integer")
    _sync(model.graph.device)
    start = time.perf_counter()
    from nd_root_lp_features import RootLPFeatureError
    try:
        lp_metadata = model.graph.begin_online_root_lp(c, total_time_limit)
    except RootLPFeatureError as error:
        elapsed = time.perf_counter()-start
        model.graph.end_online_root_lp()
        return {"method": METHOD_NAME if mode == "native" else MATCHED_METHOD_NAME,
                "mode": mode, "x": None, "objective": None, "valid": False,
                "certified": False, "globally_certified": False,
                "status": "root_lp_feature_failure", "seconds": elapsed,
                "inference_seconds": elapsed, "root_lp_seconds": error.metadata.get("seconds", elapsed),
                "root_lp_metadata": error.metadata, "solver_seconds": 0.0,
                "submip_records": [], "best_fixed": None, "best_submip_index": None,
                "integer_count": len(model.graph.integer_indices), "mean_fixed_count": 0.0,
                "seed": int(seed), "total_time_limit_seconds": total_time_limit,
                "inference_device": str(model.graph.device)}
    try:
        assignments = (model.partial_assignments(c, seed) if mode == "native" else
                       [model.matched_partial_assignment(case, c, int(K))])
        _sync(model.graph.device)
        inference_seconds = time.perf_counter()-start
        output = _run_assignments(model, case, c, assignments, seed, start,
                                 inference_seconds, total_time_limit, _solver(solve_fn), mode)
        output["root_lp_seconds"] = lp_metadata["seconds"]
        output["root_lp_metadata"] = lp_metadata
        return output
    finally:
        # Test/confirmation features never survive a timed method call. Native
        # and matched calls for the same cost therefore each pay for an LP.
        model.graph.end_online_root_lp()


def solve_matched_cost(model: NeuralDivingModel, case: Any, cost: np.ndarray, seed: int,
                       K: int, total_time_limit: float = 30.0,
                       solve_fn: Callable[..., Any] | None = None) -> dict[str, Any]:
    return solve_cost(model, case, cost, seed, total_time_limit, solve_fn, mode="matched", K=K)


def _quality(records: list[dict[str, Any]], reference_objectives: np.ndarray,
             indices: np.ndarray) -> dict[str, Any]:
    valid_count, recovered_count, certified_recovered_count = 0, 0, 0
    seconds, gaps = [], []
    per_cost = []
    for record, i in zip(records, indices):
        valid = bool(record["valid"])
        objective, ref = record["objective"], float(reference_objectives[int(i)])
        tolerance = 1e-6+1e-9*abs(ref)
        recovered = bool(valid and abs(float(objective)-ref) <= tolerance)
        certified_recovered = bool(recovered and record["certified"])
        valid_count += int(valid)
        recovered_count += int(recovered)
        certified_recovered_count += int(certified_recovered)
        seconds.append(float(record["seconds"]))
        gap = (max(0.0, float(objective)-ref)/max(abs(ref), 1.0)) if valid else None
        if gap is not None:
            gaps.append(gap)
        per_cost.append({"train_index": int(i), "valid": valid,
                         "recovered": recovered, "certified_recovered": certified_recovered,
                         "objective": objective, "reference_objective": ref,
                         "seconds": float(record["seconds"]),
                         "fixed_maps": [r["fixed"] for r in record["submip_records"]]})
    n = len(indices)
    return {"count": n, "valid_count": valid_count, "recovered_count": recovered_count,
            "certified_recovered_count": certified_recovered_count,
            "feasible_fraction": valid_count/n, "recovery_fraction": recovered_count/n,
            "certified_recovery_fraction": certified_recovered_count/n,
            "mean_seconds": float(np.mean(seconds)),
            "mean_valid_relative_gap": float(np.mean(gaps)) if gaps else None,
            "per_cost": per_cost}


def _single_validation(network: _CoverageGCN, graph: FixedBipartiteGraph,
                       config: NeuralDivingConfig, case: Any, costs: np.ndarray,
                       indices: np.ndarray, reference: np.ndarray, coverage: float,
                       model_index: int, solve_fn: Callable[..., Any]) -> dict[str, Any]:
    shell = NeuralDivingModel(graph, [network], network,
                            replace(config, coverage_targets=(coverage,)), {})
    records = []
    for index in indices:
        seed = config.seed+300001+int(index)*4099+model_index*1009
        records.append(solve_cost(shell, case, costs[int(index)], seed,
                       config.validation_time_limit, solve_fn))
    return _quality(records, reference, indices)


def _sanity_gate(graph: FixedBipartiteGraph, config: NeuralDivingConfig,
                 case: Any, costs: np.ndarray, pools: list[dict[str, Any]],
                 training_indices: np.ndarray, reference: np.ndarray,
                 solve_fn: Callable[..., Any]) -> dict[str, Any]:
    indices = training_indices[:min(config.sanity_train_count, len(training_indices))]
    seed = config.seed+700001
    network = _new_network(graph, config, seed)
    optimizer = _optimizer(network, config)
    before = _full_diagnostics(network, costs, pools, indices)
    curve = []
    best_step, best_diagnostics = 0, before
    best_state = _state(network)
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    # One gradient step per loop on this tiny subset, never on validation costs.
    for step in range(1, config.sanity_steps+1):
        rate = config.learning_rate * (1.0 if step <= config.sanity_decay_after_steps
                                     else config.sanity_learning_rate_decay)
        for group in optimizer.param_groups:
            group["lr"] = rate
        network.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = network(costs[indices])
        loss = _full_loss(logits, indices, pools)
        if not bool(torch.isfinite(loss)):
            return {"passed": False, "reason": "nonfinite_small_subset_loss",
                    "indices": indices.tolist(), "before": before}
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip))
        optimizer.step()
        if step % config.log_every == 0 or step == config.sanity_steps:
            value = _full_diagnostics(network, costs, pools, indices)
            if value["excess_nll"] < best_diagnostics["excess_nll"]:
                best_step, best_diagnostics = step, value
                best_state = _state(network)
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            curve.append({"step": step, "learning_rate": optimizer.param_groups[0]["lr"],
                          "gradient_norm_before_clip": gradient_norm,
                          "clipping_active": gradient_norm > config.gradient_clip, **value})
            print(f"ND sanity width={config.hidden_size} step={step} excess={value['excess_nll']:.5f}",
                  flush=True)
    terminal = _full_diagnostics(network, costs, pools, indices)
    if terminal["excess_nll"] < best_diagnostics["excess_nll"]:
        best_step, best_diagnostics = config.sanity_steps, terminal
        best_state = _state(network)
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    network.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer_state)
    after = _full_diagnostics(network, costs, pools, indices)
    fit_ok = after["excess_nll"] <= config.sanity_excess_nll_limit
    scope = _common_indices(case)
    fixed_count = max(1, int(round(len(scope)*config.sanity_fixed_fraction)))
    target_K = len(scope)-fixed_count
    selected = indices[:min(config.sanity_solver_count, len(indices))]
    shell = NeuralDivingModel(graph, [], network, config, {})
    records = [solve_cost(shell, case, costs[int(index)],
               config.seed+800001+int(index), config.sanity_time_limit,
               solve_fn, mode="matched", K=target_K) for index in selected]
    quality = _quality(records, reference, selected)
    feasibility_ok = quality["feasible_fraction"] >= config.sanity_feasible_fraction
    return {"passed": bool(fit_ok and feasibility_ok), "indices": indices.tolist(),
            "scope_size": len(scope), "small_subip_target_K": target_K,
            "small_subip_fixed_fraction": fixed_count/len(scope),
            "steps": config.sanity_steps, "before": before, "after": after,
            "learning_rate_schedule": {"base": config.learning_rate,
                "decay_after_steps": config.sanity_decay_after_steps,
                "decay_factor": config.sanity_learning_rate_decay},
            "best_step": best_step, "terminal": terminal,
            "best_model_and_optimizer_restored": True,
            "excess_nll_limit": config.sanity_excess_nll_limit,
            "fit_passed": bool(fit_ok), "small_subip_feasibility_passed": bool(feasibility_ok),
            "small_subip_quality": quality, "curve": curve,
            "interpretation": "tiny-subset optimization/integration gate, not a final-policy recovery target"}


def _warmup_selection(graph: FixedBipartiteGraph, config: NeuralDivingConfig,
                      costs: np.ndarray, pools: list[dict[str, Any]],
                      train: np.ndarray, validation: np.ndarray
                      ) -> tuple[dict[str, torch.Tensor], int, dict[str, Any]]:
    seed = config.seed+110003
    network = _new_network(graph, config, seed)
    optimizer, rng = _optimizer(network, config), np.random.default_rng(seed)
    best_nll, best_epoch, best_state = np.inf, 0, None
    history = []
    for epoch in range(1, config.warmup_epochs+1):
        train_loss = _train_epoch(network, optimizer, costs, pools, train, rng, config)
        diagnostics = _full_diagnostics(network, costs, pools, validation)
        nll = diagnostics["full_nll_per_integer"]
        history.append({"epoch": epoch, "train_full_nll": train_loss, **diagnostics})
        if nll < best_nll-1e-7:
            best_nll, best_epoch, best_state = nll, epoch, _state(network)
        if epoch % config.log_every == 0 or epoch == 1:
            print(f"ND warmup width={config.hidden_size} epoch={epoch} val_nll={nll:.5f}", flush=True)
        if epoch-best_epoch >= config.patience:
            break
    return best_state, best_epoch, {"network_seed": seed, "epochs": history,
                                   "selected_epoch": best_epoch,
                                   "best_validation_full_nll": best_nll}


def _coverage_selection(graph: FixedBipartiteGraph, config: NeuralDivingConfig,
                        warmup_state: dict[str, torch.Tensor], coverage: float,
                        model_index: int, case: Any, costs: np.ndarray,
                        pools: list[dict[str, Any]], training: np.ndarray,
                        validation: np.ndarray, solver_validation: np.ndarray,
                        reference: np.ndarray, solve_fn: Callable[..., Any]
                        ) -> tuple[dict[str, torch.Tensor], int, dict[str, Any]]:
    seed = config.seed+120007+model_index*1009
    network = _new_network(graph, config, seed)
    network.load_state_dict(warmup_state)
    optimizer, rng = _optimizer(network, config), np.random.default_rng(seed)
    history, solver_history = [], []
    best_score, best_epoch, best_state = None, 0, None
    # Epoch zero is an admissible warmup-only policy and uses validation only.
    for epoch in range(config.epochs+1):
        train_loss = None if epoch == 0 else _train_epoch(
            network, optimizer, costs, pools, training, rng, config, coverage)
        network.eval()
        with torch.no_grad():
            prediction, acceptance = network(costs[validation])
            loss = float(_selective_loss(prediction, acceptance, validation, pools,
                                         coverage, config.coverage_penalty))
            q = float(torch.sigmoid(acceptance).mean())
        history.append({"epoch": epoch, "train_selective_loss": train_loss,
                        "validation_selective_loss": loss, "validation_soft_coverage": q})
        if epoch % config.validation_every == 0 or epoch == config.epochs:
            quality = _single_validation(network, graph, config, case, costs,
                solver_validation, reference, coverage, model_index, solve_fn)
            # Recovery and feasibility dominate loss. Time breaks remaining ties.
            score = (quality["recovered_count"], quality["valid_count"],
                     -quality["mean_seconds"], -loss)
            solver_history.append({"epoch": epoch, "quality": quality, "score": list(score)})
            if best_score is None or score > best_score:
                best_score, best_epoch, best_state = score, epoch, _state(network)
        if epoch % config.log_every == 0:
            print(f"ND coverage width={config.hidden_size} C={coverage} epoch={epoch} "
                  f"val_loss={loss:.5f} q={q:.3f}", flush=True)
    return best_state, best_epoch, {"coverage_target": coverage, "network_seed": seed,
            "selected_epoch": best_epoch, "selection_score": list(best_score),
            "epochs": history, "solver_validations": solver_history}


def fit(case: Any, pools: list[dict[str, Any]],
        config: NeuralDivingConfig | dict[str, Any] | None = None,
        validation: Any = None, solve_fn: Callable[..., Any] | None = None) -> NeuralDivingModel:
    """Select on an internal training split, then refit all training costs.

    No test fields are accessed. Validation solver calls inspect only retained
    training costs. A failed sanity/policy gate prevents a completed model.
    """
    if validation is not None:
        raise ValueError("v2 frozen protocol uses internal training-only validation")
    config = NeuralDivingConfig() if config is None else (
        NeuralDivingConfig(**config) if isinstance(config, dict) else config)
    _setup(config)
    solver = _solver(solve_fn)
    start = time.perf_counter()
    costs = np.asarray(case.train_costs, dtype=np.float64)
    if len(costs) < 5:
        raise ValueError("training-only validation needs at least five costs")
    shuffled = np.random.default_rng(config.seed).permutation(len(costs))
    n_validation = max(1, int(round(len(costs)*config.validation_fraction)))
    validation_indices = np.sort(shuffled[:n_validation])
    training_indices = np.sort(shuffled[n_validation:])
    graph = FixedBipartiteGraph(case.model, costs[training_indices],
                              config.normalize_constraint_rows, config.device,
                              config.use_root_lp_features, config.root_lp_time_limit, config.seed)
    root_lp_metadata = graph.precompute_training_root_lp(costs)
    costs, prepared, pool_metadata = _prepare_pools(case, pools, graph.integer_indices)
    reference = np.einsum("ij,ij->i", costs, np.asarray(case.train_solutions, dtype=np.float64))
    if not np.allclose(reference, [p["objectives"].min() for p in prepared],
                       atol=1e-6, rtol=1e-6):
        raise ValueError("training pools must include the archived reference optimal labels")
    # Every checkpoint uses the same fixed, seeded subset of 24 validation costs.
    solver_validation = np.sort(np.random.default_rng(config.seed+130003).choice(
        validation_indices, min(config.validation_solver_count, n_validation), replace=False))
    candidates, gates, selected, best_width_score = [], [], None, None
    selection_start = time.perf_counter()
    for width in config.width_candidates:
        candidate_config = replace(config, hidden_size=int(width))
        gate = _sanity_gate(graph, candidate_config, case, costs, prepared,
                            training_indices, reference, solver)
        gates.append({"width": int(width), **gate})
        if not gate["passed"]:
            candidates.append({"width": int(width), "eligible": False,
                               "reason": "small_subset_sanity_gate_failed"})
            continue
        warmup_state, warmup_epochs, warmup_history = _warmup_selection(
            graph, candidate_config, costs, prepared, training_indices, validation_indices)
        networks, coverage_epochs, coverage_histories = [], [], []
        for index, coverage in enumerate(config.coverage_targets):
            state, epochs, history = _coverage_selection(graph, candidate_config,
                warmup_state, coverage, index, case, costs, prepared, training_indices,
                validation_indices, solver_validation, reference, solver)
            network = _new_network(graph, candidate_config, config.seed+index)
            network.load_state_dict(state)
            networks.append(network)
            coverage_epochs.append(epochs)
            coverage_histories.append(history)
        predictor = _new_network(graph, candidate_config, config.seed)
        predictor.load_state_dict(warmup_state)
        shell = NeuralDivingModel(graph, networks, predictor, candidate_config, {})
        records = [solve_cost(shell, case, costs[int(index)],
            config.seed+300001+int(index)*4099, config.final_validation_time_limit,
            solver) for index in solver_validation]
        quality = _quality(records, reference, solver_validation)
        warmup_nll = warmup_history["best_validation_full_nll"]
        width_score = (quality["recovered_count"], quality["valid_count"],
                       -quality["mean_seconds"], -warmup_nll, -int(width))
        eligible = quality["feasible_fraction"] >= config.require_native_validation_feasible_fraction
        candidate = {"width": int(width), "eligible": bool(eligible),
                     "warmup_epochs": warmup_epochs, "coverage_epochs": coverage_epochs,
                     "warmup_history": warmup_history, "coverage_histories": coverage_histories,
                     "native_ensemble_validation": quality, "width_selection_score": list(width_score)}
        candidates.append(candidate)
        if eligible and (best_width_score is None or width_score > best_width_score):
            best_width_score, selected = width_score, candidate
        # CPU copies used only for epoch bookkeeping; final fits start afresh.
        del networks, predictor, shell
        if config.device == "cuda":
            torch.cuda.empty_cache()
    if selected is None:
        diagnostics = {"passed": False, "width_candidates": gates,
                       "candidates": candidates,
                       "failure_reason": "no width passed both tiny-fit and native-validation gates"}
        raise TrainingSanityError(diagnostics["failure_reason"], diagnostics)
    selection_seconds = time.perf_counter()-selection_start
    chosen_config = replace(config, hidden_size=selected["width"])
    final_graph = FixedBipartiteGraph(case.model, costs,
                                    config.normalize_constraint_rows, config.device,
                                    config.use_root_lp_features, config.root_lp_time_limit, config.seed)
    # LP features depend only on each original cost/model. They are independent
    # of the training96/all120 RMS cost scale and contain no integer labels.
    final_graph.training_lp_cache = dict(graph.training_lp_cache)
    final_indices = np.arange(len(costs), dtype=np.int64)
    refit_start = time.perf_counter()
    full_seed = config.seed+110003
    predictor = _new_network(final_graph, chosen_config, full_seed)
    optimizer, rng = _optimizer(predictor, chosen_config), np.random.default_rng(full_seed)
    full_refit_losses = []
    for epoch in range(1, selected["warmup_epochs"]+1):
        value = _train_epoch(predictor, optimizer, costs, prepared, final_indices,
                             rng, chosen_config)
        full_refit_losses.append(value)
        if epoch % config.log_every == 0 or epoch == 1:
            print(f"ND all120 full width={selected['width']} epoch={epoch} nll={value:.5f}", flush=True)
    full_diagnostics = _full_diagnostics(predictor, costs, prepared, final_indices)
    full_state = _state(predictor)
    networks, refit_histories = [], []
    for index, (coverage, epochs) in enumerate(zip(
            config.coverage_targets, selected["coverage_epochs"])):
        seed = config.seed+120007+index*1009
        network = _new_network(final_graph, chosen_config, seed)
        network.load_state_dict(full_state)
        optimizer, rng = _optimizer(network, chosen_config), np.random.default_rng(seed)
        losses = []
        for epoch in range(1, epochs+1):
            value = _train_epoch(network, optimizer, costs, prepared, final_indices,
                                 rng, chosen_config, coverage)
            losses.append(value)
            if epoch % config.log_every == 0 or epoch == 1:
                print(f"ND all120 selective C={coverage} epoch={epoch} loss={value:.5f}", flush=True)
        network.eval()
        networks.append(network)
        refit_histories.append({"coverage_target": coverage, "epochs": epochs,
                                "train_selective_losses": losses})
    _sync(final_graph.device)
    refit_seconds = time.perf_counter()-refit_start
    metadata = {"method": METHOD_NAME, "matched_method": MATCHED_METHOD_NAME,
        "paper": PAPER_URL, "case_id": str(case.case_id), "config": asdict(chosen_config),
        "torch_version": torch.__version__, "training_device": config.device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "feature_version": "formulation13-plus-certified-root-lp15-v1" if config.use_root_lp_features else "formulation13",
        "root_lp_training": root_lp_metadata,
        "cuda_device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
        "cuda_sparse_determinism": "warn_only; fixed RNG seeds" if config.device == "cuda" else "CPU deterministic",
        "matrix_digest": final_graph.matrix_digest,
        "cost_scale_train_only": final_graph.cost_scale, "selection_cost_scale": graph.cost_scale,
        "train_cost_digest": _array_digest(costs),
        "training_indices": training_indices.tolist(), "validation_indices": validation_indices.tolist(),
        "solver_validation_indices": solver_validation.tolist(),
        "validation_selection_rule": "recovery count, feasibility count, shorter elapsed time, lower loss",
        "final_fit_indices": final_indices.tolist(), "final_fit_count": len(costs),
        "train_cost_count": len(costs), "integer_variable_count": len(graph.integer_indices),
        "continuous_variable_count": graph.n_variables-len(graph.integer_indices),
        "common_coordinate_indices": _common_indices(case).tolist(),
        "training_seconds": time.perf_counter()-start, "epoch_selection_seconds": selection_seconds,
        "final_refit_seconds": refit_seconds, "width_candidates": candidates,
        "selected_width": selected["width"], "selected_warmup_epochs": selected["warmup_epochs"],
        "selected_coverage_epochs": selected["coverage_epochs"],
        "final_full_predictor_diagnostics": full_diagnostics,
        "final_full_refit_losses": full_refit_losses, "final_selective_refits": refit_histories,
        "sanity_gate": {"passed": True, "width_candidates": gates,
                        "selected_width": selected["width"],
                        "final_native_validation": selected["native_ensemble_validation"],
                        "validation_is_pre_refit": True, "failure_reason": None},
        "fresh_confirmation_test_accessed": False,
        "warmup_adaptation": "complete weighted Eq11 NLL before coverage-specific Eq17 optimization",
        "matched_variant": "full predictor; MAP; exact count d-K most-confident common u/x; no v/w/p fixing",
        **pool_metadata}
    return NeuralDivingModel(final_graph, networks, predictor, chosen_config, metadata)


def load(case: Any, path: str | Path, device: str = "cpu") -> NeuralDivingModel:
    with Path(path).open("rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 3:
        raise ValueError("repaired v2 loader requires checkpoint format3")
    config = NeuralDivingConfig(**payload["config"])
    _setup(replace(config, device=device))
    metadata = payload["training_metadata"]
    if not metadata.get("sanity_gate", {}).get("passed", False):
        raise TrainingSanityError("checkpoint did not pass training gates", metadata.get("sanity_gate", {}))
    graph = FixedBipartiteGraph(case.model, np.asarray(case.train_costs)[metadata["final_fit_indices"]],
                              config.normalize_constraint_rows, device,
                              config.use_root_lp_features, config.root_lp_time_limit, config.seed)
    if graph.matrix_digest != metadata["matrix_digest"]:
        raise ValueError("checkpoint does not match the frozen model")
    if _array_digest(np.asarray(case.train_costs, dtype=np.float64)) != metadata["train_cost_digest"]:
        raise ValueError("checkpoint does not match training costs")
    if _common_indices(case).tolist() != metadata["common_coordinate_indices"]:
        raise ValueError("checkpoint common-coordinate scope does not match the case")
    networks = []
    for state in payload["state_dicts"]:
        network = _new_network(graph, replace(config, device=device), config.seed)
        network.load_state_dict(state)
        network.eval()
        networks.append(network)
    if len(networks) != len(config.coverage_targets):
        raise ValueError("checkpoint has the wrong number of coverage models")
    predictor = _new_network(graph, replace(config, device=device), config.seed)
    predictor.load_state_dict(payload["full_predictor_state"])
    predictor.eval()
    return NeuralDivingModel(graph, networks, predictor, config, metadata)


def save_metadata(model: NeuralDivingModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.training_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

