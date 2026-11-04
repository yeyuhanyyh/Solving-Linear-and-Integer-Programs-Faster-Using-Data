"""Li et al. (AAAI 2025) source-adapted preference model reduction, v2.

Source: https://arxiv.org/html/2501.00307v1, Eqs. (8)--(21), Algorithms 1--2.
Permanent bounds, residual attention, optimizer settings and readiness checks
are declared local adaptations, not verified author code or author protocol.
No test solution or test objective is accessed by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
import json
import os
import random
import zipfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


SOURCE = "https://arxiv.org/html/2501.00307v1"
IMPLEMENTATION_VERSION = "li2025_source_adapted_v2_permanent_bounds_residual_attention_final_optimizer_v5"


@dataclass(frozen=True)
class PreferenceConfig:
    """The candidate grid is fixed before any test evaluation."""

    seed: int = 20261009
    hidden_dim: int = 128
    attention_layers: int = 2
    epochs: int = 1200
    batch_size: int = 128
    learning_rates: tuple[float, ...] = (1e-4, 1e-3)
    lambda1_values: tuple[float, ...] = (0.8, 0.9)
    lambda2_values: tuple[float, ...] = (0.1, 0.2)
    top_k_values: tuple[int, ...] = (1, 3, 5)
    validation_fraction: float = 0.2
    weight_decay: float = 0.01
    lr_decay: float = 0.9
    lr_decay_epochs: int = 40
    cover_feasibility_tolerance: float = 1e-4
    cover_suboptimality_tolerance: float = 1e-4
    active_tolerance: float = 1e-7
    feasibility_tolerance: float = 1e-6
    rank_tolerance: float = 1e-10
    denominator_floor: float = 1e-12
    reward_floor: float = 1e-10
    failed_decode_penalty: float = 1e6
    stationarity_relative_tolerance: float = 1e-9
    certificate_gap_absolute_tolerance: float = 1e-7
    certificate_gap_relative_tolerance: float = 1e-10
    reduced_lp_time_limit: float = 30.0
    recovery_absolute_tolerance: float = 1e-6
    recovery_relative_tolerance: float = 1e-9
    feedforward_multiplier: int = 2
    device: str = "cpu"
    sanity_minimum_score_spread: float = 1e-4
    sanity_minimum_relative_loss_improvement: float = 0.01
    sanity_minimum_non_tie_accuracy: float = 0.55  # Legacy reference diagnostic only; not a readiness gate.
    sanity_true_difference_tolerance: float = 1e-6
    gradient_clip_norm: float = 5.0


class PairAttentionLayer(nn.Module):
    """Eq. (13) attention inside an explicitly adapted pre-norm residual block.

    Residual paths, LayerNorm and per-token MLP are local v2 choices. They must
    not be described as recovered author architecture.
    """

    def __init__(self, width: int, feedforward_multiplier: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.attention_output = nn.Linear(width, width)
        self.token_norm = nn.LayerNorm(width)
        self.token_mlp = nn.Sequential(nn.Linear(width, width * feedforward_multiplier),
            nn.GELU(), nn.Linear(width * feedforward_multiplier, width))
        self.scale = width ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        q, k, v = self.q(normalized), self.k(normalized), self.v(normalized)
        attention = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        tokens = tokens + self.attention_output(torch.matmul(attention, v))
        return tokens + self.token_mlp(self.token_norm(tokens))


class PreferenceNetwork(nn.Module):
    """Each token concatenates a cost-parameter vector and one full strategy."""

    def __init__(self, parameter_dim: int, strategies: np.ndarray, config: PreferenceConfig) -> None:
        super().__init__()
        self.register_buffer("strategies", torch.as_tensor(strategies, dtype=torch.float32))
        input_dim = parameter_dim + strategies.shape[1]
        self.input_embedding = nn.Sequential(nn.Linear(input_dim, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim))
        self.layers = nn.ModuleList([PairAttentionLayer(config.hidden_dim, config.feedforward_multiplier)
            for _ in range(config.attention_layers)])
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.GELU(), nn.Linear(config.hidden_dim // 2, 1))

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        batch = parameters.shape[0]
        count = self.strategies.shape[0]
        costs = parameters[:, None, :].expand(batch, count, parameters.shape[1])
        strategies = self.strategies[None, :, :].expand(batch, count, self.strategies.shape[1])
        tokens = self.input_embedding(torch.cat((costs, strategies), dim=-1))
        for layer in self.layers:
            tokens = layer(tokens)
        return self.output(tokens).squeeze(-1)


def preference_loss(predictions: torch.Tensor, rewards: torch.Tensor, lambda1: float, lambda2: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Original adjacent pairs, with both losses divided by their pair count.

    Eqs. (17)--(18) divide by instances only. The additional common factor
    1/(M-1) is a local normalization: it preserves their relative weighting
    and unregularized minimizers while making scales comparable across banks.
    """
    if predictions.shape[1] <= 1:
        zero = predictions.sum() * 0.0
        return zero, zero, zero
    order = torch.argsort(rewards, dim=1, descending=True, stable=True)
    target = torch.gather(rewards, 1, order)
    predicted = torch.gather(predictions, 1, order)
    target_difference = target[:, :-1] - target[:, 1:]
    predicted_difference = predicted[:, :-1] - predicted[:, 1:]
    labels = torch.where(target_difference > 0.0, torch.ones_like(target_difference), torch.full_like(target_difference, 0.5))
    ranking = F.binary_cross_entropy_with_logits(predicted_difference, labels, reduction="mean")
    difference = ((predicted_difference - target_difference) ** 2).mean()
    total = lambda1 * ranking + lambda2 * difference
    return total, ranking, difference


def greedy_set_cover(edges: np.ndarray) -> np.ndarray:
    """Algorithm 2. Stable smallest original strategy ID resolves equal counts."""
    edges = np.asarray(edges, dtype=bool)
    if edges.ndim != 2 or edges.shape[0] == 0:
        raise ValueError("SetCover requires at least one training instance")
    uncovered = np.ones(edges.shape[0], dtype=bool)
    selected: list[int] = []
    while np.any(uncovered):
        counts = edges[uncovered].sum(axis=0)
        if counts.size == 0 or counts.max() == 0:
            missing = np.flatnonzero(uncovered).tolist()
            raise ValueError(f"No near-optimal feasible strategy covers training rows {missing}; decoder audit required")
        strategy = int(np.argmax(counts))
        selected.append(strategy)
        uncovered &= ~edges[:, strategy]
    return np.asarray(selected, dtype=int)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    # Avoid modifying the process-wide interop thread count after Torch starts.
    torch.use_deterministic_algorithms(True)


def _subset_training_case(case: Any, indices: np.ndarray) -> Any:
    # No test fields exist on the object supplied to extraction/decoding.
    return SimpleNamespace(model=case.model, case_id=case.case_id, seed=case.seed,
        **{name: np.asarray(getattr(case, name))[indices] for name in ("train_features", "train_costs", "train_solutions")})


def _training_hash(case: Any) -> str:
    digest = sha256()
    for name in ("train_features", "train_costs", "train_solutions"):
        array = np.ascontiguousarray(np.asarray(getattr(case, name), dtype=np.float64))
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _model_hash(case: Any) -> str:
    digest = sha256()
    matrix = sp.csr_matrix(case.model.A)
    digest.update(str(matrix.shape).encode())
    for array in (matrix.data.astype(np.float64), matrix.indices.astype(np.int64), matrix.indptr.astype(np.int64)):
        digest.update(np.ascontiguousarray(array).tobytes())
    for name in ("row_lower", "row_upper", "col_lower", "col_upper", "integrality"):
        digest.update(np.ascontiguousarray(np.asarray(getattr(case.model, name), dtype=np.float64)).tobytes())
    return digest.hexdigest()


class InequalityStrategyBank:
    """Shared full strategies, Li Eqs. (5)--(7) inequality-reduced LP decode.

    MLOPT's active-equality primal alone is insufficient here. It is accepted
    as a fast path only with a nonnegative reduced-inequality dual certificate.
    Otherwise solve the actual reduced LP, without imposing inactive rows.
    """

    def __init__(self, case: Any, config: PreferenceConfig) -> None:
        from adapter import LinearModel
        from mlopt_baseline import StrategyBank
        self.extracted = StrategyBank(case, active_tolerance=config.active_tolerance,
            feasibility_tolerance=config.feasibility_tolerance, rank_tolerance=config.rank_tolerance)
        self.case, self.config = case, config
        self.decode_statistics = {"calls": 0, "certified_kkt_fast_paths": 0, "reduced_lp_solves": 0, "failed_reduced_lp_solves": 0, "seconds": 0.0}
        self.reduced_models = []
        for strategy in self.strategies:
            rows = sp.vstack((self.E, self.G[strategy.active_rows]), format="csr")
            lower = np.concatenate((self.beq, np.full(len(strategy.active_rows), -np.inf)))
            upper = np.concatenate((self.beq, self.h[strategy.active_rows]))
            # The Appendix lists lb/ub separately. Retaining their original
            # domain is a declared v2 adaptation, not verified author behavior.
            column_lower = np.asarray(case.model.col_lower, dtype=float).copy()
            column_upper = np.asarray(case.model.col_upper, dtype=float).copy()
            column_lower[self.integer_indices] = strategy.integer_values
            column_upper[self.integer_indices] = strategy.integer_values
            self.reduced_models.append(LinearModel(rows, lower, upper, column_lower, column_upper, np.zeros(self.n_variables, dtype=int)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.extracted, name)

    def decode(self, strategy_index: int, cost: np.ndarray, time_limit: float | None = None) -> dict[str, Any]:
        from adapter import solve
        started = perf_counter()
        self.decode_statistics["calls"] += 1
        cost = np.asarray(cost, dtype=float)
        if not np.all(np.isfinite(cost)):
            raise ValueError("A reduced LP needs a finite objective vector")
        strategy = self.strategies[int(strategy_index)]
        decoded = self.extracted.decode(strategy_index, cost)
        x = decoded.get("x")
        active_residual = decoded.get("active_equality_residual", float("inf"))
        stationarity_residual = decoded.get("stationarity_residual", float("inf"))
        stationarity_tolerance = self.config.stationarity_relative_tolerance * max(1.0, float(np.abs(cost[self.continuous_indices]).max(initial=0.0)))
        certificate = (x is not None and np.all(np.isfinite(x)) and
            active_residual <= self.config.feasibility_tolerance and
            np.isfinite(stationarity_residual) and stationarity_residual <= stationarity_tolerance)
        if certificate:
            reduced_model = self.reduced_models[int(strategy_index)]
            bound_violation = max(float(np.maximum(reduced_model.col_lower - x, 0.0).max(initial=0.0)),
                float(np.maximum(x - reduced_model.col_upper, 0.0).max(initial=0.0)))
            certificate = bound_violation <= self.config.feasibility_tolerance
        negative_dual = 0.0
        objective_gap_bound = 0.0 if not len(self.continuous_indices) else float("inf")
        if certificate and strategy.factor is not None:
            dual = strategy.factor.solve(-cost[self.continuous_indices], trans="T")
            # Permanent equality multipliers are unrestricted. All multipliers
            # on active Gx<=h rows must be nonnegative; unused rows have zero.
            inequality_rows = strategy.selected_equation_rows >= self.E.shape[0]
            negative_dual = float(np.maximum(-dual[inequality_rows], 0.0).max(initial=0.0))
            # Negative multipliers can create an arbitrarily large objective
            # error on a large box, even when numerically small. Do not certify
            # by an absolute sign tolerance: fall back to an actual LP instead.
            certificate = np.all(np.isfinite(dual)) and negative_dual == 0.0
            if certificate:
                # A small stationarity residual alone is insufficient on a
                # large box. Bound the objective gap using the dual Lagrangian
                # minimized over the permanent variable domain.
                residual_gradient = np.asarray(strategy.basis_matrix.T @ dual).ravel() + cost[self.continuous_indices]
                cont_x = x[self.continuous_indices]
                reduced_model = self.reduced_models[int(strategy_index)]
                lower_distance = np.maximum(cont_x - reduced_model.col_lower[self.continuous_indices], 0.0)
                upper_distance = np.maximum(reduced_model.col_upper[self.continuous_indices] - cont_x, 0.0)
                positive, negative = residual_gradient > 0.0, residual_gradient < 0.0
                box_error = float((residual_gradient[positive] * lower_distance[positive]).sum()
                    + (-residual_gradient[negative] * upper_distance[negative]).sum())
                equation_residual = np.asarray(strategy.equations @ x).ravel() - strategy.equations_rhs
                objective_gap_bound = max(0.0, box_error - float(dual @ equation_residual[strategy.selected_equation_rows]))
                gap_tolerance = self.config.certificate_gap_absolute_tolerance + self.config.certificate_gap_relative_tolerance * abs(float(cost @ x))
                certificate = np.isfinite(objective_gap_bound) and objective_gap_bound <= gap_tolerance
        if certificate:
            self.decode_statistics["certified_kkt_fast_paths"] += 1
            decoded["status"] = "reduced_lp_kkt_optimal" if decoded["valid"] else "reduced_lp_kkt_optimal_original_infeasible"
            decoded["reduced_optimal"] = True
            decoded["negative_dual"] = negative_dual
            decoded["decoder"] = "dual_feasible_cached_kkt"
        else:
            self.decode_statistics["reduced_lp_solves"] += 1
            remaining = self.config.reduced_lp_time_limit
            if time_limit is not None:
                remaining = min(remaining, max(0.0, time_limit - (perf_counter() - started)))
            solved = solve(self.reduced_models[int(strategy_index)], cost, seed=int(self.case.seed),
                time_limit=remaining, relax_integrality=True)
            reduced_optimal = bool(solved.optimal and solved.valid and solved.x is not None)
            if reduced_optimal:
                original = self.extracted.check_original(solved.x)
                decoded = {"x": solved.x, "objective": float(cost @ solved.x), "valid": original["valid"],
                    "status": "reduced_lp_highs_optimal" if original["valid"] else "reduced_lp_highs_optimal_original_infeasible",
                    "violation": original["violation"], "integer_residual": original["integer_residual"],
                    "reduced_optimal": True, "decoder": "reduced_inequality_lp", "reduced_solver_status": solved.status}
            else:
                self.decode_statistics["failed_reduced_lp_solves"] += 1
                decoded = {"x": None, "objective": float("inf"), "valid": False,
                    "status": "reduced_lp_" + solved.status.lower().replace(" ", "_"),
                    "violation": float("inf"), "integer_residual": float("inf"),
                    "reduced_optimal": False, "decoder": "reduced_inequality_lp", "reduced_solver_status": solved.status}
        decoded["strategy_id"] = int(strategy_index)
        decoded["negative_dual"] = negative_dual
        decoded["kkt_objective_gap_bound"] = objective_gap_bound
        decoded["original_certified"] = False
        self.decode_statistics["seconds"] += perf_counter() - started
        return decoded


def _constraint_normalizer(bank: Any, floor: float) -> float:
    rhs = [np.asarray(bank.h, dtype=float).ravel(), np.asarray(bank.beq, dtype=float).ravel()]
    finite = np.concatenate(rhs)
    finite = finite[np.isfinite(finite)]
    return max(float(np.max(np.abs(finite))) if finite.size else 0.0, floor)


def _infeasibility(bank: Any, x: np.ndarray | None, floor: float) -> float:
    if x is None or not np.all(np.isfinite(x)):
        return float("inf")
    violation = 0.0
    if bank.G.shape[0]:
        violation = max(violation, float(np.maximum(np.asarray(bank.G @ x).ravel() - bank.h, 0.0).max()))
    if bank.E.shape[0]:
        violation = max(violation, float(np.abs(np.asarray(bank.E @ x).ravel() - bank.beq).max()))
    return violation / _constraint_normalizer(bank, floor)


def _strategy_features(bank: Any, selected: np.ndarray) -> np.ndarray:
    count = int(bank.n_canonical_rows)
    features = []
    for index in selected:
        strategy = bank.strategies[int(index)]
        active = np.zeros(count, dtype=np.float32)
        active[np.asarray(strategy.active_rows, dtype=int)] = 1.0
        features.append(np.concatenate((active, np.asarray(strategy.integer_values, dtype=np.float32))))
    return np.stack(features)


def _strategy_signatures(bank: Any) -> list[dict[str, list[int]]]:
    return [{"active_rows": np.asarray(strategy.active_rows, dtype=int).tolist(), "integer_values": np.asarray(strategy.integer_values, dtype=int).tolist()} for strategy in bank.strategies]


def _pair_cache_key(case: Any, bank: Any, config: PreferenceConfig) -> str:
    digest = sha256(_training_hash(case).encode())
    digest.update(_model_hash(case).encode())
    digest.update(json.dumps(asdict(config), sort_keys=True).encode())
    digest.update(json.dumps(_strategy_signatures(bank), sort_keys=True).encode())
    for matrix in (bank.G, bank.E):
        digest.update(str(matrix.shape).encode())
        for array in (matrix.data, matrix.indices, matrix.indptr):
            digest.update(np.ascontiguousarray(array).tobytes())
    for array in (bank.h, bank.beq):
        digest.update(np.ascontiguousarray(array).tobytes())
    for filename in ("preference_baseline.py", "mlopt_baseline.py", "adapter.py", "portable_adapter.py"):
        source_path = Path(__file__).parent / filename
        if source_path.exists():
            digest.update(filename.encode())
            digest.update(source_path.read_bytes())
    return digest.hexdigest()


def _pair_data(case: Any, bank: Any, config: PreferenceConfig, cache_path: Path | None = None) -> dict[str, Any]:
    """All pairs are training parameters times training-extracted strategies."""
    cache_key = _pair_cache_key(case, bank, config)
    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                if str(cache["cache_key"].item()) == cache_key:
                    result = {name: cache[name].copy() for name in ("objectives", "p", "d", "edges", "selected", "rewards", "failed")}
                    result.update(cache_reused=True, cache_key=cache_key, original_decode_statistics=json.loads(str(cache["decode_statistics"].item())))
                    return result
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
            # An interrupted or incomplete cache never supplies training labels.
            pass
    previous_statistics = bank.decode_statistics.copy()
    costs = np.asarray(case.train_costs, dtype=float)
    true_objectives = np.sum(costs * np.asarray(case.train_solutions, dtype=float), axis=1)
    objectives = np.empty((len(costs), len(bank.strategies)))
    infeasibility = np.empty_like(objectives)
    for i, cost in enumerate(costs):
        for j in range(len(bank.strategies)):
            decoded = bank.decode(j, cost)
            objectives[i, j] = decoded["objective"]
            infeasibility[i, j] = _infeasibility(bank, decoded.get("x"), config.denominator_floor)
    denominators = np.maximum(np.abs(true_objectives), config.denominator_floor)
    suboptimality = np.abs(objectives - true_objectives[:, None]) / denominators[:, None]
    suboptimality[~np.isfinite(suboptimality)] = np.inf
    edges = (infeasibility <= config.cover_feasibility_tolerance) & (suboptimality <= config.cover_suboptimality_tolerance)
    selected = greedy_set_cover(edges)
    penalties = infeasibility[:, selected] + suboptimality[:, selected]
    failed = ~np.isfinite(penalties)
    penalties[failed] = config.failed_decode_penalty
    # A floor is necessary because Eq. (10) is otherwise +inf for exact pairs.
    rewards = -np.log(np.maximum(penalties, config.reward_floor))
    statistics = {key: bank.decode_statistics[key] - previous_statistics[key] for key in bank.decode_statistics}
    result = {"objectives": objectives, "p": infeasibility, "d": suboptimality, "edges": edges, "selected": selected, "rewards": rewards, "failed": failed,
        "cache_reused": False, "cache_key": cache_key, "original_decode_statistics": statistics}
    if cache_path is not None:
        partial = cache_path.with_name(cache_path.name + ".writing")
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **{key: result[key] for key in ("objectives", "p", "d", "edges", "selected", "rewards", "failed")},
                cache_key=cache_key, decode_statistics=json.dumps(statistics))
        os.replace(partial, cache_path)
    return result


class PreferenceBaseline:
    """Train-only validation, SetCover, pair attention, and minimum-p inference."""

    def __init__(self, case: Any, config: PreferenceConfig | None = None) -> None:
        self.case = case
        self.config = config or PreferenceConfig(seed=int(case.seed) + 20261009)
        self.bank: Any = None
        self.selected: np.ndarray | None = None
        self.network: PreferenceNetwork | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.top_k: int | None = None
        self.report: dict[str, Any] = {}

    @property
    def fit_seconds(self) -> float:
        return float(self.report.get("fit_seconds", 0.0))

    @property
    def metadata(self) -> dict[str, Any]:
        return self.report

    def _bank(self, case: Any) -> Any:
        return InequalityStrategyBank(case, self.config)

    def _train_network(self, case: Any, bank: Any, pairs: dict[str, np.ndarray], learning_rate: float, lambda1: float, lambda2: float, seed: int) -> tuple[PreferenceNetwork, np.ndarray, np.ndarray, list[dict[str, float]]]:
        _seed_everything(seed)
        features = np.asarray(case.train_features, dtype=float)
        mean, scale = features.mean(axis=0), features.std(axis=0)
        scale[scale < 1e-12] = 1.0
        requested_device = self.config.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for preference training but is unavailable")
        parameters = torch.as_tensor((features - mean) / scale, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(pairs["rewards"], dtype=torch.float32, device=device)
        network = PreferenceNetwork(features.shape[1], _strategy_features(bank, pairs["selected"]), self.config).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_epochs, gamma=self.config.lr_decay)
        generator = torch.Generator().manual_seed(seed)
        history: list[dict[str, float]] = []
        if rewards.shape[1] == 1:
            # There are no adjacent preference pairs and no strategy choice.
            return network.eval().cpu(), mean, scale, history
        network.eval()
        with torch.inference_mode():
            initial_losses = preference_loss(network(parameters), rewards, lambda1, lambda2)
        history.append({"epoch": 0, "total": float(initial_losses[0]), "ranking": float(initial_losses[1]), "difference": float(initial_losses[2]), "gradient_norm_max": 0.0})
        for epoch in range(self.config.epochs):
            network.train()
            permutation = torch.randperm(parameters.shape[0], generator=generator)
            totals = np.zeros(3)
            gradient_norm_max = 0.0
            for indices in permutation.split(self.config.batch_size):
                indices = indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = network(parameters[indices])
                loss, ranking, difference = preference_loss(outputs, rewards[indices], lambda1, lambda2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Preference loss is nonfinite; no model or result should be accepted")
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(network.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=True)
                gradient_norm_max = max(gradient_norm_max, float(gradient_norm))
                optimizer.step()
                totals += np.array([loss.item(), ranking.item(), difference.item()]) * len(indices)
            scheduler.step()
            totals /= parameters.shape[0]
            history.append({"epoch": epoch + 1, "total": float(totals[0]), "ranking": float(totals[1]), "difference": float(totals[2]), "gradient_norm_max": gradient_norm_max})
        return network.eval().cpu(), mean, scale, history

    def _train_final_network(self, case: Any, bank: Any, pairs: dict[str, np.ndarray],
            learning_rate: float, lambda1: float, lambda2: float, seed: int) -> tuple[Any, ...]:
        """Final refit with actual post-update full-objective state selection.

        Development fitting above is unchanged. Added evaluations are in eval
        mode and use no random operation. Selection uses the same objective
        and frozen final labels, before any CPU readiness or test evaluation.
        """
        start = perf_counter()
        _seed_everything(seed)
        features = np.asarray(case.train_features, dtype=float)
        mean, scale = features.mean(axis=0), features.std(axis=0)
        scale[scale < 1e-12] = 1.0
        requested_device = self.config.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for preference training but is unavailable")
        parameters = torch.as_tensor((features - mean) / scale, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(pairs["rewards"], dtype=torch.float32, device=device)
        network = PreferenceNetwork(features.shape[1], _strategy_features(bank, pairs["selected"]), self.config).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_epochs, gamma=self.config.lr_decay)
        generator = torch.Generator().manual_seed(seed)
        history, post_history = [], []
        network.eval()
        with torch.inference_mode():
            initial_losses = preference_loss(network(parameters), rewards, lambda1, lambda2)
        initial = {"epoch": 0, "total": float(initial_losses[0]), "ranking": float(initial_losses[1]),
                   "difference": float(initial_losses[2]), "gradient_norm_max": 0.0}
        if rewards.shape[1] > 1:
            history.append(initial)
        initialization_seconds = perf_counter() - start
        optimizer_seconds = evaluation_seconds = snapshot_seconds = 0.0
        best_loss, best_epoch, best_state = float("inf"), None, None
        for epoch in range(self.config.epochs if rewards.shape[1] > 1 else 0):
            update_start = perf_counter()
            network.train()
            permutation = torch.randperm(parameters.shape[0], generator=generator)
            totals = np.zeros(3)
            gradient_norm_max = 0.0
            for indices in permutation.split(self.config.batch_size):
                indices = indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = network(parameters[indices])
                loss, ranking, difference = preference_loss(outputs, rewards[indices], lambda1, lambda2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Preference loss is nonfinite; no model or result should be accepted")
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(network.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=True)
                gradient_norm_max = max(gradient_norm_max, float(gradient_norm))
                optimizer.step()
                totals += np.array([loss.item(), ranking.item(), difference.item()]) * len(indices)
            scheduler.step()
            totals /= parameters.shape[0]
            history.append({"epoch": epoch + 1, "total": float(totals[0]), "ranking": float(totals[1]),
                            "difference": float(totals[2]), "gradient_norm_max": gradient_norm_max})
            optimizer_seconds += perf_counter() - update_start
            evaluation_start = perf_counter()
            network.eval()
            with torch.inference_mode():
                values = preference_loss(network(parameters), rewards, lambda1, lambda2)
            post = {"epoch": epoch + 1, "total": float(values[0]), "ranking": float(values[1]),
                    "difference": float(values[2]), "next_learning_rate": float(optimizer.param_groups[0]["lr"])}
            if not all(np.isfinite(post[key]) for key in ("total", "ranking", "difference")):
                raise FloatingPointError("Final preference post-update objective is nonfinite")
            post_history.append(post)
            evaluation_seconds += perf_counter() - evaluation_start
            if post["total"] < best_loss:
                snapshot_start = perf_counter()
                best_loss, best_epoch = post["total"], epoch + 1
                best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
                snapshot_seconds += perf_counter() - snapshot_start
        if best_state is None:
            # No preference choice, or a deliberately zero-epoch diagnostic.
            best_loss, best_epoch = initial["total"], 0
            snapshot_start = perf_counter()
            best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
            snapshot_seconds += perf_counter() - snapshot_start
        transfer_start = perf_counter()
        network.eval().cpu()
        transfer_seconds = perf_counter() - transfer_start
        summary = {"learning_rate": learning_rate, "lambda1": lambda1, "lambda2": lambda2,
            "seed": seed, "epochs": self.config.epochs if rewards.shape[1] > 1 else 0,
            "batch_size": self.config.batch_size, "lr_decay": self.config.lr_decay,
            "lr_decay_epochs": self.config.lr_decay_epochs, "training_count": len(features),
            "strategy_count": int(rewards.shape[1]), "best_epoch": best_epoch,
            "best_full_training_objective": best_loss,
            "best_full_training_losses": post_history[best_epoch - 1] if best_epoch else initial,
            "endpoint_full_training_losses": post_history[-1] if post_history else initial,
            "objective_device": device.type, "objective_evaluation": "full_training_batch_eval_mode",
            "best_selection_scope": "post_update_epochs_only" if post_history else "initialization_without_updates",
            "history_semantics": "Epoch0 full eval; subsequent mean minibatch losses before their optimizer updates",
            "post_history_semantics": "Full training batch in original order, eval mode, after optimizer updates",
            "post_update_history": post_history,
            "timing_seconds": {"initialization": initialization_seconds, "optimizer_updates": optimizer_seconds,
                "post_update_full_evaluations": evaluation_seconds, "best_state_copies": snapshot_seconds,
                "endpoint_cpu_transfer": transfer_seconds, "fit_with_trace": perf_counter() - start}}
        return network, mean, scale, history, best_state, summary

    def _training_sanity(self, case: Any, bank: Any, pairs: dict[str, Any], network: PreferenceNetwork,
            mean: np.ndarray, scale: np.ndarray, lambda1: float, lambda2: float,
            history: list[dict[str, float]], top_k: int = 1) -> dict[str, Any]:
        """Predeclared fit checks using training pair labels, never test data.

        Readiness uses finite scores, score spread and improvements in the
        unchanged preference objective. Full adjacent-pair accuracy, recovery,
        oracle availability and cost dependence remain diagnostics. The legacy
        0.55 accuracy reference is local, not an author-native requirement.
        """
        features = np.asarray(case.train_features, dtype=float)
        parameters = torch.as_tensor((features - mean) / scale, dtype=torch.float32)
        network.eval().cpu()
        with torch.inference_mode():
            # Single-instance inference matches the later CPU timing path.
            scores = np.concatenate([network(row[None, :]).numpy() for row in parameters], axis=0)
        rewards = np.asarray(pairs["rewards"], dtype=float)
        selected = np.asarray(pairs["selected"], dtype=int)
        n, m = scores.shape
        finite = bool(np.all(np.isfinite(scores)) and np.all(np.isfinite(rewards)))
        order = np.argsort(-rewards, axis=1, kind="stable")
        target = np.take_along_axis(rewards, order, axis=1)
        prediction = np.take_along_axis(scores.astype(float), order, axis=1)
        true_difference = target[:, :-1] - target[:, 1:]
        predicted_difference = prediction[:, :-1] - prediction[:, 1:]
        non_ties = true_difference > self.config.sanity_true_difference_tolerance
        informative_pairs = int(non_ties.sum())
        accuracy = float((predicted_difference[non_ties] > 0.0).mean()) if informative_pairs else 1.0
        spread = np.ptp(scores.astype(float), axis=1)
        if m > 1 and finite:
            with torch.inference_mode():
                losses = preference_loss(torch.as_tensor(scores), torch.as_tensor(rewards, dtype=torch.float32), lambda1, lambda2)
            total, ranking, difference = [float(value) for value in losses]
            constant_difference = float(np.square(true_difference).mean())
            constant_total = lambda1 * float(np.log(2.0)) + lambda2 * constant_difference
        else:
            total = ranking = difference = constant_difference = constant_total = 0.0
        initial_total = float(history[0]["total"]) if history else 0.0
        relative_improvement = (constant_total - total) / max(abs(constant_total), self.config.denominator_floor)
        relative_initial_improvement = (initial_total - total) / max(abs(initial_total), self.config.denominator_floor)
        maximum_spread = float(spread.max(initial=0.0))
        reasons = []
        if not finite:
            reasons.append("nonfinite_training_scores_or_rewards")
        if informative_pairs:
            if maximum_spread < self.config.sanity_minimum_score_spread:
                reasons.append("collapsed_strategy_scores")
            if relative_improvement < self.config.sanity_minimum_relative_loss_improvement:
                reasons.append("no_loss_improvement_over_equal_scores")
            if relative_initial_improvement < self.config.sanity_minimum_relative_loss_improvement:
                reasons.append("no_loss_improvement_over_initial_model")
        objectives = np.asarray(pairs["objectives"], dtype=float)[:, selected]
        p = np.asarray(pairs["p"], dtype=float)[:, selected]
        originals = np.sum(np.asarray(case.train_costs, dtype=float) * np.asarray(case.train_solutions, dtype=float), axis=1)
        tolerances = self.config.recovery_absolute_tolerance + self.config.recovery_relative_tolerance * np.abs(originals)
        feasibility = p * _constraint_normalizer(bank, self.config.denominator_floor) <= self.config.feasibility_tolerance
        strict = feasibility & np.isfinite(objectives) & (np.abs(objectives - originals[:, None]) <= tolerances[:, None])
        prediction_order = np.argsort(-scores, axis=1, kind="stable")
        top_one = prediction_order[:, 0]
        candidate_positions = prediction_order[:, :min(top_k, m)]
        candidate_p = np.take_along_axis(p, candidate_positions, axis=1)
        winner_positions = candidate_positions[np.arange(n), np.argmin(candidate_p, axis=1)]
        top_one_rewards = rewards[np.arange(n), top_one]
        constant_strict = strict.sum(axis=0)
        near_optimal = np.asarray(pairs["edges"], dtype=bool)[:, selected]
        constant_near_optimal = near_optimal.sum(axis=0)
        strict_constant_position = int(np.argmax(constant_strict))
        near_constant_position = int(np.argmax(constant_near_optimal))
        mean_rewards = rewards.mean(axis=0)
        reward_constant_position = int(np.argmax(mean_rewards))
        reward_oracle_mean = float(rewards.max(axis=1).mean())
        best_constant_reward = float(mean_rewards[reward_constant_position])
        return {"passed": not reasons, "failure_reasons": reasons,
            "readiness_policy": "finite_noncollapsed_scores_and_training_loss_improvement_v4",
            "readiness_is_author_native_requirement": False,
            "readiness_thresholds": {"maximum_score_spread": self.config.sanity_minimum_score_spread,
                "relative_loss_improvement_over_equal_and_initial": self.config.sanity_minimum_relative_loss_improvement},
            "scope": "training_only", "training_count": n, "strategy_count": m,
            "informative_adjacent_pairs": informative_pairs, "non_tie_pair_accuracy": accuracy,
            "legacy_adjacent_gate_passed": bool(not informative_pairs or accuracy >= self.config.sanity_minimum_non_tie_accuracy),
            "adjacent_pair_accuracy_reference_threshold": self.config.sanity_minimum_non_tie_accuracy,
            "adjacent_pair_accuracy_reference_is_author_native_requirement": False,
            "adjacent_pair_accuracy_affects_readiness": False,
            "adjacent_pair_accuracy_diagnostic_applicable": bool(informative_pairs),
            "final_total_loss": total, "final_ranking_loss": ranking, "final_difference_loss": difference,
            "equal_score_total_loss": constant_total, "equal_score_ranking_loss": float(np.log(2.0)) if m > 1 else 0.0,
            "equal_score_difference_loss": constant_difference, "initial_total_loss": initial_total,
            "relative_loss_improvement_over_equal_scores": relative_improvement,
            "relative_loss_improvement_over_initial": relative_initial_improvement,
            "maximum_strategy_score_spread": maximum_spread, "median_strategy_score_spread": float(np.median(spread)),
            "training_top1_mean_reward_regret": float((rewards.max(axis=1) - top_one_rewards).mean()),
            "training_top1_mean_reward": float(top_one_rewards.mean()),
            "training_mean_reward_oracle": reward_oracle_mean,
            "training_best_constant_mean_reward": best_constant_reward,
            "training_best_constant_mean_reward_regret": reward_oracle_mean - best_constant_reward,
            "training_best_constant_reward_strategy_id": int(selected[reward_constant_position]),
            "training_top1_mean_reward_gain_over_best_constant": float(top_one_rewards.mean()) - best_constant_reward,
            "training_reward_oracle_gain_over_best_constant": reward_oracle_mean - best_constant_reward,
            "training_top1_distinct_predicted_strategies": int(np.unique(top_one).size),
            "training_best_constant_strict_recovery": int(constant_strict[strict_constant_position]),
            "training_best_constant_strict_strategy_id": int(selected[strict_constant_position]),
            "training_best_constant_near_optimal_recovery": int(constant_near_optimal[near_constant_position]),
            "training_best_constant_near_optimal_strategy_id": int(selected[near_constant_position]),
            "recovery_and_cost_dependence_affect_readiness": False,
            "training_top1_strict_recovery": int(strict[np.arange(n), top_one].sum()),
            "training_cached_eq21_strict_recovery": int(strict[np.arange(n), winner_positions].sum()),
            "training_cached_eq21_top_k": top_k,
            "training_pool_strict_recovery_oracle": int(strict.any(axis=1).sum()),
            "training_pool_near_optimal_coverage": int(pairs["edges"][:, selected].any(axis=1).sum()),
            "training_top1_near_optimal_recovery": int(pairs["edges"][np.arange(n), selected[top_one]].sum()),
            "degenerate_ranking": m == 1 or informative_pairs == 0}

    def _solve_with(self, bank: Any, selected: np.ndarray, network: PreferenceNetwork, mean: np.ndarray, scale: np.ndarray, cost: np.ndarray, features: np.ndarray, top_k: int, total_time_limit: float = 30.0) -> dict[str, Any]:
        start = perf_counter()
        parameters = torch.as_tensor(((np.asarray(features, dtype=float) - mean) / scale)[None, :], dtype=torch.float32)
        with torch.inference_mode():
            scores = network(parameters)[0].numpy()
        # Score ties preserve the deterministic SetCover order.
        ranking = np.argsort(-scores, kind="stable")[:min(top_k, len(selected))]
        candidates = []
        time_limit_exhausted = False
        for position in ranking:
            remaining = total_time_limit - (perf_counter() - start)
            if remaining <= 0.0:
                time_limit_exhausted = True
                break
            strategy_id = int(selected[position])
            decoded = dict(bank.decode(strategy_id, np.asarray(cost, dtype=float), time_limit=remaining))
            decoded["strategy_id"] = strategy_id
            decoded["predicted_reward"] = float(scores[position])
            decoded["normalized_infeasibility"] = _infeasibility(bank, decoded.get("x"), self.config.denominator_floor)
            if perf_counter() - start > total_time_limit:
                # A solver limit excludes Python/model-building overhead.
                # Preserve its diagnostic status but reject any late candidate.
                time_limit_exhausted = True
                decoded["late_solver_status"] = decoded.get("status")
                decoded.update(x=None, objective=float("inf"), valid=False,
                    status="prediction_wall_budget_exceeded", reduced_optimal=False,
                    normalized_infeasibility=float("inf"))
            candidates.append(decoded)
            if time_limit_exhausted:
                break
        # Eq. (21): NOT best objective among feasible candidates. If p ties,
        # Python's stable min chooses the earlier predicted-reward rank.
        winner = min(candidates, key=lambda candidate: candidate["normalized_infeasibility"]) if candidates else {}
        result = {key: winner.get(key) for key in ("x", "objective", "valid", "status", "violation", "strategy_id", "normalized_infeasibility", "reduced_optimal", "decoder", "negative_dual", "kkt_objective_gap_bound", "reduced_solver_status")}
        result["valid"] = bool(winner.get("valid", False))
        result["status"] = winner.get("status", "no_candidates")
        result["seconds"] = perf_counter() - start
        result["candidate_count"] = len(candidates)
        result["planned_candidate_count"] = len(ranking)
        result["time_limit_exhausted"] = time_limit_exhausted or (perf_counter()-start > total_time_limit)
        result["candidates"] = [{key: item.get(key) for key in ("strategy_id", "objective", "valid", "status", "predicted_reward", "normalized_infeasibility", "reduced_optimal", "decoder", "negative_dual", "kkt_objective_gap_bound", "reduced_solver_status", "late_solver_status")} for item in candidates]
        result["reduced_certified"] = bool(winner.get("reduced_optimal", False))
        result["certified"] = False
        result["certificate_scope"] = "reduced_model"
        result["original_certified"] = False
        if result["seconds"] > total_time_limit:
            result["winner_status_before_budget_rejection"] = result["status"]
            result.update(x=None, objective=float("inf"), valid=False,
                status="prediction_wall_budget_exceeded", reduced_optimal=False,
                normalized_infeasibility=float("inf"), reduced_certified=False,
                certified=False, original_certified=False)
        return result

    def fit(self, output_dir: str | Path | None = None) -> "PreferenceBaseline":
        start = perf_counter()
        destination = Path(output_dir) if output_dir is not None else None
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
        development_start = perf_counter()
        n = len(self.case.train_costs)
        if n < 5:
            raise ValueError("Preference baseline requires at least five training costs for held-out tuning")
        rng = np.random.default_rng(self.config.seed)
        permutation = rng.permutation(n)
        validation_count = max(1, int(round(n * self.config.validation_fraction)))
        validation_indices = np.sort(permutation[:validation_count])
        fit_indices = np.sort(permutation[validation_count:])
        development_case = _subset_training_case(self.case, fit_indices)
        development_bank = self._bank(development_case)
        development_pairs = _pair_data(development_case, development_bank, self.config,
            destination / "preference-development-pairs.npz" if destination is not None else None)
        validation_costs = np.asarray(self.case.train_costs)[validation_indices]
        validation_features = np.asarray(self.case.train_features)[validation_indices]
        validation_objectives = np.sum(validation_costs * np.asarray(self.case.train_solutions)[validation_indices], axis=1)
        tuning = []
        for learning_rate in self.config.learning_rates:
            for lambda1, lambda2 in ((a, b) for a in self.config.lambda1_values for b in self.config.lambda2_values):
                network, mean, scale, history = self._train_network(development_case, development_bank, development_pairs, learning_rate, lambda1, lambda2, self.config.seed)
                sanity = self._training_sanity(development_case, development_bank, development_pairs, network, mean, scale, lambda1, lambda2, history)
                if not sanity["passed"]:
                    tuning.append({"learning_rate": learning_rate, "lambda1": lambda1, "lambda2": lambda2,
                        "top_k": None, "sanity": sanity, "eligible_for_selection": False,
                        "final_training_loss": history[-1]["total"] if history else 0.0})
                    continue
                for top_k in self.config.top_k_values:
                    results = [self._solve_with(development_bank, development_pairs["selected"], network, mean, scale, c, f, top_k) for c, f in zip(validation_costs, validation_features)]
                    valid = np.array([r["valid"] for r in results], dtype=bool)
                    objectives = np.array([r["objective"] if r["objective"] is not None else np.nan for r in results], dtype=float)
                    tolerances = self.config.recovery_absolute_tolerance + self.config.recovery_relative_tolerance * np.abs(validation_objectives)
                    recovered = valid & (np.abs(objectives - validation_objectives) <= tolerances)
                    gap = np.abs(objectives - validation_objectives) / np.maximum(np.abs(validation_objectives), self.config.denominator_floor)
                    finite_valid_gap = gap[valid & np.isfinite(gap)]
                    mean_gap = float(finite_valid_gap.mean()) if finite_valid_gap.size else float("inf")
                    tuning.append({"learning_rate": learning_rate, "lambda1": lambda1, "lambda2": lambda2, "top_k": top_k, "validation_recovered": int(recovered.sum()), "validation_valid": int(valid.sum()), "validation_count": validation_count, "validation_mean_relative_gap_valid": mean_gap, "final_training_loss": history[-1]["total"] if history else 0.0, "sanity": sanity, "eligible_for_selection": True})
        eligible = [trial for trial in tuning if trial["eligible_for_selection"]]
        if not eligible:
            self.report = {"method": IMPLEMENTATION_VERSION, "configuration": asdict(self.config),
                "sanity_passed": False, "failure_stage": "all_development_fits_failed_training_sanity",
                "tuning": tuning, "training_hash": _training_hash(self.case), "model_hash": _model_hash(self.case),
                "test_information_used": False, "fit_seconds": perf_counter() - start}
            self.report["sanity_gate"] = {"passed": False, "failure_reasons": ["all_development_fits_failed_training_sanity"], "scope": "training_only"}
            if destination is not None:
                (destination / "preference-training-failed.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")
            raise RuntimeError("All preference development fits failed training sanity; inspect preference-training-failed.json. No benchmark is allowed.")
        best = max(eligible, key=lambda row: (row["validation_recovered"], row["validation_valid"], -row["validation_mean_relative_gap_valid"], -row["top_k"], -row["final_training_loss"]))
        development_selection_seconds = perf_counter() - development_start
        final_bank_start = perf_counter()
        final_training_case = _subset_training_case(self.case, np.arange(n))
        self.bank = self._bank(final_training_case)
        pairs = _pair_data(final_training_case, self.bank, self.config,
            destination / "preference-final-pairs.npz" if destination is not None else None)
        self.selected = pairs["selected"]
        final_bank_preparation_seconds = perf_counter() - final_bank_start
        self.top_k = int(best["top_k"])
        final_optimizer_start = perf_counter()
        final_optimizer_trials, selected_final_trial = [], None
        for learning_rate in self.config.learning_rates:
            trial_start = perf_counter()
            network, mean, scale, trial_history, best_state, trial = self._train_final_network(
                final_training_case, self.bank, pairs, learning_rate, best["lambda1"], best["lambda2"], self.config.seed)
            diagnostic_start = perf_counter()
            trial["endpoint_cpu_sanity"] = self._training_sanity(final_training_case, self.bank, pairs,
                network, mean, scale, best["lambda1"], best["lambda2"], trial_history, top_k=self.top_k)
            trial["timing_seconds"]["endpoint_cpu_diagnostics"] = perf_counter() - diagnostic_start
            restore_start = perf_counter()
            network.load_state_dict(best_state)
            trial["timing_seconds"]["retained_state_restore"] = perf_counter() - restore_start
            diagnostic_start = perf_counter()
            trial["retained_cpu_sanity"] = self._training_sanity(final_training_case, self.bank, pairs,
                network, mean, scale, best["lambda1"], best["lambda2"], trial_history, top_k=self.top_k)
            trial["timing_seconds"]["retained_cpu_diagnostics"] = perf_counter() - diagnostic_start
            trial["retained_objective_vs_rowwise_cpu_loss_delta"] = (
                trial["retained_cpu_sanity"]["final_total_loss"] - trial["best_full_training_objective"])
            trial["history"] = trial_history
            trial["timing_seconds"]["trial_wall"] = perf_counter() - trial_start
            trial["fit_seconds"] = trial["timing_seconds"]["trial_wall"]
            final_optimizer_trials.append(trial)
            # Readiness and recovery never choose the optimizer/state. A failed
            # minimum-objective state fails the final gate, without fallback.
            if selected_final_trial is None or trial["best_full_training_objective"] < selected_final_trial["best_full_training_objective"]:
                selected_final_trial = trial
                self.network, self.mean, self.scale, history = network, mean, scale, trial_history
            del best_state
        final_optimizer_seconds = perf_counter() - final_optimizer_start
        final_sanity = selected_final_trial["retained_cpu_sanity"]
        final_optimizer = {"learning_rate": selected_final_trial["learning_rate"],
            "lambda1": best["lambda1"], "lambda2": best["lambda2"], "top_k": self.top_k,
            "seed": self.config.seed, "retained_epoch": selected_final_trial["best_epoch"],
            "retained_full_training_objective": selected_final_trial["best_full_training_objective"],
            "selection": "minimum full-training objective across retained original-grid states; stable grid-order ties",
            "retained_state_scope": selected_final_trial["best_selection_scope"],
            "scope": "final_training_only", "readiness_used_for_selection": False,
            "recovery_used_for_selection": False, "heldout_used_for_final_learning_rate_selection": False,
            "source_adaptation": True}
        extra_trials = [trial for trial in final_optimizer_trials if trial["learning_rate"] != best["learning_rate"]]
        development_lr_trial = next(trial for trial in final_optimizer_trials if trial["learning_rate"] == best["learning_rate"])
        additional_lr_seconds = sum(trial["timing_seconds"]["trial_wall"] for trial in extra_trials)
        checkpoint_overhead_seconds = sum(development_lr_trial["timing_seconds"][key] for key in
            ("post_update_full_evaluations", "best_state_copies", "retained_state_restore", "retained_cpu_diagnostics"))
        final_optimizer_timing = {"total_wall_seconds": final_optimizer_seconds,
            "original_grid_trial_count": len(final_optimizer_trials), "additional_learning_rate_trial_count": len(extra_trials),
            "additional_learning_rate_trial_wall_seconds": additional_lr_seconds,
            "development_selected_lr_checkpoint_stage_seconds": checkpoint_overhead_seconds,
            "incremental_protocol_stage_seconds": additional_lr_seconds + checkpoint_overhead_seconds,
            "incremental_definition": "Additional LR trials in full plus added post-update evaluation, snapshot, restore and retained-state diagnostics for the development-selected LR; measured stages, not a causal baseline timing comparison"}
        labels = np.asarray(self.bank.labels, dtype=int)
        counts = np.bincount(labels, minlength=len(self.bank.strategies))
        self.report = {"method": "Li2025 preference model reduction (v2 source adaptation)", "implementation_version": IMPLEMENTATION_VERSION, "source": SOURCE, "configuration": asdict(self.config), "selected_hyperparameters": best, "training_hash": _training_hash(self.case), "model_hash": _model_hash(self.case), "training_count": n, "training_only_validation_indices": validation_indices.tolist(), "training_only_fit_indices": fit_indices.tolist(), "development_candidate_strategies": len(development_bank.strategies), "development_setcover_strategies": len(development_pairs["selected"]), "final_candidate_strategies": len(self.bank.strategies), "final_setcover_strategies": len(self.selected), "selected_strategy_ids": self.selected.tolist(), "setcover_uncovered_training_rows": int((~pairs["edges"][:, self.selected].any(axis=1)).sum()), "training_instance_strategy_pairs": int(pairs["objectives"].size), "selected_training_preference_pairs": int(n * max(len(self.selected) - 1, 0)), "failed_selected_pair_decodes": int(pairs["failed"].sum()), "good_turing_missing_mass_estimate": float((counts == 1).sum() / n), "good_turing_exploration_stop_used": False, "final_training_epochs": self.config.epochs if len(self.selected) > 1 else 0, "tuning": tuning, "history": history, "fit_seconds": perf_counter() - start, "test_information_used": False, "sanity_passed": final_sanity["passed"], "training_sanity": final_sanity, "selection_rule": "minimum normalized original-constraint violation among predicted top-k, Eq. (21)", "b_normalization": "infinity norm of finite canonical constraint RHS, including equality RHS", "decode": "selected inequalities plus permanent original variable bounds and all integer values; dual-certified cached KKT or actual HiGHS reduced LP", "adaptations": ["permanent_original_variable_bounds_unverified_author_domain", "residual_layernorm_token_mlp_attention", "pair_mean_loss_normalization", "independent_lambda2_grid", "uniform_1200_epochs_batch128_decay_every40_and_gradient_clipping", "local_finite_spread_loss_readiness_with_adjacent_accuracy_diagnostic_only", "fixed_120_training_budget_no_adaptive_good_turing"], "training_device": self.config.device, "inference_device": "cpu", "native_full_integer_strategy": True, "matched_dimension_variant": False, "development_decode_statistics": development_bank.decode_statistics.copy(), "final_decode_statistics": self.bank.decode_statistics.copy()}
        self.report["pair_cache"] = {"development": {key: development_pairs[key] for key in ("cache_reused", "cache_key", "original_decode_statistics")},
            "final": {key: pairs[key] for key in ("cache_reused", "cache_key", "original_decode_statistics")}}
        self.report["sanity_gate"] = final_sanity
        self.report["development_selected_hyperparameters"] = best
        self.report["actual_final_optimizer"] = final_optimizer
        self.report["final_selected_lr"] = final_optimizer["learning_rate"]
        self.report["selected_state_loss"] = final_sanity["final_total_loss"]
        self.report["selected_state_loss_semantics"] = "Fresh rowwise CPU evaluation after restoring the selected actual retained state"
        self.report["history_semantics"] = "Complete terminal pre-update history of the selected LR; its last row is not the retained-state loss"
        self.report["final_training_epochs_semantics"] = "Epochs per nondegenerate LR fit; every configured epoch is executed for each original-grid LR"
        self.report["total_final_training_epochs"] = sum(trial["epochs"] for trial in final_optimizer_trials)
        self.report["final_optimizer_trials"] = final_optimizer_trials
        self.report["final_optimizer_timing"] = final_optimizer_timing
        self.report["development_selection_seconds"] = development_selection_seconds
        self.report["final_bank_preparation_seconds"] = final_bank_preparation_seconds
        self.report["final_optimizer_search_seconds"] = final_optimizer_seconds
        self.report["final_training_optimizer_steps"] = sum(
            trial["epochs"] * int(np.ceil(n / self.config.batch_size)) for trial in final_optimizer_trials)
        self.report["adaptations"].append("uniform_final_original_learning_rate_grid_and_actual_best_post_update_training_state")
        self.report["requested_training_device"] = self.config.device
        self.report["training_device"] = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else ("cpu" if self.config.device == "auto" else self.config.device)
        if output_dir is not None:
            self.save(output_dir)
            np.savez_compressed(Path(output_dir) / "preference-pair-data.npz", p=pairs["p"], d=pairs["d"], edges=pairs["edges"], selected=self.selected, rewards=pairs["rewards"], mean=self.mean, scale=self.scale)
        if not final_sanity["passed"]:
            raise RuntimeError("Final preference fit failed training sanity; saved diagnostic checkpoint is ineligible for benchmarking.")
        return self

    def save(self, output_dir: str | Path) -> None:
        if self.network is None or self.selected is None:
            raise RuntimeError("No fitted preference model to save")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "preference-training.json").write_text(json.dumps(self.report, indent=2, ensure_ascii=True), encoding="utf-8")
        with (destination / "preference-network.pt").open("wb") as handle:
            torch.save({"state_dict": {name: tensor.detach().cpu() for name, tensor in self.network.state_dict().items()}, "configuration": asdict(self.config), "implementation_version": IMPLEMENTATION_VERSION, "sanity_passed": bool(self.report.get("sanity_passed", False)), "pair_cache_key": self.report["pair_cache"]["final"]["cache_key"], "training_hash": _training_hash(self.case), "model_hash": _model_hash(self.case), "selected_strategy_ids": self.selected.tolist(), "strategy_signatures": _strategy_signatures(self.bank), "mean": self.mean, "scale": self.scale, "top_k": self.top_k}, handle)

    def solve(self, cost: np.ndarray, features: np.ndarray, top_k: int | None = None, total_time_limit: float = 30.0) -> dict[str, Any]:
        if self.network is None or self.selected is None:
            raise RuntimeError("Call fit() before solve()")
        if not self.report.get("sanity_passed", False):
            raise RuntimeError("A failed or unchecked preference fit cannot be benchmarked")
        return self._solve_with(self.bank, self.selected, self.network, self.mean, self.scale, cost, features, int(top_k or self.top_k), total_time_limit=total_time_limit)

    def predict(self, features: np.ndarray, cost: np.ndarray) -> dict[str, Any]:
        return self.solve(cost, features)


def fit_case(case: Any, output_dir: str | Path | None = None, config: PreferenceConfig | None = None) -> PreferenceBaseline:
    return PreferenceBaseline(case, config=config).fit(output_dir=output_dir)


def load_fitted(case: Any, output_dir: str | Path) -> PreferenceBaseline:
    """Restore a trusted local checkpoint; rebuild strategy LU caches offline.

    Source training data must match exactly. This function never trains or
    selects parameters and must complete before timed inference starts.
    """
    destination = Path(output_dir)
    with (destination / "preference-network.pt").open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
    if checkpoint.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("Checkpoint was not produced by this v2 implementation")
    if not checkpoint.get("sanity_passed", False):
        raise ValueError("Checkpoint failed or did not record the training sanity gate")
    fitted = PreferenceBaseline(case, config=PreferenceConfig(**checkpoint["configuration"]))
    _seed_everything(fitted.config.seed)
    if _training_hash(case) != checkpoint["training_hash"]:
        raise ValueError("Checkpoint training features/costs/solutions differ from supplied data")
    if _model_hash(case) != checkpoint["model_hash"]:
        raise ValueError("Checkpoint original model or variable domain differs from supplied data")
    fitted.bank = fitted._bank(_subset_training_case(case, np.arange(len(case.train_costs))))
    if _strategy_signatures(fitted.bank) != checkpoint["strategy_signatures"]:
        raise ValueError("Checkpoint strategy bank differs from supplied training data")
    if _pair_cache_key(case, fitted.bank, fitted.config) != checkpoint["pair_cache_key"]:
        raise ValueError("Checkpoint source/dependency/configuration pin differs from this implementation")
    fitted.selected = np.asarray(checkpoint["selected_strategy_ids"], dtype=int)
    fitted.mean = np.asarray(checkpoint["mean"], dtype=float)
    fitted.scale = np.asarray(checkpoint["scale"], dtype=float)
    fitted.top_k = int(checkpoint["top_k"])
    fitted.network = PreferenceNetwork(len(fitted.mean), _strategy_features(fitted.bank, fitted.selected), fitted.config)
    fitted.network.load_state_dict(checkpoint["state_dict"])
    fitted.network.eval()
    fitted.report = json.loads((destination / "preference-training.json").read_text(encoding="utf-8"))
    return fitted
