"""Paper-based MLOPT for frozen, cost-parametric MILPs (P = 0).

This is a bounded-label-budget implementation, not the author's package or its
Good--Turing sampling run.  See docs/mlopt-method.md for the source mapping.
No test cost, test solution, or test optimal value is read by ``fit``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from time import perf_counter
from types import SimpleNamespace
from typing import Any
import json
import math
import pickle

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits


AUTHOR_COMMIT = "48cb9f8b004b648d0c6bbfb586623a6696ebbcca"
PAPER = "https://arxiv.org/html/1907.02206v4"


@dataclass
class Config:
    active_tolerance: float = 1e-7
    feasibility_tolerance: float = 1e-6
    rank_tolerance: float = 1e-10
    pruning_alpha: float = 0.05
    pruning_suboptimality: float = 1e-4
    pruning_max_iterations: int = 20
    n_best: int = 10
    validation_fraction: float = 0.2
    hidden_candidates: tuple = ((32, 32, 32), (64, 64, 64))
    learning_rates: tuple = (1e-3, 1e-2)
    epoch_candidates: tuple = (50, 150)
    dropout: float = 0.1
    batch_size: int = 32
    seed: int = 0
    device: str = "auto"


def _config(config: Any = None) -> Config:
    if config is None:
        return Config()
    if isinstance(config, Config):
        return config
    if isinstance(config, dict):
        unknown = set(config) - set(Config.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown MLOPT configuration keys: {sorted(unknown)}")
        return Config(**config)
    raise TypeError("config must be a Config or a dictionary")


@dataclass
class Strategy:
    id: int
    active_rows: np.ndarray
    integer_values: np.ndarray
    representative_training_index: int
    x: np.ndarray | None = None
    decoder_status: str = "unbuilt"
    decoder_rank: int = 0
    equality_residual: float = float("inf")
    original_violation: float = float("inf")
    integer_residual: float = float("inf")
    selected_equation_rows: np.ndarray | None = None
    rhs: np.ndarray | None = None
    equations: Any = None
    equations_rhs: np.ndarray | None = None
    basis_matrix: Any = None
    factor: Any = None


def canonical_constraints(model):
    """Represent each finite row/variable side, retaining equalities separately.

    Canonical row identities are stable in the supplied model's ordering.
    Bounds are constraints too; no domain constraint is lost from a strategy.
    """
    A = sp.csr_matrix(model.A, dtype=float)
    n = A.shape[1]
    lo, hi = np.asarray(model.row_lower), np.asarray(model.row_upper)
    equal = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    E, beq = A[equal], hi[equal].astype(float)
    blocks, rhs, identities = [], [], []
    upper = np.flatnonzero(np.isfinite(hi) & ~equal)
    lower = np.flatnonzero(np.isfinite(lo) & ~equal)
    if len(upper):
        blocks.append(A[upper]); rhs.append(hi[upper])
        identities.extend(("row_upper", int(i)) for i in upper)
    if len(lower):
        blocks.append(-A[lower]); rhs.append(-lo[lower])
        identities.extend(("row_lower", int(i)) for i in lower)
    identity = sp.eye(n, format="csr")
    clo, chi = np.asarray(model.col_lower), np.asarray(model.col_upper)
    upper = np.flatnonzero(np.isfinite(chi))
    lower = np.flatnonzero(np.isfinite(clo))
    if len(upper):
        blocks.append(identity[upper]); rhs.append(chi[upper])
        identities.extend(("col_upper", int(i)) for i in upper)
    if len(lower):
        blocks.append(-identity[lower]); rhs.append(-clo[lower])
        identities.extend(("col_lower", int(i)) for i in lower)
    G = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, n))
    h = np.concatenate(rhs).astype(float) if rhs else np.empty(0)
    return G, h, E, beq, identities


class StrategyBank:
    """Full strategies with cached, independent-row Eq. (10) decoding.

    Integer-coordinate elimination leaves an equality system on continuous
    coordinates.  Pivoted QR drops only redundant rows.  With P=0, full column
    rank is required: otherwise the KKT decoder is genuinely singular and is
    reported as a failure, rather than repaired with a ridge or an LP solver.
    """
    def __init__(self, case, active_tolerance=1e-7,
                 feasibility_tolerance=1e-6, rank_tolerance=1e-10):
        started = perf_counter()
        self.case = case
        self.model = case.model
        self.G, self.h, self.E, self.beq, self.row_identities = canonical_constraints(self.model)
        self.n_canonical_rows = self.G.shape[0]
        self.n_variables = self.G.shape[1]
        self.integer_indices = np.flatnonzero(np.asarray(self.model.integrality) != 0)
        self.continuous_indices = np.flatnonzero(np.asarray(self.model.integrality) == 0)
        self.active_tolerance = float(active_tolerance)
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.rank_tolerance = float(rank_tolerance)
        self.strategies: list[Strategy] = []
        labels, seen = [], {}
        solutions = np.asarray(case.train_solutions, dtype=float)
        if solutions.ndim != 2 or solutions.shape[1] != self.n_variables:
            raise ValueError("Full original-variable training solutions are required")
        with threadpool_limits(limits=1):
            for i, x in enumerate(solutions):
                validation = self.check_original(x)
                if not validation["valid"]:
                    raise ValueError(f"Invalid original training solution {i}: {validation}")
                active = np.flatnonzero(np.abs(self.G @ x - self.h) <= self.active_tolerance)
                integers = np.rint(x[self.integer_indices]).astype(np.int64)
                key = (tuple(active.tolist()), tuple(integers.tolist()))
                if key not in seen:
                    j = len(self.strategies)
                    seen[key] = j
                    strategy = Strategy(j, active, integers, i)
                    self._build_decoder(strategy)
                    self.strategies.append(strategy)
                labels.append(seen[key])
        self.labels = np.asarray(labels, dtype=np.int64)
        self.fit_seconds = perf_counter() - started
        self.diagnostics = {
            "n_samples": len(solutions), "n_full_strategies": len(self.strategies),
            "n_integer_coordinates": len(self.integer_indices),
            "n_continuous_coordinates": len(self.continuous_indices),
            "n_canonical_inequalities": self.G.shape[0],
            "n_permanent_equalities": self.E.shape[0],
            "decoder_status_counts": {status: sum(s.decoder_status == status for s in self.strategies)
                                      for status in sorted({s.decoder_status for s in self.strategies})},
            "max_representative_reconstruction_error": max(
                (float(np.max(np.abs(s.x-solutions[s.representative_training_index])))
                 if s.x is not None else float("inf")) for s in self.strategies),
            "offline_decoder_seconds": self.fit_seconds,
        }

    def check_original(self, x):
        if x is None:
            return {"valid": False, "violation": float("inf"), "integer_residual": float("inf")}
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_variables,) or not np.all(np.isfinite(x)):
            return {"valid": False, "violation": float("inf"), "integer_residual": float("inf")}
        inequality = float(np.max(self.G @ x - self.h, initial=0.0))
        equality = float(np.max(np.abs(self.E @ x - self.beq), initial=0.0))
        integrality = float(np.max(np.abs(x[self.integer_indices] - np.rint(x[self.integer_indices])), initial=0.0))
        violation = max(inequality, equality, 0.0)
        return {"valid": bool(violation <= self.feasibility_tolerance and integrality <= self.feasibility_tolerance),
                "violation": violation, "integer_residual": integrality}

    def _equations(self, strategy):
        R = sp.vstack([self.E, self.G[strategy.active_rows]], format="csr")
        b = np.concatenate([self.beq, self.h[strategy.active_rows]])
        rhs = b - R[:, self.integer_indices] @ strategy.integer_values
        return R, b, np.asarray(rhs)

    def _build_decoder(self, strategy):
        R, b, rhs = self._equations(strategy)
        strategy.equations, strategy.equations_rhs = R, b
        nc = len(self.continuous_indices)
        x = np.zeros(self.n_variables, dtype=float)
        x[self.integer_indices] = strategy.integer_values
        if nc:
            C = R[:, self.continuous_indices].toarray()
            if not C.shape[0]:
                strategy.decoder_status = "rank_deficient"; return
            _, triangular, pivots = la.qr(C.T, mode="economic", pivoting=True, check_finite=False)
            diagonal = np.abs(np.diag(triangular))
            scale = float(np.max(diagonal, initial=0.0))
            rank = int(np.sum(diagonal > self.rank_tolerance * max(1.0, scale)))
            strategy.decoder_rank = rank
            if rank != nc:
                strategy.decoder_status = "rank_deficient"; return
            independent = np.asarray(pivots[:nc], dtype=np.int64)
            B = sp.csc_matrix(C[independent])
            strategy.basis_matrix = B
            try:
                strategy.factor = splu(B)
                strategy.rhs = rhs[independent].copy()
                strategy.selected_equation_rows = independent
                x[self.continuous_indices] = strategy.factor.solve(strategy.rhs)
            except (RuntimeError, ValueError):
                strategy.decoder_status = "factorization_failure"; return
        residual = float(np.max(np.abs(R @ x - b), initial=0.0))
        strategy.equality_residual = residual
        strategy.x = x
        check = self.check_original(x)
        strategy.original_violation = check["violation"]
        strategy.integer_residual = check["integer_residual"]
        if residual > self.feasibility_tolerance:
            strategy.decoder_status = "inconsistent_active_equalities"
        elif not check["valid"]:
            strategy.decoder_status = "original_constraint_violation"
        else:
            strategy.decoder_status = "decoded"

    def strategy_encoding(self, strategy_index):
        strategy = self.strategies[int(strategy_index)]
        active = np.zeros(self.G.shape[0], dtype=float)
        active[strategy.active_rows] = 1
        return np.concatenate([active, strategy.integer_values.astype(float)])

    def decode(self, strategy_index, cost):
        """Cached-factor decoding and all original constraints checked every call.

        The primal is cost-independent only because this benchmark changes cost
        and fixes the entire integer assignment and a full-rank active system.
        The reduced equality KKT stationarity solve still uses the new cost.
        """
        strategy = self.strategies[int(strategy_index)]
        cost = np.asarray(cost, dtype=float)
        if cost.shape != (self.n_variables,):
            raise ValueError("cost must be the full original objective vector")
        if not np.all(np.isfinite(cost)):
            raise ValueError("cost must contain only finite values")
        if strategy.x is None:
            return {"x": None, "objective": float("inf"), "valid": False,
                    "status": strategy.decoder_status, "violation": float("inf"),
                    "integer_residual": float("inf"), "strategy_id": strategy.id}
        x = np.zeros(self.n_variables, dtype=float)
        x[self.integer_indices] = strategy.integer_values
        stationarity_residual = 0.0
        if strategy.factor is not None:
            x[self.continuous_indices] = strategy.factor.solve(strategy.rhs)
            nu = strategy.factor.solve(-cost[self.continuous_indices], trans="T")
            R, b = strategy.equations, strategy.equations_rhs
            B = strategy.basis_matrix
            stationarity_residual = float(np.max(np.abs(B.T @ nu + cost[self.continuous_indices]), initial=0.0))
            active_residual = float(np.max(np.abs(R @ x - b), initial=0.0))
        else:
            R, b = strategy.equations, strategy.equations_rhs
            active_residual = float(np.max(np.abs(R @ x - b), initial=0.0))
        check = self.check_original(x)
        stationarity_tolerance = self.feasibility_tolerance * (1+float(np.max(np.abs(cost), initial=0.0)))
        if active_residual > self.feasibility_tolerance:
            status = "inconsistent_active_equalities"
        elif not check["valid"]:
            status = "original_constraint_violation"
        elif stationarity_residual > stationarity_tolerance:
            status = "kkt_stationarity_failure"
        else:
            status = "decoded"
        valid = status == "decoded"
        return {"x": x, "objective": float(cost @ x), "valid": valid,
                "status": status,
                "violation": check["violation"], "integer_residual": check["integer_residual"],
                "active_equality_residual": active_residual,
                "stationarity_residual": stationarity_residual,
                "stationarity_tolerance": stationarity_tolerance, "strategy_id": strategy.id}

    def pair_matrices(self):
        """All training-cost/full-strategy pairs, including invalid raw candidates.

        Fixed structure means each cached decoded primal is constant across
        costs, so matrix multiplication is equivalent to repeated Eq. (10)
        primal decoding; it never reads held-out test data.
        """
        n, m = len(self.case.train_costs), len(self.strategies)
        objective = np.full((n, m), np.inf)
        violation = np.full((n, m), np.inf)
        valid = np.zeros((n, m), dtype=bool)
        for j, strategy in enumerate(self.strategies):
            if strategy.x is not None:
                objective[:, j] = np.asarray(self.case.train_costs) @ strategy.x
                violation[:, j] = strategy.original_violation
                valid[:, j] = strategy.decoder_status == "decoded"
        return objective, violation, valid

    def __getstate__(self):
        state = dict(self.__dict__)
        strategies = []
        for strategy in self.strategies:
            copy = Strategy(**{k: v for k, v in strategy.__dict__.items() if k != "factor"})
            strategies.append(copy)
        state["strategies"] = strategies
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        with threadpool_limits(limits=1):
            for strategy in self.strategies:
                if strategy.basis_matrix is not None:
                    strategy.factor = splu(strategy.basis_matrix)


def prune_strategies(bank: StrategyBank, config: Config):
    """Paper Algorithms 2--3: frequency selection with per-sample guarantees."""
    labels = bank.labels
    counts = np.bincount(labels, minlength=len(bank.strategies))
    order = np.argsort(-counts, kind="stable")
    train_costs = np.asarray(bank.case.train_costs)
    train_solutions = np.asarray(bank.case.train_solutions)
    optimal = np.einsum("ij,ij->i", train_costs, train_solutions)
    alpha = config.pruning_alpha
    iterations, decode_count = [], 0
    for iteration in range(config.pruning_max_iterations):
        target = math.ceil((1-alpha)*len(labels))
        selected, covered = [], 0
        for j in order:
            selected.append(int(j)); covered += int(counts[j])
            if covered > target:
                break
        selected = np.asarray(selected, dtype=np.int64)
        inverse = {int(j): k for k, j in enumerate(selected)}
        reassigned = np.asarray([inverse.get(int(j), -1) for j in labels], dtype=np.int64)
        discarded = np.flatnonzero(reassigned < 0)
        accepted, worst_gap = True, 0.0
        for i in discarded:
            best_objective, best_label = np.inf, -1
            for new_label, j in enumerate(selected):
                decoded = bank.decode(j, train_costs[i]); decode_count += 1
                if decoded["valid"] and decoded["objective"] < best_objective:
                    best_objective, best_label = decoded["objective"], new_label
            denominator = abs(float(optimal[i]))
            excess = max(0.0, float(best_objective-optimal[i]))
            gap = excess/denominator if denominator > 0 else (0.0 if excess == 0 else float("inf"))
            worst_gap = max(worst_gap, float(gap))
            # Paper Eq. (7): F_ij <= f_i* + epsilon |f_i*|.  At f_i*=0
            # there is no absolute-error exemption; failing pruning safely
            # keeps more strategies rather than admitting cost degradation.
            acceptable = best_objective <= optimal[i] + config.pruning_suboptimality*denominator
            if best_label < 0 or not acceptable:
                accepted = False; break
            reassigned[i] = best_label
        iterations.append({"iteration": iteration, "alpha": alpha,
                           "selected": len(selected), "discarded_samples": len(discarded),
                           "accepted": accepted, "worst_reassignment_gap": worst_gap})
        if accepted:
            return selected, reassigned, {"iterations": iterations, "candidate_decodes": decode_count,
                                         "n_retained": len(selected), "fallback_to_unpruned": False}
        alpha /= 2
    return np.arange(len(bank.strategies)), labels.copy(), {
        "iterations": iterations, "candidate_decodes": decode_count,
        "n_retained": len(bank.strategies), "fallback_to_unpruned": True}


def _train_network(features, labels, n_classes, hidden, lr, epochs, config, seed):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config.device in ("cpu", "cuda"):
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MLOPT CUDA training requested but CUDA is unavailable")
        device = torch.device(config.device)
    else:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    mean = np.asarray(features, dtype=float).mean(axis=0)
    scale = np.asarray(features, dtype=float).std(axis=0)
    scale[scale < 1e-12] = 1
    x = torch.tensor((np.asarray(features)-mean)/scale, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    layers, dimension = [], x.shape[1]
    for width in hidden:
        layers.extend([nn.Linear(dimension, int(width)), nn.ReLU(), nn.Dropout(config.dropout)])
        dimension = int(width)
    layers.append(nn.Linear(dimension, int(n_classes)))
    network = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(lr))
    loss_function = nn.CrossEntropyLoss()
    network.train()
    for _ in range(int(epochs)):
        indices = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), config.batch_size):
            batch = indices[start:start+config.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(x[batch]), y[batch])
            loss.backward(); optimizer.step()
    network.eval()
    # Dropout is identity at evaluation.  NumPy evaluates the identical dense
    # affine/ReLU network without framework dispatch in the timed inference.
    weights = [(layer.weight.detach().cpu().numpy().copy(), layer.bias.detach().cpu().numpy().copy())
               for layer in network if isinstance(layer, nn.Linear)]
    return weights, mean, scale


class MLOPTModel:
    def __init__(self, bank, selected, weights, mean, scale, config):
        self.bank, self.selected = bank, np.asarray(selected, dtype=np.int64)
        self.weights, self.mean, self.scale = weights, mean, scale
        self.config = config
        self.fit_seconds = 0.0
        self.metadata = {}

    def predict(self, costparams, obj):
        started = perf_counter()
        costparams = np.asarray(costparams, dtype=float)
        if costparams.shape != self.mean.shape or not np.all(np.isfinite(costparams)):
            raise ValueError("costparams must be a finite vector with the fitted feature shape")
        obj = np.asarray(obj, dtype=float)
        if obj.shape != (self.bank.n_variables,) or not np.all(np.isfinite(obj)):
            raise ValueError("obj must be a finite full original objective vector")
        values = ((costparams-self.mean)/self.scale).astype(np.float32)
        for i, (weight, bias) in enumerate(self.weights):
            values = weight @ values + bias
            if i != len(self.weights)-1:
                values = np.maximum(values, 0)
        ranked = np.argsort(-values, kind="stable")[:min(self.config.n_best, len(self.selected))]
        best = None
        candidate_diagnostics = []
        for local_label in ranked:
            decoded = self.bank.decode(self.selected[local_label], obj)
            candidate_diagnostics.append({k: decoded[k] for k in
                ("strategy_id", "objective", "valid", "status", "violation")})
            if decoded["valid"] and (best is None or decoded["objective"] < best["objective"]):
                best = decoded
        if best is None:
            result = {"x": None, "objective": float("inf"), "valid": False,
                      "status": "no_feasible_predicted_strategy", "certified": False,
                      "violation": float("inf")}
        else:
            result = dict(best)
            result["status"] = "feasible_strategy_decoded"
            result["certified"] = False
        result["seconds"] = perf_counter()-started
        result["diagnostics"] = {"n_candidates": len(ranked), "candidates": candidate_diagnostics,
                                 "original_optimality_certified": False,
                                 "full_solver_fallback": False}
        return result

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)


def _training_subset(case, indices):
    # A deliberately test-free object: fitting cannot accidentally see any
    # frozen test costs or their reference objective values.
    return SimpleNamespace(model=case.model,
        train_features=np.asarray(case.train_features)[indices],
        train_costs=np.asarray(case.train_costs)[indices],
        train_solutions=np.asarray(case.train_solutions)[indices],
        case_id=getattr(case, "case_id", "unknown"))


def fit(case, config=None):
    config = _config(config)
    started = perf_counter()
    n = len(case.train_features)
    if n < 2 or len(case.train_costs) != n or len(case.train_solutions) != n:
        raise ValueError("At least two complete original training solutions are required")
    training_arrays = (np.asarray(case.train_features), np.asarray(case.train_costs),
                       np.asarray(case.train_solutions))
    if any(array.ndim != 2 or not np.all(np.isfinite(array)) for array in training_arrays):
        raise ValueError("Training features, costs, and complete solutions must be finite matrices")
    if training_arrays[1].shape != training_arrays[2].shape:
        raise ValueError("Training cost and complete-solution matrices must have the same shape")
    case_seed = int.from_bytes(sha256(str(getattr(case, "case_id", "case")).encode()).digest()[:4], "little")
    seed = (int(config.seed)+case_seed) % (2**31-1)
    order = np.random.default_rng(seed).permutation(n)
    n_validation = max(1, int(round(config.validation_fraction*n)))
    n_validation = min(n-1, n_validation)
    validation_indices, train_indices = order[:n_validation], order[n_validation:]
    validation_case = _training_subset(case, train_indices)
    bank = StrategyBank(validation_case, config.active_tolerance,
                        config.feasibility_tolerance, config.rank_tolerance)
    selected, labels, pruning = prune_strategies(bank, config)
    trial_results, chosen, best_score = [], None, None
    with threadpool_limits(limits=1):
        for hidden in config.hidden_candidates:
            for learning_rate in config.learning_rates:
                for epochs in config.epoch_candidates:
                    network = _train_network(validation_case.train_features, labels, len(selected),
                                             hidden, learning_rate, epochs, config, seed)
                    model = MLOPTModel(bank, selected, *network, config)
                    correct, feasible, normalized_gaps = 0, 0, []
                    for i in validation_indices:
                        prediction = model.predict(case.train_features[i], case.train_costs[i])
                        reference = float(np.asarray(case.train_costs[i]) @ np.asarray(case.train_solutions[i]))
                        if prediction["valid"]:
                            feasible += 1
                            delta = abs(prediction["objective"]-reference)
                            correct += delta <= 1e-6+1e-9*abs(reference)
                            normalized_gaps.append(delta/max(1e-8, abs(reference)))
                        else:
                            normalized_gaps.append(1e6)
                    score = (int(correct), int(feasible), -float(np.mean(normalized_gaps)))
                    trial = {"hidden": list(hidden), "learning_rate": learning_rate,
                             "epochs": epochs, "validation_n": n_validation,
                             "validation_recovered": int(correct), "validation_feasible": int(feasible),
                             "validation_mean_gap_with_failure_penalty": float(np.mean(normalized_gaps))}
                    trial_results.append(trial)
                    if best_score is None or score > best_score:
                        best_score, chosen = score, trial
        final_case = _training_subset(case, np.arange(n))
        final_bank = StrategyBank(final_case, config.active_tolerance,
                                 config.feasibility_tolerance, config.rank_tolerance)
        final_selected, final_labels, final_pruning = prune_strategies(final_bank, config)
        network = _train_network(final_case.train_features, final_labels, len(final_selected),
                                 chosen["hidden"], chosen["learning_rate"], chosen["epochs"], config, seed)
    model = MLOPTModel(final_bank, final_selected, *network, config)
    model.fit_seconds = perf_counter()-started
    model.metadata = {
        "method": "MLOPT-native-v2-paper-fixed120", "paper": PAPER,
        "author_source_commit": AUTHOR_COMMIT, "config": asdict(config),
        "training_cost_budget": n, "adaptive_exploration_stopping_claimed": False,
        "source_of_labels": "frozen full original optimizers",
        "validation_source": "only an internal split of the training costs",
        "training_indices": train_indices.tolist(), "validation_indices": validation_indices.tolist(),
        "validation_trials": trial_results, "selected_hyperparameters": chosen,
        "validation_bank": bank.diagnostics, "validation_pruning": pruning,
        "final_bank": final_bank.diagnostics, "final_pruning": final_pruning,
        "n_retained_strategies": len(final_selected), "fit_seconds": model.fit_seconds,
        "training_device": ("cuda" if config.device == "auto" and __import__("torch").cuda.is_available()
                            else "cpu" if config.device == "auto" else config.device),
        "prediction_device": "cpu",
        "native_compression_dimension_parameter": None,
        "solver_calls_at_prediction": 0, "kkt_regularization": False,
        "decoder": "integer elimination, pivoted-QR independent rows, cached sparse LU; P=0",
    }
    return model


def self_test():
    """Nontrivial feasible, redundant, and singular LP decode checks."""
    model = SimpleNamespace(A=sp.csr_matrix([[1., 1.], [2., 2.]]),
        row_lower=np.array([1., 2.]), row_upper=np.array([1., 2.]),
        col_lower=np.zeros(2), col_upper=np.ones(2), integrality=np.array([1, 0]))
    case = SimpleNamespace(model=model, train_solutions=np.array([[0., 1.], [1., 0.]]),
        train_costs=np.array([[1., 2.], [2., 1.]]), train_features=np.array([[1.], [2.]]),
        case_id="synthetic_decoder_test")
    bank = StrategyBank(case)
    assert all(s.decoder_status == "decoded" for s in bank.strategies)
    for j, strategy in enumerate(bank.strategies):
        decoded = bank.decode(j, np.array([3., 4.]))
        assert decoded["valid"] and np.allclose(decoded["x"], strategy.x)
        assert decoded["stationarity_residual"] < 1e-12
    selected, labels, _ = prune_strategies(bank, Config())
    assert len(selected) == 2 and len(labels) == 2
    unbounded = SimpleNamespace(A=sp.csr_matrix((0, 1)), row_lower=np.empty(0),
        row_upper=np.empty(0), col_lower=np.array([-np.inf]), col_upper=np.array([np.inf]),
        integrality=np.array([0]))
    singular_case = SimpleNamespace(model=unbounded, train_solutions=np.array([[0.]]),
        train_costs=np.array([[0.]]), train_features=np.array([[0.]]))
    singular = StrategyBank(singular_case)
    assert singular.strategies[0].decoder_status == "rank_deficient"
    assert singular.decode(0, np.array([0.]))["x"] is None
    return {"redundant_equalities": "pass", "stationarity": "pass", "singular_without_ridge": "pass"}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
