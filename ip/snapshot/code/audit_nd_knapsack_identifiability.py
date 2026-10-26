"""Training-only analytical knapsack audit; no solver or training imports.

Reads only the exported training archive, eight training pools,
the frozen source text, and training-blocked diagnostics. Root LP values are
obtained by the fractional-knapsack ratio theorem, without a solver call.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
CASE = "knapsack_seed101_a0.05"
INDICES = np.asarray([2, 3, 4, 5, 6, 8, 9, 10])


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entropy(r):
    out = np.zeros_like(r, dtype=float)
    middle = (r > 0) & (r < 1)
    out[middle] = -r[middle] * np.log(r[middle]) - (1-r[middle]) * np.log1p(-r[middle])
    return out


def _layer_norm(v):
    """Unit affine LayerNorm proxy, with PyTorch's default epsilon.

    This is applied to input vectors, not unknown trained MLP messages. It
    isolates conditioning of the aggregate-and-normalize operation only.
    """
    centered = v - v.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)


def _summary(values):
    return {name: float(value) for name, value in zip(
        ("min", "q25", "median", "q75", "max"), np.quantile(values, [0, .25, .5, .75, 1]))}


def _lp_by_ratio(c, weights, capacity):
    profit = -c
    assert np.all(profit > 0) and np.all(weights > 0)
    ratio = profit / weights
    order = np.argsort(-ratio, kind="stable")
    assert len(np.unique(ratio)) == len(ratio), "LP order ties need a separate audit"
    used = 0.
    x = np.zeros(len(weights), dtype=float)
    for j in order:
        x[j] = min(1., max(0., (capacity-used)/weights[j]))
        used += x[j] * weights[j]
        if used >= capacity:
            break
    fractional = np.flatnonzero((x > 1e-10) & (x < 1-1e-10))
    assert len(fractional) == 1, "this analytical feature construction requires one fractional item"
    y = -float(ratio[fractional[0]])
    rc = c - weights * y
    primal = float(c @ x)
    dual = y * capacity + float(rc[rc < 0].sum())
    assert abs(primal-dual) <= 1e-8
    assert abs(weights@x-capacity) <= 1e-8
    return x, y, rc, fractional, ratio


def _raw_features(a, bounds, c, cost_scale):
    rlo, rup, clo, cup, integer = bounds
    n, m = a.shape[1], a.shape[0]
    features = np.zeros((n+m, 28), dtype=np.float32)
    features[:n, 0] = 1
    features[n:, 1] = 1
    features[:n, 2] = c.astype(np.float32) / cost_scale
    features[:n, 3] = integer != 0
    finite = np.concatenate([v[np.isfinite(v)] for v in (clo, cup, rlo, rup)])
    scale = max(float(np.log1p(abs(finite)).max()), 1.)
    def transform(v):
        out = np.zeros_like(v, dtype=float)
        good = np.isfinite(v)
        out[good] = np.sign(v[good])*np.log1p(abs(v[good]))/scale
        return out
    features[:n, 4] = transform(clo)
    features[:n, 5] = transform(cup)
    features[:n, 6] = np.isfinite(clo)
    features[:n, 7] = np.isfinite(cup)
    features[n:, 8] = transform(rlo)
    features[n:, 9] = transform(rup)
    features[n:, 10] = np.isfinite(rlo)
    features[n:, 11] = np.isfinite(rup)
    degree = np.concatenate((np.diff(a.tocsc().indptr), np.diff(a.indptr)))
    features[:, 12] = np.log1p(degree)/max(float(np.log1p(degree).max()), 1.)
    return features


def main():
    folder = ROOT / "data" / CASE
    source = ROOT / "code" / "neural_diving_baseline.py"
    blocked_path = ROOT / "audit" / "remote-attempt2" / "logs" / "training-blocked.json"
    source_sha = _sha(source)
    with np.load(folder/"training.npz", allow_pickle=False) as data:
        assert all("test" not in key.lower() and "confirm" not in key.lower() for key in data.files)
        names = ("A_data", "A_indices", "A_indptr", "A_shape", "row_lower", "row_upper",
                 "col_lower", "col_upper", "integrality", "train_costs", "train_solutions")
        arrays = {key: data[key].copy() for key in names}
    a = sparse.csr_matrix((arrays["A_data"], arrays["A_indices"], arrays["A_indptr"]),
                         shape=tuple(arrays["A_shape"]))
    assert a.shape == (1, 200), "this audit applies only to one-capacity-row knapsack"
    weights = a.toarray()[0]
    capacity = float(arrays["row_upper"][0])
    costs = np.asarray(arrays["train_costs"], dtype=np.float64)
    optimum_labels = np.rint(arrays["train_solutions"]).astype(int)
    blocked = json.loads(blocked_path.read_text(encoding="utf-8"))
    gates = blocked["diagnostics"]["width_candidates"]
    assert all(np.array_equal(g["indices"], INDICES) for g in gates)
    # Match runner_common.py:78-79 as pure hash arithmetic, without importing it.
    version = "independent-confirmation-v2-20261009"
    seed = int.from_bytes(hashlib.sha256((version+":"+CASE).encode()).digest()[:4],
                          "little") % (2**31-1)
    shuffled = np.random.default_rng(seed).permutation(len(costs))
    fit_indices = np.sort(shuffled[24:])
    assert np.array_equal(fit_indices[:8], INDICES)
    cost_scale = float(np.sqrt(np.mean(costs[fit_indices]**2)))
    bounds = tuple(arrays[k] for k in ("row_lower", "row_upper", "col_lower", "col_upper", "integrality"))
    row_norm = float(np.linalg.norm(weights))
    _, weight_inverse, weight_counts = np.unique(weights, return_inverse=True, return_counts=True)
    repeated_weights = [np.flatnonzero(weight_inverse == group)
                        for group in np.flatnonzero(weight_counts > 1)]
    records, input_features, marginals_all, lp_round_all = [], [], [], []
    for index in INDICES:
        pool_path = folder / "pools" / f"{CASE}_train{int(index):03d}.npz"
        with np.load(pool_path, allow_pickle=False) as pool:
            labels = np.rint(pool["solutions"]).astype(int)
            objective = pool["objectives"].copy()
        energy_weights = np.exp(-(objective-objective.min()))
        energy_weights /= energy_weights.sum()
        marginal = energy_weights @ labels
        assert np.all((marginal >= -1e-12) & (marginal <= 1+1e-12))
        marginal = np.clip(marginal, 0., 1.)  # only summation roundoff
        c = costs[index]
        x, y, rc, fractional, ratio = _lp_by_ratio(c, weights, capacity)
        lp_round = (x >= .5).astype(int)
        expected_error = np.where(lp_round, 1-marginal, marginal)
        optimum_error = lp_round != optimum_labels[index]
        feat = _raw_features(a, bounds, c, cost_scale)
        n = 200
        root = np.zeros((n+1, 15), dtype=float)
        root[:n, 0] = x
        root[:n, 1] = np.abs(x-np.rint(x))
        norm_c = float(np.linalg.norm(c))
        root[:n, 2] = rc/norm_c
        root[:n, 3] = np.abs(x) <= 1e-6
        root[:n, 4] = np.abs(x-1) <= 1e-6
        root[:n, 5] = x <= 1e-10
        root[fractional, 6] = 1
        root[:n, 7] = x >= 1-1e-10
        root[n, 10] = y/(row_norm*norm_c)
        root[n, 13] = 1
        root[n, 14] = float(weights@c)/(row_norm*norm_c)
        feat[:, 13:] = root.astype(np.float32)
        assert np.isfinite(feat).all()
        # Pure input-space proxy: first-layer identity messages, no trained
        # weights. Actual architecture uses a learned MLP before aggregation.
        input64 = feat.astype(float)
        row_message = input64[n]
        variable_aggregate = input64[:n] + weights[:, None] * row_message
        row_aggregate = row_message + np.sum(weights[:, None] * input64[:n], axis=0)
        ln_variable = _layer_norm(variable_aggregate)
        ln_common = _layer_norm(row_message)
        distance_to_common = np.linalg.norm(ln_variable-ln_common, axis=1)
        # Relative own-message scale is before centering; it is descriptive,
        # while the O(1/w) asymptotic statement uses centered vectors.
        own_to_row = np.linalg.norm(input64[:n], axis=1)/(weights*np.linalg.norm(row_message))
        float32_aggregate = feat[:n] + weights.astype(np.float32)[:, None]*feat[n]
        float32_ln = _layer_norm(float32_aggregate)
        float32_error = np.max(abs(float32_ln.astype(float)-ln_variable), axis=1)
        roundoff_distances = cdist(float32_ln, float32_ln)
        np.fill_diagonal(roundoff_distances, np.inf)
        repeated_weight_records = []
        for members in repeated_weights:
            opposite_map = bool(np.any(marginal[members] >= .5) and np.any(marginal[members] < .5))
            repeated_weight_records.append({"weight": float(weights[members[0]]),
                "items": members.tolist(), "weighted_marginals": marginal[members].tolist(),
                "opposite_map_labels": opposite_map,
                "normalized_cost_span": float(np.ptp(feat[members, 2])),
                "normalized_reduced_cost_span": float(np.ptp(root[members, 2]))})
        augmented = np.column_stack((feat[:n], weights.astype(np.float32)))
        # Exact within-instance node/edge collisions are necessary to force
        # indistinguishability in this star; raw features are retained by concat.
        _, inverse, counts = np.unique(augmented, axis=0, return_inverse=True, return_counts=True)
        conflict_groups = []
        for group in np.flatnonzero(counts > 1):
            members = np.flatnonzero(inverse == group)
            if np.ptp(marginal[members]) > 1e-10:
                conflict_groups.append(members.tolist())
        # Exact collision tests above use original coefficients. For near
        # distances, use dimensionless coefficients to avoid unit dominance.
        scaled_augmented = np.column_stack((feat[:n], weights/weights.max()))
        distances = cdist(scaled_augmented, scaled_augmented)
        np.fill_diagonal(distances, np.inf)
        opposite = (marginal[:, None] >= .5) != (marginal[None, :] >= .5)
        conflicting_distances = np.where(opposite, distances, np.inf)
        closest = np.unravel_index(np.argmin(conflicting_distances), conflicting_distances.shape)
        ln_opposite_distances = np.where(opposite, roundoff_distances, np.inf)
        ln_closest = np.unravel_index(np.argmin(ln_opposite_distances), ln_opposite_distances.shape)
        sorted_cost = np.sort(feat[:n, 2])
        spacing_ulps = np.diff(sorted_cost) / np.maximum(
            abs(np.spacing(sorted_cost[:-1])), abs(np.spacing(sorted_cost[1:])))
        records.append({"train_index": int(index), "pool_size": len(labels),
            "effective_pool_size": float(1/(energy_weights@energy_weights)),
            "weighted_marginal_entropy": float(_entropy(marginal).mean()),
            "fractional_item": int(fractional[0]), "fractional_value": float(x[fractional[0]]),
            "lp_round_vs_optimum_count": int(optimum_error.sum()),
            "lp_round_vs_optimum_indices": np.flatnonzero(optimum_error).tolist(),
            "lp_round_weighted_expected_error": float(expected_error.mean()),
            "lp_round_weighted_expected_error_count": float(expected_error.sum()),
            "lp_round_capacity_slack": float(capacity-weights@lp_round),
            "exact_augmented_node_groups": len(counts), "exact_conflicting_groups": conflict_groups,
            "nearest_conflicting_nodes": [int(q) for q in closest],
            "nearest_conflicting_distance_with_dimensionless_edge": float(conflicting_distances[closest]),
            "min_normalized_cost_spacing": float(np.diff(np.sort(feat[:n, 2])).min()),
            "min_normalized_cost_spacing_float32_ulps": float(spacing_ulps.min()),
            "min_nonzero_normalized_reduced_cost": float(np.min(abs(root[:n, 2])[abs(root[:n, 2]) > 1e-12])),
            "signed_row_dual": y, "normalized_row_dual": float(root[n, 10]),
            "repeated_weight_groups": repeated_weight_records,
            "repeated_weight_opposite_map_group_count": sum(r["opposite_map_labels"] for r in repeated_weight_records),
            "identity_message_proxy": {
                "scope": "input identity messages only; not the trained MLP activations",
                "own_to_weighted_row_message_norm_ratio": _summary(own_to_row),
                "layer_norm_distance_to_common_row_vector": _summary(distance_to_common),
                "large_weight_500_plus_node_count": int((weights >= 500).sum()),
                "large_weight_500_plus_max_distance_to_common": float(distance_to_common[weights >= 500].max()),
                "row_aggregate_max_absolute_coordinate": float(abs(row_aggregate).max()),
                "row_aggregate_mean_own_to_neighbors_scale": float(np.linalg.norm(row_message)/np.linalg.norm(row_aggregate-row_message)),
                "float32_vs_float64_layer_norm_max_absolute_error": float(float32_error.max()),
                "nearest_opposite_map_nodes_after_float32_aggregation_layer_norm": [int(q) for q in ln_closest],
                "nearest_opposite_map_layer_norm_distance": float(ln_opposite_distances[ln_closest]),
                "exact_float32_layer_norm_vectors": int(len(np.unique(float32_ln, axis=0)))},
            "optimal_label_differences_away_from_fractional_item":
                np.flatnonzero(optimum_error & ~np.isin(np.arange(n), fractional)).tolist(),
            "disagreement_items": [{"item": int(j), "lp_value": float(x[j]),
                "lp_rounded": int(lp_round[j]), "optimal_label": int(optimum_labels[index, j]),
                "weighted_marginal": float(marginal[j]), "weight": float(weights[j]),
                "ratio_margin": float(ratio[j]+y), "normalized_reduced_cost": float(rc[j]/norm_c)}
                for j in np.flatnonzero(optimum_error)]})
        input_features.append(feat[:n])
        marginals_all.append(marginal)
        lp_round_all.append(lp_round)
    all_features = np.concatenate(input_features)
    all_marginals = np.concatenate(marginals_all)
    report = {"scope": "eight archived training costs; analytical fractional LP only",
        "case_id": CASE, "indices": INDICES.tolist(), "training_archive_has_no_test_arrays": True,
        "solver_calls": 0, "training_calls": 0, "original_or_confirmation_test_accessed": False,
        "actual_capacity_row_count": a.shape[0], "variable_count": a.shape[1],
        "weight_range": [float(weights.min()), float(weights.max())],
        "distinct_weights": len(np.unique(weights)), "capacity": capacity,
        "repeated_weight_group_count": len(repeated_weights),
        "repeated_weight_node_count": int(sum(len(members) for members in repeated_weights)),
        "self_to_edge_coefficient_ratio": _summary(1/weights),
        "constraint_neighbor_weight_sum": float(weights.sum()),
        "formal_runner_seed": seed, "selection_cost_scale": cost_scale,
        "root_feature_source": "analytical ratio-order solution; original HiGHS cache not stored locally",
        "feature_count": all_features.shape[1], "source_sha256": source_sha,
        "training_archive_sha256": _sha(folder/"training.npz"),
        "gate_log_sha256": _sha(blocked_path),
        "marginal_entropy_floor": float(_entropy(all_marginals).mean()),
        "lp_round_weighted_expected_map_bit_error": float(np.mean([
            np.where(pred, 1-r, r).mean() for pred, r in zip(lp_round_all, marginals_all)])),
        "pooled_fractional_item_count": len(records), "per_cost": records,
        "lp_round_vs_archived_optimum_total_bits": sum(r["lp_round_vs_optimum_count"] for r in records),
        "lp_round_vs_archived_optimum_bits_away_from_fractional_item": sum(
            len(r["optimal_label_differences_away_from_fractional_item"]) for r in records),
        "gate": [{"width": g["width"], "best_step": g["best_step"],
                  "best_excess_nll": g["after"]["excess_nll"],
                  "best_expected_map_bit_error": g["after"]["expected_map_bit_error"],
                  "late_curve": [{k: point.get(k) for k in ("step", "learning_rate", "excess_nll", "gradient_norm_before_clip")}
                                 for point in g["curve"][-6:]]} for g in gates],
        "interpretation": {"entropy_floor": "valid unrestricted Bernoulli lower bound; not a guarantee for finite shared GCN",
            "lp_rounding": "near the gate's bit-error scale, but neither an exact numerical match nor proof that ND predicts these same errors; 18 of 19 optimal-label disagreements are on integral LP variables",
            "exact_symmetry": "no within-cost input-plus-edge collision with conflicting weighted labels was found",
            "layer_norm": "large star edge weights can suppress learned own-node messages by O(1/w), but raw/root inputs remain in every concatenated embedding",
            "coefficient_fidelity": "raw coefficient aggregation followed by LayerNorm and concat is the audited Nair architecture; this audit identifies conditioning, not a mathematical implementation bug",
            "root_dual_fidelity": "row dual divided by row norm and objective norm matches the audited Gasse official feature convention; its scale is diagnostic, not an implementation bug",
            "precision": "positive layer scaling is mostly removed by LayerNorm; this is conditioning, not proof of float32 information loss",
            "near_symmetry_limit": "actual trained message activations/checkpoints are unavailable; numerical collapse cannot be measured from raw graph alone"}}
    assert _sha(source) == source_sha
    output = ROOT / "analysis" / "nd-knapsack-identifiability-training-audit-20261010.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("actual_capacity_row_count", "weight_range", "marginal_entropy_floor",
        "lp_round_weighted_expected_map_bit_error", "gate", "source_sha256")}))


if __name__ == "__main__":
    main()
