"""Cost draws for two predetermined UC feature layouts.

The sampler has no access to optimization labels, fitted models or outcomes.
All laws and layouts use the same six latent quantile columns for a given
version, stream, phase and physical seed. Lognormal draws are never clipped,
truncated or redrawn; unrepresentable factors fail explicitly.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral
from typing import Any

import numpy as np
from scipy.special import ndtri


FEATURE_DIMENSION = 6
N_GENERATORS = 6
N_PERIODS = 24
_LAYOUTS = ("first4_whole_day", "first2_three_time_blocks")
_LAWS = {"log_normal_sigma1": ("log_normal", 1.0),
         "log_uniform_A5": ("log_uniform", 5.0)}
_SCENARIOS = {
    f"{law}_{layout}": (distribution, parameter, layout)
    for layout in _LAYOUTS
    for law, (distribution, parameter) in _LAWS.items()
}


def scenario_fields(scenario: str | Mapping[str, Any]) -> dict[str, Any]:
    """Validate one of the four predetermined law/layout combinations.

    Feature indices are zero based. Whole-day features 0--3 affect generators
    0--3 in every period; columns 4--5 equal one. Block features have index
    2*b+g for block b=0,1,2 and generator g=0,1. The three time blocks are
    [0,8), [8,16) and [16,24); generators 2--5 have factor one throughout.
    """
    if isinstance(scenario, str):
        identifier, supplied = scenario, {}
    elif isinstance(scenario, Mapping):
        identifier, supplied = scenario.get("scenario_id"), scenario
    else:
        raise ValueError("scenario must be a predetermined identifier or mapping")
    if not isinstance(identifier, str) or identifier not in _SCENARIOS:
        raise ValueError(f"Unknown predetermined scenario: {identifier!r}")
    distribution, parameter, layout = _SCENARIOS[identifier]
    if layout == "first4_whole_day":
        varying_features, varying_generators = list(range(4)), list(range(4))
        mapping = [{"feature_id": g, "generator_id": g,
                    "period_start": 0, "period_stop_exclusive": N_PERIODS}
                   for g in range(FEATURE_DIMENSION)]
        blocks = [[0, N_PERIODS]]
        description = ("(rho_0,rho_1,rho_2,rho_3,1,1); feature g multiplies "
                       "generator g's four cost components in all 24 periods")
    else:
        varying_features, varying_generators = list(range(6)), [0, 1]
        blocks = [[0, 8], [8, 16], [16, 24]]
        mapping = [{"feature_id": 2 * b + g, "generator_id": g,
                    "period_start": start, "period_stop_exclusive": stop}
                   for b, (start, stop) in enumerate(blocks) for g in (0, 1)]
        description = ("(rho_0,0,rho_0,1,rho_1,0,rho_1,1,rho_2,0,rho_2,1) "
                       "in block-major order, where rho_b,g multiplies generator "
                       "g's four cost components in periods [8b,8b+8); "
                       "generators 2--5 have factor one")
    fields: dict[str, Any] = {
        "scenario_id": identifier,
        "distribution": distribution,
        "feature_dimension": FEATURE_DIMENSION,
        "feature_layout": layout,
        "feature_layout_definition": description,
        "varying_feature_ids": varying_features,
        "fixed_feature_ids": [j for j in range(FEATURE_DIMENSION)
                              if j not in varying_features],
        "varying_generator_ids": varying_generators,
        "fixed_generator_ids": [g for g in range(N_GENERATORS)
                                if g not in varying_generators],
        "fixed_generator_factor": 1.0,
        "fixed_feature_factor": 1.0,
        "feature_index_mapping": mapping,
        "time_blocks": blocks,
        "period_indexing": "zero based; block stop is exclusive",
        "cost_components_sharing_factor": ["fuel", "on", "startup", "shutdown"],
        "median_factor": 1.0,
        "clipping_applied": False,
        "truncation_applied": False,
        "resampling_applied": False,
    }
    if distribution == "log_uniform":
        fields.update({
            "scale_A": parameter,
            "factor_bounds": [1.0 / parameter, parameter],
            "distribution_definition": "rho = exp((2 U - 1) log A), U IID Uniform(0,1)",
            "mean_factor": (parameter - 1.0 / parameter) / (2.0 * math.log(parameter)),
            "normal_transform": None,
        })
        if "sigma" in supplied:
            raise ValueError("A log-uniform scenario cannot specify sigma")
    else:
        fields.update({
            "sigma": parameter,
            "factor_bounds": None,
            "distribution_definition": "rho = exp(sigma Phi^{-1}(U)), U IID Uniform(0,1)",
            "mean_factor": math.exp(0.5 * parameter * parameter),
            "normal_transform": "scipy.special.ndtri on the open 53-bit uniform grid",
        })
        if "scale_A" in supplied:
            raise ValueError("A lognormal scenario cannot specify scale_A")
    # A caller may add protocol prose, but redundant scientific fields must agree.
    for key in fields:
        if key in supplied and supplied[key] != fields[key]:
            raise ValueError(f"{key} disagrees with predetermined scenario {identifier}")
    if "layout" in supplied and supplied["layout"] != layout:
        raise ValueError(f"layout disagrees with predetermined scenario {identifier}")
    return fields


def select_scenarios(protocol: Mapping[str, Any],
                     identifiers: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Return validated protocol scenarios, optionally in explicitly given order.

    Selection is by identifiers only; this module cannot inspect any outcomes.
    A selected identifier must already be present in the supplied protocol.
    """
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    scenarios = protocol.get("scenarios")
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)) or not scenarios:
        raise ValueError("protocol scenarios must be a nonempty sequence")
    validated = [scenario_fields(scenario) for scenario in scenarios]
    by_id = {fields["scenario_id"]: fields for fields in validated}
    if len(by_id) != len(validated):
        raise ValueError("protocol scenarios contain duplicate identifiers")
    if identifiers is None:
        return validated
    if not isinstance(identifiers, Sequence) or isinstance(identifiers, (str, bytes)) or not identifiers:
        raise ValueError("scenario identifiers must be a nonempty sequence")
    if any(not isinstance(identifier, str) for identifier in identifiers):
        raise ValueError("scenario identifiers must be strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("selected scenario identifiers must be unique")
    absent = [identifier for identifier in identifiers if identifier not in by_id]
    if absent:
        raise ValueError(f"Selected scenarios are absent from protocol: {absent!r}")
    return [dict(by_id[identifier]) for identifier in identifiers]


def _array_digest(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype="<f8", order="C")
    header = json.dumps({"shape": list(canonical.shape), "dtype": "<f8"},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\n" + canonical.tobytes(order="C")).hexdigest()


def _stream_for_phase(protocol: Mapping[str, Any], phase: str) -> int:
    if "sampler_streams" in protocol:
        streams = protocol["sampler_streams"]
        if not isinstance(streams, Mapping) or phase not in streams:
            raise ValueError("sampler_streams must contain the requested phase")
        stream = streams[phase]
    else:
        stream = protocol.get("sampler_stream")
    if isinstance(stream, bool) or not isinstance(stream, Integral) or stream < 0:
        raise ValueError("sampler stream must be a nonnegative integer")
    return int(stream)


def sample_features(protocol: Mapping[str, Any], phase: str,
                    physical_seed: int, count: int,
                    scenario: str | Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return float64 factors of shape (count,6) and JSON-compatible provenance.

    The seed hash excludes law, layout and count. Thus layouts/laws share the
    six quantile columns, and smaller training banks are exact prefixes. Phase
    and stream are both hashed. Use sampler_streams.train=2026101207 and
    sampler_streams.test=2026101208; fresh confirmation changes only test to
    2026101209. A single sampler_stream is also accepted for an explicit phase.

    Quantiles are IID PCG64 integers in {1,...,2**53-1}, divided by 2**53.
    Both endpoints are excluded by construction, without clipping or rejection.
    This finite numerical grid approximates the continuous laws stated above.
    """
    fields = scenario_fields(scenario)
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    version = protocol.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("protocol version must be a nonempty string")
    if phase not in {"train", "test"}:
        raise ValueError("phase must be train or test")
    stream = _stream_for_phase(protocol, phase)
    if isinstance(physical_seed, bool) or not isinstance(physical_seed, Integral):
        raise ValueError("physical_seed must be an integer")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    seed_fields = {"version": version, "sampler_stream": stream,
                   "family": "uc", "phase": phase, "physical_seed": int(physical_seed)}
    material = json.dumps(seed_fields, sort_keys=True, separators=(",", ":"))
    seed_digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    seed = int.from_bytes(bytes.fromhex(seed_digest)[:16], "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    uniforms = rng.integers(1, 2**53, size=(int(count), FEATURE_DIMENSION),
                            dtype=np.int64).astype(np.float64) / float(2**53)
    varying = fields["varying_feature_ids"]
    features = np.ones((int(count), FEATURE_DIMENSION), dtype=np.float64)
    selected_uniforms = uniforms[:, varying]
    try:
        with np.errstate(over="raise", invalid="raise", under="raise", divide="raise"):
            if fields["distribution"] == "log_uniform":
                factors = np.exp((2.0 * selected_uniforms - 1.0) * math.log(fields["scale_A"]))
            else:
                factors = np.exp(fields["sigma"] * ndtri(selected_uniforms))
    except FloatingPointError as error:
        raise RuntimeError(f"Unrepresentable cost factor in {fields['scenario_id']}; no resampling") from error
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise RuntimeError(f"Nonfinite or nonpositive cost factor in {fields['scenario_id']}; no resampling")
    features[:, varying] = factors
    metadata = {
        **fields,
        "phase": phase,
        "physical_seed": int(physical_seed),
        "count": int(count),
        "sampler_stream": stream,
        "rng": "numpy.Generator(numpy.PCG64)",
        "uniform_grid": "{1,...,2**53-1}/2**53; no endpoints",
        "uniform_grid_is_discrete": True,
        "seed_material": material,
        "seed_sha256": seed_digest,
        "latent_uniform_sha256": _array_digest(uniforms),
        "feature_sha256": _array_digest(features),
        "sampler_only": True,
        "optimization_labels_accessed": False,
        "model_or_test_outcomes_accessed": False,
    }
    json.dumps(metadata, allow_nan=False)
    return features, metadata


def sample_diversity(train_features: Any, test_features: Any | None = None) -> dict[str, Any]:
    """Count exact row duplicates and overlaps, retaining every IID draw."""
    train = np.asarray(train_features, dtype=float)
    if train.ndim != 2 or train.shape[1] != FEATURE_DIMENSION or not len(train):
        raise ValueError("training features must have nonempty shape (n,6)")
    if not np.isfinite(train).all() or np.any(train <= 0):
        raise ValueError("training features must be finite and positive")
    train_rows = Counter(tuple(float(x) for x in row) for row in train)
    result: dict[str, Any] = {
        "feature_dimension": FEATURE_DIMENSION,
        "training_count": int(len(train)),
        "training_unique_full_vectors": len(train_rows),
        "training_duplicate_rows": int(len(train) - len(train_rows)),
        "training_coordinate_min": train.min(axis=0).tolist(),
        "training_coordinate_max": train.max(axis=0).tolist(),
        "test_count": None,
        "test_unique_full_vectors": None,
        "test_duplicate_rows": None,
        "shared_unique_full_vectors": None,
        "test_rows_overlapping_training": None,
        "train_test_overlap_pairs": None,
        "iid_duplicates_retained": True,
    }
    if test_features is not None:
        test = np.asarray(test_features, dtype=float)
        if test.ndim != 2 or test.shape[1] != FEATURE_DIMENSION or not len(test):
            raise ValueError("test features must have nonempty shape (m,6)")
        if not np.isfinite(test).all() or np.any(test <= 0):
            raise ValueError("test features must be finite and positive")
        test_rows = Counter(tuple(float(x) for x in row) for row in test)
        shared = set(train_rows) & set(test_rows)
        result.update({
            "test_count": int(len(test)),
            "test_unique_full_vectors": len(test_rows),
            "test_duplicate_rows": int(len(test) - len(test_rows)),
            "test_coordinate_min": test.min(axis=0).tolist(),
            "test_coordinate_max": test.max(axis=0).tolist(),
            "shared_unique_full_vectors": len(shared),
            "test_rows_overlapping_training": sum(test_rows[row] for row in shared),
            "train_test_overlap_pairs": sum(train_rows[row] * test_rows[row] for row in shared),
        })
    return result
