"""Tiny synthetic root-LP contracts; never loads a research case or labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from nd_root_lp_features import ROOT_FEATURE_NAMES, RootLPFeatureError, RootLPFeatureExtractor


def _model(a, row_lower, row_upper, col_lower, col_upper, integer=True):
    matrix = sparse.csr_matrix(np.asarray(a, dtype=float).reshape(-1, 1))
    return SimpleNamespace(A=matrix, row_lower=np.asarray(row_lower, dtype=float),
        row_upper=np.asarray(row_upper, dtype=float),
        col_lower=np.asarray([col_lower], dtype=float),
        col_upper=np.asarray([col_upper], dtype=float),
        integrality=np.asarray([int(integer)]))


def run_checks():
    cases = [
        ("fractional_binary_domain_lower_row", _model([1.], [.5], [np.inf], 0., 1.), 1., .5, .5),
        ("upper_row_max_as_min", _model([1.], [-np.inf], [.5], 0., 1.), -2., .5, -1.),
        ("negative_coefficient_lower_row", _model([-1.], [-.5], [np.inf], 0., 1.), -2., .5, -1.),
        ("negative_upper_row_min", _model([-1.], [-2.], [-1.], -5., 5., False), 1., 1., 1.),
        ("column_lower_negative_bound", _model([], [], [], -1., 3., False), 2., -1., -2.),
        ("column_upper_max_as_min", _model([], [], [], -1., 3., False), -2., 3., -6.),
        ("equality_negative_bound", _model([2.], [-.5], [-.5], -1., 1., False), 4., -.25, -1.),
        ("zero_objective", _model([1.], [.5], [1.], 0., 1.), 0., None, 0.),
        ("zero_coefficient_row", _model([0.], [-np.inf], [2.], 0., 1.), 1., 0., 0.),
    ]
    records = []
    for name, model, cost, primal, objective in cases:
        extractor = RootLPFeatureExtractor(model, seed=1909, time_limit=2.)
        features, metadata = extractor.extract(np.asarray([cost]))
        assert metadata["certified"] and metadata["optimal"]
        assert metadata["value_valid"] and metadata["dual_valid"] and metadata["basis_valid"]
        assert features.shape == (1 + model.A.shape[0], len(ROOT_FEATURE_NAMES))
        assert features.dtype == np.float32 and np.isfinite(features).all()
        assert abs(metadata["primal_objective"] - objective) <= 1e-8
        assert abs(metadata["dual_objective"] - objective) <= 1e-8
        if primal is not None:
            assert abs(float(features[0, 0]) * extractor.col_scale[0] - primal) <= 1e-7
        if name == "fractional_binary_domain_lower_row":
            assert features[0, 1] == .5 and metadata["integer_residual_diagnostic"] == .5
            assert features[1, 10] > 0 and features[1, 12] == 1
        elif name == "upper_row_max_as_min":
            assert features[1, 10] < 0 and features[1, 13] == 1
        elif name == "negative_coefficient_lower_row":
            assert features[1, 10] > 0 and features[1, 12] == 1
        elif name == "column_lower_negative_bound":
            assert features[0, 2] > 0 and features[0, 3] == 1
        elif name == "column_upper_max_as_min":
            assert features[0, 2] < 0 and features[0, 4] == 1
        elif name == "zero_coefficient_row":
            assert extractor.row_norm[0] == 1., "zero rows use Gasse's unit norm convention"
            assert features[1, 11] == 2.
        records.append({"name": name, "certified": metadata["certified"],
            "primal_objective": metadata["primal_objective"],
            "dual_objective": metadata["dual_objective"],
            "integer_residual_diagnostic": metadata["integer_residual_diagnostic"],
            "stationarity_residual": metadata["dual_stationarity_residual"]})
    # The original finite bound creates a float64-finite slack that cannot be
    # represented as a finite float32 feature; it must fail before returning.
    overflow = RootLPFeatureExtractor(_model([0.], [-np.inf], [1e40], 0., 1.),
                                      seed=1909, time_limit=2.)
    try:
        overflow.extract(np.asarray([1.]))
    except RootLPFeatureError as error:
        assert "float32 range" in str(error)
        assert error.metadata["certified"]
        records.append({"name": "float32_overflow_rejected", "passed": True,
                        "primal_dual_certificate_valid": True})
    else:
        raise AssertionError("nonfinite float32 features were returned")
    source = Path(__file__).with_name("nd_root_lp_features.py")
    return {"passed": True, "scope": "one-variable synthetic LPs only",
        "synthetic_lp_solves": len(cases) + 1, "real_case_solves": 0,
        "training_calls": 0, "confirmation_test_accessed": False,
        "extractor_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_checks()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
