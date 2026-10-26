"""Paper-based reimplementation of Xavier--Qiu--Ahmed affine predictions.

This is a new implementation of Section 3.3 / Table 1 of arXiv:1902.01697,
not the original authors' experimental code. Binary coordinates are time-major:
``i = t * n_units + g``. Features are supplied by the caller. For an objective-
only synthetic benchmark, pass the COMPLETE observed cost vector, not labels,
solutions, or a subset selected after examining test outcomes.

Choices not completely specified by the paper: StandardScaler fitted within
each training fold; linear SVC(C=1); five stratified folds; pooled OOF metrics;
skip candidates having fewer than five examples in either class; deterministic
fixed-value-first arbitration. No validation/test labels are accepted here.

Main interface::

    model = XQAAffineBaseline(random_state=42).fit(X, y, n_times=T, n_units=G)
    equalities, info = model.predict_equalities(feature)
    for equality in equalities:
        row, rhs = equality.as_row(T * G)

The caller must keep all original variable domains and constraints. Returned
equalities restrict the feasible set and do not certify global optimality.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Equality:
    """One binary-coordinate equation: x[i]=rhs or x[i]-x[j]=0."""

    kind: str
    i: int
    j: int | None = None
    rhs: int = 0

    def as_row(self, n_binary: int) -> tuple[np.ndarray, int]:
        row = np.zeros(n_binary, dtype=float)
        row[self.i] = 1.0
        if self.j is not None:
            row[self.j] = -1.0
        return row, self.rhs

    def satisfied_by(self, y: np.ndarray) -> np.ndarray:
        flat = np.asarray(y)
        if self.j is None:
            return flat[..., self.i] == self.rhs
        return flat[..., self.i] == flat[..., self.j]


@dataclass(frozen=True)
class Thresholds:
    z_fix: float
    z_min: float
    z_max: float
    alpha_recall: float
    alpha_precision: float


TABLE1 = {
    "fix0": Thresholds(1.0, .25, .75, .90, .90),
    "fix1": Thresholds(1.0, .25, .75, .75, .75),
    "next_equal": Thresholds(.975, .025, .975, .50, .75),
}


def precision_threshold(positive_frequency: float, kind: str) -> float:
    alpha = TABLE1[kind].alpha_precision
    return max(positive_frequency, 1.0 - positive_frequency) * (1.0 - alpha) + alpha


def frequency_action(positive_frequency: float, kind: str) -> str:
    """Table 1 action, before checking class counts or OOF scores."""
    threshold = TABLE1[kind]
    if positive_frequency >= threshold.z_fix:
        return "unconditional"
    if positive_frequency < threshold.z_min or positive_frequency > threshold.z_max:
        return "skip_frequency"
    return "classify"


def candidate_dictionary(n_times: int, n_units: int) -> list[Equality]:
    if n_times < 1 or n_units < 1:
        raise ValueError("n_times and n_units must be positive integers")
    result = []
    for t in range(n_times):
        for g in range(n_units):
            i = t * n_units + g
            result.extend((Equality("fix0", i, rhs=0), Equality("fix1", i, rhs=1)))
            if t + 1 < n_times:
                result.append(Equality("next_equal", i, (t + 1) * n_units + g))
    return result


def _validate_training(
    X: Any, y: Any, n_times: int, n_units: int
) -> tuple[np.ndarray, np.ndarray]:
    candidate_dictionary(n_times, n_units)
    features = np.array(X, dtype=float, copy=True)
    binary = np.asarray(y)
    if binary.ndim == 3:
        if binary.shape[1:] != (n_times, n_units):
            raise ValueError("3D y must have shape (n_samples, n_times, n_units)")
        binary = binary.reshape(binary.shape[0], -1)
    if binary.ndim != 2 or binary.shape[1] != n_times * n_units:
        raise ValueError("y must be time-major and have n_times * n_units columns")
    if features.ndim != 2 or features.shape[0] != binary.shape[0] or not len(features):
        raise ValueError("X must be a nonempty 2D array with one row per training solution")
    if not features.shape[1] or not np.isfinite(features).all():
        raise ValueError("Features must be finite and contain at least one column")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("Training decisions must be exactly binary")
    return features, np.array(binary, dtype=np.int8, copy=True)


def _new_classifier(C: float, random_state: int):
    # Lazy import allows use of SparseUnanimousBaseline without scikit-learn.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=C, random_state=random_state))


@dataclass
class CandidateFit:
    equality: Equality
    status: str
    frequency: float
    positives: int
    precision_required: float
    recall_required: float
    oof_precision: float | None = None
    oof_recall: float | None = None
    classifier: Any = None

    def record(self) -> dict[str, Any]:
        return {
            "equality": asdict(self.equality),
            "status": self.status,
            "frequency": self.frequency,
            "positives": self.positives,
            "precision_required": self.precision_required,
            "recall_required": self.recall_required,
            "oof_precision": self.oof_precision,
            "oof_recall": self.oof_recall,
        }


def _arbitrate(proposed: list[CandidateFit]) -> tuple[list[Equality], dict[str, int]]:
    """At most one equation originating from each commitment variable.

    Prefer fixed values over equality to the next period. If both fixed values
    are predicted, prefer higher OOF positive precision, then higher positive
    frequency, then fix0. An unconditional proposal uses empirical frequency
    as its precision score. This rule is our recorded reproduction choice.
    """
    by_source: dict[int, list[CandidateFit]] = {}
    for record in proposed:
        by_source.setdefault(record.equality.i, []).append(record)
    selected: list[Equality] = []
    conflicts = fixed_conflicts = 0
    for i in sorted(by_source):
        options = by_source[i]
        conflicts += len(options) > 1
        fixed = [record for record in options if record.equality.j is None]
        fixed_conflicts += len(fixed) > 1
        pool = fixed or options
        winner = max(
            pool,
            key=lambda record: (
                record.oof_precision if record.oof_precision is not None else record.frequency,
                record.frequency,
                record.equality.kind == "fix0",
            ),
        )
        selected.append(winner.equality)
    return selected, {
        "variables_with_competing_proposals": int(conflicts),
        "variables_with_opposite_fixed_values": int(fixed_conflicts),
        "proposals_removed_by_arbitration": len(proposed) - len(selected),
    }


def equalities_consistent(equalities: list[Equality], n_binary: int) -> bool:
    """Check dictionary equations only, without accessing a test solution."""
    parent = list(range(n_binary))

    def find(i: int) -> int:
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for equality in equalities:
        if equality.j is not None:
            parent[find(equality.i)] = find(equality.j)
    fixed: dict[int, int] = {}
    for equality in equalities:
        if equality.j is None:
            representative = find(equality.i)
            if representative in fixed and fixed[representative] != equality.rhs:
                return False
            fixed[representative] = equality.rhs
    return True


class XQAAffineBaseline:
    """Nominal paper-based affine predictor, with Table 1 thresholds."""

    def __init__(self, *, random_state: int = 0, C: float = 1.0, n_splits: int = 5):
        if C <= 0 or n_splits < 2:
            raise ValueError("C must be positive and n_splits at least two")
        self.random_state = int(random_state)
        self.C = float(C)
        self.n_splits = int(n_splits)

    def fit(self, X: Any, y: Any, n_times: int, n_units: int) -> "XQAAffineBaseline":
        started = perf_counter()
        features, binary = _validate_training(X, y, n_times, n_units)
        self.n_features_in_ = features.shape[1]
        self.n_binary_ = n_times * n_units
        self.n_times_ = n_times
        self.n_units_ = n_units
        self.records_: list[CandidateFit] = []
        for equality in candidate_dictionary(n_times, n_units):
            labels = equality.satisfied_by(binary).astype(np.int8)
            positives = int(labels.sum())
            frequency = positives / len(labels)
            threshold = TABLE1[equality.kind]
            action = frequency_action(frequency, equality.kind)
            record = CandidateFit(
                equality=equality,
                status=action,
                frequency=frequency,
                positives=positives,
                precision_required=precision_threshold(frequency, equality.kind),
                recall_required=threshold.alpha_recall,
            )
            if action == "classify":
                if min(positives, len(labels) - positives) < self.n_splits:
                    record.status = "skip_class_count"
                else:
                    from sklearn.model_selection import StratifiedKFold

                    splitter = StratifiedKFold(
                        n_splits=self.n_splits, shuffle=True, random_state=self.random_state
                    )
                    oof = np.zeros(len(labels), dtype=np.int8)
                    for train_indices, valid_indices in splitter.split(features, labels):
                        classifier = _new_classifier(self.C, self.random_state)
                        classifier.fit(features[train_indices], labels[train_indices])
                        oof[valid_indices] = classifier.predict(features[valid_indices])
                    true_positives = int(np.sum((oof == 1) & (labels == 1)))
                    record.oof_precision = true_positives / max(1, int(oof.sum()))
                    record.oof_recall = true_positives / positives
                    if (
                        record.oof_precision >= record.precision_required
                        and record.oof_recall >= record.recall_required
                    ):
                        record.status = "accepted_classifier"
                        record.classifier = _new_classifier(self.C, self.random_state)
                        record.classifier.fit(features, labels)
                    else:
                        record.status = "skip_oof_gate"
            self.records_.append(record)
        self.fit_seconds_ = perf_counter() - started
        self.diagnostics_ = {
            "method": "XQA-affine-paper-based-reimplementation",
            "n_train": len(features),
            "n_features": self.n_features_in_,
            "n_times": n_times,
            "n_units": n_units,
            "n_candidates": len(self.records_),
            "candidate_status_counts": dict(Counter(record.status for record in self.records_)),
            "fit_seconds": self.fit_seconds_,
            "C": self.C,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "scaling": "StandardScaler fitted separately within every training fold",
            "layout": "time-major i=t*n_units+g",
            "arbitration": "fixed values first; precision, frequency, then fix0",
        }
        counts = self.diagnostics_["candidate_status_counts"]
        self.training_stats = {
            "candidates": len(self.records_),
            "constant": counts.get("unconditional", 0),
            "trained": counts.get("accepted_classifier", 0) + counts.get("skip_oof_gate", 0),
            "accepted": counts.get("accepted_classifier", 0),
            "skipped": sum(v for k, v in counts.items() if k.startswith("skip_")),
            "elapsed_seconds": self.fit_seconds_,
            **self.diagnostics_,
        }
        self.train_diag = self.training_stats
        return self

    def candidate_records(self) -> list[dict[str, Any]]:
        return [record.record() for record in self.records_]

    def predict_equalities(self, feature: Any) -> tuple[list[Equality], dict[str, Any]]:
        started = perf_counter()
        features = np.asarray(feature, dtype=float)
        if features.shape != (self.n_features_in_,) or not np.isfinite(features).all():
            raise ValueError("Prediction needs one finite feature vector with the training dimension")
        proposed = []
        for record in self.records_:
            if record.status == "unconditional" or (
                record.status == "accepted_classifier"
                and int(record.classifier.predict(features.reshape(1, -1))[0]) == 1
            ):
                proposed.append(record)
        selected, conflicts = _arbitrate(proposed)
        consistent = equalities_consistent(selected, self.n_binary_)
        if not consistent:
            raise RuntimeError("Internal error: arbitrated dictionary equations are inconsistent")
        info = {
            "proposals": len(proposed),
            "selected": len(selected),
            "equalities_consistent": consistent,
            "prediction_seconds": perf_counter() - started,
            **conflicts,
        }
        return selected, info


class SparseUnanimousBaseline:
    """All dictionary equations satisfied by every supplied training solution.

    Unlike the conditional method, this retains all unanimous equations,
    including redundant ones. No at-most-one arbitration is necessary because
    every training decision witnesses consistency. It does NOT calculate all
    affine equations satisfied by the data; only the fixed sparse dictionary.
    """

    def fit(self, X: Any, y: Any, n_times: int, n_units: int) -> "SparseUnanimousBaseline":
        started = perf_counter()
        features, binary = _validate_training(X, y, n_times, n_units)
        self.n_features_in_ = features.shape[1]
        self.n_binary_ = n_times * n_units
        self.equalities_ = [
            equality
            for equality in candidate_dictionary(n_times, n_units)
            if np.all(equality.satisfied_by(binary))
        ]
        self.fit_seconds_ = perf_counter() - started
        self.diagnostics_ = {
            "method": "unanimous-sparse-dictionary",
            "n_train": len(features),
            "n_equalities": len(self.equalities_),
            "fit_seconds": self.fit_seconds_,
            "layout": "time-major i=t*n_units+g",
        }
        self.training_stats = {
            "candidates": len(candidate_dictionary(n_times, n_units)),
            "constant": len(self.equalities_),
            "trained": 0,
            "accepted": len(self.equalities_),
            "skipped": len(candidate_dictionary(n_times, n_units)) - len(self.equalities_),
            "elapsed_seconds": self.fit_seconds_,
            **self.diagnostics_,
        }
        self.train_diag = self.training_stats
        return self

    def predict_equalities(self, feature: Any) -> tuple[list[Equality], dict[str, Any]]:
        started = perf_counter()
        features = np.asarray(feature, dtype=float)
        if features.shape != (self.n_features_in_,) or not np.isfinite(features).all():
            raise ValueError("Prediction needs one finite feature vector with the training dimension")
        return list(self.equalities_), {
            "selected": len(self.equalities_),
            "equalities_consistent": equalities_consistent(self.equalities_, self.n_binary_),
            "prediction_seconds": perf_counter() - started,
        }


def fit(X_features: Any, y_binary: Any, n_times: int, n_units: int, **kwargs: Any) -> XQAAffineBaseline:
    """Convenience wrapper returning a fitted nominal XQA model."""
    return XQAAffineBaseline(**kwargs).fit(X_features, y_binary, n_times, n_units)
