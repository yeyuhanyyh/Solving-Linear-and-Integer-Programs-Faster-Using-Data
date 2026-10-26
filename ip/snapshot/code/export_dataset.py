"""Locally export the frozen synthetic training data; never solve or download.

This is a one-time LOCAL build utility. It reads the existing v1 experiment
and its exact FrozenCase adapter. The portable experiment consumes only the
resulting root-relative data and copied synthetic controls, not this utility
or the old local directories. No old test features, costs, solutions or
objectives are exported. Source provenance may include hashes of old files
that contain test data; those files' contents are never copied.

Run with the existing local scientific Python runtime::

    python code/export_dataset.py

Every file is published atomically without replacing an existing different
file. Repeating an identical export verifies and accepts the existing files.
The dataset manifest is published last and is the completion marker.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "synthetic-ip-training-v2.1"
CONTROL_NAMES = ("experiment_core.py", "xqa_baseline.py", "uc_model.py")
TRAINING_KEYS = (
    "A_data", "A_indices", "A_indptr", "A_shape", "row_lower", "row_upper",
    "col_lower", "col_upper", "integrality", "variable_names", "row_names",
    "train_features", "train_costs", "train_solutions", "train_objectives",
    "commitment_indices", "binary_indices", "base_cost", "n_times", "n_units",
    "affine_anchor", "affine_basis", "affine_normals",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        raise TypeError("Absolute/local paths must not enter exported metadata")
    return value


def json_bytes(value) -> bytes:
    return (json.dumps(json_safe(value), sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    """Avoid timestamps and object arrays so repeated exports have equal hashes."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for key in sorted(arrays):
            array = np.asarray(arrays[key])
            if array.dtype.hasobject:
                raise TypeError(f"Object arrays are forbidden in portable NPZ: {key}")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            entry = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=6)
    return output.getvalue()


def publish(path: Path, payload: bytes) -> str:
    """Atomic, exclusive publication; refuse silent replacement, including races."""
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_bytes(payload)
    if path.exists():
        if not path.is_file() or sha256_file(path) != expected:
            raise FileExistsError(f"Refusing to overwrite a different export: {path}")
        return expected
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Windows rename is atomic and refuses an existing destination;
            # POSIX rename would replace it, so use exclusive hard-link there.
            # Hard-links are disallowed by some Windows workspace policies.
            if os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or sha256_file(path) != expected:
                raise FileExistsError(f"A different export appeared concurrently: {path}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"Published file failed its SHA256 check: {path}")
        return expected
    finally:
        temporary.unlink(missing_ok=True)


def load_local_adapter(v1_root: Path, old_root: Path):
    """Load only trusted, existing local source; no networking or code acquisition."""
    source = v1_root / "code" / "adapter.py"
    if not source.is_file():
        raise FileNotFoundError(f"The local v1 adapter is missing: {source}")
    spec = importlib.util.spec_from_file_location("frozen_v1_export_adapter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the local FrozenCase adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    module.EXPERIMENT_ROOT = v1_root
    module.RESEARCH_ROOT = old_root.parent
    module.OLD_ROOT = old_root
    module.FINAL_ROOT = old_root / "speed_revision_20260924"
    # Make an accidental solve an explicit failure. load_case itself reads and
    # regenerates the synthetic matrices; no optimization is necessary.
    def forbid_solve(*args, **kwargs):
        raise RuntimeError("The dataset exporter may not solve or collect new labels")
    module.solve = forbid_solve
    module.collect_training_pools = forbid_solve
    return module


def source_key(path: Path, research_root: Path) -> str:
    return path.resolve().relative_to(research_root.resolve()).as_posix()


def training_arrays(case) -> dict[str, np.ndarray]:
    model, base = case.model, case.original_base
    result = {
        "A_data": model.A.data.copy(), "A_indices": model.A.indices.copy(),
        "A_indptr": model.A.indptr.copy(), "A_shape": np.asarray(model.A.shape, dtype=np.int64),
        **{key: np.asarray(getattr(model, key)).copy() for key in
           ("row_lower", "row_upper", "col_lower", "col_upper", "integrality")},
        "variable_names": np.asarray(model.variable_names, dtype=str),
        "row_names": np.asarray(model.row_names, dtype=str),
        **{key: np.asarray(getattr(case, key)).copy() for key in
           ("train_features", "train_costs", "train_solutions", "train_objectives", "commitment_indices")},
        "binary_indices": np.asarray(base["binary_indices"]).copy(),
        "base_cost": np.asarray(base["base_cost"]).copy(),
        "n_times": np.asarray(base["n_times"], dtype=np.int64),
        "n_units": np.asarray(base["n_units"], dtype=np.int64),
        **{f"affine_{key}": np.asarray(case.affine[key]).copy() for key in ("anchor", "basis", "normals")},
    }
    if set(result) != set(TRAINING_KEYS):
        raise RuntimeError("Training schema drift")
    if case.n_train != 120 or result["train_costs"].shape != (120, model.n_variables):
        raise RuntimeError(f"Incomplete frozen training data: {case.case_id}")
    if result["train_solutions"].shape != result["train_costs"].shape:
        raise RuntimeError("Training solution/cost shapes differ")
    if result["train_features"].shape != (120, 6 if case.family == "uc" else 8):
        raise RuntimeError("Full observed cost-parameter features are required")
    if result["train_objectives"].shape != (120,) or not all(r["certified"] for r in case.training_records):
        raise RuntimeError("All 120 original training labels must be certified")
    if not np.array_equal(result["commitment_indices"], result["binary_indices"]):
        raise RuntimeError("Original binary/commitment coordinate mapping changed")
    return result


def verify_npz(path: Path, expected: dict[str, np.ndarray]):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise RuntimeError(f"Exported training schema differs: {path}")
        for name, array in expected.items():
            actual = archive[name]
            if actual.dtype != array.dtype or actual.shape != array.shape or not np.array_equal(actual, array):
                raise RuntimeError(f"Exported array differs from exact v1 FrozenCase: {name}")


def verify_pool(path: Path, case, index: int, adapter) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"solutions", "objectives", "source_tags", "metadata"}:
            raise RuntimeError(f"Unexpected incumbent pool keys: {path}")
        solutions, objectives = archive["solutions"].copy(), archive["objectives"].copy()
        tags = archive["source_tags"].tolist()
        metadata = json.loads(str(archive["metadata"]))
    count, n = len(solutions), case.model.n_variables
    if not 1 <= count <= 64 or solutions.shape != (count, n) or objectives.shape != (count,):
        raise RuntimeError(f"Invalid pool dimensions or 64-candidate cap: {path}")
    if len(tags) != count or not np.isfinite(solutions).all() or not np.isfinite(objectives).all():
        raise RuntimeError(f"Incomplete or nonfinite pool arrays: {path}")
    required = dict(case_id=case.case_id, train_index=index, max_solutions=64,
                    time_limit=30.0, solver_seed=case.seed + index,
                    cost_sha256=sha256_bytes(case.train_costs[index].tobytes()),
                    input_fingerprint=sha256_bytes(json.dumps(case.source_hashes, sort_keys=True).encode()),
                    cached_optimum_included=True)
    for key, value in required.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Frozen pool metadata mismatch: {path.name}/{key}")
    assignments = np.rint(solutions[:, case.model.integer_indices]).astype(np.int64)
    if len(set(map(tuple, assignments))) != count:
        raise RuntimeError(f"Pool repeats an assignment across all integer variables: {path}")
    allowed_tags = {"cached_optimum", "mip_solution_callback", "final_solver_solution"}
    if not set(tags) <= allowed_tags or tags.count("cached_optimum") != 1:
        raise RuntimeError(f"Pool must retain exactly one cached certified optimum: {path}")
    cached = tags.index("cached_optimum")
    if not np.array_equal(solutions[cached], case.train_solutions[index]):
        raise RuntimeError(f"Pool cached optimum differs from the frozen training label: {path}")
    if abs(objectives[cached] - case.train_objectives[index]) > adapter.recovery_tolerance(case.train_objectives[index]):
        raise RuntimeError(f"Pool cached optimal objective changed: {path}")
    for solution, objective in zip(solutions, objectives):
        if not adapter.validate_solution(case.model, solution)["valid"]:
            raise RuntimeError(f"Original feasibility/integrality failed in pool: {path}")
        if abs(float(case.train_costs[index] @ solution) - objective) > adapter.recovery_tolerance(objective):
            raise RuntimeError(f"Candidate objective differs from its frozen training cost: {path}")
    seconds = float(metadata["collection_seconds"])
    if not np.isfinite(seconds) or seconds < 0:
        raise RuntimeError(f"Invalid pool offline collection time: {path}")
    return dict(train_index=index, candidates=count, collection_seconds=seconds,
                cached_certified_optimum_included=True)


def export_dataset(root: Path, v1_root: Path, old_root: Path) -> dict:
    root, v1_root, old_root = root.resolve(), v1_root.resolve(), old_root.resolve()
    research_root = old_root.parent
    adapter = load_local_adapter(v1_root, old_root)
    cases = adapter.list_case_ids()
    if len(cases) != 18 or len(set(cases)) != 18:
        raise RuntimeError("Expected the exact 18 Appendix settings/seed cases")
    file_hashes, input_hashes, case_entries = {}, {}, []
    provenance_sources = [v1_root / "code" / "adapter.py", v1_root / "protocol.json"]
    provenance_sources += [old_root / "code" / name for name in CONTROL_NAMES]
    for name in ("frozen-input-audit.json", "pool-audit.json"):
        path = v1_root / "sources" / name
        if path.is_file():
            provenance_sources.append(path)
    for path in provenance_sources:
        input_hashes[source_key(path, research_root)] = sha256_file(path)
    for name in CONTROL_NAMES:
        path = root / "code" / "controls" / name
        file_hashes[path.relative_to(root).as_posix()] = publish(path, (old_root / "code" / name).read_bytes())
    for case_id in cases:
        case = adapter.load_case(case_id, verify=True)
        arrays = training_arrays(case)
        folder = root / "data" / case_id
        training_path = folder / "training.npz"
        entry_files = {}
        # Different NumPy versions can use different valid NPY headers for
        # identical arrays. Accept an existing pack only after exact schema,
        # dtype, shape and value verification, then retain its actual hash.
        if training_path.exists():
            verify_npz(training_path, arrays)
            training_hash = sha256_file(training_path)
        else:
            training_hash = publish(training_path, deterministic_npz(arrays))
        entry_files[training_path.relative_to(root).as_posix()] = training_hash
        verify_npz(training_path, arrays)
        case_sources = {str(key).replace("\\", "/"): value for key, value in case.source_hashes.items()}
        input_hashes.update(case_sources)
        exact = [[str(int(value)) for value in row] for row in case.affine["normals_exact"].tolist()]
        affine = {key: json_safe(value) for key, value in case.affine.items()
                  if key not in ("anchor", "basis", "normals", "normals_exact")}
        affine["normals_exact"] = exact
        metadata = dict(schema_version=SCHEMA_VERSION, case_id=case.case_id, family=case.family,
                        seed=case.seed, amplitude=case.amplitude, metadata=json_safe(case.metadata),
                        original_base_metadata=json_safe(case.original_base["metadata"]),
                        n_times=int(case.original_base["n_times"]), n_units=int(case.original_base["n_units"]),
                        variable_names=list(case.model.variable_names), row_names=list(case.model.row_names),
                        # Pool fingerprints depend on the exact original key
                        # strings, including their historical separators.
                        source_hashes=dict(case.source_hashes), affine=affine, training_labels=120,
                        training_labels_certified=True, original_physical_and_training_arrays_verified=True,
                        old_test_contents_exported=False)
        metadata_path = folder / "metadata.json"
        entry_files[metadata_path.relative_to(root).as_posix()] = publish(metadata_path, json_bytes(metadata))
        source_pool_folder = v1_root / "pools" / case_id
        expected_names = {f"{case_id}_train{i:03d}.npz" for i in range(120)}
        if {p.name for p in source_pool_folder.glob("*.npz")} != expected_names:
            raise RuntimeError(f"Expected exactly the frozen 120 training pools: {case_id}")
        summaries = []
        for index in range(120):
            source = source_pool_folder / f"{case_id}_train{index:03d}.npz"
            summary = verify_pool(source, case, index, adapter)
            target = folder / "pools" / source.name
            original_hash = sha256_file(source)
            target_hash = publish(target, source.read_bytes())
            if target_hash != original_hash:
                raise RuntimeError("Pool copy changed original source bytes")
            summaries.append(summary)
            source_relative = source_key(source, research_root)
            input_hashes[source_relative] = original_hash
            case_sources[source_relative] = original_hash
            entry_files[target.relative_to(root).as_posix()] = target_hash
        summary = dict(case_id=case_id, training_objectives=120, test_objectives_used=0,
                       max_solutions=64, time_limit=30.0, cached_certified_optima_included=True,
                       all_integer_assignments_distinct=True,
                       original_feasibility_integrality_and_objectives_verified=True,
                       total_feasible_labels=sum(value["candidates"] for value in summaries),
                       stored_collection_seconds=sum(value["collection_seconds"] for value in summaries),
                       distinct_assignments_per_cost=[value["candidates"] for value in summaries])
        path = folder / "pools" / "collection-summary.json"
        entry_files[path.relative_to(root).as_posix()] = publish(path, json_bytes(summary))
        file_hashes.update(entry_files)
        case_entries.append(dict(case_id=case_id, family=case.family, seed=case.seed,
                                 amplitude=case.amplitude, directory=f"data/{case_id}",
                                 training_npz=f"data/{case_id}/training.npz",
                                 metadata_json=f"data/{case_id}/metadata.json",
                                 pools_directory=f"data/{case_id}/pools", training_objectives=120,
                                 pool_count=120, max_solutions=64, pool_time_limit_seconds=30.0,
                                 total_pool_candidates=summary["total_feasible_labels"],
                                 pool_collection_seconds=summary["stored_collection_seconds"],
                                 file_sha256=entry_files, input_source_sha256=case_sources,
                                 exact_v1_array_equality=True, cached_certified_optima_verified=True))
        print(json.dumps(dict(case=case_id, status="exported_and_verified", pools=120,
                              candidates=summary["total_feasible_labels"])), flush=True)
    # Verify every input has stayed frozen during export. Hash-only provenance
    # includes old trials; no test data content is present in the output pack.
    for name, expected in input_hashes.items():
        if sha256_file(research_root / name) != expected:
            raise RuntimeError(f"An input changed during export: {name}")
    for name, expected in file_hashes.items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"An exported file changed during export: {name}")
    fingerprint = sha256_bytes(json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode())
    manifest = dict(schema_version=SCHEMA_VERSION, export_kind="training_only_synthetic_dataset",
                    cases=case_entries, case_count=18, training_objectives=2160, pool_files=2160,
                    training_npz_keys=list(TRAINING_KEYS), pool_max_solutions=64,
                    pool_time_limit_seconds=30.0, old_test_contents_exported=False,
                    test_features_exported=False, test_costs_exported=False,
                    test_solutions_exported=False, test_objectives_exported=False,
                    cached_certified_training_optima_included=True,
                    copied_controls=list(CONTROL_NAMES),
                    file_sha256=file_hashes, input_source_sha256=input_hashes,
                    dataset_fingerprint=fingerprint,
                    fingerprint_definition="SHA256 of compact sorted JSON of root-relative file_sha256",
                    output_bytes=sum((root / name).stat().st_size for name in file_hashes),
                    local_only_exporter="code/export_dataset.py",
                    remote_bundle_excludes_local_only_exporter=True,
                    old_test_hash_provenance_only=True,
                    no_training_no_solving_no_network=True,
                    exact_v1_physical_and_training_array_equality=True)
    manifest_path = root / "data" / "dataset-manifest.json"
    publish(manifest_path, json_bytes(manifest))
    print(json.dumps(dict(status="all_18_cases_exported", manifest="data/dataset-manifest.json",
                          dataset_fingerprint=fingerprint, output_bytes=manifest["output_bytes"])), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--v1-root", type=Path, default=ROOT.parent / "ip-paper-baselines-20261009")
    parser.add_argument("--old-root", type=Path, default=ROOT.parent / "ip-comparison-20260924")
    arguments = parser.parse_args()
    export_dataset(arguments.root, arguments.v1_root, arguments.old_root)


if __name__ == "__main__":
    main()
