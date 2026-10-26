"""Independent, read-only canonical-completion audit of a finished cohort.

No driver/runtime, model, solver, or torch is imported. The audit reads no
development NPZ bank. Only --audit opens real inputs. --self-test is an entirely
synthetic software check. There is no default operation.

Gate order: verify the complete frozen training/model cohort and every paired
development completion registry/existence; verify ALL registered trial bytes;
only then load model-only NPZ fields and parse any trial. An incomplete or
inconsistent cohort produces no output report. The output is a new immutable
JSON file, and no original completion marker or result is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from scipy import sparse

METHODS = ("ours", "neural_diving")
FEAS_TOL = 1e-6
FIX_TOL = 1e-6
BUDGET = 30.0
MODEL_FIELDS = ("A_data", "A_indices", "A_indptr", "A_shape", "row_lower",
                "row_upper", "col_lower", "col_upper", "integrality",
                "cost_offset", "cost_feature_coefficients")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_sha(value):
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def no_duplicate_keys(items):
    result = {}
    for key, value in items:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise AuditError(f"Nonfinite JSON constant: {value}")


def parse_json(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_keys,
                      parse_constant=reject_constant)


def inside(folder, name):
    folder = Path(folder).resolve()
    require(isinstance(name, str) and not Path(name).is_absolute(), "Expected a relative artifact name")
    path = (folder / name).resolve()
    require(path.is_relative_to(folder) and path != folder, f"Artifact escapes directory: {name}")
    return path


def positive_integer(value, name):
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"Expected positive integer {name}")
    return value


class Sources:
    def __init__(self):
        self.hashes = {}

    def remember(self, path, digest):
        key = str(Path(path).resolve())
        if key in self.hashes:
            require(self.hashes[key] == digest, f"Source changed during audit: {key}")
        self.hashes[key] = digest
        return digest

    def verify(self, path, expected=None):
        digest = self.remember(path, file_sha(path))
        if expected is not None:
            require(valid_sha(expected) and digest == expected, f"Source SHA256 mismatch: {path}")
        return digest

    def read(self, path):
        raw = Path(path).read_bytes()
        self.remember(path, hashlib.sha256(raw).hexdigest())
        return parse_json(raw)

    def recheck(self):
        for name, expected in self.hashes.items():
            require(file_sha(name) == expected, f"Source changed before report publication: {name}")


def recovery_tolerance(reference):
    return 1e-6 + 1e-9 * abs(float(reference))


def close_value(saved, computed, name):
    require((saved is None) == (computed is None), f"Null/value mismatch: {name}")
    if computed is not None:
        require(isinstance(saved, (int, float)) and not isinstance(saved, bool)
                and math.isfinite(float(saved))
                and abs(float(saved) - computed) <= 1e-10 * max(1.0, abs(computed)),
                f"Derived numeric mismatch: {name}")


def pilot_guard(entry, train_hash, manifest_sha, driver_hash, seal_hash, protocol_hash, shared_hashes, method):
    return dict(case_id=entry["case_id"], method=method, training_array_hash=train_hash,
                source_hashes={name: entry["files"][name] for name in ("training.npz", "metadata.json")},
                target_K=entry["K"], pilot_runner_sha256=driver_hash,
                runtime_seal_sha256=seal_hash, pilot_protocol_sha256=protocol_hash,
                selection_manifest_sha256=manifest_sha, all_shared_source_sha256=shared_hashes,
                cost_map="signed affine; offset + coefficients @ feature",
                development_used_in_training=False, new_confirmation_created=False)


def check_bank_bindings(path, manifest_sha, cases, sources, producer_seal_sha):
    """Check the existing attach_fixed_development_banks.py JSON receipt.

    Called only AFTER all models/completions are ready. No referenced bank NPZ
    is opened or hashed here; byte identity is the producer's sealed attestation.
    """
    if path is None:
        return dict(producer_receipts_crosschecked=False,
                    scope="Bank SHA is bound only by paired completion/row guards; no prior producer receipt was supplied",
                    cases={c["entry"]["case_id"]: c["completions"]["ours"]["document"]["guard"]["development_bank_sha256"] for c in cases})
    path = Path(path).resolve()
    require(path.suffix.lower() == ".json", "--bank-bindings must be a JSON attachment receipt")
    document = sources.read(path)
    require(document.get("status") == "ATTACHED_EXISTING_FIXED_DEVELOPMENT_BANKS"
            and document["manifest_sha256"] == manifest_sha, "Attachment receipt status/manifest differs")
    require(valid_sha(producer_seal_sha) and document["producer_data_seal_sha256"] == producer_seal_sha,
            "Attachment receipt and cohort bind different producer data seals")
    require(all(document.get(key) is False for key in ("original_files_modified", "training_files_modified", "new_sampling", "confirmation")),
            "Attachment receipt admits changed files, new samples, or confirmation")
    entries = document["entries"]
    require(isinstance(entries, list) and len(entries) == len(cases), "Attachment receipt case count differs")
    bindings = {entry["case_id"]: entry for entry in entries}
    require(len(bindings) == len(entries) and set(bindings) == {c["entry"]["case_id"] for c in cases},
            "Attachment receipt must cover exactly the entire cohort")
    evidence = {}
    for case in cases:
        cid = case["entry"]["case_id"]
        binding = bindings[cid]
        bank_sha = case["completions"]["ours"]["document"]["guard"]["development_bank_sha256"]
        require(binding["sha256"] == bank_sha and binding.get("byte_identical") is True
                and binding.get("array_contents_decoded") is False, f"{cid}: attachment/completion bank binding differs")
        source, destination = Path(binding["source"]), Path(binding["destination"])
        require(source.is_absolute() and destination.is_absolute()
                and destination.resolve() == (case["directory"] / "development.npz").resolve(),
                f"{cid}: attachment destination is not the driver's original case bank alias")
        evidence[cid] = dict(development_bank_sha256=bank_sha, source=str(source), destination=str(destination),
                             byte_identity_attested_by_producer=True, bank_bytes_independently_read=False)
    return dict(producer_receipts_crosschecked=True, bindings_path=str(path),
                bindings_sha256=sources.hashes[str(path)], producer_data_seal_sha256=producer_seal_sha, cases=evidence)


def preflight(manifest_path, driver_root, results, expected_count, certificate_mode, bank_bindings, sources):
    """No trial JSON or development NPZ is opened in this function."""
    manifest = sources.read(manifest_path)
    require(manifest.get("schema") == "generic-binary-primary-cohort-v2", "Wrong cohort schema")
    require(manifest.get("frozen_before_any_fit") is True and manifest.get("training_only") is True
            and manifest.get("confirmation_included") is False, "Cohort is not frozen training-only")
    entries = manifest.get("cases")
    require(isinstance(entries, list) and entries, "Empty cohort")
    ids = [entry["case_id"] for entry in entries]
    require(len(set(ids)) == len(ids), "Duplicate cohort case")
    require(all(isinstance(cid, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", cid)
                and cid not in (".", "..") for cid in ids), "Unsafe case_id")
    if "source_data_seal" in manifest or "source_data_seal_sha256" in manifest:
        seal_source = Path(manifest["source_data_seal"])
        require(seal_source.is_absolute(), "Source-data seal must use an absolute path")
        # Bind this receipt only; do not follow any development-bank references.
        sources.verify(seal_source, manifest["source_data_seal_sha256"])
    driver_hash = sources.verify(driver_root / "generic_binary_pilot.py")
    seal_path = driver_root / "runtime-seal.json"
    seal = sources.read(seal_path)
    require(seal.get("schema") == "generic-binary-ND-runtime-v2", "Wrong runtime seal")
    shared_hashes = seal.get("file_sha256")
    require(isinstance(shared_hashes, dict) and shared_hashes, "Missing sealed sources")
    for name, expected in shared_hashes.items():
        sources.verify(inside(driver_root, name), expected)
    protocol_hash = sources.verify(driver_root / "GENERIC_BINARY_PROTOCOL.md")
    manifest_sha = sources.hashes[str(manifest_path)]
    seal_hash = sources.hashes[str(seal_path)]
    cases, issues = [], []
    for entry in entries:
        cid = entry["case_id"]
        try:
            D = positive_integer(entry["D"], f"{cid}.D")
            G = positive_integer(entry["feature_dimension"], f"{cid}.feature_dimension")
            n = positive_integer(entry["n_train"], f"{cid}.n_train")
            require(D == 3 * G and entry["K"] == G, f"{cid}: require D=3G, declared full K=G")
            directory = Path(entry["directory"])
            require(directory.is_absolute(), f"{cid}: source directory must be absolute")
            directory = directory.resolve()
            names = set(entry["files"])
            require({"training.npz", "metadata.json"} <= names <= {"training.npz", "metadata.json", "affine.json"},
                    f"{cid}: source list must contain training artifacts only")
            for name, expected in entry["files"].items():
                sources.verify(inside(directory, name), expected)
            meta = sources.read(directory / "metadata.json")
            require(meta["case_id"] == cid and meta["D"] == D and meta["K"] == G
                    and meta["n_train"] == n, f"{cid}: training metadata identity/dimensions differ")
            require(meta.get("training_only") is True and meta.get("development_costs_exported") is False,
                    f"{cid}: training metadata is not training-only")
            require(meta["training_npz_sha256"] == entry["files"]["training.npz"], f"{cid}: training NPZ pin differs")
            if "global_optimizer_hull_K" in meta:
                require(meta["global_optimizer_hull_K"] == G, f"{cid}: global K declaration differs")
            receipt_path = results / "training-case-receipts" / f"{cid}.json"
            receipt = sources.read(receipt_path)
            training_hash = receipt["training_guard"]["training_array_hash"]
            require(valid_sha(training_hash), f"{cid}: missing training-array fingerprint")
            make_guard = lambda method: pilot_guard(entry, training_hash, manifest_sha, driver_hash,
                                                   seal_hash, protocol_hash, shared_hashes, method)
            require(receipt["training_guard"] == make_guard("case"), f"{cid}: Ours training guard differs")
            require(receipt["source"] == make_guard("case")["source_hashes"], f"{cid}: receipt training sources differ")
            cert = receipt["original_labels_certified"]
            require(cert["guard"] == make_guard("training_oracle") and cert["certified_label_count"] == n
                    and cert["all_labels_globally_certified"] is True, f"{cid}: training certification receipt incomplete")
            require(receipt["exact_affine"]["rank"] == G, f"{cid}: Ours archived rank is not full G")
            if "affine.json" in entry["files"]:
                selected_affine = sources.read(directory / "affine.json")
                for key in ("anchor", "basis", "normals", "normals_exact", "rank", "representation"):
                    require(selected_affine[key] == receipt["exact_affine"][key],
                            f"{cid}: Ours fitted affine receipt differs from the frozen prefix evidence: {key}")
            model_dir = results / "models" / cid / "neural_diving"
            marker = sources.read(model_dir / "complete.json")
            require(marker["guard"] == make_guard("neural_diving"), f"{cid}: ND guard differs")
            require(marker.get("sanity_gate_passed") is True and marker["training_label_count"] == n,
                    f"{cid}: ND readiness/label count incomplete")
            artifacts = marker.get("artifact_sha256")
            require(isinstance(artifacts, dict) and {"complete-predictor.pt", "metadata.json", "start.json"} <= set(artifacts),
                    f"{cid}: ND artifact registry incomplete")
            for name, expected in artifacts.items():
                sources.verify(inside(model_dir, name), expected)
            case = dict(entry=entry, meta=meta, directory=directory, D=D, G=G, n=n,
                        affine=receipt["exact_affine"], guards={method: make_guard(method) for method in METHODS},
                        oracle_guard=make_guard("development_oracle"), completions={}, certificates={})
            paired_count = None
            paired_bank_hash = None
            for method in METHODS:
                folder = results / "development" / cid / method
                complete_path = folder / "complete.json"
                completed = sources.read(complete_path)
                guard, summary = completed["guard"], completed["summary"]
                require(guard["training"] == case["guards"][method] and guard["budget"] == BUDGET,
                        f"{cid}/{method}: development guard/budget differs")
                require(valid_sha(guard["development_bank_sha256"]), f"{cid}/{method}: missing bank hash")
                count = positive_integer(summary["count"], f"{cid}/{method}.count")
                require(summary["case_id"] == cid and summary["method"] == method and summary["phase"] == "development"
                        and summary["n_train"] == n and summary["K"] == G and summary["D"] == D,
                        f"{cid}/{method}: completion identity/dimensions differ")
                if paired_count is not None:
                    require(count == paired_count and guard["development_bank_sha256"] == paired_bank_hash,
                            f"{cid}: methods do not bind the same complete bank/count")
                paired_count, paired_bank_hash = count, guard["development_bank_sha256"]
                declared_count = entry.get("development_count", manifest.get("development_count", expected_count))
                if expected_count is not None:
                    require(count == expected_count, f"{cid}/{method}: explicit expected count differs")
                if declared_count is not None:
                    require(count == positive_integer(declared_count, "development_count"), f"{cid}: manifest count differs")
                names = {f"dev{i:03d}.json" for i in range(count)}
                registry = completed["test_record_sha256"]
                require(isinstance(registry, dict) and set(registry) == names
                        and all(valid_sha(value) for value in registry.values()), f"{cid}/{method}: incomplete exact registry")
                present = {path.name for path in folder.glob("dev*.json") if re.fullmatch(r"dev[0-9]+\.json", path.name)}
                require(present == names, f"{cid}/{method}: missing or unregistered trial files")
                for name in names:
                    require(inside(folder, name).is_file(), f"{cid}/{method}: missing trial {name}")
                case["completions"][method] = dict(folder=folder, document=completed, count=count,
                                                    count_authority="manifest/explicit" if declared_count is not None else "paired completion summaries")
            for i in range(paired_count):
                path = results / "development-certificates" / cid / f"dev{i:03d}.json"
                if certificate_mode != "none" and path.is_file():
                    case["certificates"][i] = path
                elif certificate_mode == "required":
                    raise AuditError(f"{cid}: required reference certificate missing at {i}")
            cases.append(case)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            issues.append(f"{cid}: {type(error).__name__}: {error}")
    require(not issues and len(cases) == len(entries),
            "Preflight failed before opening any trial: " + json.dumps(issues, ensure_ascii=True))
    bank_evidence = check_bank_bindings(bank_bindings, manifest_sha, cases, sources, manifest.get("source_data_seal_sha256"))
    # Global barrier: no raw trial has been opened, including for hashing, above.
    for case in cases:
        for item in case["completions"].values():
            for name, expected in item["document"]["test_record_sha256"].items():
                sources.verify(inside(item["folder"], name), expected)
        for path in case["certificates"].values():
            sources.verify(path)
    return manifest, cases, bank_evidence


def original_check(model, x):
    if x is None:
        return dict(valid=False, reason="no_solution", original_violation=None, integrality_residual=None)
    x = np.asarray(x, dtype=float)
    if x.shape != (model["D"],) or not np.isfinite(x).all():
        return dict(valid=False, reason="bad_shape_or_nonfinite", original_violation=None, integrality_residual=None)
    activity = model["A"] @ x
    violation = max(float(np.max(model["row_lower"] - activity, initial=0)),
                    float(np.max(activity - model["row_upper"], initial=0)),
                    float(np.max(model["col_lower"] - x, initial=0)),
                    float(np.max(x - model["col_upper"], initial=0)))
    residual = float(np.max(np.abs(x - np.rint(x)), initial=0))
    return dict(valid=bool(violation <= FEAS_TOL and residual <= FEAS_TOL),
                reason="checked", original_violation=violation, integrality_residual=residual)


def infer_model(data, entry):
    """Infer roles from costs and original rows, never from column positions."""
    D, G = entry["D"], entry["feature_dimension"]
    A = sparse.csr_matrix((data["A_data"], data["A_indices"], data["A_indptr"]), shape=tuple(data["A_shape"]))
    A.check_format(full_check=True)
    A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
    require(A.shape == (entry["rows"], D) and np.isfinite(A.data).all(), "Model matrix shape/values differ")
    model = dict(A=A, D=D, G=G)
    for key, shape in (("row_lower", (A.shape[0],)), ("row_upper", (A.shape[0],)),
                       ("col_lower", (D,)), ("col_upper", (D,)), ("integrality", (D,))):
        value = np.asarray(data[key])
        require(value.shape == shape and not np.isnan(value).any(), f"Bad model array: {key}")
        model[key] = value
    require(np.all(model["row_lower"] <= model["row_upper"]), "Inconsistent row bounds")
    require(np.array_equal(model["col_lower"], np.zeros(D)) and np.array_equal(model["col_upper"], np.ones(D))
            and np.array_equal(model["integrality"], np.ones(D)), "Original domains are not binary [0,1]")
    offset = np.asarray(data["cost_offset"], dtype=float)
    coefficient = np.asarray(data["cost_feature_coefficients"], dtype=float)
    require(D == 3 * G and offset.shape == (D,) and coefficient.shape == (D, G)
            and np.isfinite(offset).all() and np.isfinite(coefficient).all(), "Bad original affine objective map")
    C = []
    for feature in range(G):
        columns = np.flatnonzero(coefficient[:, feature] != 0)
        require(len(columns) == 1 and coefficient[columns[0], feature] == -1.0,
                "Each feature must price exactly one C at coefficient -1")
        C.append(int(columns[0]))
    require(len(set(C)) == G, "Features do not identify distinct C variables")
    Cset, basic = set(C), set(range(D)) - set(C)
    require(np.array_equal(offset[C], np.zeros(G)) and np.all(offset[list(basic)] == -.25)
            and not np.any(coefficient[list(basic)]), "Original AB=.25 / C=xi profits differ")
    capacity, pairs, global_rows, conflict_edges = {}, {}, [], set()
    used_basic = set()
    for r in range(A.shape[0]):
        start, end = A.indptr[r:r + 2]
        columns, values = A.indices[start:end], A.data[start:end]
        bs = [int(j) for j in columns if int(j) in basic]
        cs = [int(j) for j in columns if int(j) in Cset]
        lo, upper = float(model["row_lower"][r]), float(model["row_upper"][r])
        if not bs:
            require(len(cs) == len(columns) == 2 and np.all(values == 1.0)
                    and upper == 1.0 and np.isneginf(lo), "Nonlocal rows must be original two-C conflict inequalities")
            edge = tuple(sorted(cs))
            require(edge not in conflict_edges, "Duplicate original conflict edge")
            conflict_edges.add(edge)
            global_rows.append(r)
            continue
        require(len(cs) == 1, "A local row must identify exactly one C")
        c = cs[0]
        weights = {int(j): float(w) for j, w in zip(columns, values)}
        if len(bs) == 2 and len(columns) == 3:
            require(c not in capacity and c not in pairs, "Duplicate or mixed local formulation")
            wa, wb, wc = weights[bs[0]], weights[bs[1]], weights[c]
            require(wa > 0 and wb > 0 and math.isfinite(upper)
                    and abs(wa + wb - upper) <= 1e-12 * max(1.0, abs(upper))
                    and max(wa, wb) < wc <= upper and lo <= min(wc, upper), "Local substitute capacity is invalid")
            require(not (set(bs) & used_basic), "A basic item belongs to several blocks")
            capacity[c] = dict(basic=sorted(bs), rows=[r], weights=[wa, wb, wc], capacity=upper)
            used_basic.update(bs)
        elif len(bs) == 1 and len(columns) == 2:
            require(c not in capacity and weights[c] == weights[bs[0]] == 1.0
                    and upper == 1.0 and lo <= 1.0, "Invalid pairwise local hull row")
            require(bs[0] not in used_basic, "A basic item occurs in duplicate/multiple local rows")
            pairs.setdefault(c, dict(basic=[], rows=[]))["basic"].append(bs[0])
            pairs[c]["rows"].append(r)
            used_basic.add(bs[0])
        else:
            raise AuditError("Unrecognized row involving basic items")
    require(used_basic == basic, "Some basic items are ungrouped")
    require(global_rows, "Expected a genuinely coupled conflict graph")
    if capacity and not pairs and set(capacity) == Cset:
        formulation, groups = "capacity", capacity
    elif pairs and not capacity and set(pairs) == Cset and all(len(p["basic"]) == 2 for p in pairs.values()):
        formulation, groups = "pairwise_hull", pairs
    else:
        raise AuditError("Incomplete/mixed local bundle structure")
    declared = entry.get("local_formulation")
    require(declared is None or declared == formulation, "Frozen declared local formulation differs")
    require(entry.get("case_schema", "generic_binary") in ("generic_binary", "coupled-bundle-uniform-gross-v1"),
            "Unexpected frozen physical schema")
    model.update(offset=offset, coefficient=coefficient, C=np.asarray(C),
                 basics=[sorted(groups[c]["basic"]) for c in C], local_formulation=formulation,
                 local_evidence=[dict(feature=g, C=c, **groups[c]) for g, c in enumerate(C)],
                 other_C_only_rows=global_rows)
    all_ab = np.zeros(D)
    all_ab[list(basic)] = 1
    require(original_check(model, all_ab)["valid"], "The original all-AB fallback is not feasible")
    return model


def modular_rank(matrix, prime=2147483647):
    """A nonzero modular minor certifies rational rank, without a solver."""
    a = [[int(value) % prime for value in row] for row in np.asarray(matrix, dtype=object).tolist()]
    if not a:
        return 0
    rank = 0
    for column in range(len(a[0])):
        pivot = next((r for r in range(rank, len(a)) if a[r][column]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        inverse = pow(a[rank][column], -1, prime)
        a[rank] = [(v * inverse) % prime for v in a[rank]]
        for r in range(rank + 1, len(a)):
            factor = a[r][column]
            if factor:
                a[r] = [(v - factor * p) % prime for v, p in zip(a[r], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def check_affine_receipt(model, affine):
    D, G = model["D"], model["G"]
    anchor = np.asarray(affine["anchor"], dtype=float)
    basis = np.asarray(affine["basis"], dtype=float)
    require(anchor.shape == (D,) and basis.shape == (D, G) and np.isfinite(basis).all(), "Bad affine receipt shape")
    require(np.isin(anchor, (0, 1)).all() and np.array_equal(basis, np.rint(basis)), "Affine basis is not archived integer differences")
    require(modular_rank(basis) == G, "Archived basis lacks independently certified full rank G")
    for c, (a, b) in zip(model["C"], model["basics"]):
        require(anchor[a] + anchor[c] == anchor[b] + anchor[c] == 1,
                "Archived anchor violates bundle optimizer relations")
        require(np.array_equal(basis[a] + basis[c], np.zeros(G))
                and np.array_equal(basis[b] + basis[c], np.zeros(G)), "Archived basis leaves the bundle mode space")
    normals = np.asarray(affine["normals_exact"], dtype=object)
    require(normals.shape == (D - G, D)
            and all(isinstance(v, int) and not isinstance(v, bool) for v in normals.flat), "Bad exact normals")
    integer_basis = np.rint(basis).astype(np.int64).astype(object)
    require(modular_rank(normals) == D - G
            and np.all(normals @ integer_basis == 0), "Archived exact normals/basis disagree")
    require(original_check(model, anchor)["valid"], "Archived Ours anchor is not original-feasible")
    return dict(rank=G, modular_rank_prime=2147483647, known_mode_space_equals_archived_space=True,
                training_labels_not_decompressed=True)


def normalize_fixed(fixed, D, K):
    if fixed is None:
        return None
    require(isinstance(fixed, dict) and len(fixed) == D - K, "ND must fix exactly D-K distinct original coordinates")
    normalized = {}
    for key, value in fixed.items():
        require(isinstance(key, str) and re.fullmatch(r"0|[1-9][0-9]*", key), "Malformed fixing column")
        j = int(key)
        require(0 <= j < D and value in (0, 1) and not isinstance(value, bool), "Nonbinary/out-of-range fixing")
        normalized[j] = int(value)
    return normalized


def canonical_diagnostic(model, x, cost, feature, reference, fixed):
    """Only called on an original-valid candidate; never replaces that candidate."""
    z = np.rint(x[model["C"]]).astype(int)
    canonical = np.zeros(model["D"])
    canonical[model["C"]] = z
    for value, (a, b) in zip(z, model["basics"]):
        canonical[a] = canonical[b] = 1 - value
    checked = original_check(model, canonical)
    require(checked["valid"], "Canonical completion is not original-feasible despite verified geometry")
    raw_profit, reference_profit = -float(cost @ x), -float(reference)
    canonical_profit = .5 * model["G"] + float((feature - .5) @ z)
    require(abs(canonical_profit + float(cost @ canonical)) <= 1e-10 * max(1.0, abs(canonical_profit)),
            "Canonical gross-profit formula differs from original cost@point")
    completion_loss = canonical_profit - raw_profit
    tolerance = recovery_tolerance(reference)
    rounding_allowance = FEAS_TOL * float(np.abs(cost).sum())
    require(completion_loss >= -tolerance - rounding_allowance, "Basic completion decreased original gross profit")
    selection_loss = reference_profit - canonical_profit
    require(selection_loss >= -tolerance, "Canonical feasible profit beats the archived original optimum")
    total_loss = reference_profit - raw_profit
    require(abs(total_loss - selection_loss - completion_loss) <= 1e-10 * max(1.0, reference_profit),
            "Total profit-loss decomposition failed")
    violations = [] if fixed is None else [dict(column=j, imposed=value, canonical_value=int(canonical[j]))
                                           for j, value in fixed.items() if abs(canonical[j] - value) > FIX_TOL]
    return dict(point=canonical.tolist(), C_assignment=z.tolist(), original_check=checked,
                profit=canonical_profit, ratio=canonical_profit / reference_profit,
                basic_completion_loss=completion_loss, upgrade_selection_loss=selection_loss,
                total_profit_loss=total_loss, basic_completion_ratio_loss=completion_loss / reference_profit,
                upgrade_selection_ratio_loss=selection_loss / reference_profit,
                total_ratio_loss=total_loss / reference_profit,
                completion_nonnegative_within_tolerance=completion_loss >= -tolerance - rounding_allowance,
                rounding_allowance=rounding_allowance, violates_original_fixed=bool(violations),
                fixed_violations=violations, replaces_primary_return=False)


def verify_trial(row, case, method, index, model, path, sources):
    completed = case["completions"][method]["document"]
    payload = {key: value for key, value in row.items() if key != "record_payload_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(row.get("record_payload_sha256") == digest, f"{path}: payload checksum differs")
    require(row["guard"] == completed["guard"] and row["case_id"] == case["entry"]["case_id"]
            and row["method"] == method and row["index"] == index, f"{path}: trial identity/guard differs")
    require(row["n_train"] == case["n"] and row["D"] == case["D"] and row["K"] == case["G"]
            and row["experiment_phase"] == "development" and row["confirmation"] is False,
            f"{path}: trial shape/phase differs")
    feature, cost = np.asarray(row["feature"], dtype=float), np.asarray(row["cost"], dtype=float)
    require(feature.shape == (case["G"],) and cost.shape == (case["D"],)
            and np.isfinite(feature).all() and np.isfinite(cost).all(), f"{path}: nonfinite/wrong-shaped inputs")
    require(np.all((feature >= .1) & (feature <= 2.5)), f"{path}: feature outside declared Uniform support")
    require(np.allclose(model["offset"] + model["coefficient"] @ feature, cost, rtol=1e-14, atol=1e-12)
            and np.all(cost < 0), f"{path}: original signed affine costs differ")
    reference = float(row["reference_objective"])
    require(math.isfinite(reference) and reference < 0, f"{path}: need strictly positive optimal gross profit")
    tol = recovery_tolerance(reference)
    require(-reference >= .5 * case["G"] - tol
            and -reference <= .5 * case["G"] + float(np.maximum(feature - .5, 0).sum()) + tol,
            f"{path}: reference violates simple original-profit bounds")
    seconds = float(row["seconds"])
    require(math.isfinite(seconds) and seconds >= 0 and row["budget_seconds"] == BUDGET,
            f"{path}: elapsed time/budget differs")
    x = None if row["solution"] is None else np.asarray(row["solution"], dtype=float)
    original = original_check(model, x)
    fixed = normalize_fixed(row["fixed"], case["D"], case["G"])
    if method == "ours":
        require(fixed is None, f"{path}: Ours unexpectedly has coordinate fixings")
        actual_K = case["G"]
        restriction = bool(original["valid"] and all(int(round(x[a])) + int(round(x[c])) == 1
                            and int(round(x[b])) + int(round(x[c])) == 1
                            for c, (a, b) in zip(model["C"], model["basics"])))
        if original["valid"]:
            normals = np.asarray(case["affine"]["normals_exact"], dtype=object)
            anchor = np.asarray(case["affine"]["anchor"], dtype=object)
            exact = bool(np.all(normals @ (np.rint(x).astype(np.int64).astype(object) - anchor) == 0))
            require(exact == restriction, f"{path}: known mode relations and Ours archived slice disagree")
    else:
        actual_K = None if fixed is None else case["D"] - len(fixed)
        restriction = bool(original["valid"] and fixed is not None
                           and all(abs(x[j] - value) <= FIX_TOL for j, value in fixed.items()))
    timely = seconds <= BUDGET
    valid = bool(original["valid"] and restriction and timely and actual_K == case["G"])
    raw_objective = float(cost @ x) if original["valid"] else None
    objective = raw_objective if valid else None
    ratio = objective / reference if valid else None
    recovered = bool(valid and abs(objective - reference) <= tol)
    for key, expected in dict(original_feasible=original["valid"], restriction_valid=restriction,
                              timely=timely, valid=valid, recovered=recovered).items():
        require(row.get(key) is expected, f"{path}: derived flag differs: {key}")
    require(row.get("actual_K") == actual_K, f"{path}: actual_K differs")
    for key, value in dict(raw_objective=raw_objective, objective=objective, profit_ratio=ratio).items():
        close_value(row.get(key), value, f"{path}/{key}")
    if "objective_ratio" in row:
        close_value(row["objective_ratio"], ratio, f"{path}/objective_ratio")
    if raw_objective is not None:
        require(raw_objective >= reference - tol, f"{path}: original-valid return beats reference")
        require(-raw_objective >= -FEAS_TOL * float(np.abs(cost).sum()),
                f"{path}: original gross-profit ratio is negative beyond binary feasibility tolerance")
    result = dict(case_id=case["entry"]["case_id"], method=method, index=index, D=case["D"],
                  G=case["G"], K=case["G"], actual_K=actual_K, n_train=case["n"],
                  source_path=str(path), source_sha256=sources.hashes[str(path)],
                  source_record_payload_sha256=digest, cost=cost.tolist(), feature=feature.tolist(),
                  development_bank_sha256=completed["guard"]["development_bank_sha256"],
                  reference_objective=reference, reference_profit=-reference, solution=row["solution"],
                  fixed=row["fixed"], original_check=original, restriction_valid=restriction,
                  seconds=seconds, budget_seconds=BUDGET, timely=timely, primary_valid=valid,
                  primary_recovered=recovered, primary_objective=objective, primary_ratio=ratio,
                  raw_profit=None if raw_objective is None else -raw_objective,
                  candidate_missing=x is None, status=row.get("status", row.get("raw", {}).get("status")),
                  canonical=None, canonical_in_same_primary_subset=valid)
    if original["valid"]:
        result["canonical"] = canonical_diagnostic(model, x, cost, feature, reference, fixed)
    return result


def mean(values):
    return float(np.mean(values)) if values else None


def aggregate(records):
    primary = [row for row in records if row["primary_valid"]]
    N = len(records)
    canonical = [row["canonical"] for row in primary]
    require(all(value is not None for value in canonical), "Primary-valid candidate lacks canonical diagnostic")
    primary_ratio_loss = mean([1.0 - row["primary_ratio"] for row in primary])
    selection_ratio_loss = mean([v["upgrade_selection_ratio_loss"] for v in canonical])
    completion_ratio_loss = mean([v["basic_completion_ratio_loss"] for v in canonical])
    if primary:
        require(abs(primary_ratio_loss - selection_ratio_loss - completion_ratio_loss) <= 1e-12,
                "Mean of per-test normalized loss decomposition failed")
    return dict(count=N, recovery_count=sum(row["primary_recovered"] for row in records),
                recovery_percent=100.0 * sum(row["primary_recovered"] for row in records) / N if N else None,
                ratio_denominator=len(primary), profit_ratio_mean_valid=mean([row["primary_ratio"] for row in primary]),
                negative_primary_profit_count=sum(row["primary_ratio"] < 0 for row in primary),
                invalid_primary_count=N - len(primary), missing_candidate_count=sum(row["candidate_missing"] for row in records),
                original_invalid_count=sum(not row["original_check"]["valid"] for row in records),
                restriction_invalid_count=sum(not row["restriction_valid"] for row in records),
                late_count=sum(not row["timely"] for row in records), missing_record_count=0,
                canonical_diagnostic_count_all_original_valid=sum(row["canonical"] is not None for row in records),
                canonical_same_primary_denominator=len(primary),
                canonical_ratio_mean_same_primary_subset=mean([v["ratio"] for v in canonical]),
                basic_completion_loss_mean_same_primary_subset=mean([v["basic_completion_loss"] for v in canonical]),
                upgrade_selection_loss_mean_same_primary_subset=mean([v["upgrade_selection_loss"] for v in canonical]),
                basic_completion_ratio_loss_mean_same_primary_subset=completion_ratio_loss,
                upgrade_selection_ratio_loss_mean_same_primary_subset=selection_ratio_loss,
                mean_one_minus_primary_ratio_same_primary_subset=primary_ratio_loss,
                total_ratio_loss_mean_same_primary_subset=mean([v["total_ratio_loss"] for v in canonical]),
                canonical_violates_fixed_count_same_primary_subset=sum(v["violates_original_fixed"] for v in canonical),
                canonical_replaces_primary_return=False,
                count_note="Invalid/missing-candidate/late counts may overlap; all N remain in the Recovery denominator.")


def check_primary_summary(actual, completed):
    for key in ("count", "recovery_count", "ratio_denominator"):
        require(actual[key] == completed[key], f"Recomputed primary summary differs: {key}")
    for key in ("recovery_percent", "profit_ratio_mean_valid"):
        close_value(completed[key], actual[key], f"primary summary/{key}")
    if "negative_returned_profit_count" in completed:
        require(completed["negative_returned_profit_count"] == actual["negative_primary_profit_count"],
                "Recomputed negative-profit count differs")


def check_certificate(case, index, feature_row, sources):
    path = case["certificates"].get(index)
    if path is None:
        return None
    cert = sources.read(path)
    guard = dict(training=case["oracle_guard"],
                 development_bank_sha256=case["completions"]["ours"]["document"]["guard"]["development_bank_sha256"])
    require(cert["guard"] == guard and cert["index"] == index, f"{path}: reference certificate guard/index differs")
    oracle = cert["oracle"]
    require(oracle.get("certified") is True and oracle.get("valid") is True,
            f"{path}: original oracle is not certified and valid")
    close_value(cert["reference"], feature_row["reference_objective"], f"{path}/reference")
    require(abs(float(oracle["objective"]) - feature_row["reference_objective"])
            <= recovery_tolerance(feature_row["reference_objective"]), f"{path}: oracle/reference disagree")
    return dict(path=str(path), sha256=sources.hashes[str(path)], certified=True,
                scope="Archived oracle receipt checked; no independent optimization or oracle solution reconstruction")


def audit(args):
    manifest_path, driver_root, results, output = (getattr(args, name).resolve()
                                                   for name in ("manifest", "driver_root", "results", "output"))
    require(not output.exists(), f"Output must be new: {output}")
    sources = Sources()
    manifest, cases, bank_evidence = preflight(manifest_path, driver_root, results, args.expected_count,
                                             args.certificates, args.bank_bindings, sources)
    records, summaries, geometry, pairs, comparisons = [], [], [], [], []
    for case in cases:
        # Only these model/physical arrays are decompressed. No training labels,
        # training features/costs, development bank, or weights are deserialized.
        npz_path = case["directory"] / "training.npz"
        sources.verify(npz_path, case["entry"]["files"]["training.npz"])
        with np.load(npz_path, allow_pickle=False) as stream:
            data = {key: stream[key].copy() for key in MODEL_FIELDS}
        model = infer_model(data, case["entry"])
        affine_evidence = check_affine_receipt(model, case["affine"])
        geometry.append(dict(case_id=case["entry"]["case_id"], D=case["D"], G=case["G"], K=case["G"],
                             local_formulation=model["local_formulation"], local_mapping=model["local_evidence"],
                             other_C_only_rows=model["other_C_only_rows"], affine_evidence=affine_evidence,
                             proof="Fixing C, fill both positive-profit basic items to 1-C. Verified local rows permit both complete modes; all other original rows depend only on unchanged C."))
        by_method = {}
        for method in METHODS:
            item, current = case["completions"][method], []
            for index in range(item["count"]):
                path = inside(item["folder"], f"dev{index:03d}.json")
                row = sources.read(path)
                current.append(verify_trial(row, case, method, index, model, path, sources))
            summary = aggregate(current)
            check_primary_summary(summary, item["document"]["summary"])
            summary.update(case_id=case["entry"]["case_id"], method=method,
                           count_authority=item["count_authority"])
            summaries.append(summary); records.extend(current); by_method[method] = current
        intersection = []
        for index, (ours, nd) in enumerate(zip(by_method["ours"], by_method["neural_diving"])):
            require(ours["cost"] == nd["cost"] and ours["feature"] == nd["feature"]
                    and ours["reference_objective"] == nd["reference_objective"],
                    f"{case['entry']['case_id']}/{index}: paired original cost/feature/reference differs")
            certificate = check_certificate(case, index, ours, sources)
            paired_valid = ours["primary_valid"] and nd["primary_valid"]
            difference = ours["primary_ratio"] - nd["primary_ratio"] if paired_valid else None
            if paired_valid:
                intersection.append((ours["primary_ratio"], nd["primary_ratio"]))
            pairs.append(dict(case_id=case["entry"]["case_id"], index=index, paired_inputs_identical=True,
                              reference_objective=ours["reference_objective"], reference_certificate=certificate,
                              both_primary_valid=paired_valid, paired_ratio_difference_ours_minus_ND=difference))
        comparisons.append(dict(case_id=case["entry"]["case_id"], all_paired_query_count=len(by_method["ours"]),
                                paired_primary_valid_intersection_N=len(intersection),
                                ours_ratio_mean_on_intersection=mean([a for a, _ in intersection]),
                                ND_ratio_mean_on_intersection=mean([b for _, b in intersection]),
                                mean_paired_ratio_difference_ours_minus_ND=mean([a - b for a, b in intersection]),
                                subset_rule="Intersection of original primary-valid timely returns; never difference of means on different subsets"))
    sources.recheck()
    return dict(schema="coupled-bundle-canonical-completion-audit-v1", status="COMPLETE_AUDITED",
                manifest_sha256=sources.hashes[str(manifest_path)], audit_script_sha256=file_sha(Path(__file__)),
                gate=dict(expected_cases=len(cases), ND_models_verified=len(cases), paired_method_completions_verified=2 * len(cases),
                          all_complete_before_any_trial_open=True, all_trial_byte_hashes_before_any_trial_parse=True),
                operations=dict(training=False, inference=False, solving=False, runtime_imports=False,
                                development_npz_bank_read=False, raw_results_modified=False),
                metric_rules=dict(primary="Original saved candidate only, original feasibility + its restriction + strict wall seconds<=30; no invalid zero-imputation",
                                  recovery_denominator="All predetermined N per case/method",
                                  canonical="Diagnostic completion only; its means use exactly the same primary-valid subset",
                                  decomposition="1-primary_ratio = upgrade_selection_ratio_loss + basic_completion_ratio_loss",
                                  gross_profit="0.5*G + sum((xi_g-0.5)*C_g); the real basic return is retained"),
                limitations=["Training NPZ whole-file hashes are checked, but training feature/cost/solution arrays are never decompressed; their fingerprint is cross-bound through saved guards and receipts.",
                             "Reference optimality is checked through optional archived certified oracle receipts, not a new solve.",
                             "Without an independent manifest development_count or --expected-count, N is bound to paired completion summaries and exact on-disk registries.",
                             "Sub-tolerance floating-point loss differences are retained without clipping; binary rounding allowance is reported."],
                missing_record_count=0, record_count=len(records), paired_query_count=len(pairs),
                bank_binding_evidence=bank_evidence,
                geometry=geometry, aggregates=summaries, records=records, paired_inputs=pairs,
                paired_comparisons=comparisons,
                source_sha256=dict(sorted(sources.hashes.items())))


def publish_new(path, report):
    raw = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Output must remain new: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".writing-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def self_test():
    """Synthetic arithmetic/matrix fixtures only; no path or real data is read."""
    A = sparse.csr_matrix([[1, 1, 1.5, 0, 0, 0], [0, 0, 0, 1, 1, 1.5], [0, 0, 1, 0, 0, 1]])
    data = dict(A_data=A.data, A_indices=A.indices, A_indptr=A.indptr, A_shape=np.array(A.shape),
                row_lower=np.full(3, -np.inf), row_upper=np.array([2., 2., 1.]),
                col_lower=np.zeros(6), col_upper=np.ones(6), integrality=np.ones(6),
                cost_offset=np.array([-.25, -.25, 0, -.25, -.25, 0]),
                cost_feature_coefficients=np.array([[0, 0], [0, 0], [-1, 0], [0, 0], [0, 0], [0, -1]]))
    entry = dict(D=6, feature_dimension=2, rows=3, K=2)
    model = infer_model(data, entry)
    feature = np.array([1., .2]); cost = model["offset"] + model["coefficient"] @ feature
    x = np.array([0., 0., 1., 1., 0., 0.]); fixed = {0: 0, 1: 0, 2: 1, 4: 0}
    result = canonical_diagnostic(model, x, cost, feature, -1.5, fixed)
    require(result["point"] == [0., 0., 1., 1., 1., 0.] and result["profit"] == 1.5
            and result["basic_completion_loss"] == .25 and result["violates_original_fixed"]
            and result["fixed_violations"] == [dict(column=4, imposed=0, canonical_value=1)]
            and not result["replaces_primary_return"], "Canonical fixture failed")
    require(abs((-cost @ x) / 1.5 - 5 / 6) < 1e-14, "Primary fixture ratio changed")
    valid = dict(primary_valid=True, primary_recovered=False, primary_ratio=5 / 6,
                 canonical=result, candidate_missing=False, original_check=dict(valid=True),
                 restriction_valid=True, timely=True)
    late = dict(valid, primary_valid=False, primary_ratio=None, timely=False)
    missing = dict(valid, primary_valid=False, primary_ratio=None, canonical=None,
                   candidate_missing=True, original_check=dict(valid=False), restriction_valid=False)
    stats = aggregate([valid, late, missing])
    require(stats["count"] == 3 and stats["ratio_denominator"] == 1 and stats["recovery_count"] == 0
            and stats["late_count"] == stats["missing_candidate_count"] == 1
            and stats["canonical_ratio_mean_same_primary_subset"] == 1.0, "Invalid/late denominator fixture failed")
    # Independent permutation verifies that mapping does not depend on 3g+j.
    perm = np.array([5, 0, 4, 2, 1, 3]); Ap = A[:, perm].tocsr()
    permuted = dict(data, A_data=Ap.data, A_indices=Ap.indices, A_indptr=Ap.indptr,
                    cost_offset=data["cost_offset"][perm], cost_feature_coefficients=data["cost_feature_coefficients"][perm])
    pm = infer_model(permuted, entry)
    pr = canonical_diagnostic(pm, x[perm], cost[perm], feature, -1.5, None)
    require(pr["point"] == np.asarray(result["point"])[perm].tolist(), "Permuted-coordinate fixture failed")
    hull_A = sparse.csr_matrix([[1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1]])
    hull_data = dict(data, A_data=hull_A.data, A_indices=hull_A.indices, A_indptr=hull_A.indptr,
                     A_shape=np.array(hull_A.shape), row_lower=np.full(5, -np.inf), row_upper=np.ones(5))
    hull = infer_model(hull_data, dict(entry, rows=5, local_formulation="pairwise_hull"))
    hr = canonical_diagnostic(hull, x, cost, feature, -1.5, fixed)
    require(hull["local_formulation"] == "pairwise_hull" and hr["point"] == result["point"]
            and hr["fixed_violations"] == result["fixed_violations"], "Pairwise local hull fixture failed")
    print(json.dumps(dict(self_test="PASS", synthetic_only=True, real_inputs_read=False,
                          primary_ratio=5 / 6, canonical_ratio=result["ratio"], completion_loss=.25,
                          completion_violates_fixed=True, all_N_denominator=3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--audit", action="store_true", help="Explicitly audit an entirely completed frozen cohort")
    operation.add_argument("--self-test", action="store_true", help="Synthetic software fixture only; no real inputs")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--driver-root", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, help="New, non-existing report JSON")
    parser.add_argument("--expected-count", type=int, help="Optional independent expected development queries per case")
    parser.add_argument("--bank-bindings", type=Path,
                        help="Optional existing fixed-development-bank attachment receipt; crosscheck every case SHA against both method guards")
    parser.add_argument("--certificates", choices=("none", "if-present", "required"), default="if-present")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if any(getattr(args, name) is None for name in ("manifest", "driver_root", "results", "output")):
        parser.error("--audit requires --manifest, --driver-root, --results and --output")
    if args.expected_count is not None and args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    try:
        report = audit(args)
        publish_new(args.output.resolve(), report)
    except (AuditError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(json.dumps(dict(status="AUDIT_BLOCKED", output_written=False,
                              error=f"{type(error).__name__}: {error}"), ensure_ascii=True), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(dict(status=report["status"], record_count=report["record_count"],
                          output=str(args.output.resolve()), report_sha256=file_sha(args.output.resolve())), ensure_ascii=True))


if __name__ == "__main__":
    main()
