"""Independent, fail-closed audit of the fixed twelve-case confirmation.

This tool never imports a producer, predictor, IP solver or evaluator. All 24
model roles are checked before opening a fresh bank. The reference optimum is
recomputed by a separate exact independent-set recursion. C-completion remains
an accounting diagnostic and never changes the returned point or main ratio.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import numpy as np
import sympy as sym

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent
GENERIC = FOLLOWUP / "objective-gap-designs-20261025/adapter/generic-binary-v2"
TOOLS = FOLLOWUP / "objective-gap-designs-20261025/coupled-bundle/confirmation-v1"
DEFAULT_RUN = HERE / "uniform-gross-confirmation"
DRIVER_SHA = "d7ff552f55775f90387444e8cbc6290ee371ca94960f7280f42c762c8768e8c2"
METHODS = ("ours", "neural_diving")
SETTINGS = ((16, 37), (20, 45), (32, 36), (32, 69))
SEEDS = (1009, 1013, 1019)
NAMESPACE = "coupled-bundle-independent-uniform-gross-confirmation-v1-20261025"
BINDINGS = {}


def require(value, label):
    if not value:
        raise ValueError(label)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    require(expected is None or value == expected, "File checksum differs: " + str(path))
    BINDINGS[str(path)] = value
    return value


def read(path):
    bind(path)
    return json.loads(Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def pack(path):
    bind(path)
    with np.load(path, allow_pickle=False) as stream:
        return {key: stream[key].copy() for key in stream.files}


def canonical(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def verify_payload(row, label):
    expected = row.get("record_payload_sha256")
    payload = {key: value for key, value in row.items() if key != "record_payload_sha256"}
    require(expected == hashlib.sha256(canonical(payload)).hexdigest(), label + ": payload checksum")


def same_number(value, expected, label):
    require((value is None and expected is None) or
        (value is not None and expected is not None and math.isfinite(float(value)) and
         math.isfinite(float(expected)) and
         abs(float(value) - float(expected)) <= 1e-10 * max(1., abs(float(expected)))), label)


def tolerance(reference):
    return 1e-6 + 1e-9 * abs(float(reference))


def near_reference(value, reference, label):
    require(value is not None and math.isfinite(float(value)) and
            abs(float(value) - float(reference)) <= tolerance(reference), label)


def check_map(mapping, label, relative_to=None):
    require(bool(mapping), label + ": empty checksum map")
    result = {}
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            require(relative_to is not None, label + ": unrooted path")
            path = Path(relative_to) / path
        result[str(path.resolve())] = bind(path, expected)
    return result


def array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def domain(case, point):
    if point is None:
        return False
    point = np.asarray(point, dtype=float)
    if point.shape != (case["D"],) or not np.isfinite(point).all():
        return False
    return bool(np.all(point >= -1e-6) and np.all(point <= 1. + 1e-6) and
        np.all(np.abs(point - np.rint(point)) <= 1e-6) and
        np.all(case["A"] @ point >= case["lower"] - 1e-6) and
        np.all(case["A"] @ point <= case["upper"] + 1e-6))


def affine_contains(case, point):
    integers = np.rint(point).astype(np.int64).astype(object)
    return bool(np.all(case["N"] @ (integers - case["anchor"]) == 0))


def derive_graph(A, lower, upper, G):
    """Derive graph and local feasible triples from actual original rows."""
    require(A.shape[1] == 3 * G and len(A) >= G and np.isneginf(lower).all(),
            "Original row dimensions/lower bounds differ")
    for g in range(G):
        cols = np.flatnonzero(A[g])
        require(np.array_equal(cols, np.arange(3 * g, 3 * g + 3)), "Unexpected bundle row")
        wa, wb, wc = A[g, cols]
        require(wa > 0 and wb > 0 and max(wa, wb) < wc < wa + wb and
                upper[g] == wa + wb, "Bundle capacity no longer has AB-or-C geometry")
        feasible = [bits for bits in itertools.product((0, 1), repeat=3)
                    if np.dot(A[g, cols], bits) <= upper[g] + 1e-10]
        require(set(feasible) == {(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                 (1, 1, 0), (0, 0, 1)}, "Unexpected local binary feasible set")
    edges = []
    for r in range(G, len(A)):
        cols = np.flatnonzero(A[r])
        require(len(cols) == 2 and all(int(j) % 3 == 2 for j in cols) and
                np.array_equal(A[r, cols], np.ones(2)) and upper[r] == 1,
                "Unexpected original conflict row")
        edges.append(tuple(sorted(int(j) // 3 for j in cols)))
    require(len(edges) == len(set(edges)), "Duplicate conflict edges")
    return tuple(sorted(edges))


def independent_optimum(features, edges):
    """Exact MWIS by component splitting and first-bit include/exclude.

    This deliberately uses no producer oracle, memo digest, solver output or
    producer vertex selection. Negative-premium vertices can always be omitted.
    """
    features = np.asarray(features, dtype=float)
    G = len(features)
    require(np.isfinite(features).all(), "Nonfinite MWIS weights")
    premium = features - .5
    adjacency = [0] * G
    for u, v in edges:
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u

    @lru_cache(maxsize=None)
    def visit(mask):
        if not mask:
            return 0., 0
        # Splitting disconnected subgraphs supplies an independent certificate.
        first = mask & -mask
        component, frontier = 0, first
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            component |= bit
            v = bit.bit_length() - 1
            frontier |= adjacency[v] & mask & ~component
        if component != mask:
            lv, lm = visit(component)
            rv, rm = visit(mask ^ component)
            return lv + rv, lm | rm
        v = first.bit_length() - 1
        without, wo_mask = visit(mask ^ first)
        included, in_mask = visit(mask & ~first & ~adjacency[v])
        included += float(premium[v])
        if included > without:
            return included, in_mask | first
        return without, wo_mask

    eligible = sum(1 << j for j in range(G) if premium[j] > 0)
    value, chosen = visit(eligible)
    C = np.asarray([(chosen >> j) & 1 for j in range(G)], dtype=np.int64)
    witness = np.column_stack((1 - C, 1 - C, C)).reshape(-1)
    return -(.5 * G + value), witness, visit.cache_info().currsize


def verify_protocol(protocol):
    expected = dict(schema="coupled-bundle-twelve-case-shared-physical-confirmation-protocol-v2",
        phase="independent_confirmation", expected_case_count=12, expected_physical_bank_count=9,
        expected_model_role_count=24, expected_neural_model_count=12, test_count_per_case=120,
        test_count_per_physical_bank=120, distinct_objective_count=1080, case_query_count=1440,
        method_return_count=2880, method_ids=list(METHODS), physical_seeds=list(SEEDS),
        xi_lower=.1, xi_upper=2.5, base_project_gross_return=.25, rng_namespace=NAMESPACE,
        rng_algorithm="numpy.random.Generator(numpy.random.PCG64(seed128))")
    for key, value in expected.items():
        require(protocol.get(key) == value, "Protocol differs: " + key)
    require({(s["G"], s["n_train"], s["D"], s["K"]) for s in protocol["settings"]} ==
            {(G, n, 3 * G, G) for G, n in SETTINGS}, "Protocol settings differ")


def preflight(freeze_path, expected_sha):
    """No fresh path is read or inspected anywhere in this function."""
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") is not None,
            "An explicit reviewed lowercase freeze SHA256 is required")
    bind(freeze_path, expected_sha)
    freeze = read(freeze_path)
    for key, value in dict(schema="coupled-bundle-all-twelve-models-freeze-v1", frozen=True,
        case_count=12, model_role_count=24, ND_model_count=12, physical_bank_count=9,
        confirmation_not_generated_before_freeze=True).items():
        require(freeze.get(key) == value, "Incomplete global freeze: " + key)
    sources = check_map(freeze["source_sha256"], "All globally frozen sources")
    for name in ("fresh_bank_cli.py", "fresh-confirmation-protocol.json", "confirmation_runner.py",
                 "verify_evaluator_equivalence.py"):
        path = str((TOOLS / name).resolve())
        require(sources.get(path) == sha(path), "Required fresh source absent from global freeze: " + name)
    # The scientific freeze governs sampling/prediction. This separate post-hoc
    # validator is pinned by its caller-supplied source SHA, not a science role.
    require(freeze.get("scientific_evaluation_equivalence", {}).get("passed") is True,
            "Original scientific evaluator equivalence gate missing")
    protocol = read(freeze["protocol_path"])
    bind(freeze["protocol_path"], freeze["protocol_sha256"])
    verify_protocol(protocol)
    registry = read(freeze["registry_path"])
    require(registry.get("schema") == "coupled-bundle-twelve-case-confirmation-registry-v1" and
            registry.get("all_twelve_cases_confirmed_regardless_of_development_gap") is True,
            "Fixed all-case registry differs")
    for source_seal in registry["source_seals"]:
        seal = read(source_seal)
        require(str(Path(source_seal).resolve()) in sources, "Registry seal absent from global sources")
        check_map(seal.get("source_sha256", seal.get("file_sha256")), "Registry source seal", Path(source_seal).parent)
    bind(GENERIC / "generic_binary_pilot.py", DRIVER_SHA)
    runtime = read(GENERIC / "runtime-seal.json")
    check_map(runtime["file_sha256"], "Generic shared runtime", GENERIC)
    cases = {}
    for cohort in registry["cohorts"]:
        manifest_path = Path(cohort["manifest_path"])
        manifest = read(manifest_path)
        require(manifest.get("schema") == "generic-binary-primary-cohort-v2" and
                manifest.get("training_only") is True and manifest.get("frozen_before_any_fit") is True and
                manifest.get("confirmation_included") is False, "Training-only manifest scope differs")
        data_seal_path = Path(manifest["source_data_seal"])
        bind(data_seal_path, manifest["source_data_seal_sha256"])
        data_seal = read(data_seal_path)
        for key in ("files", "source_hashes"):
            if key in data_seal:
                check_map(data_seal[key], "Training producer seal " + key, data_seal_path.parent)
        for e in manifest["cases"]:
            cid = e["case_id"]
            require(cid not in cases, "Duplicate model case")
            directory = Path(e["directory"])
            training_files = check_map(e["files"], "Selected training files", directory)
            a = pack(directory / "training.npz")
            meta = read(directory / "metadata.json")
            affine = read(directory / "affine.json")
            G = a["train_features"].shape[1]
            D = a["train_costs"].shape[1]
            n = len(a["train_costs"])
            seed = int(meta["seed"])
            require((G, n) in SETTINGS and seed in SEEDS and D == 3 * G and
                    e["D"] == D and e["K"] == G and e["n_train"] == n and
                    meta["case_id"] == cid and meta["n_train"] == n, "Training case identity differs")
            offset = np.tile([-.25, -.25, 0.], G)
            coefficient = np.zeros((D, G))
            coefficient[3 * np.arange(G) + 2, np.arange(G)] = -1
            require(np.array_equal(a["cost_offset"], offset) and
                    np.array_equal(a["cost_feature_coefficients"], coefficient), "Original cost map differs")
            require(np.array_equal(np.asarray(meta["metadata"]["cost_offset"]), offset) and
                    np.array_equal(np.asarray(meta["metadata"]["cost_feature_coefficients"]), coefficient),
                    "Metadata cost map differs")
            require(np.array_equal(a["train_costs"], offset + a["train_features"] @ coefficient.T) and
                    np.isfinite(a["train_features"]).all() and
                    np.all((a["train_features"] >= .1) & (a["train_features"] <= 2.5)),
                    "Training feature/cost law differs")
            require(np.array_equal(a["col_lower"], np.zeros(D)) and
                    np.array_equal(a["col_upper"], np.ones(D)) and
                    np.array_equal(a["integrality"], np.ones(D)), "Original binary bounds differ")
            A = np.zeros(tuple(a["A_shape"]))
            for r in range(len(A)):
                for k in range(int(a["A_indptr"][r]), int(a["A_indptr"][r + 1])):
                    A[r, int(a["A_indices"][k])] += a["A_data"][k]
            edges = derive_graph(A, a["row_lower"], a["row_upper"], G)
            require(edges == tuple(sorted(tuple(sorted(edge)) for edge in meta["metadata"]["edges"])),
                    "Original constraints and metadata graph disagree")
            Y = a["train_solutions"]
            require(Y.shape == (n, D) and np.isfinite(Y).all() and
                    np.array_equal(Y, np.rint(Y)), "Malformed integer training labels")
            Y = Y.astype(np.int64)
            N = np.asarray([[sym.Rational(v) for v in row] for row in affine["normals_exact"]], dtype=object)
            anchor = np.asarray([sym.Rational(v) for v in affine["anchor"]], dtype=object)
            basis = sym.Matrix(affine["basis"])
            rank = sym.Matrix((Y[1:] - Y[0]).T.tolist()).rank()
            require(rank == G == affine["rank"] and N.shape == (D - G, D) and
                    sym.Matrix(N.tolist()).rank() == D - G and basis.shape == (D, G) and
                    basis.rank() == G and sym.Matrix(N.tolist()) * basis == sym.zeros(D-G, G) and
                    np.all((Y.astype(object) - anchor) @ N.T == 0), "Exact learned affine space differs")
            case = dict(cid=cid, G=G, D=D, n=n, K=G, seed=seed, A=A,
                lower=a["row_lower"], upper=a["row_upper"], N=N, anchor=anchor,
                edges=edges, offset=offset, coefficient=coefficient, affine=affine,
                folder=directory, model_root=Path(cohort["model_root"]), arrays=a)
            baseline = np.tile([1, 1, 0], G)
            for j in range(-1, G):
                witness = baseline.copy()
                if j >= 0:
                    witness[3*j:3*j+3] = [0, 0, 1]
                require(domain(case, witness) and affine_contains(case, witness), "Global geometry witness absent")
            for feature, cost, x, reference in zip(a["train_features"], a["train_costs"], Y, a["train_objectives"]):
                require(domain(case, x), "Original training label infeasible")
                near_reference(float(cost @ x), reference, "Original training objective differs")
                optimum, witness, _ = independent_optimum(feature, edges)
                near_reference(reference, optimum, "Training label fails independent global optimum")
                require(domain(case, witness), "Independent training witness infeasible")
            training_guard = dict(case_id=cid, training_array_hash=array_hash(*[a[k] for k in
                ("train_features", "train_costs", "train_solutions", "train_objectives")]),
                source_hashes={name: sha(directory / name) for name in ("training.npz", "metadata.json")},
                target_K=G, pilot_runner_sha256=DRIVER_SHA, runtime_seal_sha256=sha(GENERIC / "runtime-seal.json"),
                pilot_protocol_sha256=sha(GENERIC / "GENERIC_BINARY_PROTOCOL.md"),
                selection_manifest_sha256=sha(manifest_path), all_shared_source_sha256=runtime["file_sha256"],
                cost_map="signed affine; offset + coefficients @ feature", development_used_in_training=False,
                new_confirmation_created=False)
            case["training_guard"] = training_guard
            out = case["model_root"]
            nd_folder = out / "models" / cid / "neural_diving"
            marker = read(nd_folder / "complete.json")
            require(marker.get("guard") == dict(training_guard, method="neural_diving") and
                    marker.get("sanity_gate_passed") is True and marker.get("training_label_count") == n and
                    "complete-predictor.pt" in marker.get("artifact_sha256", {}), "Incomplete ND training role")
            nd_files = {str((nd_folder / "complete.json").resolve()): sha(nd_folder / "complete.json")}
            nd_files.update(check_map(marker["artifact_sha256"], "Completed ND artifacts", nd_folder))
            receipt_path = out / "training-case-receipts" / (cid + ".json")
            receipt = read(receipt_path)
            require(receipt.get("training_guard") == dict(training_guard, method="case"), "Ours training receipt guard differs")
            for key in ("anchor", "basis", "normals_exact", "normals", "rank", "representation"):
                require(receipt["exact_affine"][key] == affine[key], "Ours representation receipt differs")
            cert_folder = out / "training-certificates" / cid
            cert_complete = read(cert_folder / "complete.json")
            oracle_guard = dict(training_guard, method="training_oracle")
            require(cert_complete.get("guard") == oracle_guard and
                    cert_complete.get("certified_label_count") == n and
                    cert_complete.get("all_labels_globally_certified") is True, "Incomplete Ours training certification")
            ours_files = {str(receipt_path.resolve()): sha(receipt_path),
                          str((cert_folder / "complete.json").resolve()): sha(cert_folder / "complete.json")}
            for i, reference in enumerate(a["train_objectives"]):
                p = cert_folder / f"train{i:03d}.json"
                row = read(p)
                require(row.get("guard") == oracle_guard and row.get("index") == i, "Training certificate identity differs")
                near_reference(row["source_objective"], reference, "Training source certificate differs")
                oracle = row["oracle"]
                require(oracle.get("valid") is True and oracle.get("certified") is True, "Uncertified training label")
                near_reference(oracle["objective"], reference, "Training certificate primal differs")
                near_reference(oracle["dual_bound"], reference, "Training certificate dual differs")
                ours_files[str(p.resolve())] = sha(p)
            methods = dict(ours=dict(guard=dict(training_guard, method="ours"), files=ours_files),
                           neural_diving=dict(guard=dict(training_guard, method="neural_diving"), files=nd_files))
            # Rebuild every frozen entry, rather than accepting a subset of roles.
            expected = dict(case_id=cid, manifest_path=str(manifest_path.resolve()),
                training_directory=str(directory.resolve()), model_root=str(out.resolve()),
                G=G, D=D, K=G, n_train=n, seed=seed, physical_id=f"physical_G{G}_seed{seed}",
                physical_model_sha256=array_hash(a["A_data"], a["A_indices"], a["A_indptr"],
                    a["row_lower"], a["row_upper"], a["col_lower"], a["col_upper"], a["integrality"],
                    np.asarray(meta["metadata"]["cost_offset"]), np.asarray(meta["metadata"]["cost_feature_coefficients"])),
                training_files=training_files, methods=methods)
            require(freeze["cases"].get(cid) == expected, "Entire frozen model role entry differs: " + cid)
            case["freeze_entry"] = expected
            cases[cid] = case
    require(len(cases) == 12 and set(cases) == set(freeze["cases"]) and
            {(c["G"], c["n"], c["seed"]) for c in cases.values()} ==
            {(G, n, seed) for G, n in SETTINGS for seed in SEEDS}, "Incomplete fixed twelve-case population")
    physical = defaultdict(list)
    for cid, case in cases.items():
        physical[case["freeze_entry"]["physical_id"]].append(cid)
    require(len(physical) == 9, "Incorrect physical-bank population")
    for pid, ids in physical.items():
        require(len({cases[cid]["freeze_entry"]["physical_model_sha256"] for cid in ids}) == 1,
                "Paired original physical models differ")
        if len(ids) == 2:
            short, long = sorted((cases[cid] for cid in ids), key=lambda c: c["n"])
            require(short["G"] == 32 and (short["n"], long["n"]) == (36, 69), "Unexpected paired prefixes")
            for key in ("train_features", "train_costs", "train_solutions", "train_objectives"):
                require(np.array_equal(short["arrays"][key], long["arrays"][key][:36]), "Training prefix pairing differs")
    return freeze, protocol, cases, physical


def load_banks(freeze, protocol, cases, physical, freeze_sha):
    banks = {}
    total_states = 0
    for pid, ids0 in sorted(physical.items()):
        ids = sorted(ids0)
        case = cases[ids[0]]
        G, D, seed = case["G"], case["D"], case["seed"]
        folder = Path(freeze["bank_root"]) / pid
        meta = read(folder / "metadata.json")
        ref_path = folder / "reference-certificates.json"
        npz_path = folder / "confirmation.npz"
        label = NAMESPACE + "|G=" + str(G) + "|physical_seed=" + str(seed)
        seed128 = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:16], "little")
        bindings = {cid: dict(K=cases[cid]["K"], n_train=cases[cid]["n"],
                    training_files=freeze["cases"][cid]["training_files"],
                    model_role_files={m: freeze["cases"][cid]["methods"][m]["files"] for m in METHODS}) for cid in ids}
        required = dict(schema="shared-physical-confirmation-bank-v2", physical_id=pid, G=G, D=D,
            seed=seed, count=120, case_ids=ids, case_bindings=bindings, phase="independent_confirmation",
            confirmation=True, model_freeze_sha256=freeze_sha, protocol_sha256=freeze["protocol_sha256"],
            npz_sha256=bind(npz_path), reference_certificates_sha256=bind(ref_path), rng_namespace=NAMESPACE,
            rng_algorithm=protocol["rng_algorithm"], rng_seed128=seed128, rng_seed_input=label,
            rng_seed_independent_of_training_count_and_models=True,
            all_global_guards_revalidated_before_rng=True, generator_sha256=sha(TOOLS / "fresh_bank_cli.py"),
            source_freeze_count=12, independently_certified_original_references=True)
        for key, value in required.items():
            require(meta.get(key) == value, pid + ": bank metadata differs: " + key)
        bank = pack(npz_path)
        require(set(bank) == {"features", "costs", "solutions", "objectives"}, "Fresh NPZ keys differ")
        for key, shape in dict(features=(120, G), costs=(120, D), solutions=(120, D), objectives=(120,)).items():
            require(bank[key].shape == shape and np.isfinite(bank[key]).all(), "Fresh array shape/finite differs")
        # This RNG reconstruction happens only after the complete all-role gate.
        expected_features = np.random.Generator(np.random.PCG64(seed128)).uniform(.1, 2.5, size=(120, G))
        require(np.array_equal(bank["features"], expected_features), "Fresh RNG namespace/law pairing differs")
        expected_costs = np.full((120, D), -.25)
        expected_costs[:, 2::3] = expected_features
        expected_costs[:, 2::3] *= -1
        require(np.array_equal(bank["costs"], expected_costs) and np.all(bank["objectives"] < 0),
                "Original gross objective map/sign differs")
        cert = read(ref_path)
        guard = dict(model_freeze_sha256=freeze_sha, protocol_sha256=freeze["protocol_sha256"])
        for key, value in dict(schema="coupled-bundle-confirmation-reference-certificates-v1",
            physical_id=pid, case_ids=ids, count=120, guard=guard, all_original_IP_certified=True,
            all_independent_references_agree=True, original_solver_threads=1,
            original_constraints_unmodified=True, no_full_optimizer_map_enumeration=True).items():
            require(cert.get(key) == value, pid + ": reference manifest differs: " + key)
        expected_files = {f"reference-records/ref{i:03d}.json" for i in range(120)}
        require(set(cert["record_file_sha256"]) == expected_files and
                {"reference-records/" + p.name for p in (folder / "reference-records").glob("ref*.json")} == expected_files,
                "Reference population differs")
        for i in range(120):
            path = folder / f"reference-records/ref{i:03d}.json"
            bind(path, cert["record_file_sha256"][f"reference-records/ref{i:03d}.json"])
            row = read(path)
            verify_payload(row, pid + f"/reference/{i}")
            require(row.get("physical_id") == pid and row.get("case_ids") == ids and
                    row.get("index") == i and row.get("guard") == guard, "Reference row identity differs")
            feature, cost, point, ref = (bank[k][i] for k in ("features", "costs", "solutions", "objectives"))
            require(np.array_equal(row["feature"], feature) and np.array_equal(row["cost"], cost) and
                    np.array_equal(row["solution"], point), "Reference row and NPZ differ")
            optimum, independent_x, states = independent_optimum(feature, case["edges"])
            total_states += states
            require(domain(case, point) and domain(case, independent_x), "Original reference witness infeasible")
            near_reference(float(cost @ point), ref, "Reference original objective differs")
            near_reference(ref, optimum, "Independent exact MWIS reference disagrees")
            near_reference(float(cost @ independent_x), optimum, "Independent MWIS witness accounting differs")
            oracle = row["highs"]
            require(oracle.get("valid") is True and oracle.get("certified") is True, "Missing HiGHS certificate")
            for value in (row["objective"], row["independent_objective"], oracle["objective"], oracle["dual_bound"],
                          row["independent_mwis"]["objective"]):
                near_reference(value, optimum, "Reference primal/dual/producer oracle disagreement")
            require(domain(case, row["independent_solution"]), "Producer independent witness infeasible")
            near_reference(float(cost @ row["independent_solution"]), optimum, "Producer witness objective differs")
            producer = row["independent_mwis"]
            producer_C = np.rint(np.asarray(row["independent_solution"])[2::3]).astype(np.int64)
            selected = sum(int(value) << j for j, value in enumerate(producer_C))
            require(producer.get("selected_mask") == selected and
                    producer.get("selected_vertices") == np.flatnonzero(producer_C).tolist() and
                    producer.get("positive_premium_mask") == sum(1 << j for j in range(G) if feature[j] > .5) and
                    producer.get("negative_and_zero_premiums_removed") is True and
                    producer.get("full_optimizer_map_enumeration") is False and
                    producer.get("frozen_resource_limits") == protocol["independent_reference_limits"] and
                    1 <= producer.get("memo_state_count", 0) <= protocol["independent_reference_limits"]["maximum_memo_states"],
                    "Producer MWIS witness/mask/protocol certificate differs")
            require(row.get("warm_start_used") is False and row.get("objective_is_original_gross_profit_negative") is True and
                    row.get("no_artificial_objective_shift") is True and row.get("timing_not_isolated") is True and
                    row.get("original_training_label_generation_seconds") is None,
                    "Reference objective/timing provenance differs")
            for timer in ("cold_Full_wall_seconds", "cold_Full_cpu_seconds"):
                require(math.isfinite(float(row[timer])) and float(row[timer]) >= 0, "Invalid reference generation timer")
        banks[pid] = (bank, meta)
    return banks, total_states


def score(case, method, row, cost, feature, reference):
    """Independently score the actual point, including late and invalid returns."""
    D, K, G = case["D"], case["K"], case["G"]
    x = None if row["solution"] is None else np.asarray(row["solution"], dtype=float)
    feasible = domain(case, x)
    fixed = row["fixed"]
    if method == "ours":
        actual = K
        require(fixed is None, "Ours unexpectedly has coordinate fixings")
        restriction = bool(feasible and affine_contains(case, x))
    else:
        require(method == "neural_diving", "Unexpected method")
        require(fixed is None or isinstance(fixed, dict), "Malformed fixing mapping")
        actual = None if fixed is None else D - len(fixed)
        norm = {}
        if fixed is not None:
            for j, value in fixed.items():
                col = int(j)
                require(str(col) == str(j) and 0 <= col < D and col not in norm and
                        value in (0, 1), "Malformed original variable fixing")
                norm[col] = value
            require(len(norm) == D - K, "Returned fixing restriction does not match K")
        restriction = bool(feasible and fixed is not None and
                           all(abs(float(x[j]) - value) <= 1e-6 for j, value in norm.items()))
    seconds = float(row["seconds"])
    require(math.isfinite(seconds) and seconds >= 0, "Invalid end-to-end evaluation wall time")
    timely = seconds <= 30.
    valid = bool(feasible and restriction and timely and actual == K)
    raw = float(cost @ x) if feasible else None
    objective = raw if valid else None
    ratio = objective / float(reference) if valid else None
    recovered = bool(valid and abs(objective - reference) <= tolerance(reference))
    if valid:
        require(math.isfinite(ratio) and ratio <= 1. + tolerance(reference) / abs(reference),
                "Returned gross ratio exceeds original certified optimum")
    completion_profit = completion_ratio = None
    if feasible:
        C = np.rint(x[2::3]).astype(np.int64)
        completion = np.column_stack((1 - C, 1 - C, C)).reshape(-1)
        require(domain(case, completion), "Diagnostic C-completion infeasible in original model")
        completion_profit = .5 * G + float((feature - .5) @ C)
        same_number(-float(cost @ completion), completion_profit, "Diagnostic completion accounting differs")
        require(completion_profit >= -raw - 1e-6 * float(np.sum(np.abs(cost))) - 1e-8, "Diagnostic completion lowers gross profit beyond original-domain rounding tolerance")
        completion_ratio = completion_profit / (-float(reference))
    return dict(actual_K=actual, original_feasible=feasible, restriction_valid=restriction,
        timely=timely, valid=valid, recovered=recovered, raw_objective=raw, objective=objective,
        reference_objective=float(reference), profit_ratio=ratio, completion_ratio=completion_ratio,
        completion_profit=completion_profit)


def audit_returns(freeze, cases, banks, freeze_path, results_root):
    cells, diagnostics = [], []
    all_scores = defaultdict(list)
    for cid, case in sorted(cases.items()):
        entry = freeze["cases"][cid]
        bank, meta = banks[entry["physical_id"]]
        for method in METHODS:
            folder = results_root / cid / method
            receipt = read(folder / "complete.json")
            guard = dict(training=dict(case["training_guard"], method=method),
                model_freeze_sha256=sha(freeze_path), protocol_sha256=freeze["protocol_sha256"],
                confirmation_bank_sha256=meta["npz_sha256"],
                confirmation_metadata_sha256=sha(Path(freeze["bank_root"]) / entry["physical_id"] / "metadata.json"),
                reference_certificates_sha256=meta["reference_certificates_sha256"],
                wrapper_sha256=sha(TOOLS / "confirmation_runner.py"),
                model_role_files_sha256=entry["methods"][method]["files"], budget=30., phase="independent_confirmation")
            require(receipt.get("guard") == guard, "Completed confirmation role guard differs")
            expected_files = {f"test{i:03d}.json" for i in range(120)}
            require(set(receipt["test_record_sha256"]) == expected_files and
                    {p.name for p in folder.glob("test*.json")} == expected_files, "Method return population differs")
            scores = []
            for i in range(120):
                path = folder / f"test{i:03d}.json"
                bind(path, receipt["test_record_sha256"][path.name])
                row = read(path)
                verify_payload(row, cid + "/" + method + "/" + str(i))
                for key, value in dict(guard=guard, case_id=cid, method=method, index=i,
                    n_train=case["n"], K=case["K"], D=case["D"], budget_seconds=30.,
                    experiment_phase="independent_confirmation", confirmation=True).items():
                    require(row.get(key) == value, "Method row identity differs: " + key)
                cost, feature, reference = bank["costs"][i], bank["features"][i], float(bank["objectives"][i])
                require(np.array_equal(row["cost"], cost) and np.array_equal(row["feature"], feature),
                        "Actual evaluation objective differs from fixed bank")
                s = score(case, method, row, cost, feature, reference)
                for key in ("actual_K", "original_feasible", "restriction_valid", "timely", "valid", "recovered"):
                    require(row[key] == s[key], "Independently derived return flag differs: " + key)
                for key in ("raw_objective", "objective", "reference_objective", "profit_ratio"):
                    same_number(row[key], s[key], "Independently derived original objective metric differs: " + key)
                scores.append(s)
                diagnostics.append(dict(case_id=cid, method=method, index=i, primary_valid=s["valid"],
                    primary_ratio=s["profit_ratio"], primary_gross_profit=None if s["objective"] is None else -s["objective"],
                    candidate_C_completion_gross_profit=s["completion_profit"],
                    candidate_C_completion_ratio=s["completion_ratio"],
                    primary_eligible_C_completion_ratio=s["completion_ratio"] if s["valid"] else None,
                    diagnostic_does_not_replace_return=True))
            good = [s for s in scores if s["valid"]]
            ratios = [s["profit_ratio"] for s in good]
            recovered = sum(s["recovered"] for s in scores)
            summary_expected = dict(case_id=cid, method=method, phase="independent_confirmation",
                n_train=case["n"], K=case["K"], D=case["D"], count=120, recovery_count=recovered,
                recovery_percent=100. * recovered / 120,
                profit_ratio_mean_valid=math.fsum(ratios) / len(ratios) if ratios else None,
                ratio_denominator=len(ratios), negative_returned_profit_count=sum(r < 0 for r in ratios),
                independent_confirmation=True)
            summary = receipt["summary"]
            require(set(summary) == set(summary_expected), "Unexpected confirmation summary schema")
            for key, value in summary_expected.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) or value is None:
                    same_number(summary[key], value, "Independent summary differs: " + key)
                else:
                    require(summary[key] == value, "Summary identity differs: " + key)
            cell = dict(case_id=cid, G=case["G"], n_train=case["n"], seed=case["seed"],
                method=method, D=case["D"], K=case["K"], total=120, recovered=recovered,
                recovery_percent=100. * recovered / 120, ratio_denominator=len(good),
                profit_ratio_mean_valid=summary_expected["profit_ratio_mean_valid"],
                C_completion_ratio_mean_primary_valid=math.fsum(s["completion_ratio"] for s in good) / len(good) if good else None,
                invalid_count=120-len(good), overbudget_count=sum(not s["timely"] for s in scores))
            cells.append(cell)
            all_scores[(case["G"], case["n"], method)].extend(scores)
    pooled = []
    for (G, n, method), scores in sorted(all_scores.items()):
        require(len(scores) == 360, "Every setting/method requires the full three-seed 360 denominator")
        good = [s for s in scores if s["valid"]]
        recovered = sum(s["recovered"] for s in scores)
        pooled.append(dict(G=G, n_train=n, D=3*G, K=G, method=method, total=360,
            recovered=recovered, recovery_percent=100.*recovered/360, ratio_denominator=len(good),
            profit_ratio_mean_valid=math.fsum(s["profit_ratio"] for s in good)/len(good) if good else None,
            C_completion_ratio_mean_primary_valid=math.fsum(s["completion_ratio"] for s in good)/len(good) if good else None,
            invalid_count=360-len(good), overbudget_count=sum(not s["timely"] for s in scores)))
    require(len(cells) == 24 and len(pooled) == 8 and len(diagnostics) == 2880,
            "Final confirmation population/cell count differs")
    return cells, pooled, diagnostics


def self_test():
    edges = ((0, 1), (1, 2), (1, 3))
    # Includes tied and negative premiums, with an independent brute oracle.
    for feature in ([.1, .2, .3, .4], [.5, .5, .5, .5], [1., 2., 1.2, 2.5], [2.5, .1, 2.4, 2.]):
        optimum, x, _ = independent_optimum(feature, edges)
        values = []
        for C in itertools.product((0, 1), repeat=4):
            if all(C[u] + C[v] <= 1 for u, v in edges):
                values.append(.5*4 + sum((v-.5)*b for v, b in zip(feature, C)))
        same_number(optimum, -max(values), "Toy exact MWIS versus brute-force mismatch")
        require(all(x[3*u+2] + x[3*v+2] <= 1 for u, v in edges), "Toy MWIS witness conflict")
    case = dict(D=3, G=1, K=1, A=np.asarray([[1., 1., 1.5]]),
                lower=np.asarray([-np.inf]), upper=np.asarray([2.]),
                N=np.asarray([[1, -1, 0], [1, 0, 1]], dtype=object),
                anchor=np.asarray([1, 1, 0], dtype=object))
    cost, feature = np.asarray([-.25, -.25, -2.]), np.asarray([2.])
    point = [0, 0, 1]
    base = dict(solution=point, fixed=None, seconds=1.)
    require(score(case, "ours", base, cost, feature, -2.)["recovered"], "Toy exact return")
    late = score(case, "ours", dict(base, seconds=30.000001), cost, feature, -2.)
    require(not late["valid"] and late["objective"] is None and not late["recovered"], "Late return in primary score")
    invalid = score(case, "ours", dict(base, solution=[1,1,1]), cost, feature, -2.)
    require(not invalid["original_feasible"] and not invalid["restriction_valid"] and invalid["profit_ratio"] is None,
            "Invalid original point restriction semantics")
    s = score(case, "neural_diving", dict(base, fixed={"0":0,"1":0}), cost, feature, -2.)
    require(s["valid"] and s["actual_K"] == 1, "Toy exact K variable fixing")
    bad_fix = score(case, "neural_diving", dict(base, fixed={"0":1,"1":0}), cost, feature, -2.)
    require(not bad_fix["valid"] and not bad_fix["restriction_valid"], "Toy violated fixing accepted")
    require(sum(s["recovered"] for s in (late, invalid)) == 0 and
            [s for s in (late, invalid) if s["valid"]] == [], "Zero-valid denominator fixture")
    # Original feasibility tolerates 1e-6 per coordinate; diagnostics must too.
    near_integer = score(case, "neural_diving", dict(base, solution=[1.+4e-7,1.+4e-7,0.], fixed={"0":1,"1":1}), np.asarray([-.25,-.25,-.1]), np.asarray([.1]), -.5)
    require(near_integer["valid"] and near_integer["recovered"], "Near-integer return diagnostic is stricter than accepted domain")
    row = dict(a=[1,2], b=None)
    row["record_payload_sha256"] = hashlib.sha256(canonical(row)).hexdigest()
    verify_payload(row, "Toy payload")
    print(json.dumps(dict(status="SELF_TEST_PASS", fresh_banks_read=False, solver_or_training_run=False,
                          source_sha256=sha(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_RUN / "root_all_models_freeze.json")
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--expected-audit-source-sha256")
    parser.add_argument("--results", type=Path, default=DEFAULT_RUN / "results")
    parser.add_argument("--report", type=Path, default=DEFAULT_RUN / "independent-confirmation-audit.json")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(re.fullmatch(r"[0-9a-f]{64}", args.expected_audit_source_sha256 or "") is not None and
            sha(__file__) == args.expected_audit_source_sha256,
            "Pin this independent validator with --expected-audit-source-sha256 before running")
    freeze, protocol, cases, physical = preflight(args.freeze, args.expected_freeze_sha256)
    if args.preflight_only:
        print(json.dumps(dict(status="ALL_24_MODEL_ROLES_VALIDATED", cases=12, physical_banks=9,
                              fresh_banks_read=False, source_sha256=sha(__file__))))
        return
    banks, states = load_banks(freeze, protocol, cases, physical, args.expected_freeze_sha256)
    cells, pooled, diagnostics = audit_returns(freeze, cases, banks, args.freeze, args.results)
    # Detect source/model mutation during the audit; do not publish a stale pass.
    preflight(args.freeze, args.expected_freeze_sha256)
    report = dict(schema="uniform-gross-independent-confirmation-audit-v1", status="PASS",
        discrepancy_count=0, audited_at_utc=datetime.now(timezone.utc).isoformat(),
        model_freeze_sha256=args.expected_freeze_sha256, protocol_sha256=freeze["protocol_sha256"],
        audit_source_sha256=sha(__file__), all_model_roles_validated_before_any_bank=True,
        case_count=12, model_role_count=24, physical_bank_count=9, distinct_objective_count=1080,
        case_query_count=1440, raw_method_returns=2880, recovery_denominator_per_setting_method=360,
        independent_reference_memo_states=states, per_case_cells=cells, cells=pooled,
        per_return_C_completion_diagnostics=diagnostics, source_and_artifact_bindings=BINDINGS,
        new_solver_or_training_run=False, no_favorable_seed_or_winner_selection=True,
        timing_scope="30s returned wall time is a primary validity condition. Cold reference-generation timings are not isolated solve-speed measurements and support no speedup claim.",
        diagnostic_scope="Canonical C-completion can violate the learned baseline fixings. It is an extra original-feasible accounting diagnostic and never replaces a returned point, primary ratio or full recovery denominator.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="PASS", discrepancy_count=0, raw_method_returns=2880,
        physical_references=1080, cells=pooled, report=str(args.report), report_sha256=sha(args.report),
        source_sha256=sha(__file__)), allow_nan=False))


if __name__ == "__main__":
    main()
