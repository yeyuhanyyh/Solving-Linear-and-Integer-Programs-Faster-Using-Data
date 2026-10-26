"""Generate the pre-specified, independent stream 2 confirmation banks.

Import and --validate-only perform no objective RNG creation, bank inspection,
bank read, or solve. An explicit expected freeze digest and semantic validation
of all 15 frozen model roles precede every physical-bank boundary. Original
gross-profit costs and the sealed cold-Full / independent-MWIS reference path
are retained. Existing or incomplete banks are never redrawn or overwritten.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import time
import uuid
import zipfile


TOOLS = Path(__file__).resolve().parent
PROTOCOL = TOOLS / "stream2-protocol.json"
RUNNER = TOOLS / "stream2_runner.py"
ORIGINAL_GENERATOR = TOOLS.parents[1] / "confirmation-v1" / "fresh_bank_cli.py"
ORIGINAL_GENERATOR_SHA256 = "62a6557c74bfcc791c74bc462252f3dc8cd7ce67d4e98c2eb5e9f2f4908eab09"
METHODS = ("ours", "neural_diving_original", "neural_diving_continued", "svm", "mlopt")
SEEDS = (1009, 1013, 1019)
RNG_NAMESPACE = "coupled-bundle-independent-uniform-gross-stream2-fixed-budget-v1-20261025"
TEST_COUNT = 120


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def _verify_source_bindings(freeze):
    required = (Path(__file__).resolve(), PROTOCOL.resolve(), RUNNER.resolve(), ORIGINAL_GENERATOR.resolve())
    for path in required:
        if freeze["source_sha256"].get(str(path)) != sha(path):
            raise ValueError("Stream 2 generator/protocol/runner/reference source is not globally frozen: " + str(path))
    if sha(ORIGINAL_GENERATOR) != ORIGINAL_GENERATOR_SHA256:
        raise ValueError("The sealed original dual-reference generator changed")
    if Path(freeze["protocol_path"]).resolve() != PROTOCOL.resolve() or freeze["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Globally frozen stream 2 protocol differs")


def load_validated_runner(freeze_path, expected_freeze_sha256):
    """Finish the global 15-role gate before any bank directory is inspected."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_freeze_sha256):
        raise ValueError("A literal lowercase expected freeze SHA256 is required")
    freeze_path = Path(freeze_path)
    if sha(freeze_path) != expected_freeze_sha256:
        raise ValueError("Stream 2 freeze differs from the explicitly expected digest")
    initial = read(freeze_path)
    if initial.get("schema") != "stream2-all-five-role-models-freeze-v1" or initial.get("frozen") is not True:
        raise ValueError("Need the completed stream 2 all-five-role global freeze")
    # Verify the actual runner and generator before importing the runner. Only
    # source files and the freeze itself are read before the semantic gate.
    _verify_source_bindings(initial)
    spec = importlib.util.spec_from_file_location("guarded_stream2_confirmation_runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    freeze, cases = runner.validate_global_freeze(freeze_path, expected_sha=expected_freeze_sha256)
    if sha(freeze_path) != expected_freeze_sha256 or freeze != initial:
        raise ValueError("Stream 2 freeze changed while validating all 15 roles")
    _verify_source_bindings(freeze)
    if freeze.get("case_count") != 3 or freeze.get("model_role_count") != 15 or freeze.get("physical_bank_count") != 3:
        raise ValueError("All three cases / 15 roles / three physical banks are required")
    protocol = read(PROTOCOL)
    required_protocol = dict(expected_case_count=3, expected_model_role_count=15, expected_physical_bank_count=3,
        test_count_per_case=TEST_COUNT, method_ids=list(METHODS), physical_seeds=list(SEEDS),
        xi_lower=0.1, xi_upper=2.5, base_project_gross_return=0.25, rng_namespace=RNG_NAMESPACE,
        rng_algorithm="numpy.random.Generator(numpy.random.PCG64(seed128))")
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError("The fixed stream 2 protocol differs: " + key)
    entries = freeze["cases"]
    actual = {(int(e["G"]), int(e["D"]), int(e["K"]), int(e["n_train"]), int(e["seed"])) for e in entries.values()}
    expected = {(32, 96, 32, 36, seed) for seed in SEEDS}
    if len(cases) != 3 or set(cases) != set(entries) or actual != expected:
        raise ValueError("The complete fixed G32 / n36 stream 2 cohort is required")
    physical = {}
    for cid, entry in entries.items():
        pid = f"stream2_G32_seed{entry['seed']}"
        case = cases[cid]
        if entry["physical_id"] != pid or set(entry["methods"]) != set(METHODS):
            raise ValueError("Stream 2 physical identity or required model roles differ")
        if not entry.get("training_files") or not entry.get("physical_model_sha256"):
            raise ValueError("Training and original physical-model bindings are required")
        if any(not entry["methods"][method].get("files") for method in METHODS):
            raise ValueError("An actual model-role artifact binding is absent")
        if (case.n_train != 36 or case.seed != entry["seed"] or case.train_features.shape[1] != 32
                or case.model.n_variables != 96 or int(case.affine["rank"]) != 32):
            raise ValueError("Loaded training case differs from the fixed stream 2 cohort")
        physical.setdefault(pid, []).append(cid)
    if len(physical) != 3 or any(len(ids) != 1 for ids in physical.values()):
        raise ValueError("Each stream 2 seed requires its own single shared five-role bank")
    return runner, freeze, cases, protocol, physical


def _reference_source(freeze):
    """Load only the fixed original graph and exact-MWIS helpers after the gate."""
    path = ORIGINAL_GENERATOR.resolve()
    if sha(path) != ORIGINAL_GENERATOR_SHA256 or freeze["source_sha256"].get(str(path)) != ORIGINAL_GENERATOR_SHA256:
        raise ValueError("Original independent-reference helpers are not frozen verbatim")
    spec = importlib.util.spec_from_file_location("stream2_sealed_original_reference_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_npz_new(path, arrays, np):
    """Stable NPZ bytes; no timestamp or model-dependent ordering enters RNG."""
    with Path(path).open("xb") as target:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for key in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
                member = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(member, buffer.getvalue())


def case_bindings(ids, freeze):
    return {cid: dict(K=freeze["cases"][cid]["K"], n_train=freeze["cases"][cid]["n_train"],
        training_files=freeze["cases"][cid]["training_files"],
        model_role_files={method: freeze["cases"][cid]["methods"][method]["files"] for method in METHODS}) for cid in ids}


def generate_bank(pid, ids, runner, freeze, cases, protocol, freeze_path, expected_sha):
    # Revalidate the full 15-role population and explicit digest before any
    # existing-bank inspection, partial-bank inspection, or objective RNG.
    live, _ = runner.validate_global_freeze(freeze_path, expected_sha=expected_sha)
    if sha(freeze_path) != expected_sha or live != freeze:
        raise ValueError("Global stream 2 freeze changed before physical-bank boundary")
    _verify_source_bindings(live)
    import numpy as np
    d = runner.enable()
    import experiment_core as core
    reference_source = _reference_source(freeze)
    entry = freeze["cases"][ids[0]]
    case = cases[ids[0]]
    G, D, edges = reference_source.physical_graph(case, np)
    seed = int(entry["seed"])
    if (G, D, seed, pid) != (32, 96, seed, f"stream2_G32_seed{seed}") or seed not in SEEDS or len(ids) != 1:
        raise ValueError("Unexpected stream 2 physical bank")
    bank_root = Path(freeze["bank_root"])
    if not bank_root.is_absolute():
        raise ValueError("The frozen stream 2 bank root must be absolute")
    final = bank_root / pid
    # This is the first bank-directory inspection. Every existing complete
    # bank is verified through the frozen loader and is never regenerated.
    if final.exists():
        for cid in ids:
            runner.load_confirmation_bank(cases[cid], freeze_path)
        return dict(physical_id=pid, status="already_complete_verified", case_ids=ids)
    if any(bank_root.glob(pid + ".partial-*")):
        raise FileExistsError("An incomplete stream 2 bank requires explicit review: " + pid)
    partial = bank_root / (pid + ".partial-" + uuid.uuid4().hex)
    partial.mkdir(parents=True, exist_ok=False)
    guard = dict(model_freeze_sha256=expected_sha, protocol_sha256=freeze["protocol_sha256"])
    write_new(partial / "start.json", dict(physical_id=pid, case_ids=ids, method_ids=list(METHODS), guard=guard,
        all_case_guards_revalidated_before_rng=True, all_15_model_roles_revalidated_before_rng=True,
        created_utc=utc(), rng_namespace=protocol["rng_namespace"], stream_id="stream2"))
    label = protocol["rng_namespace"] + "|G=" + str(G) + "|physical_seed=" + str(seed)
    seed128 = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:16], "little")
    rng = np.random.Generator(np.random.PCG64(seed128))
    features = rng.uniform(protocol["xi_lower"], protocol["xi_upper"], (120, G))
    costs = d.signed_costs_from_features(case, features)
    expected_costs = np.full((120, D), -0.25)
    expected_costs[:, 2::3] = -features
    if not np.array_equal(costs, expected_costs):
        raise ValueError("Original signed-affine gross-profit map differs from the fixed law")
    solutions, objectives, record_hashes = [], [], {}
    # The dual-oracle path below is the original sealed generator's path:
    # independent memoized MWIS plus a cold, unmodified original binary IP.
    for index, (feature, cost) in enumerate(zip(features, costs)):
        independent_wall = time.perf_counter()
        independent_cpu = time.process_time()
        independent, witness = reference_source.independent_mwis(feature, edges, protocol["independent_reference_limits"])
        independent["cpu_seconds"] = time.process_time() - independent_cpu
        independent["wall_seconds"] = time.perf_counter() - independent_wall
        witness = np.asarray(witness, dtype=float)
        if not core.validate_solution(case.original_base, witness)["valid"]:
            raise ValueError("Independent MWIS witness violates original binary IP")
        reference = float(independent["objective"])
        tolerance = 1e-6 + 1e-9 * abs(reference)
        if abs(float(cost @ witness) - reference) > tolerance:
            raise ValueError("Independent original gross-profit objective differs")
        full_wall, full_cpu = time.perf_counter(), time.process_time()
        highs, x = core.solve(case.original_base, cost, time_limit=30.0, seed=seed + index)
        original_valid = bool(x is not None and core.validate_solution(case.original_base, x)["valid"])
        objective = float(cost @ x) if original_valid else None
        if not (original_valid and highs.get("valid") is True and highs.get("certified") is True):
            raise ValueError("Original cold Full IP reference was not certified")
        if abs(objective - reference) > tolerance or abs(float(highs["dual_bound"]) - reference) > tolerance:
            raise ValueError("HiGHS primal/dual and independent MWIS disagree")
        full_cpu = time.process_time() - full_cpu
        full_wall = time.perf_counter() - full_wall
        row = dict(physical_id=pid, case_ids=ids, index=index, guard=guard, feature=feature, cost=cost,
            solution=x, objective=objective, independent_objective=reference, highs=highs,
            independent_mwis=independent, independent_solution=witness,
            cold_Full_wall_seconds=full_wall, cold_Full_cpu_seconds=full_cpu,
            cold_Full_timer_includes="model construction, original integer solve and original/reference validation",
            cold_Full_timer_excludes="independent MWIS calculation and serialization", timing_not_isolated=True,
            original_training_label_generation_seconds=None, warm_start_used=False,
            objective_is_original_gross_profit_negative=True, no_artificial_objective_shift=True,
            completed_utc=utc())
        row["record_payload_sha256"] = hashlib.sha256(canonical(row)).hexdigest()
        relative = f"reference-records/ref{index:03d}.json"
        write_new(partial / relative, row)
        record_hashes[relative] = sha(partial / relative)
        solutions.append(x)
        objectives.append(objective)
    write_npz_new(partial / "confirmation.npz", dict(features=features, costs=costs,
                  solutions=np.asarray(solutions), objectives=np.asarray(objectives)), np)
    certificates = dict(schema="stream2-confirmation-reference-certificates-v1", physical_id=pid,
        case_ids=ids, count=120, guard=guard, all_original_IP_certified=True, all_independent_references_agree=True,
        reference_method="cold original HiGHS binary IP plus independent memoized MWIS",
        original_solver_threads=1, original_constraints_unmodified=True, no_full_optimizer_map_enumeration=True,
        record_file_sha256=record_hashes)
    write_new(partial / "reference-certificates.json", certificates)
    meta = dict(schema="stream2-shared-physical-confirmation-bank-v1", physical_id=pid, G=G, D=D, seed=seed, count=120,
        case_ids=ids, case_bindings=case_bindings(ids, freeze), phase="independent_confirmation", confirmation=True,
        method_ids=list(METHODS), model_role_ids=list(METHODS), stream_id="stream2",
        model_freeze_sha256=expected_sha, protocol_sha256=freeze["protocol_sha256"],
        npz_sha256=sha(partial / "confirmation.npz"), reference_certificates_sha256=sha(partial / "reference-certificates.json"),
        rng_namespace=protocol["rng_namespace"], rng_algorithm=protocol["rng_algorithm"], rng_seed128=seed128,
        rng_seed_input=label, rng_seed_independent_of_training_count_and_models=True,
        all_global_guards_revalidated_before_rng=True, all_15_model_roles_revalidated_before_rng=True,
        generator_sha256=sha(Path(__file__)), original_reference_generator_sha256=ORIGINAL_GENERATOR_SHA256,
        source_freeze_count=3, model_role_count=15, independently_certified_original_references=True, created_utc=utc())
    write_new(partial / "metadata.json", meta)
    # The whole population must still match before the bank is published.
    live, _ = runner.validate_global_freeze(freeze_path, expected_sha=expected_sha)
    if sha(freeze_path) != expected_sha or live != freeze:
        raise ValueError("Global stream 2 freeze changed before publishing the physical bank")
    _verify_source_bindings(live)
    resolved_root, resolved_partial, resolved_final = bank_root.resolve(), partial.resolve(), final.resolve()
    if resolved_partial.parent != resolved_root or resolved_final.parent != resolved_root or resolved_final.exists():
        raise ValueError("Stream 2 publication must remain directly within the frozen bank root without overwriting a bank")
    os.rename(resolved_partial, resolved_final)
    for cid in ids:
        runner.load_confirmation_bank(cases[cid], freeze_path)
    return dict(physical_id=pid, status="created_and_evaluator_verified", case_ids=ids,
                npz_sha256=meta["npz_sha256"], metadata_sha256=sha(final / "metadata.json"), count=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    runner, freeze, cases, protocol, physical = load_validated_runner(args.freeze, args.expected_freeze_sha256)
    if args.validate_only:
        print(json.dumps(dict(status="GLOBAL_FREEZE_VALIDATED_NO_RNG_OR_BANK_ACCESS", cases=3,
            method_roles=15, physical_banks=3, expected_freeze_sha256=args.expected_freeze_sha256)))
        return
    rows = []
    for pid in sorted(physical, key=lambda value: freeze["cases"][physical[value][0]]["seed"]):
        row = generate_bank(pid, physical[pid], runner, freeze, cases, protocol, args.freeze, args.expected_freeze_sha256)
        rows.append(row)
        print(json.dumps(dict(event="stream2_physical_bank_complete", **row)), flush=True)
    print(json.dumps(dict(status="ALL_THREE_STREAM2_PHYSICAL_BANKS_COMPLETE", case_count=3,
        physical_bank_count=3, distinct_objectives=360, case_queries=360, expected_method_returns=1800,
        model_freeze_sha256=args.expected_freeze_sha256, banks=rows)), flush=True)


if __name__ == "__main__":
    main()
