"""Guarded fresh shared-bank generator. SOURCE ONLY until globally frozen.

No RNG, bank access or solve occurs on import or under --validate-only.
The required expected freeze digest and all-case semantic/source validation
precede any confirmation-bank access or objective RNG creation.
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
PROTOCOL = TOOLS / "fresh-confirmation-protocol.json"
RUNNER = TOOLS / "confirmation_runner.py"
METHODS = ("ours", "neural_diving")
SEEDS = (1009, 1013, 1019)
SETTINGS = {(16, 37), (20, 45), (32, 69), (32, 36)}


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


def load_validated_runner(freeze_path, expected_freeze_sha256):
    """This must run before directory inspection, bank IO or any RNG creation."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_freeze_sha256):
        raise ValueError("A literal lowercase expected freeze SHA256 is required")
    if sha(freeze_path) != expected_freeze_sha256:
        raise ValueError("All-model freeze digest differs from the explicitly expected digest")
    # Loading this audited wrapper performs no generation. Its validator checks
    # all 12 checkpoints, Ours representations, labels, source and data guards.
    spec = importlib.util.spec_from_file_location("guarded_confirmation_runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    freeze, cases = runner.validate_global_freeze(Path(freeze_path), semantic=True)
    if sha(freeze_path) != expected_freeze_sha256:
        raise ValueError("Freeze changed while validating the all-model population")
    for path in (Path(__file__).resolve(), PROTOCOL.resolve(), RUNNER.resolve(), (TOOLS / "verify_evaluator_equivalence.py").resolve()):
        if freeze["source_sha256"].get(str(path)) != sha(path):
            raise ValueError("Generator/protocol/evaluator source is not in the all-model freeze: " + str(path))
    if Path(freeze["protocol_path"]).resolve() != PROTOCOL.resolve() or freeze["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Globally frozen fresh protocol differs")
    if freeze.get("case_count") != 12 or freeze.get("model_role_count") != 24 or freeze.get("ND_model_count") != 12:
        raise ValueError("The full 12-case / 24-role / 12-ND freeze is required")
    protocol = read(PROTOCOL)
    if protocol["expected_case_count"] != 12 or protocol["test_count_per_case"] != 120 or protocol["method_ids"] != list(METHODS):
        raise ValueError("Unexpected fixed protocol population")
    entries = freeze["cases"]
    expected = {(G, n, seed) for G, n in SETTINGS for seed in SEEDS}
    actual = {(int(e["G"]), int(e["n_train"]), int(e["seed"])) for e in entries.values()}
    if len(cases) != 12 or set(cases) != set(entries) or actual != expected:
        raise ValueError("Global case cohort is incomplete or differs")
    physical = {}
    for cid, entry in entries.items():
        pid = f"physical_G{entry['G']}_seed{entry['seed']}"
        if entry["physical_id"] != pid or set(entry["methods"]) != set(METHODS):
            raise ValueError("Physical identity or required method roles differ")
        physical.setdefault(pid, []).append(cid)
    if len(physical) != 9:
        raise ValueError("Need exactly nine physical objective banks")
    for pid, ids in physical.items():
        ids.sort()
        expected_count = 2 if entries[ids[0]]["G"] == 32 else 1
        if len(ids) != expected_count:
            raise ValueError("G32 n36/n69 must share one physical bank per seed")
    return runner, freeze, cases, protocol, physical


def physical_graph(case, np):
    """Derive the independent-set graph from original frozen matrix rows."""
    G = int(case.train_features.shape[1])
    D = int(case.model.n_variables)
    if D != 3 * G:
        raise ValueError("Expected three original item variables per bundle")
    A = case.model.A.tocsr()
    lower, upper = case.model.row_lower, case.model.row_upper
    edges = []
    for index in range(A.shape[0]):
        row = A.getrow(index)
        columns, values = row.indices, row.data
        if index < G:
            if not np.array_equal(columns, np.arange(3 * index, 3 * index + 3)):
                raise ValueError("Original bundle-capacity row layout differs")
            a, b, c = values
            if not (a > 0 and b > 0 and max(a, b) < c < a + b and upper[index] == a + b and np.isneginf(lower[index])):
                raise ValueError("Original bundle proof assumptions differ")
        else:
            if len(columns) != 2 or not np.all(columns % 3 == 2) or not np.array_equal(values, np.ones(2)) or upper[index] != 1 or not np.isneginf(lower[index]):
                raise ValueError("Original conflict graph row differs")
            edges.append(tuple(sorted(map(int, columns // 3))))
    if len(set(edges)) != len(edges):
        raise ValueError("Duplicate original graph rows")
    return G, D, sorted(edges)


def independent_mwis(feature, edges, resource_limits):
    """Exact recurrence over bitmasks; never enumerate the full solution map."""
    G = len(feature)
    premiums = [float(value) - 0.5 for value in feature]
    neighbors = [0] * G
    for u, v in edges:
        neighbors[u] |= 1 << v
        neighbors[v] |= 1 << u
    positive = sum(1 << i for i, value in enumerate(premiums) if value > 0)
    memo = {0: (0.0, 0)}
    started = time.perf_counter()
    new_states = 0

    def best(mask):
        nonlocal new_states
        if mask in memo:
            return memo[mask]
        new_states += 1
        if new_states + 1 > int(resource_limits["maximum_memo_states"]):
            raise RuntimeError("Independent MWIS reference exhausted its frozen state limit; retain partial bank, do not replace costs")
        if new_states == 1 or new_states % int(resource_limits["new_states_check_interval"]) == 0:
            if time.perf_counter() - started > float(resource_limits["wall_seconds"]):
                raise RuntimeError("Independent MWIS reference exhausted its frozen wall limit; retain partial bank, do not replace costs")
        vertices = [i for i in range(G) if mask & (1 << i)]
        v = max(vertices, key=lambda i: ((neighbors[i] & mask).bit_count(), premiums[i], -i))
        bit = 1 << v
        without = best(mask & ~bit)
        tail_score, tail_set = best(mask & ~(bit | neighbors[v]))
        with_v = (premiums[v] + tail_score, tail_set | bit)
        result = with_v if with_v[0] > without[0] else without if without[0] > with_v[0] else min(with_v, without, key=lambda r: r[1])
        memo[mask] = result
        return result

    score, selected = best(positive)
    if time.perf_counter() - started > float(resource_limits["wall_seconds"]):
        raise RuntimeError("Independent MWIS reference exceeded its frozen wall limit; retain partial bank, do not replace costs")
    chosen = [(selected >> i) & 1 for i in range(G)]
    if any(chosen[u] and chosen[v] for u, v in edges):
        raise ValueError("MWIS recurrence returned a conflicting set")
    objective = -(0.5 * G + sum(premiums[i] for i in range(G) if chosen[i]))
    witness = [value for bit in chosen for value in (1 - bit, 1 - bit, bit)]
    state_digest = hashlib.sha256(canonical([[mask, value[0], value[1]] for mask, value in sorted(memo.items())])).hexdigest()
    if time.perf_counter() - started > float(resource_limits["wall_seconds"]):
        raise RuntimeError("Independent MWIS reference exceeded its frozen wall limit while certifying memo states; retain partial bank, do not replace costs")
    return dict(objective=objective, selected_mask=selected, selected_vertices=[i for i in range(G) if chosen[i]],
                positive_premium_mask=positive, memo_state_count=len(memo), memo_states_sha256=state_digest,
                recurrence="max(exclude v, premium[v] + recurse without v and its neighbors)",
                negative_and_zero_premiums_removed=True, full_optimizer_map_enumeration=False,
                new_state_evaluations=new_states, frozen_resource_limits=resource_limits), witness


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
    # Revalidate the complete population and explicit digest at every physical
    # boundary, before inspecting an existing bank or creating the next RNG.
    live, _ = runner.validate_global_freeze(freeze_path, semantic=True)
    if sha(freeze_path) != expected_sha or live != freeze:
        raise ValueError("Global freeze changed before physical-bank boundary")
    import numpy as np
    d = runner.enable()
    import experiment_core as core
    entry = freeze["cases"][ids[0]]
    case = cases[ids[0]]
    G, D, edges = physical_graph(case, np)
    seed = int(entry["seed"])
    bank_root = Path(freeze["bank_root"])
    final = bank_root / pid
    # Existing complete banks are verified by the same frozen evaluator. They
    # are never redrawn or overwritten; incomplete banks require review.
    if final.exists():
        for cid in ids:
            runner.load_confirmation_bank(cases[cid], freeze_path)
        return dict(physical_id=pid, status="already_complete_verified", case_ids=ids)
    if any(bank_root.glob(pid + ".partial-*")):
        raise FileExistsError("An incomplete fresh bank requires explicit review: " + pid)
    partial = bank_root / (pid + ".partial-" + uuid.uuid4().hex)
    partial.mkdir(parents=True, exist_ok=False)
    guard = dict(model_freeze_sha256=expected_sha, protocol_sha256=freeze["protocol_sha256"])
    write_new(partial / "start.json", dict(physical_id=pid, case_ids=ids, guard=guard,
        all_case_guards_revalidated_before_rng=True, created_utc=utc(), rng_namespace=protocol["rng_namespace"]))
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
    for index, (feature, cost) in enumerate(zip(features, costs)):
        independent_wall = time.perf_counter()
        independent_cpu = time.process_time()
        independent, witness = independent_mwis(feature, edges, protocol["independent_reference_limits"])
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
    certificates = dict(schema="coupled-bundle-confirmation-reference-certificates-v1", physical_id=pid,
        case_ids=ids, count=120, guard=guard, all_original_IP_certified=True, all_independent_references_agree=True,
        reference_method="cold original HiGHS binary IP plus independent memoized MWIS",
        original_solver_threads=1, original_constraints_unmodified=True, no_full_optimizer_map_enumeration=True,
        record_file_sha256=record_hashes)
    write_new(partial / "reference-certificates.json", certificates)
    meta = dict(schema="shared-physical-confirmation-bank-v2", physical_id=pid, G=G, D=D, seed=seed, count=120,
        case_ids=ids, case_bindings=case_bindings(ids, freeze), phase="independent_confirmation", confirmation=True,
        model_freeze_sha256=expected_sha, protocol_sha256=freeze["protocol_sha256"],
        npz_sha256=sha(partial / "confirmation.npz"), reference_certificates_sha256=sha(partial / "reference-certificates.json"),
        rng_namespace=protocol["rng_namespace"], rng_algorithm=protocol["rng_algorithm"], rng_seed128=seed128,
        rng_seed_input=label, rng_seed_independent_of_training_count_and_models=True,
        all_global_guards_revalidated_before_rng=True, generator_sha256=sha(Path(__file__)),
        source_freeze_count=12, independently_certified_original_references=True, created_utc=utc())
    write_new(partial / "metadata.json", meta)
    # This bank must be coherent with the still-frozen full model population.
    runner.validate_global_freeze(freeze_path, semantic=True)
    if sha(freeze_path) != expected_sha:
        raise ValueError("Global freeze changed before publishing the physical bank")
    resolved_root, resolved_partial, resolved_final = bank_root.resolve(), partial.resolve(), final.resolve()
    if resolved_partial.parent != resolved_root or resolved_final.parent != resolved_root or resolved_final.exists():
        raise ValueError("Fresh bank publication targets must stay directly within the frozen absolute bank root and must not overwrite an existing bank")
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
        print(json.dumps(dict(status="GLOBAL_FREEZE_VALIDATED_NO_RNG_OR_BANK_ACCESS", cases=12, method_roles=24,
                              neural_models=12, physical_banks=9, expected_freeze_sha256=args.expected_freeze_sha256)))
        return
    order = sorted(physical, key=lambda pid: ({32: 0, 16: 1, 20: 2}[freeze["cases"][physical[pid][0]]["G"]],
                                              freeze["cases"][physical[pid][0]]["seed"]))
    rows = []
    for pid in order:
        row = generate_bank(pid, physical[pid], runner, freeze, cases, protocol, args.freeze, args.expected_freeze_sha256)
        rows.append(row)
        print(json.dumps(dict(event="physical_bank_complete", **row)), flush=True)
    print(json.dumps(dict(status="ALL_NINE_PHYSICAL_BANKS_COMPLETE", case_count=12, physical_bank_count=9,
                          distinct_objectives=1080, case_queries=1440, expected_method_returns=2880,
                          model_freeze_sha256=args.expected_freeze_sha256, banks=rows)), flush=True)


if __name__ == "__main__":
    main()
