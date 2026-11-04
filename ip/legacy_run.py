"""Portable entry points for the immutable IP experiment release.

Use a separate process per problem/seed/method: archived modules intentionally
use stable import names, and the two problem runtimes must not be mixed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "snapshot"
FOLLOWUP = SNAPSHOT / "ip-followup-20261025"
UC = FOLLOWUP / "ip-ratio-003-uc-20261025/nrel-startup-original-fuel-v1"
DESIGNS = FOLLOWUP / "objective-gap-designs-20261025"
KNAP = FOLLOWUP / "objective-gap-followup-20261025"
GENERIC = DESIGNS / "adapter/generic-binary-v2"
SEEDS = {"uc": (101, 211, 307), "knapsack": (1009, 1013, 1019)}
METHODS = ("full", "ours", "svm", "mlopt", "neural-diving", "preference-learning", "nd-continued")
_PREFIXES = None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            return None
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def import_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_artifacts():
    """No solver, numpy, torch, pickle or training import is needed."""
    manifest = read(ROOT / "artifact-manifest.json")
    for entry in manifest["files"]:
        path = (ROOT / entry["destination"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Archive path escapes release")
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("Archive artifact changed: " + entry["destination"])
    return {"status": "PASS", "files": len(manifest["files"]),
            "bytes": sum(e["bytes"] for e in manifest["files"]),
            "manifest_sha256": sha(ROOT / "artifact-manifest.json")}


def canonical_source_paths(value):
    """Normalize only source-location strings, preserving every hash/value.

    Original UC model guards used absolute source paths as dictionary keys.
    All corresponding real relocated bytes were checked above and by the
    original helper. Canonicalizing the keys lets its original equality guard
    compare that same data rather than a machine-specific checkout location.
    """
    global _PREFIXES
    if _PREFIXES is None:
        _PREFIXES = (read(ROOT / "artifact-manifest.json")["source_root"].replace("/", "\\"), str(SNAPSHOT.resolve()))
    old, local = _PREFIXES
    if isinstance(value, dict):
        return {canonical_source_paths(k): canonical_source_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical_source_paths(v) for v in value]
    if isinstance(value, str) and value.startswith(local):
        return old + value[len(local):].replace("/", "\\")
    return value


def relocated_path(text):
    global _PREFIXES
    if _PREFIXES is None:
        _PREFIXES = (read(ROOT / "artifact-manifest.json")["source_root"].replace("/", "\\"), str(SNAPSHOT.resolve()))
    old = _PREFIXES[0].replace("\\", "/").rstrip("/")
    text = str(text).replace("\\", "/")
    if text == old or text.startswith(old + "/"):
        return str(SNAPSHOT / text[len(old):].lstrip("/"))
    return text


def relocate_values(value):
    if isinstance(value, dict):
        return {relocate_values(k): relocate_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_values(v) for v in value]
    if isinstance(value, str):
        return relocated_path(value)
    return value


def set_threads():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def signed_cost_map(case, features):
    import numpy as np
    return np.asarray(case.metadata["cost_offset"]) + np.asarray(features) @ np.asarray(case.metadata["cost_feature_coefficients"]).T


@contextmanager
def load_context(problem, seed, method, output):
    """Reuse archived constructors, with path-only relocation around them."""
    import numpy as np
    if problem == "uc":
        confirm = import_path(UC / "confirmation-v1/confirm_n32.py", "release_uc_confirmation")
        helper, runtime = confirm.load_helpers()
        original_binding = helper._case_binding
        helper._case_binding = lambda case: canonical_source_paths(original_binding(case))
        # Original load_case verifies NPZ hashes, labels, UC physics, domains,
        # affine metadata and both SVM/MLOPT checkpoint source bindings.
        case, models, fit, offset, coefficient, files = confirm.load_case(seed, helper, runtime)
        sys.path.insert(0, str(FOLLOWUP))
        if method in ("neural-diving", "nd-continued"):
            sys.path.insert(0, str(FOLLOWUP / "uc_bundle_frozen/runtime"))
        import adapter
        import portable_adapter
        import uc_adapters as data
        import train_uc_baselines as trainer
        import experiment_core as core
        metadata = dict(case.metadata, cost_offset=offset.tolist(), cost_feature_coefficients=coefficient.tolist(),
                        nd_cost_offset=offset, nd_cost_coefficients=coefficient)
        metadata["baseline_seed"] = seed
        if method in ("neural-diving", "nd-continued"):
            metadata["baseline_seed"] = int.from_bytes(hashlib.sha256((case.case_id + "|nd-uc-supplement-v1").encode()).digest()[:4], "little") % (2**31 - 1)
        case = replace(case, metadata=metadata)
        for module in (adapter, portable_adapter, data):
            module.costs_from_features = signed_cost_map
        context = dict(helper=helper, runtime=runtime, models=models, trainer=trainer, data=data, core=core, adapter=adapter, source_files=files)
        validate_geometry(case)
        yield case, context
    else:
        generic = import_path(GENERIC / "generic_binary_pilot.py", "release_generic_binary")
        generic.enable_runtime()
        generic.bind_signed_cost_map()
        manifest_path = DESIGNS / "coupled-bundle/scale-extension-g32-v1/short-prefix-n36/sealed-cohort-v1/training-only-manifest.json"
        original_manifest = read(manifest_path)
        # Only path values change. Each original training SHA is still checked
        # by validate_selection; schema, signed cost map, rank and feasibility
        # assertions in the original load_case are all retained.
        relocated = relocate_values(original_manifest)
        local_manifest = output / "relocated-training-manifest.json"
        write_new(local_manifest, relocated)
        entry = next(e for e in relocated["cases"] if int(e["physical_seed"]) == seed)
        case = generic.load_case(entry["directory"], local_manifest)
        case.metadata["selection_manifest_sha256"] = sha(manifest_path)
        case.metadata["selection_manifest_path"] = str(manifest_path)
        import adapter
        import train_uc_baselines as trainer
        import uc_adapters as data
        import experiment_core as core
        extra = KNAP / "extra-baselines-coupled-v1"
        sys.path.insert(0, str(extra))
        import svm_mlopt_components as components
        context = dict(generic=generic, trainer=trainer, data=data, adapter=adapter, core=core,
                       components=components, source_manifest=manifest_path)
        validate_geometry(case)
        yield case, context


def validate_geometry(case):
    """Independently recompute the exact affine restriction from the labels."""
    import numpy as np
    import sympy as sp
    y = np.rint(case.train_solutions[:, case.commitment_indices]).astype(int)
    anchor = np.asarray(case.affine["anchor"], dtype=int)
    N = np.asarray(case.affine["normals_exact"], dtype=object)
    exact = np.asarray([[sp.Rational(v) for v in row] for row in N], dtype=object)
    rank = int(sp.Matrix((y[1:] - y[0]).T.tolist()).rank())
    if rank != int(case.affine["rank"]) or not np.array_equal(anchor, y[0]):
        raise ValueError("Original affine anchor/rank differs from training labels")
    if exact.shape != (y.shape[1] - rank, y.shape[1]) or np.any((y.astype(object) - anchor) @ exact.T != 0):
        raise ValueError("Original affine equations do not contain all training labels")
    if sp.Matrix(exact.tolist()).rank() != len(exact):
        raise ValueError("Affine equations do not have the declared independent dimension")


def checkpoint_path(problem, method, case):
    cid = case.case_id
    if problem == "uc":
        if method in ("svm", "mlopt"):
            return case.source_directory / "baseline-models" / (method + ".pkl")
        if method == "neural-diving":
            return UC / "nd-supplement-v1/models" / cid / "complete-predictor.pt"
        if method == "nd-continued":
            return UC / "nd-supplement-v1/continued-models" / cid / "continued-predictor.pt"
        return UC / "li-supplement-v1/run-v1/models" / cid / "li"
    if method in ("svm", "mlopt"):
        return KNAP / "extra-baselines-coupled-v1/results/models" / cid / method / "model.pkl"
    if method == "neural-diving":
        return KNAP / "uniform-gross-g32-n36-pilot-results/models" / cid / "neural_diving/complete-predictor.pt"
    if method == "nd-continued":
        return DESIGNS / "coupled-bundle/training-budget-control-v1/continued-models" / cid / "continued-predictor.pt"
    return KNAP / "li-supplement-v1/models" / cid / "li"


def load_predictor(problem, method, case, context, custom=None):
    if method in ("full", "ours"):
        return None
    path = Path(custom) if custom else checkpoint_path(problem, method, case)
    if method in ("neural-diving", "nd-continued"):
        return context["trainer"].load_nd_complete(case, path)
    if method == "preference-learning":
        import preference_baseline as li
        return li.load_fitted(context["adapter"].training_case(case), path)
    if problem == "uc":
        if custom:
            with path.open("rb") as stream:
                model = pickle.load(stream)
            if model._new_uc_source_binding != context["helper"]._case_binding(case):
                raise ValueError("Fresh checkpoint binding differs from current training/source")
            return model
        return context["models"]["SVM" if method == "svm" else "MLOPT"]
    return context["components"].load_component(path, case, method)


def load_bank(problem, case):
    import numpy as np
    if problem == "uc":
        path = UC / "confirmation-v1/results-v1" / case.case_id / "confirmation.npz"
    else:
        path = KNAP / "uniform-gross-stream2-training-budget-control/banks" / f"stream2_G32_seed{case.seed}" / "confirmation.npz"
    with np.load(path, allow_pickle=False) as stream:
        bank = {k: stream[k].copy() for k in stream.files}
    if len(bank["costs"]) != 120 or not np.allclose(signed_cost_map(case, bank["features"]), bank["costs"], rtol=1e-14, atol=1e-10):
        raise ValueError("Frozen bank or explicit objective mapping differs")
    return bank, path


def solve_one(problem, method, case, ctx, predictor, cost, feature, reference, index, budget):
    import numpy as np
    adapter = ctx["adapter"]
    begin = time.perf_counter()
    fixed = None
    diagnostics = {}
    if method in ("full", "ours"):
        raw, x = ctx["core"].solve(case.original_base, cost, affine=case.affine if method == "ours" else None,
                                        seed=case.seed + index, time_limit=budget)
    elif problem == "uc" and method in ("svm", "mlopt"):
        fn = ctx["helper"].predict_svm if method == "svm" else ctx["helper"].predict_mlopt
        row = fn(predictor, case, cost, feature, case.seed + index, budget)
        raw, x, fixed = row["raw"], row.get("solution"), row.get("fixed")
        diagnostics = row.get("diagnostic", {})
    elif method in ("neural-diving", "nd-continued"):
        import neural_diving_baseline as nd
        raw = nd.solve_matched_cost(predictor, case, cost, seed=case.seed + index,
                                   K=int(case.affine["rank"]), total_time_limit=budget, solve_fn=adapter.solve)
        x, fixed = raw.pop("x", None), raw.get("best_fixed")
    else:
        if method == "preference-learning":
            fixed, diagnostics = ctx["trainer"].propose_li_native_anchor(predictor, case, cost, feature, budget)
        else:
            fn = ctx["components"].propose_svm if method == "svm" else ctx["components"].propose_mlopt
            fixed, diagnostics = fn(predictor, case, cost, feature)
        remaining = budget - (time.perf_counter() - begin)
        if fixed is None or remaining <= 0:
            raw, x = dict(valid=False, certified=False, status="no_anchor_or_budget"), None
        else:
            result = adapter.solve(case.model, cost, fixed=fixed, seed=case.seed + index, time_limit=remaining)
            raw, x = result.to_dict(include_solution=False), result.x
    original = bool(x is not None and adapter.validate_solution(case.model, x)["valid"])
    if problem == "uc" and original:
        original = bool(ctx["runtime"]["uc_model"].validate_uc_solution(case.original_base, x)["valid"])
    if method == "full":
        actual_K, restricted = len(case.commitment_indices), True
    elif method == "ours":
        actual_K = int(case.affine["rank"])
        restricted = bool(original and np.max(np.abs(np.asarray(case.affine["normals"], dtype=float) @ (np.asarray(x)[case.commitment_indices] - case.affine["anchor"])), initial=0) <= 1e-6)
    else:
        actual_K = None if fixed is None else len(case.commitment_indices) - len(fixed)
        restricted = bool(original and fixed is not None and actual_K == int(case.affine["rank"]) and
                          all(abs(x[int(j)] - v) <= 1e-6 for j, v in fixed.items()))
    objective = float(cost @ np.asarray(x)) if original else None
    seconds = time.perf_counter() - begin
    valid = bool(original and restricted and seconds <= budget)
    ratio = None
    if valid:
        ratio = float(reference) / objective if problem == "uc" else objective / float(reference)
        if ratio > 1 + adapter.recovery_tolerance(reference) / abs(reference):
            raise ValueError("Returned restricted solution beats certified original optimum")
    return clean(dict(case_id=case.case_id, seed=case.seed, index=index, method=method,
                      K=actual_K, target_K=int(case.affine["rank"]), solution=x, fixed=fixed,
                      cost=cost, feature=feature, reference_objective=reference, raw_objective=objective,
                      objective=objective if valid else None, objective_ratio=ratio, original_feasible=original,
                      restriction_valid=restricted, timely=seconds <= budget, valid=valid, seconds=seconds,
                      recovered=bool(valid and abs(objective - reference) <= adapter.recovery_tolerance(reference)),
                      budget_seconds=budget, raw=raw, diagnostics=diagnostics,
                      release_rerun=True, published_archive_unchanged=True))


def new_output(args):
    if args.output:
        output = Path(args.output).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "runs" / f"{args.command}-{args.problem}-{args.seed}-{args.method}-{stamp}"
    if output == ROOT or output.is_relative_to(SNAPSHOT) or output.is_relative_to(ROOT / "provenance") or output.is_relative_to(ROOT / "results"):
        raise ValueError("New runs must not overwrite archive/provenance/published results")
    output.mkdir(parents=True, exist_ok=False)
    return output


def train_new(problem, method, case, ctx, output, device):
    """Call original fitting algorithms; no archived completion marker reused."""
    begin, cpu = time.perf_counter(), time.process_time()
    artifact = None
    metadata = {}
    if method == "ours":
        import numpy as np
        affine = ctx["core"].fit_affine(np.rint(case.train_solutions[:, case.commitment_indices]).astype(int))
        write_new(output / "affine.json", affine)
        metadata = {"K": affine["rank"]}
        artifact = output / "affine.json"
    elif method in ("neural-diving", "preference-learning"):
        # Original train_one executes the original full schedule and publishes
        # fresh partial->complete directories. Its guard is replaced only with
        # a release-specific data/source identity, never an old marker.
        trainer = ctx["trainer"]
        trainer.model_guard = lambda current, name, bundle: dict(
            release_manifest_sha256=sha(ROOT / "artifact-manifest.json"),
            release_runner_sha256=sha(__file__), method=name,
            case_id=current.case_id, training_array_hash=current.dataset_fingerprint,
            source_hashes=current.source_hashes, K=int(current.affine["rank"]), fresh_fit=True)
        name = "neural_diving" if method == "neural-diving" else "li"
        trainer.train_one(case, name, output, FOLLOWUP / "uc_bundle_frozen", device)
        folder = output / "models" / case.case_id / name
        artifact = folder / "complete-predictor.pt" if name == "neural_diving" else folder
        metadata = read(folder / "metadata.json")
    elif method in ("svm", "mlopt"):
        if problem == "knapsack":
            fn = ctx["components"].fit_svm if method == "svm" else ctx["components"].fit_mlopt
            model, metadata = fn(case) if method == "svm" else fn(case, device=device)
            artifact = output / "model.pkl"
            ctx["components"].save_component(model, artifact)
        else:
            runtime = ctx["runtime"]
            if method == "svm":
                model, metadata = runtime["fast_baselines"].fit_coordinate_svm_small(case)
            else:
                config = runtime["mlopt_baseline"].Config(device=device,
                    seed=int(case.metadata.get("baseline_seed", case.seed)), n_best=10,
                    hidden_candidates=((32, 32, 32), (64, 64, 64)), learning_rates=(1e-3, 1e-2),
                    epoch_candidates=(50, 150), dropout=.1, batch_size=32, validation_fraction=.2)
                model = runtime["mlopt_baseline"].fit(ctx["adapter"].training_case(case), config=config)
                metadata = model.metadata
            model._new_uc_source_binding = ctx["helper"]._case_binding(case)
            artifact = output / "model.pkl"
            with artifact.open("xb") as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError("Train supports ours, svm, mlopt, neural-diving or preference-learning")
    report = dict(status="COMPLETE", method=method, case_id=case.case_id,
                  wrapper_wall_seconds=time.perf_counter() - begin, wrapper_cpu_seconds=time.process_time() - cpu,
                  shared_training_optimum_generation_included=False, artifact=str(artifact), metadata=metadata,
                  original_published_learning_times_unchanged=True)
    write_new(output / "fresh-fit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-artifacts", help="Verify every archived byte using stdlib only")
    for command in ("smoke", "evaluate", "train"):
        p = sub.add_parser(command)
        p.add_argument("--problem", required=True, choices=SEEDS)
        p.add_argument("--seed", required=True, type=int)
        p.add_argument("--method", choices=METHODS, default="ours")
        p.add_argument("--output", type=Path, help="New, nonexistent output directory (default ip/runs/)")
        if command != "train":
            p.add_argument("--checkpoint", type=Path, help="Optional newly fitted checkpoint, instead of archived one")
            p.add_argument("--budget", type=float, default=30.)
        if command == "evaluate":
            p.add_argument("--limit", type=int, default=120, help="First 1..120 frozen tests; use120 for full bank")
        if command == "train":
            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    receipt = verify_artifacts()
    if args.command == "verify-artifacts":
        print(json.dumps(receipt)); return
    if args.seed not in SEEDS[args.problem]:
        parser.error("Select one of the fixed physical seeds " + str(SEEDS[args.problem]))
    if args.command == "train" and args.method in ("full", "nd-continued"):
        parser.error("No fitting for full; archived continued checkpoints are evaluation-only in this wrapper")
    if args.command != "train" and (not math.isfinite(args.budget) or args.budget <= 0):
        parser.error("Budget must be finite and positive")
    if args.command != "train" and args.checkpoint and args.method in ("full", "ours"):
        parser.error("--checkpoint is for fitted predictors; ours uses the verified affine restriction for the supplied labels")
    if args.command == "evaluate" and not 1 <= args.limit <= 120:
        parser.error("Limit must be between1 and120")
    output = new_output(args)
    write_new(output / "start.json", dict(arguments=vars(args), archive_verification=receipt,
                                         runner_sha256=sha(__file__), started_utc=datetime.now(timezone.utc).isoformat()))
    set_threads()
    with load_context(args.problem, args.seed, args.method, output) as (case, context):
        if args.command == "train":
            report = train_new(args.problem, args.method, case, context, output, args.device)
        else:
            predictor = load_predictor(args.problem, args.method, case, context, args.checkpoint)
            if args.command == "smoke":
                features, costs, refs = case.train_features[:1], case.train_costs[:1], case.train_objectives[:1]
                bank_path = None
            else:
                bank, bank_path = load_bank(args.problem, case)
                features, costs, refs = bank["features"][:args.limit], bank["costs"][:args.limit], bank["objectives"][:args.limit]
            rows = []
            for i, (feature, cost, reference) in enumerate(zip(features, costs, refs)):
                row = solve_one(args.problem, args.method, case, context, predictor, cost, feature, reference, i, args.budget)
                write_new(output / f"test{i:03d}.json", row)
                rows.append(row)
                print(json.dumps(dict(event="test_complete", index=i, valid=row["valid"], recovered=row["recovered"], objective_ratio=row["objective_ratio"])), flush=True)
            ratios = [r["objective_ratio"] for r in rows if r["objective_ratio"] is not None]
            report = dict(status="COMPLETE", count=len(rows), case_id=case.case_id, method=args.method,
                          recovery_percent=100 * sum(r["recovered"] for r in rows) / len(rows),
                          objective_ratio_mean=sum(ratios) / len(ratios) if ratios else None,
                          ratio_denominator=len(ratios), source_bank=None if bank_path is None else str(bank_path),
                          training_query_only=args.command == "smoke", published_results_unchanged=True)
            write_new(output / "summary.json", report)
    display = {k: v for k, v in report.items() if k != "metadata"}
    print(json.dumps(clean(dict(output=str(output), summary=display)), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
