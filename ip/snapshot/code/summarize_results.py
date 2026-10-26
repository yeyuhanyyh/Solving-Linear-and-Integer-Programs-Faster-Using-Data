"""Verify every raw trial and summarize all six frozen Appendix C.3 settings.

Recovery and speed must be read together: the all-trial speedup also includes
fast failed predictions. A second speedup is supplied on paired recovery only.
No failed objective or slow case is excluded from the primary summary.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import adapter
from runner_common import ROOT, METHODS, MODEL_FOR_METHOD, clean, dump, event, read, freeze, selected, checkpoint, digest, object_digest
from run_benchmark import existing_trials, freeze_confirmation_cohort


def geomean(values: list[float]) -> float | None:
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all() or np.any(data <= 0):
        raise RuntimeError("Timings and paired ratios must be strictly positive and finite")
    return float(np.exp(np.mean(np.log(data))))


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    if not n:
        return None, None
    z = 1.959963984540054
    rate = k/n
    den = 1+z*z/n
    mid = (rate+z*z/(2*n))/den
    half = z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
    return float(mid-half), float(mid+half)


def verify_row(case: adapter.FrozenCase, row: dict[str, Any], test_costs, reference_objectives) -> None:
    i = int(row["test_index"])
    reference = float(reference_objectives[i])
    if row["reference_objective"] != reference:
        raise RuntimeError(f"Saved reference changed: {case.case_id}/{i}/{row['method']}")
    seconds = row["seconds"]
    if seconds is None or not np.isfinite(seconds) or seconds <= 0 or seconds != row["online_seconds"]:
        raise RuntimeError("Invalid outer elapsed measurement")
    if row["method_time_limit_seconds"] != 30. or row["method_wall_budget_exceeded"] != (seconds > 30.):
        raise RuntimeError("Saved outer method budget differs from elapsed time")
    if row["valid"] and row["method_wall_budget_exceeded"]:
        raise RuntimeError("An over-budget candidate cannot count as a usable method output")
    if row["full_fallback_used"]:
        raise RuntimeError("Hidden Full fallback is forbidden")
    if row["method"] != "full" and row["original_global_solver_certified"]:
        raise RuntimeError("A restricted/strategy method was mislabeled globally solver certified")
    if row["method"] == "full" and not (row["original_global_solver_certified"] and row["quality_recovered"]):
        raise RuntimeError("Fresh Full failed its original reference check")
    x = row["solution"]
    if x is None:
        if any(row[key] for key in ("valid","original_feasible","quality_recovered",
                "certified_recovered","table_compatible_recovered","solved_model_certified",
                "raw_solved_model_certificate","original_global_solver_certified")):
            raise RuntimeError("A missing solution cannot be feasible, recovered or certified")
        return
    x = np.asarray(x, dtype=float)
    checked = adapter.validate_solution(case.model, x)
    if bool(checked["valid"]) != bool(row["original_feasible"]):
        raise RuntimeError("Saved original feasibility differs from independent recheck")
    objective = float(test_costs[i] @ x)
    if row["objective"] is None or abs(objective-row["objective"]) > 1e-8+1e-12*abs(objective):
        raise RuntimeError("Saved primal objective differs from returned solution")
    quality = bool(row["valid"] and abs(objective-reference) <= adapter.recovery_tolerance(reference))
    if quality != row["quality_recovered"]:
        raise RuntimeError("Saved quality recovery differs from independent scoring")
    if row["valid"] and not (checked["valid"] and row["restriction_check"]["valid"]):
        raise RuntimeError("Usable solution lacks original and learned-constraint validity")
    if row["certified_recovered"] != bool(quality and row["solved_model_certified"]):
        raise RuntimeError("Solved-model certified recovery flag is inconsistent")
    expected_table = bool(quality and (row["solved_model_certified"] if row["method"] in ("full", "ours", "svm_native") else True))
    if row["table_compatible_recovered"] != expected_table:
        raise RuntimeError("Table-compatible control recovery flag is inconsistent")
    if row["method"] == "full" and not (row["original_global_solver_certified"] and quality):
        raise RuntimeError("Fresh Full failed its original reference check")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(clean(v), ensure_ascii=True) if isinstance(v, (dict, list, tuple)) else v) for k, v in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-prefix", default="paper-baselines")
    args = parser.parse_args()
    cases = selected(args.cases, adapter.list_case_ids())
    methods = selected(args.methods, METHODS)
    if "full" not in methods:
        raise ValueError("Full is required for paired timing summaries")
    frozen = freeze(create=False)
    if not (ROOT/"results"/"confirmation-model-freeze.json").exists():
        raise RuntimeError("No final-model cohort proof exists; summary cannot create a new confirmation freeze")
    cohort=freeze_confirmation_cohort(frozen)
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    fitted: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    diagnostics = {}
    missing = []
    raw_hashes = {}
    case_summaries = []
    for cid in cases:
        case = adapter.load_case(cid, verify=True)
        costs_path=ROOT/"confirmation"/cid/"costs.npz"
        refs_path=ROOT/"confirmation"/cid/"full-references.jsonl"
        if not costs_path.exists() or not refs_path.exists():
            missing.extend(dict(case=cid,test_index=i,method=m) for i in range(40) for m in methods)
            if args.allow_partial:
                continue
            raise RuntimeError(f"No independent confirmation data/reference generated for {cid}")
        with np.load(costs_path,allow_pickle=False) as data:
            test_costs=data["costs"].copy()
            test_features=data["features"].copy()
        guard=read(ROOT/"confirmation"/cid/"guard.json")
        if (guard["case"]!=cid or guard["freeze_fingerprint"]!=frozen["fingerprint"]
                or guard["model_cohort_fingerprint"]!=cohort["fingerprint"]
                or guard["costs_npz_sha256"]!=digest(costs_path)):
            raise RuntimeError(f"Independent confirmation cost/model guard changed: {cid}")
        if (test_costs.shape!=(40,case.model.n_variables) or test_features.shape!=(40,case.train_features.shape[1])
                or not np.isfinite(test_costs).all() or not np.isfinite(test_features).all()
                or not np.array_equal(test_costs,adapter.costs_from_features(case,test_features))):
            raise RuntimeError(f"Independent confirmation features/cost map is invalid: {cid}")
        guard_fingerprint=object_digest(guard)
        raw_hashes[costs_path.relative_to(ROOT).as_posix()]=digest(costs_path)
        raw_hashes[refs_path.relative_to(ROOT).as_posix()]=digest(refs_path)
        refs={}
        for line in refs_path.read_text(encoding="utf-8").splitlines():
            record=json.loads(line)
            i=int(record["test_index"])
            if (i in refs or i not in range(40) or record["case"]!=cid
                    or record["guard_fingerprint"]!=guard_fingerprint
                    or not record["result"]["certified"] or not record["result"]["valid"]):
                raise RuntimeError("Duplicate, mismatched or uncertified independent Full reference")
            x=np.asarray(record["solution"],dtype=float)
            objective=float(record["result"]["objective"])
            if (not np.isfinite(objective) or not adapter.validate_solution(case.model,x)["valid"]
                    or abs(float(test_costs[i]@x)-objective)>adapter.recovery_tolerance(objective)):
                raise RuntimeError("Independent Full reference primal/objective is invalid")
            refs[i]=record
        if set(refs)!=set(range(40)):
            raise RuntimeError(f"Independent Full references are incomplete: {cid}")
        reference_objectives=np.asarray([refs[i]["result"]["objective"] for i in range(40)])
        path = ROOT / "results" / cid / "trials.jsonl"
        records = existing_trials(path, cid, frozen["fingerprint"])
        case_manifest=read(ROOT/"results"/cid/"case-manifest.json") if records else None
        if case_manifest is not None and (case_manifest["case"]!=cid or case_manifest["freeze_fingerprint"]!=frozen["fingerprint"]):
            raise RuntimeError("Timed case manifest changed")
        if path.exists():
            raw_hashes[str(path.relative_to(ROOT))] = digest(path)
        for method in methods:
            for i in range(40):
                if (i, method) not in records:
                    missing.append(dict(case=cid, test_index=i, method=method))
            if method != "full":
                model_checkpoint = checkpoint(case, MODEL_FOR_METHOD[method], frozen)
                if model_checkpoint is not None:
                    if any(r["method"]==method for r in records.values()) and (case_manifest is None
                            or case_manifest["method_checkpoints"].get(method)!=object_digest(model_checkpoint)):
                        raise RuntimeError("Timed model checkpoint differs from its case manifest")
                    metadata = read(ROOT / "models" / cid / MODEL_FOR_METHOD[method] / "metadata.json")
                    fitted[case.family, case.amplitude, method].append(dict(case=cid, checkpoint=model_checkpoint, metadata=metadata))
                    diagnostics[f"{cid}/{method}"] = metadata
        for row in records.values():
            if row["method"] in methods:
                if row["confirmation_guard_fingerprint"]!=guard_fingerprint or row["test_kind"]!="independent_fresh_confirmation":
                    raise RuntimeError("Timed row belongs to a different confirmation stream/cohort")
                verify_row(case, row, test_costs, reference_objectives)
                groups[case.family, case.amplitude].append(row)
        case_summaries.append(dict(case=cid, records=len(records), expected_records=40*len(methods),
                                   requested_records=sum(r["method"] in methods for r in records.values())))
    if missing and not args.allow_partial:
        raise RuntimeError(f"Raw experiment is incomplete ({len(missing)} missing); no complete summary generated. First missing: {missing[:12]}")
    summaries = []
    wide = []
    for (family, amplitude), group in sorted(groups.items()):
        index = {(row["case"], row["test_index"], row["method"]): row for row in group}
        wide_row: dict[str, Any] = dict(family=family, amplitude=amplitude)
        for method in methods:
            rows = sorted((row for row in group if row["method"] == method), key=lambda row: (row["case"], row["test_index"]))
            paired = [(row, index[row["case"], row["test_index"], "full"]) for row in rows
                      if (row["case"], row["test_index"], "full") in index]
            recovered_pair = [(row, full) for row, full in paired if row["quality_recovered"] and full["quality_recovered"]]
            times = [float(row["seconds"]) for row in rows]
            quality_count = sum(row["quality_recovered"] for row in rows)
            intervals = wilson(quality_count, len(rows))
            fits = fitted[family, amplitude, method]
            if method == "full":
                fit_times, pool_times, completion_times = [], [], []
            else:
                fit_times = [f["checkpoint"]["fit_seconds"] for f in fits]
                pool_times = [f["checkpoint"]["source_pool_collection_seconds"] for f in fits]
                completion_times = [f["checkpoint"]["conditional_lp_completion_seconds"] for f in fits]
            status = Counter(str(row["status"]) for row in rows)
            normalized_gap = [abs(row["objective"]-row["reference_objective"])/max(abs(row["reference_objective"]), 1e-12)
                              for row in rows if row["valid"] and row["objective"] is not None]
            summary = dict(family=family, amplitude=amplitude, method=method, seeds=sorted(set(row["seed"] for row in rows)),
                test_count=len(rows), expected_test_count=120,
                quality_recovered=quality_count, recovery_fraction=(quality_count/len(rows) if rows else None),
                recovery_wilson95_lower=intervals[0], recovery_wilson95_upper=intervals[1],
                table_compatible_recovered=sum(row["table_compatible_recovered"] for row in rows),
                solved_model_certified_recovered=sum(row["certified_recovered"] for row in rows),
                original_feasible=sum(row["original_feasible"] for row in rows),
                usable_feasible=sum(row["valid"] for row in rows),
                solved_model_certified=sum(row["solved_model_certified"] for row in rows),
                raw_solved_model_certificates=sum(row["raw_solved_model_certificate"] for row in rows),
                original_global_solver_certified=sum(row["original_global_solver_certified"] for row in rows),
                mean_seconds=float(np.mean(times)) if times else None,
                median_seconds=float(np.median(times)) if times else None,
                geomean_seconds=geomean(times),
                paired_full_count=len(paired),
                paired_geomean_speedup_all_trials=geomean([full["seconds"]/row["seconds"] for row, full in paired]),
                recovered_pair_count=len(recovered_pair),
                paired_geomean_speedup_recovered_intersection=geomean([full["seconds"]/row["seconds"] for row, full in recovered_pair]),
                mean_relative_objective_error_usable=float(np.mean(normalized_gap)) if normalized_gap else None,
                max_relative_objective_error_usable=float(np.max(normalized_gap)) if normalized_gap else None,
                offline_cases=len(fits), offline_fit_mean_seconds=float(np.mean(fit_times)) if fit_times else 0.,
                historical_pool_collection_mean_seconds_provenance_only=float(np.mean(pool_times)) if pool_times else 0.,
                remote_conditional_lp_completion_mean_seconds=float(np.mean(completion_times)) if completion_times else 0.,
                shared_training_label_count_per_case=(120 if method != "full" else 0),
                statuses=dict(status), incomplete=len(rows)!=120,
                model_key=(MODEL_FOR_METHOD[method] if method!="full" else None),
                shared_native_matched_nd_fit=method.startswith("neural_diving_"),
                speedup_all_trials_includes_failures=True)
            summaries.append(summary)
            wide_row[f"{method}_recovered"] = f"{quality_count}/{len(rows)}"
            wide_row[f"{method}_table_compatible_recovered"] = sum(row["table_compatible_recovered"] for row in rows)
            wide_row[f"{method}_feasible"] = sum(row["valid"] for row in rows)
            wide_row[f"{method}_local_certified"] = sum(row["solved_model_certified"] for row in rows)
            wide_row[f"{method}_mean_seconds"] = summary["mean_seconds"]
            wide_row[f"{method}_all_trial_speedup"] = summary["paired_geomean_speedup_all_trials"]
            wide_row[f"{method}_recovered_speedup"] = summary["paired_geomean_speedup_recovered_intersection"]
        wide.append(wide_row)
    prefix = ROOT / "analysis" / args.output_prefix
    result = dict(freeze_fingerprint=frozen["fingerprint"], partial=bool(missing), requested_cases=cases,
        requested_methods=methods, missing=missing, raw_sha256=raw_hashes, case_inventory=case_summaries,
        scoring="Primal quality recovery at 1e-6+1e-9*abs(original objective), original feasible and learned-restriction valid. Original global solver certificates are separately scoped.",
        controls_table_compatibility="Full/Ours/SVM additionally require their solved-model certificate for table_compatible_recovered; new strategy/heuristic quality is not discarded solely for missing original global certificate.",
        timing="Fresh serial outer wall timer: prediction, construction, all candidate/subproblem solves, original feasibility and integrality, learned-restriction checks. Physics and scoring/serialization outside; all offline fits/pools finished beforehand.",
        speedup_caution="All-trial speedups retain every failure; paired recovered-intersection speedups are additional diagnostics and do not replace the complete comparison.",
        rows=summaries, six_setting_table=wide)
    dump(prefix.with_suffix(".json"), result)
    write_csv(prefix.with_suffix(".csv"), summaries)
    write_csv(prefix.with_name(prefix.name+"-six-settings.csv"), wide)
    dump(prefix.with_name(prefix.name+"-training-diagnostics.json"), diagnostics)
    lines = ["# Additional paper baselines on the frozen Appendix C.3 settings", "", 
        ("**Partial run: missing trials are listed in the JSON.**" if missing else "All requested frozen test objectives and methods are retained."), "",
        "Recovery means a valid primal solution reaches the certified original optimal value. A restricted solver certificate is reported separately. All-trial speedups include failures; recovery counts must be read alongside speed.", "",
        "| Family | Amplitude | Method | Recovered | Feasible | Local certificates | All-trial speedup | Recovered-pair speedup | Mean time (s) |", 
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    def formatted(value: float | None) -> str:
        return "—" if value is None else f"{value:.4g}"
    for row in summaries:
        lines.append(f"| {row['family']} | {row['amplitude']:g} | {row['method']} | {row['quality_recovered']}/{row['test_count']} | {row['usable_feasible']} | {row['solved_model_certified']} | {formatted(row['paired_geomean_speedup_all_trials'])} | {formatted(row['paired_geomean_speedup_recovered_intersection'])} | {formatted(row['mean_seconds'])} |")
    lines += ["", "Training used 120 frozen certified original labels per case. Native and matched Neural Diving share one fitted model. Remote GPU fits and conditional-LP completion are reported separately. Historical Windows callback-pool collection is provenance only and is not combined into a same-host offline speed comparison. All 40 confirmation costs per case were generated after every final model was frozen; no old test contents entered this pack or any fitting routine.", ""]
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    event(event="summary_complete", rows=len(summaries), setting_rows=len(wide), partial=bool(missing), output_prefix=prefix)


if __name__ == "__main__":
    main()
