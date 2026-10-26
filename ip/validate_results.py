#!/usr/bin/env python3
"""Rebuild IP Tables 3 and 4 from frozen records, using only Python's stdlib.

No solver, model pickle, PyTorch, NumPy, network or RNG is used. The small NPY
reader supports the numeric C-order arrays in this release only. This checks
saved primal feasibility and arithmetic against the recorded Full reference
bank; it does not rerun a solver to obtain a new global-optimality certificate.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
import struct
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

UC_REL = Path("ip-followup-20261025/ip-ratio-003-uc-20261025/nrel-startup-original-fuel-v1")
FOLLOW_REL = Path("ip-followup-20261025/objective-gap-followup-20261025")
KN_REL = FOLLOW_REL / "uniform-gross-stream2-training-budget-control"
KN_TRAIN_REL = Path("ip-followup-20261025/objective-gap-designs-20261025/coupled-bundle/scale-extension-g32-v1/short-prefix-n36/training-candidates")
METHODS = ("Full", "Ours", "SVM", "Neural Diving", "MLOPT", "Preference Learning")
EXPECTED = {
    "unit_commitment": {
        "Full": (360, 360, "1.000000", None),
        "Ours": (320, 360, "0.999094", (4, "0.0165", "0.0008")),
        "SVM": (65, 341, "0.924027", (3, "0.130", "0.049")),
        "Neural Diving": (268, 359, "0.994492", (1, "236.2", "24.8")),
        "MLOPT": (74, 360, "0.920463", (3, "1.066", "0.152")),
        "Preference Learning": (79, 360, "0.906016", (1, "662.9", "199.4")),
    },
    "knapsack": {
        "Full": (360, 360, "1.000000", None),
        "Ours": (360, 360, "1.000000", (4, "0.1146", "0.0106")),
        "SVM": (4, 358, "0.886799", (3, "0.264", "0.098")),
        "Neural Diving": (52, 359, "0.960871", (1, "239.4", "18.0")),
        "MLOPT": (35, 360, "0.953409", (3, "1.538", "0.731")),
        "Preference Learning": (51, 360, "0.953540", (1, "713.1", "255.9")),
    },
}
MATRIX_KEYS = ("A_data", "A_indices", "A_indptr", "A_shape", "row_lower",
               "row_upper", "col_lower", "col_upper", "integrality",
               "cost_offset", "cost_feature_coefficients")


def npy(blob):
    """Decode this release's numeric NPY files without loading any executable."""
    if blob[:6] != b"\x93NUMPY":
        raise ValueError("Not an NPY array")
    version = tuple(blob[6:8])
    offset = 10 if version == (1, 0) else 12
    length = struct.unpack("<H" if offset == 10 else "<I", blob[8:offset])[0]
    header = ast.literal_eval(blob[offset:offset + length].decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("Unexpected Fortran-order array")
    dtype, shape = header["descr"], header["shape"]
    codes = {"f8": "d", "f4": "f", "i8": "q", "i4": "i", "i2": "h",
             "i1": "b", "u8": "Q", "u4": "I", "u2": "H", "u1": "B", "b1": "?"}
    if dtype[1:] not in codes or dtype[0] not in "<>=|":
        raise ValueError("Unsupported dtype: " + str(dtype))
    endian = ">" if dtype[0] == ">" else "<"
    values = [v[0] for v in struct.iter_unpack(endian + codes[dtype[1:]], blob[offset + length:])]
    if len(values) != math.prod(shape):
        raise ValueError("NPY payload length mismatch")
    if len(shape) == 1:
        return values
    if len(shape) == 2:
        return [values[i:i + shape[1]] for i in range(0, len(values), shape[1])]
    raise ValueError("Unexpected NPY shape: " + str(shape))


class Review:
    def __init__(self, root):
        self.root = root
        self.errors, self.sources, self.check_count = [], {}, 0

    def check(self, condition, message):
        self.check_count += 1
        if not condition:
            self.errors.append(message)

    def source(self, path):
        content = path.read_bytes()
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError:
            relative = path.name
        self.sources[relative] = hashlib.sha256(content).hexdigest()
        return content

    def read(self, path):
        return json.loads(self.source(path).decode("utf-8"))

    def npz(self, path, keys):
        self.source(path)
        with zipfile.ZipFile(path) as archive:
            return {key: npy(archive.read(key + ".npy")) for key in keys}


def feasible(x, a):
    if x is None or len(x) != a["A_shape"][1] or not all(math.isfinite(v) for v in x):
        return False
    for j, value in enumerate(x):
        if value < a["col_lower"][j] - 1e-6 or value > a["col_upper"][j] + 1e-6:
            return False
        if a["integrality"][j] and abs(value - round(value)) > 1e-6:
            return False
    for row in range(a["A_shape"][0]):
        lo, hi = a["A_indptr"][row:row + 2]
        value = math.fsum(a["A_data"][k] * x[a["A_indices"][k]] for k in range(lo, hi))
        if value < a["row_lower"][row] - 1e-6 or value > a["row_upper"][row] + 1e-6:
            return False
    return True


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def record(review, raw, a, affine, bank, index, seed, method, problem, scope):
    tag = f"{problem}/{seed}/{method}/{index}"
    review.check(raw["index"] == index, tag + ": test index")
    review.check(raw["cost"] == bank["costs"][index] and raw["feature"] == bank["features"][index], tag + ": frozen bank cost/features")
    reference = bank["objectives"][index]
    review.check(raw["reference_objective"] == reference, tag + ": Full reference")
    if "record_payload_sha256" in raw:
        payload = {k: v for k, v in raw.items() if k != "record_payload_sha256"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        review.check(digest == raw["record_payload_sha256"], tag + ": payload digest")
    x, cost = raw.get("solution"), bank["costs"][index]
    original = feasible(x, a)
    restriction = False
    if method == "Ours":
        normals = [[float(Fraction(v)) for v in row] for row in affine["normals_exact"]]
        anchor = affine["anchor"]
        restriction = original and all(abs(dot(row, [x[j] - anchor[j] for j in range(scope)])) <= 1e-6 for row in normals)
        actual_k = affine["rank"]
    else:
        fixed = raw.get("fixed")
        actual_k = scope - len(fixed) if fixed is not None else None
        if fixed is not None:
            review.check(all(0 <= int(j) < scope and v in (0, 1) for j, v in fixed.items()), tag + ": coordinate scope")
            restriction = original and all(abs(x[int(j)] - v) <= 1e-6 for j, v in fixed.items())
    timely = raw["seconds"] <= 30.0
    valid = bool(original and restriction and actual_k == affine["rank"] and timely)
    value = dot(cost, x) if original else None
    tolerance = 1e-6 + 1e-9 * abs(reference)
    recovered = bool(valid and abs(value - reference) <= tolerance)
    ratio = None if not valid else (reference / value if problem == "unit_commitment" else value / reference)
    review.check(raw["actual_K"] == actual_k, tag + ": matched K")
    review.check(raw["original_feasible"] == original, tag + ": original primal feasibility")
    if "restriction_valid" in raw:
        review.check(raw["restriction_valid"] == restriction, tag + ": affine/fixing restriction")
    review.check(raw["timely"] == timely and raw["valid"] == valid, tag + ": valid timely return")
    review.check(raw["recovered"] == recovered, tag + ": Recovery")
    saved_ratio = raw.get("objective_ratio", raw.get("profit_ratio"))
    review.check((ratio is None and saved_ratio is None) or (ratio is not None and saved_ratio is not None and abs(ratio - saved_ratio) < 2e-12), tag + ": objective ratio")
    if valid:
        review.check(abs(value - raw["objective"]) <= tolerance and value >= reference - tolerance, tag + ": original objective arithmetic")
        review.check(0 < ratio <= 1 + 1e-9, tag + ": ratio direction/range")
    return dict(seed=seed, index=index, valid=valid, recovered=recovered, ratio=ratio, K=actual_k)


def require_tests(review, folder):
    files = sorted(folder.glob("test*.json"))
    review.check([f.name for f in files] == [f"test{i:03d}.json" for i in range(120)], str(folder.name) + ": complete 120-test sequence")
    return files


def validate_case_bank(review, training, bank_path, problem, seed, expected_k):
    a = review.npz(training / "training.npz", MATRIX_KEYS)
    affine = review.read(training / "affine.json")
    scope = 48 if problem == "unit_commitment" else 96
    review.check(affine["rank"] == expected_k and len(affine["normals_exact"]) == scope - expected_k, f"{problem}/{seed}: affine rank/dimensions")
    bank = review.npz(bank_path, ("features", "costs", "solutions", "objectives"))
    review.check(all(len(values) == 120 for values in bank.values()), f"{problem}/{seed}: bank sizes")
    for index, (feature, cost, x, value) in enumerate(zip(bank["features"], bank["costs"], bank["solutions"], bank["objectives"])):
        tag = f"{problem}/{seed}/Full/{index}"
        review.check(feasible(x, a), tag + ": original feasibility")
        review.check(abs(dot(cost, x) - value) <= 1e-6 + 1e-9 * abs(value), tag + ": original objective")
        mapped = [a["cost_offset"][j] + dot(row, feature) for j, row in enumerate(a["cost_feature_coefficients"])]
        review.check(all(abs(c - v) < 1e-10 for c, v in zip(cost, mapped)), tag + ": declared cost map")
        review.check(value > 0 if problem == "unit_commitment" else value < 0, tag + ": objective sign")
        if problem == "knapsack":
            review.check(all(cost[j] == -0.25 for j in range(96) if j % 3 != 2), tag + ": full basic-item profit retained")
    return a, affine, bank


def collect(review):
    pools = {problem: {method: [] for method in METHODS + ("Neural Diving continued",)} for problem in EXPECTED}
    uc = review.root / UC_REL
    for seed, K in ((101, 14), (211, 14), (307, 17)):
        cid = f"uc_startupheavy_G4T12_all_units_nrel_startup_seed{seed}_n32"
        training = uc / f"run-v1/settings/G4T12_all_units_nrel_startup_seed{seed}/n32"
        base = uc / "confirmation-v1/results-v1" / cid
        a, affine, bank = validate_case_bank(review, training, base / "confirmation.npz", "unit_commitment", seed, K)
        for index, path in enumerate(require_tests(review, base / "raw-records")):
            container = review.read(path)
            full = container["Full"]
            review.check(full["certified"] and full["optimal"] and abs(full["gap"]) <= 1e-9
                         and full["cold_wall_seconds"] <= 30
                         and full["objective"] == bank["objectives"][index],
                         f"UC/{seed}/Full/{index}: recorded timely global certificate")
            pools["unit_commitment"]["Full"].append(dict(seed=seed, index=index, valid=True, recovered=True, ratio=1.0))
            for method in ("Ours", "SVM", "MLOPT"):
                raw = dict(container["methods"][method], cost=container["cost"], feature=container["feature"])
                pools["unit_commitment"][method].append(record(review, raw, a, affine, bank, index, seed, method, "unit_commitment", 48))
        for method, relative in (("Neural Diving", "nd-supplement-v1/results"), ("Neural Diving continued", "nd-supplement-v1/continued-results"), ("Preference Learning", "li-supplement-v1/same-bank-results-v2")):
            for index, path in enumerate(require_tests(review, uc / relative / cid)):
                pools["unit_commitment"][method].append(record(review, review.read(path), a, affine, bank, index, seed, method, "unit_commitment", 48))
    kn = review.root / KN_REL
    for seed in (1009, 1013, 1019):
        cid = f"coupledbundle_uniformgross_random32_b0.5_seed{seed}_n36"
        training = review.root / KN_TRAIN_REL / f"random32_b0.5_seed{seed}/n36"
        a, affine, bank = validate_case_bank(review, training, kn / f"banks/stream2_G32_seed{seed}/confirmation.npz", "knapsack", seed, 32)
        reference_folder = kn / f"banks/stream2_G32_seed{seed}"
        certificates = review.read(reference_folder / "reference-certificates.json")
        review.check(certificates["count"] == 120 and certificates["all_original_IP_certified"], f"knapsack/{seed}: Full certificate completeness")
        for index in range(120):
            path = reference_folder / f"reference-records/ref{index:03d}.json"
            full = review.read(path)
            review.check(hashlib.sha256(path.read_bytes()).hexdigest() == certificates["record_file_sha256"]["reference-records/" + path.name],
                         f"knapsack/{seed}/Full/{index}: reference digest")
            review.check(full["highs"]["optimal"] and abs(full["highs"]["gap"]) <= 1e-9
                         and full["cold_Full_wall_seconds"] <= 30
                         and full["objective"] == bank["objectives"][index]
                         and abs(full["objective"] - full["independent_objective"]) <= 1e-6 + 1e-9 * abs(full["objective"])
                         and full["cost"] == bank["costs"][index], f"knapsack/{seed}/Full/{index}: recorded timely global reference")
            pools["knapsack"]["Full"].append(dict(seed=seed, index=index, valid=True, recovered=True, ratio=1.0))
        roles = {"Ours": "ours", "SVM": "svm", "MLOPT": "mlopt", "Neural Diving": "neural_diving_original", "Neural Diving continued": "neural_diving_continued", "Preference Learning": "li"}
        for method, role in roles.items():
            base = review.root / FOLLOW_REL / "li-supplement-v1/results" if role == "li" else kn / "results"
            for index, path in enumerate(require_tests(review, base / cid / role)):
                pools["knapsack"][method].append(record(review, review.read(path), a, affine, bank, index, seed, method, "knapsack", 96))
    return pools


def resolve_original_path(snapshot, name):
    normalized = name.replace("\\", "/")
    marker = "/ip-paper-baselines-v2-20261009/"
    if marker not in normalized:
        raise ValueError("Unexpected original provenance path")
    return snapshot / normalized.split(marker, 1)[1]


def learning_times(review, provenance):
    source = review.read(provenance / "current-table3-4-learning-time-audit.json")
    result = {}
    for problem, key in (("unit_commitment", "uc"), ("knapsack", "knapsack")):
        result[problem] = {}
        for row in source[key]["per_seed"]:
            for old_name, item in row["sources"].items():
                method = "Preference Learning" if old_name == "Li" else old_name
                saved = review.read(resolve_original_path(review.root, item["path"]))
                for field in item["field"].split("."):
                    saved = saved[field]
                review.check(saved == item["seconds"] == row["fit_wall_seconds"][old_name], f"{problem}/{row['seed']}/{method}: source fitting time")
                result[problem].setdefault(method, []).append(saved)
    return result


def continued_training_metadata(review):
    result = {}
    for problem, seeds in (("unit_commitment", (101, 211, 307)), ("knapsack", (1009, 1013, 1019))):
        cases = []
        for seed in seeds:
            if problem == "unit_commitment":
                cid = f"uc_startupheavy_G4T12_all_units_nrel_startup_seed{seed}_n32"
                path = review.root / UC_REL / "nd-supplement-v1/continued-models" / cid / "metadata.json"
            else:
                cid = f"coupledbundle_uniformgross_random32_b0.5_seed{seed}_n36"
                path = review.root / "ip-followup-20261025/objective-gap-designs-20261025/coupled-bundle/training-budget-control-v1/continued-models" / cid / "metadata.json"
            meta = review.read(path)
            final = meta["trajectory"][-1]
            review.check(meta["optimizer_steps"] == meta["final_checkpoint_step"] == final["step"] == 2000,
                         f"{problem}/{seed}: fixed final 2000-update checkpoint")
            review.check(meta["optimizer_state_reset"] and meta["training_only"], f"{problem}/{seed}: continued-training control scope")
            bit_error = final["diagnostics"]["expected_map_bit_error"]
            cases.append(dict(seed=seed, final_training_bit_error=bit_error))
        errors = [case["final_training_bit_error"] for case in cases]
        if problem == "unit_commitment":
            review.check(all(value == 0 for value in errors), "UC: published zero continued-training bit errors")
        else:
            review.check((f"{100*min(errors):.2f}", f"{100*max(errors):.2f}") == ("0.12", "1.48"),
                         "knapsack: published continued-training bit-error range")
        result[problem] = cases
    return result


def summarize(review, pools, timing):
    tables, controls = {}, {}
    for problem, methods in pools.items():
        rows = []
        for method in METHODS:
            if method == "Full":
                review.check(len(methods["Full"]) == 360, problem + ": 360 recorded certified Full references")
                recovered = sum(row["recovered"] for row in methods["Full"])
                N = sum(row["valid"] for row in methods["Full"])
                ratio = statistics.mean(row["ratio"] for row in methods["Full"])
                K = [48] * 3 if problem == "unit_commitment" else [96] * 3
            else:
                data = methods[method]
                review.check(len(data) == 360, problem + "/" + method + ": all-test denominator 360")
                recovered = sum(row["recovered"] for row in data)
                ratios = [row["ratio"] for row in data if row["valid"]]
                N, ratio = len(ratios), statistics.mean(ratios)
                K = [14, 14, 17] if problem == "unit_commitment" else [32] * 3
            exp_recovered, exp_N, exp_ratio, exp_time = EXPECTED[problem][method]
            review.check((recovered, N, f"{ratio:.6f}") == (exp_recovered, exp_N, exp_ratio), problem + "/" + method + ": published Recovery and ratio cells")
            row = dict(method=method, K_by_instance=K, tests=360, recovered=recovered,
                       recovery_percent=100 * recovered / 360, ratio_valid_returns=N,
                       objective_ratio=ratio, learning_time_seconds=None,
                       learning_time_sample_sd_seconds=None, learning_time_by_instance=None)
            if exp_time is not None:
                times = timing[problem][method]
                review.check(len(times) == 3, problem + "/" + method + ": three learning times")
                mean, sd = statistics.mean(times), statistics.stdev(times)
                decimals, expected_mean, expected_sd = exp_time
                review.check((f"{mean:.{decimals}f}", f"{sd:.{decimals}f}") == (expected_mean, expected_sd), problem + "/" + method + ": published learning mean/sample-SD cells")
                row.update(learning_time_seconds=mean, learning_time_sample_sd_seconds=sd, learning_time_by_instance=times)
            rows.append(row)
        tables[problem] = rows
        original = {(r["seed"], r["index"]): r for r in methods["Neural Diving"]}
        continued = methods["Neural Diving continued"]
        pairs = [(original[r["seed"], r["index"]], r) for r in continued if r["valid"] and original[r["seed"], r["index"]]["valid"]]
        prior = statistics.mean(a["ratio"] for a, b in pairs)
        after = statistics.mean(b["ratio"] for a, b in pairs)
        expected = ("0.994928", "0.993391") if problem == "unit_commitment" else ("0.961690", "0.956717")
        review.check((f"{prior:.6f}", f"{after:.6f}") == expected, problem + ": published paired continued-training control")
        controls[problem] = dict(additional_optimizer_updates=2000, total_test_costs=360,
                                paired_valid_returns=len(pairs), original_objective_ratio=prior,
                                continued_objective_ratio=after,
                                aggregation="Same costs where both versions returned valid timely solutions; no selection by objective value")
    return tables, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--snapshot-root", type=Path, default=base / "snapshot")
    parser.add_argument("--provenance-root", type=Path, default=base / "provenance")
    parser.add_argument("--output-dir", type=Path, default=base / "results")
    args = parser.parse_args()
    review = Review(args.snapshot_root.resolve())
    pools = collect(review)
    timing = learning_times(review, args.provenance_root)
    tables, controls = summarize(review, pools, timing)
    continued_metadata = continued_training_metadata(review)
    for problem in controls:
        controls[problem]["final_training_diagnostics_by_instance"] = continued_metadata[problem]
    source_list = "\n".join(path + " " + digest for path, digest in sorted(review.sources.items()))
    output = dict(schema="raw-validated-current-ip-paper-results-v1", status="PASS" if not review.errors else "FAIL",
                  checks=review.check_count, errors=review.errors, tables=tables, continued_neural_diving_control=controls,
                  recorded_source_files=len(review.sources), source_list_sha256=hashlib.sha256(source_list.encode()).hexdigest(),
                  metrics=dict(recovery="All 360 test costs, original feasible return within 30 seconds, absolute objective error <=1e-6+1e-9*abs(Full).",
                               unit_commitment_ratio="Full original positive total cost / returned original total cost.",
                               knapsack_ratio="Returned original gross profit / Full optimal gross profit, including all basic-item values.",
                               ratio_mean="Arithmetic mean over each method's feasible, restriction-valid returns within 30 seconds; counts are recorded for auditability.",
                               learning_time="Post-label fitting, model selection and refitting. Arithmetic mean and sample standard deviation across three recorded instance timings. Shared training-optimum generation excluded."),
                  validation_scope="Reads frozen NPZ numeric arrays and JSON records only. Recomputes primal constraints, bounds, integrality, affine/fixing restrictions, cost-map inputs, c dot x, metric decisions and timing aggregation. Reuses recorded Full reference bank; no new global optimality proof or solver/training execution.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for number, problem in ((3, "unit_commitment"), (4, "knapsack")):
        with (args.output_dir / f"table{number}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Method", "K", "Recovery (%)", "Objective ratio", "Learning time (s)"])
            for row in tables[problem]:
                values = row["K_by_instance"]
                K = str(values[0]) if len(set(values)) == 1 else f"{min(values)}--{max(values)}"
                fmt = EXPECTED[problem][row["method"]][3]
                fit = "---" if fmt is None else f"{row['learning_time_seconds']:.{fmt[0]}f} +/- {row['learning_time_sample_sd_seconds']:.{fmt[0]}f}"
                writer.writerow([row["method"], K, f"{row['recovery_percent']:.2f}", f"{row['objective_ratio']:.6f}", fit])
    print(json.dumps(dict(status=output["status"], checks=review.check_count, errors=review.errors[:20], source_files=len(review.sources), output=str(args.output_dir)), indent=2))
    return int(bool(review.errors))


if __name__ == "__main__":
    sys.exit(main())
