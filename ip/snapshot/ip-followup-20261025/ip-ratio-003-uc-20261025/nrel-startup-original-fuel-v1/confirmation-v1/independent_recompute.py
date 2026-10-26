"""One-pass independent arithmetic review of completed UC confirmation records.

No experiment runtime, pickle, model, solver or RNG is imported or executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import sympy as sp
from scipy import sparse

METHODS = ("Ours", "SVM", "MLOPT")
SEEDS = (101, 211, 307)
TOL = 1e-6


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics(x, physical, G=4, T=12):
    if x is None:
        return False
    x = np.asarray(x, dtype=float)
    if x.shape != (4*G*T,) or not np.isfinite(x).all():
        return False
    u, v, w, p = x.reshape(4, T, G)
    cap, low, up, down = (np.asarray(physical[k]) for k in ("capacity", "minimum_output", "ramp_up", "ramp_down"))
    if (np.max(np.abs(x[:3*G*T]-np.rint(x[:3*G*T]))) > TOL or
            np.min(x) < -TOL or np.max(x[:3*G*T]) > 1+TOL):
        return False
    if np.max(abs(p.sum(axis=1)-np.asarray(physical["demand"]))) > TOL:
        return False
    if np.max(p-cap*u) > TOL or np.max(low*u-p) > TOL or np.max(v+w) > 1+TOL:
        return False
    previous_u = np.vstack((physical["initial_commitment"], u[:-1]))
    previous_p = np.vstack((physical["initial_generation"], p[:-1]))
    if np.max(abs(u-previous_u-v+w)) > TOL:
        return False
    if np.max(p-previous_p-up*previous_u-cap*v) > TOL or np.max(previous_p-p-down*u-cap*w) > TOL:
        return False
    for t in range(T):
        for g in range(G):
            if v[max(0, t-int(physical["minimum_up_time"][g])+1):t+1, g].sum() > u[t,g]+TOL:
                return False
            if w[max(0, t-int(physical["minimum_down_time"][g])+1):t+1, g].sum() > 1-u[t,g]+TOL:
                return False
    return True


def feasible(x, arrays, physical):
    if not physics(x, physical):
        return False
    x = np.asarray(x, dtype=float)
    A = arrays["A"]
    ax = A @ x
    return bool(np.max(arrays["row_lower"]-ax) <= TOL and np.max(ax-arrays["row_upper"]) <= TOL
                and np.max(arrays["col_lower"]-x) <= TOL and np.max(x-arrays["col_upper"]) <= TOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent/"results-v1")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"independent-confirmation-review.json")
    args = parser.parse_args()
    summary = read(args.results/"summary.json")
    assert summary["status"] == "COMPLETE", "Only completed confirmation is reviewed"
    assert (summary["case_count"], summary["Full_reference_count"], summary["method_return_count"]) == (3, 360, 1080)
    freeze_path = Path(__file__).parent/"global-freeze.json"
    freeze = read(freeze_path)
    assert digest(freeze_path) == summary["source_freeze_sha256"]
    assert len(freeze["cases"]) == 3 and freeze["model_role_count"] == 9
    source_hashes = {str(args.results/"summary.json"): digest(args.results/"summary.json"), str(freeze_path): digest(freeze_path)}
    for name, expected in summary["files"].items():
        assert digest(args.results/name) == expected, "Result hash mismatch: "+name
    errors, trials, cases = [], [], []
    def check(ok, where):
        if not bool(ok):
            errors.append(where)
    for entry in sorted(summary["cases"], key=lambda x:x["seed"]):
        cid, seed = entry["case_id"], entry["seed"]
        assert seed in SEEDS and entry["count"] == 120
        folder = args.results/cid
        source = Path(freeze["cases"][cid]["learning_timer_source"]).parent
        meta, affine = read(source/"metadata.json"), read(source/"affine.json")
        physical = meta["metadata"]
        K = int(affine["rank"])
        with np.load(source/"training.npz", allow_pickle=False) as z:
            keys = ("A_data", "A_indices", "A_indptr", "A_shape", "row_lower", "row_upper", "col_lower", "col_upper", "integrality", "train_solutions", "cost_offset", "cost_feature_coefficients")
            arrays = {key:z[key].copy() for key in keys}
        arrays["A"] = sparse.csr_matrix((arrays["A_data"], arrays["A_indices"], arrays["A_indptr"]), shape=tuple(arrays["A_shape"]))
        assert arrays["A"].shape[1] == 192 and np.array_equal(arrays["integrality"], np.r_[np.ones(144), np.zeros(48)])
        training_u = np.rint(arrays["train_solutions"][:, :48]).astype(int)
        exact_K = len(sp.Matrix((training_u[1:]-training_u[0]).T.tolist()).rref()[1])
        check(exact_K == K == entry["K"] == freeze["cases"][cid]["K"], cid+": independent training rank")
        check(np.all((np.asarray(physical["fuel_cost"]) >= 22)&(np.asarray(physical["fuel_cost"]) <= 45)), cid+": original fuel range")
        rate = np.asarray(physical["startup_cost"])/physical["capacity"]
        check(np.allclose(rate, 22+(np.asarray(physical["original_startup_draw"])-220)*38/630, rtol=0, atol=1e-12), cid+": declared startup mapping")
        base = np.concatenate([np.tile(physical[key],12) for key in ("no_load_cost", "startup_cost", "shutdown_cost", "fuel_cost")])
        coeff = np.zeros((192,12))
        for component in range(4):
            for t in range(12):
                for g in range(4):
                    j=component*48+t*4+g
                    coeff[j,3*g+t//4]=base[j]
        check(np.array_equal(coeff, arrays["cost_feature_coefficients"]) and np.array_equal(arrays["cost_offset"], np.zeros(192)), cid+": full unshifted cost mapping")
        normals = np.asarray(affine["normals_exact"], dtype=object).reshape(48-K,48)
        anchor = np.asarray(affine["anchor"], dtype=object)
        with np.load(folder/"confirmation.npz", allow_pickle=False) as z:
            bank = {key:z[key].copy() for key in ("features", "costs", "solutions", "objectives")}
        assert bank["costs"].shape == (120,192) and bank["features"].shape == (120,12)
        refs = read(folder/"Full-reference-certificates.json")
        assert len(refs) == 120
        listed = sorted((folder/"raw-records").glob("test*.json"))
        assert [p.name for p in listed] == [f"test{i:03d}.json" for i in range(120)]
        for i, path in enumerate(listed):
            raw = read(path)
            feature, cost = np.asarray(raw["feature"]), np.asarray(raw["cost"])
            full, full_x = raw["Full"], np.asarray(raw["Full_solution"])
            where = f"{cid}/test{i:03d}"
            check(raw["index"] == i and raw["source_freeze_sha256"] == summary["source_freeze_sha256"], where+": identity")
            check(np.array_equal(feature, bank["features"][i]) and np.array_equal(cost, bank["costs"][i]) and np.array_equal(full_x, bank["solutions"][i]), where+": bank/raw inputs")
            check(np.all((feature >= 1/3)&(feature <= 3)) and np.allclose(cost, coeff@feature, rtol=1e-14, atol=1e-12), where+": law and original cost map")
            reference = float(cost@full_x)
            tol = 1e-6+1e-9*abs(reference)
            check(feasible(full_x, arrays, physical), where+": independent Full feasibility")
            check(full == refs[i] and full["certified"] and full["valid"] and reference > 0, where+": certified reference record")
            check(max(abs(reference-full["objective"]),abs(reference-full["dual_bound"]),abs(reference-bank["objectives"][i])) <= tol, where+": Full objective/primal/dual")
            assert set(raw["methods"]) == set(METHODS)
            for method, row in raw["methods"].items():
                tag = where+"/"+method
                x = row.get("solution")
                primal_ok = feasible(x, arrays, physical)
                restriction_ok = False
                actual_K = K if method == "Ours" else None
                if method == "Ours" and primal_ok:
                    u = np.rint(np.asarray(x)[:48]).astype(object)
                    restriction_ok = bool(np.all(normals@(u-anchor) == 0))
                if method != "Ours" and row.get("fixed") is not None:
                    fixed = {int(j):v for j,v in row["fixed"].items()}
                    actual_K = 48-len(fixed)
                    check(all(0 <= j < 48 and v in (0,1) for j,v in fixed.items()), tag+": common-u fixing scope")
                    restriction_ok = bool(primal_ok and all(abs(x[j]-v)<=TOL for j,v in fixed.items()))
                check(actual_K == row.get("actual_K"), tag+": actual K")
                timely = row["seconds"] <= 30
                valid = bool(primal_ok and restriction_ok and timely and actual_K == K and row["raw"].get("valid", False))
                value = float(cost@x) if primal_ok else None
                ratio = reference/value if valid else None
                recovered = bool(valid and abs(value-reference) <= tol)
                check(row["valid"] == valid, tag+": independently recomputed validity")
                check(row["reference_objective"] == reference and row["confirmation"] is True and row["phase"] == "independent_confirmation", tag+": phase/reference")
                check(row["recovered"] == recovered, tag+": recovery")
                check((ratio is None and row["objective_ratio"] is None) or (ratio is not None and abs(ratio-row["objective_ratio"])<1e-12), tag+": ratio")
                if valid:
                    check(abs(value-row["objective"])<=tol and value >= reference-tol, tag+": original returned objective")
                trials.append(dict(case_id=cid, seed=seed, index=i, method=method, K=K, actual_K=actual_K,
                                   valid=valid, recovered=recovered, original_cost=value, reference=reference,
                                   ratio=ratio, timely=timely, original_feasible=primal_ok))
        cases.append(dict(seed=seed, case_id=cid, n_fit=32, previously_collected_exploratory_labels=64, common_u_D=48, K=K, original_D=192, original_binary_D=144, reference_count=120))
    cells = []
    for method in METHODS:
        rows = [r for r in trials if r["method"]==method]
        ratios = [r["ratio"] for r in rows if r["valid"]]
        cell = dict(method=method, N=len(rows), ratio_N=len(ratios), recovered=sum(r["recovered"] for r in rows),
                    recovery_percent=100*sum(r["recovered"] for r in rows)/len(rows),
                    original_cost_ratio_mean=float(np.mean(ratios)) if ratios else None,
                    invalid=sum(not r["valid"] for r in rows), late=sum(not r["timely"] for r in rows))
        expected = next(c for c in summary["cells"] if c["method"]==method)
        check(cell["N"] == expected["total"] == 360 and cell["ratio_N"] == expected["ratio_denominator"] and cell["recovered"] == expected["recovered"], method+": aggregate counts")
        check(abs(cell["original_cost_ratio_mean"]-expected["objective_ratio_mean"])<1e-12, method+": pooled raw ratio mean")
        cells.append(cell)
    output = dict(status="PASS" if not errors else "FAIL", errors=errors, source_hashes=source_hashes, cases=cases,
                  cells=cells, trial_count=len(trials), reference_count=360, trials=trials,
                  reference_solver_rerun=False, models_or_pickles_loaded=False, timing_claim=False,
                  interpretation="Independent raw arithmetic, original ranged-row checks and separately coded UC physics reproduce the metrics. Original total operating/startup cost is retained. K describes only commitment u; startup/shutdown remain binary and dispatch continuous. NREL inspires a capital/maintenance lower-bound scale only; this is a synthetic robustness test, not plant calibration.")
    data=(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")
    if args.output.exists():
        assert args.output.read_bytes() == data, "Refusing to overwrite a different audit"
    else:
        args.output.write_bytes(data)
    print(json.dumps(dict(status=output["status"], errors=errors[:10], error_count=len(errors), cells=cells, output=str(args.output)), ensure_ascii=True, indent=2))
    return int(bool(errors))


if __name__=="__main__":
    raise SystemExit(main())
