#!/usr/bin/env python3
"""Verify and reaggregate the recorded manuscript LP timing tables.

Uses only the Python standard library. Does not solve LPs or train models.
Run from any directory:
    python results/paper_lp/verify_tables.py
    python results/paper_lp/verify_tables.py --output-dir out/paper_lp
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "sources"
CASE_NAMES = {
    "packing": "Packing",
    "mincostflow": "MinCostFlow",
    "random_lp_A": "RandomLP A",
    "random_lp_B": "RandomLP B",
    "random_lp_C": "RandomLP C",
    "random_lp_D": "RandomLP D",
    "grow7": "GROW7",
    "sc205": "SC205",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def mean_sd(rows: list[dict[str, str]], key: str, scale: float = 1.0):
    values = [float(row[key]) * scale for row in rows]
    return statistics.mean(values), statistics.stdev(values)


def table1() -> list[dict]:
    ours = {
        row["case"]: row
        for row in read_csv(SOURCE_DIR / "figure3_new_ours_timing_summary.csv")
    }
    baselines = [
        row for row in read_csv(SOURCE_DIR / "figure3_baseline_timing_raw.csv")
        if float(row["rho"]) == 0.5
    ]
    sc205_sga = [
        row for row in read_csv(SOURCE_DIR / "figure3_sc205_sga_timing_raw.csv")
        if float(row["rho"]) == 0.5
    ]
    output = []
    for case in ("packing", "random_lp_A", "random_lp_C", "sc205"):
        source = ours[case]
        dimensions = list(map(int, source["K_by_repeat"].split(",")))
        assert len(dimensions) == 3 and len(set(dimensions)) == 1
        row = {
            "Instance": CASE_NAMES[case],
            "d": int(source["d_full"]),
            "K": dimensions[0],
            "Ours learning mean (s)": float(source["learning_s_mean"]),
            "Ours learning SD (s)": float(source["learning_s_repeat_sd"]),
        }
        for original, display in (("DataDriven", "DataDrivenProj"), ("FCNNc", "FCNN")):
            group = sc205_sga if case == "sc205" and original == "DataDriven" else [
                item for item in baselines
                if item["case"] == case and item["method"] == original
            ]
            assert len(group) == 3
            assert {int(item["K"]) for item in group} == {dimensions[0]}
            for metric, key, unit, scale in (
                ("learning", "learning_time_s", "s", 1.0),
                ("solve", "solve_time_mean_s", "ms", 1000.0),
            ):
                mean, sd = mean_sd(group, key, scale)
                row[f"{display} {metric} mean ({unit})"] = mean
                row[f"{display} {metric} SD ({unit})"] = sd
        for display, original in (("Full", "full"), ("Ours", "ours")):
            row[f"{display} solve mean (ms)"] = 1000 * float(source[f"{original}_solver_s_mean_mean"])
            row[f"{display} solve SD (ms)"] = 1000 * float(source[f"{original}_solver_s_mean_repeat_sd"])
        output.append(row)
    return output


def table2() -> list[dict]:
    raw = read_csv(SOURCE_DIR / "matched_k_timing_seed_level.csv")
    seeds = (17, 1000020, 2000023)
    assert len(raw) == 72
    output = []
    for case, name in CASE_NAMES.items():
        groups = {
            display: [row for row in raw if row["case"] == case and row["method"] == original]
            for display, original in (
                ("Ours", "Ours-matched-K"),
                ("FCNN", "FCNN-c-adapted-matched-K"),
                ("Full", "Full-LP"),
            )
        }
        assert all(len(group) == 3 for group in groups.values())
        def dimensions(method):
            return [int(next(row for row in groups[method] if int(row["seed"]) == seed)["K"]) for seed in seeds]
        ks = dimensions("Ours")
        assert ks == dimensions("FCNN")
        d = int(groups["Ours"][0]["d_full"])
        assert dimensions("Full") == [d] * 3
        assert all(int(row["n_train"]) == int(row["n_test"]) == 20 for group in groups.values() for row in group)
        row = {"Instance": name, "d": d, "K by seed (17,1000020,2000023)": ",".join(map(str, ks))}
        for method in ("Ours", "FCNN"):
            mean, sd = mean_sd(groups[method], "learning_s")
            row[f"{method} learning mean (s)"] = mean
            row[f"{method} learning SD (s)"] = sd
        for method in ("Full", "Ours", "FCNN"):
            mean, sd = mean_sd(groups[method], "solver_s_mean", 1000.0)
            row[f"{method} solve mean (ms)"] = mean
            row[f"{method} solve SD (ms)"] = sd
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Optionally write regenerated CSVs to a separate directory.")
    args = parser.parse_args()
    provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    for name, record in provenance["sources"].items():
        path = SOURCE_DIR / name
        assert path.stat().st_size == record["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], name
    computed = {"table1.csv": table1(), "table2.csv": table2()}
    cells = 0
    for name, rows in computed.items():
        recorded = read_csv(HERE / name)
        assert len(rows) == len(recorded)
        for row, saved in zip(rows, recorded):
            assert set(row) == set(saved)
            for key, value in row.items():
                if isinstance(value, (float, int)):
                    assert math.isclose(value, float(saved[key]), rel_tol=1e-12, abs_tol=1e-12), (name, row["Instance"], key)
                else:
                    assert value == saved[key], (name, row["Instance"], key)
                cells += 1
    for check in provenance["manuscript_cells"]:
        filename = "table1.csv" if check["table"] == "tab:runtime-rho" else "table2.csv"
        row = next(item for item in computed[filename] if item["Instance"] == check["instance"])
        unit = "s" if check["metric"] == "learning" else "ms"
        for statistic in ("mean", "sd"):
            key = f"{check['method']} {check['metric']} {'SD' if statistic == 'sd' else 'mean'} ({unit})"
            assert abs(row[key] - check[f"display_{statistic}"]) < 0.000500001, (filename, key)
    if args.output_dir:
        destination = args.output_dir.resolve()
        if destination == HERE:
            parser.error("Use a separate output directory to preserve the recorded tables.")
        destination.mkdir(parents=True, exist_ok=True)
        for name, rows in computed.items():
            with (destination / name).open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({"status": "PASS", "tables": 2, "table_cells_reaggregated": cells, "manuscript_mean_sd_pairs_checked": len(provenance["manuscript_cells"]), "source_files_sha256_checked": len(provenance["sources"]), "new_experiments": False}))


if __name__ == "__main__":
    main()
