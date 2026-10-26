# LP timing results in the manuscript

This directory contains the saved records for Tables 1 and 2 of **Solving
Linear and Integer Programs Faster Using Data**. The tables use the current
manuscript names **Ours**, **DataDrivenProj**, and **FCNN**. Original method
identifiers are preserved in the source CSVs.

Verify the source hashes, reaggregate the recorded timings, and check all 68
mean/standard-deviation pairs against the manuscript:

```bash
python results/paper_lp/verify_tables.py
```

To write regenerated CSVs separately:

```bash
python results/paper_lp/verify_tables.py --output-dir out/paper_lp
```

This command uses only the Python standard library. It does not train models
or solve new LPs. The earlier `results/timing/` files describe different
historical workloads and remain unchanged.

## Table 1

`table1.csv` reports the outside-mass comparison at `rho = 0.5`, with 40
training costs before prior filtering and a common reduced dimension `K`.
Entries are means and sample standard deviations over three timing repeats
on fixed data. Learning is in seconds and solver time is in milliseconds.

The Ours/Full follow-up uses 20 test LPs. The archived baseline records use
12 test LPs. The DataDrivenProj timing column uses SGA for all four rows,
including the separate SC205 SGA record. The corresponding historical
accuracy plot uses PCA for SC205.

## Table 2

`table2.csv` reports the controlled comparison with 20 training LPs and 20
test LPs per seed. Methods use the same compact LP, training/test costs,
and reduced solver dimension within each instance and seed. Entries are
means and sample standard deviations across independent data seeds
`17, 1000020, 2000023`.

Each `K` triple lists the deployed solver dimensions for these three seeds.
The saved protocol uses `K = max(1, raw learned rank)`, allowing a
one-dimensional solver representation when the learned optimizer span has
rank zero. Raw learned ranks remain available in the seed-level CSV.

FCNN is an adapted cost-only neural projection baseline. These timings use
two full-LP warm-start solves, 120 training epochs, 2,400 projected training
solves, and 726 validation solves. This initialization differs from the
historical plain-random accuracy runs.

## Files and provenance

- `sources/` contains byte-identical copies of the saved records.
- `provenance.json` records source hashes, historical relative source paths,
  the manuscript source hash, metric definitions, and displayed table cells.
- `verify_tables.py` regenerates the two CSVs from their recorded sources.

Solver times measure the HiGHS solve. Neural forward passes, per-query
matrix construction, and solver preparation are recorded separately and
are not included in these table columns.
