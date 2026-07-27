# Reproduction Code for Main-Text Figures 2-4

This repository contains a compact reproduction package for the main-text
Figures 2-4 and the corresponding runtime benchmarks.
Figures are generated from code-produced raw outputs; fixed experimental
settings and artifact hashes are recorded with the corresponding results.

## Setup

Tested with Python 3.14 on Windows and CPU:

```bash
python -m pip install -r requirements.txt
```

SciPy/HiGHS solves all LPs on CPU. CUDA can accelerate FCNN tensor operations,
but not the LP solves.

## Reproduce Figures 2-4

Run the one-click reproduction:

```bash
python reproduce_figures.py
```

The default command verifies every bundled GROW7/SC205 `K=50` checkpoint by
SHA-256, loads the models, reruns inference and all fixed validation/test LP
solves, checks the reproduced objective ratios against the archived values,
updates those eight Figure 4 rows, and then validates and plots Figures 2-4.
The remaining panels are rendered from bundled raw outputs; use `--rerun`
below to retrain their models from source.

For an instant plot-only pass over the bundled raw CSV files, use:

```bash
python reproduce_figures.py --plot-only
```

Rerun and render one figure:

```bash
python reproduce_figures.py --rerun --only figure2
python reproduce_figures.py --rerun --only figure3
python reproduce_figures.py --rerun --only figure4
```

Rerun all three figures sequentially:

```bash
python reproduce_figures.py --rerun
```

Figure 2 is relatively quick. Complete Figure 3 and Figure 4 neural workloads
can take several hours on a laptop CPU. These commands use the archived
experiment implementations so that the main-text figures remain reproducible.

| Experiment | Raw results | Figure |
|---|---|---|
| Figure 2 | `results/fresh/figure2_rank_growth_raw.csv` | `figures/figure2_fresh_rank_growth.png` |
| Figure 3 | `results/fresh/figure3_timing_raw.csv` | `figures/figure3_fresh_legacy_execution.png` |
| Figure 4 | `results/fresh/figure4_timing_raw.csv` | `figures/figure4_fresh_legacy_execution.png` |

## Runtime

Run the documented three-repeat timing benchmark:

```bash
python benchmark_runtime.py --scope all --repeats 3 --force
```

Refresh only the improved Ours rows for Figure 3 with one wall-clock run:

```bash
python benchmark_runtime.py --scope figure3 --repeats 1 --force
```

The Figure 3 baselines are read unchanged from the committed three-repeat
legacy timing CSV. The new Ours row is one fresh wall-clock run and reports
HiGHS solver time separately from online preparation. Its exact learned span
is never truncated: 18 of 20 points use the archived baseline `K`; Packing at
`rho=0.1,0.3` learns `K=37,29`, while the archived baselines use `K=28,24`.
These dimensions and the baseline-source hash are recorded in the raw CSV.

Run only Figure 4 timing:

```bash
python benchmark_runtime.py --scope figure4 --repeats 3 --force
```

The default Ours implementation is in `code/core/ours.py`. For each training
cost it solves the compact full LP once, then obtains the
optimizer-difference span by one SVD. This is the generic implementation for
the continuous-cost setting, where the optimizer is unique almost surely.

For Figure 4, Ours is padded without truncation to the same `K` used by the
committed FCNN-c row. The command reruns Ours only. FCNN-c and Full LP are
read from the SHA-256-checked cache in `results/timing/cache/`; row provenance,
the raw learned rank, 20 training LP solves, and the cache source hash are
recorded in the generated CSV.

The archived ellipsoid implementation can still be timed explicitly:

```bash
python benchmark_runtime.py --scope figure4 --legacy-ours --repeats 3 --force
```

Runtime outputs report:

- offline representation-learning time;
- reduced-objective preparation time where applicable;
- reduced-LP solve-call time;
- end-to-end online time;
- full-LP solve time;
- the number of wall-clock repeats and available sample standard deviations.

The Figure 3 plot labels its one fresh Ours run and three archived baseline runs.

The runtime benchmark uses the 180-edge MinCostFlow case.

## Example Outputs

### Figure 2: Learned Dimension

![Figure 2 rank growth](figures/figure2_fresh_rank_growth.png)

### Figure 3: Objective Ratio versus Prior Misspecification

![Figure 3 accuracy](figures/figure3_fresh_legacy_execution.png)

### Figure 4: Objective Ratio versus Training Samples

![Figure 4 accuracy](figures/figure4_fresh_legacy_execution.png)


### Figure 3 Runtime

![Figure 3 runtime](figures/figure3_runtime_new_ours.svg)

### Figure 4 Runtime

![Figure 4 runtime](figures/figure4_runtime_matched_k.png)

## Experimental Protocol

### Figure 2

Figure 2 processes nested training prefixes from 0 to 25 samples on Packing,
MinCostFlow, RandomLP A-D, GROW7, and SC205. It records only the learned
dimension; this package does not include a Figure 2 runtime benchmark.

### Figure 3

Figure 3 uses 40 training LPs and
`rho = 0.1, 0.3, 0.5, 0.7, 0.9`. Packing, RandomLP A, and RandomLP C use the
archived SGA baseline; SC205 uses PCA. At each point, all methods use the
dimension learned by Ours.

### Figure 4

Figure 4 uses training sizes `20, 40, 60, 80` and FCNN-c dimensions
`K = 10, 20, 30, 50`. For each instance, one maximum pool is generated and
the training data are exact nested prefixes:

```text
first 20 samples subset first 40 subset first 60 subset first 80.
```

Validation and test sets remain fixed across the four prefixes. Each FCNN is
trained from scratch, so nested data do not mathematically guarantee monotone
FCNN test performance. No monotonic postprocessing is used. Curves show the
mean and one sample standard deviation over 12 held-out LPs.

Re-evaluate the checkpoints and regenerate Figure 4 with:

```bash
python reproduce_figures.py
```

To retrain the same deterministic GROW7 experiment from the archived code:

```bash
python code/runners/run_figure4_timing.py --repeats 1 --cases grow7 --n-list 20,40,60,80 --k-list 10,20,30,50 --out-dir results/reproduce_grow7_n80 --force
```

Fixed random seeds and hyperparameters are recorded in the output metadata.
Exact floating-point values can vary across machines, solver versions, and
thread schedules.

## Package Layout

- `reproduce_figures.py`: one-click checkpoint verification, LP re-evaluation,
  Figure 2-4 plotting, and optional full-rerun entry point.
- `benchmark_runtime.py`: one-click repeated timing entry point.
- `code/core/ours.py`: default Ours implementation.
- `code/core/`: archived LP, projection, SGA/PCA, and FCNN-c implementations.
- `code/experiments/`: instance, prior, and training-protocol definitions.
- `code/runners/`: Figure 2-4 and Figure 3-4 runtime drivers.
- `data/netlib/selected/`: GROW7 and SC205 MPS files.
- `data/checkpoints/figure4_netlib_k50/`: hashed Netlib FCNN-c states and provenance.
- `results/fresh/`: raw rows used by the accuracy figures.
- `results/timing/`: three-repeat raw timing rows.
- `results/timing/cache/`: frozen runtime baseline and provenance manifest.
- `figures/`: plots generated only from raw result files.
