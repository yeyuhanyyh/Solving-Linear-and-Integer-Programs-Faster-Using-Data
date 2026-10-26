# Solving Linear and Integer Programs Faster Using Data

Research code and experimental results for the paper by **Yuhan Ye and Omar Bennouna**.
We learn affine restrictions for LPs and IPs that share a feasible region and have varying objective coefficients.

## Getting started

Clone the repository with long-path support on Windows:

```bash
git clone -c core.longpaths=true https://github.com/yeyuhanyyh/Solving-Linear-and-Integer-Programs-Faster-Using-Data.git
cd Solving-Linear-and-Integer-Programs-Faster-Using-Data
```

Verify and rebuild the reported LP and IP tables without installing a solver or retraining models:

```bash
python results/paper_lp/verify_tables.py
python ip/run.py verify-artifacts
python ip/validate_results.py
```

These commands use the Python standard library. The repository includes source code, synthetic data, fitted models, individual test records, and the recorded learning times.

## Notation and methods

The notation and displayed method names below follow the current manuscript.

| Symbol | Meaning in the experiments |
|---|---|
| $c$ | Objective cost vector |
| $x_0+Uz$ | Learned affine restriction, with anchor $x_0$ and basis $U$ |
| $d$ | Original LP dimension after preprocessing |
| $n$ | Number of training costs. Table 1 counts costs before prior filtering |
| $K$ | Reduced LP variable count. For IPs, the learned affine dimension or the number of common integer coordinates left free by a baseline |
| $D$ | Size of the integer block being compressed: 48 for UC and 96 for knapsack |
| $\rho$ | Target probability mass excluded by the estimated cost prior $\widehat{\mathcal C}$ |

**Ours** is our affine compression method. The LP baselines are **DataDrivenProj** (a shared learned projection), **FCNN** (a cost-dependent neural projection), and **Rand** (random variable selection). **OursExact** denotes the known-prior method in the supplementary LP experiments. **Full** solves the original formulation.

The IP baselines are **SVM**, **Neural Diving**, **MLOPT**, and **Preference Learning**. Their implementations, training settings, and dimension-matching rules are described in [the IP documentation](ip/README.md). Archived code and data retain their original internal identifiers.

## LP results

The following timings reproduce Tables 1 and 2 of the current manuscript. Learning time is in seconds. LP solve time is the HiGHS solver time in milliseconds. The [saved records and verification script](results/paper_lp/) retain the separate preparation and prediction measurements.

### Table 1: outside-mass comparison

Here $\rho=0.5$ and $n=40$ before prior filtering. All compressed methods use the same $K$ for each instance. Entries are means ± sample standard deviations over three timing repetitions on fixed data.

**Learning time (s)**

| Instance | $d$ | $K$ | Ours | DataDrivenProj | FCNN |
|---|---:|---:|---:|---:|---:|
| Packing | 360 | 21 | 0.104 ± 0.006 | 3.314 ± 0.106 | 37.998 ± 0.247 |
| RandomLP A | 140 | 9 | 0.246 ± 0.001 | 0.913 ± 0.039 | 2.864 ± 0.010 |
| RandomLP C | 220 | 10 | 0.690 ± 0.001 | 1.955 ± 0.250 | 4.929 ± 0.077 |
| SC205 | 203 | 21 | 0.110 ± 0.004 | 6.632 ± 0.095 | 18.635 ± 0.243 |

**LP solve time (ms)**

| Instance | Full | Ours | DataDrivenProj | FCNN |
|---|---:|---:|---:|---:|
| Packing | 3.381 ± 0.112 | 2.720 ± 0.067 | 9.192 ± 1.278 | 5.755 ± 0.347 |
| RandomLP A | 10.790 ± 0.685 | 1.944 ± 0.031 | 5.227 ± 0.303 | 5.655 ± 1.693 |
| RandomLP C | 30.216 ± 0.082 | 2.893 ± 0.062 | 11.809 ± 1.428 | 9.286 ± 2.038 |
| SC205 | 4.273 ± 0.527 | 4.251 ± 0.219 | 4.740 ± 0.116 | 2.336 ± 0.237 |

The Ours/Full follow-up uses 20 test LPs, while the archived baseline records use 12. The [protocol](results/paper_lp/README.md) identifies the source runs and the DataDrivenProj implementation.

### Table 2: matched-dimension comparison

Both methods use $n=20$ training LPs and 20 test LPs per seed. Each $K$ triple gives the reduced dimensions for the three independent seeds. Entries are means ± sample standard deviations across those seeds.

**Learning time (s)**

| Instance | $d$ | $K$ by seed | Ours | FCNN |
|---|---:|---:|---:|---:|
| Packing | 360 | (19,19,19) | 0.084 ± 0.005 | 40.066 ± 1.766 |
| MinCostFlow | 360 | (3,1,1) | 0.189 ± 0.017 | 25.973 ± 5.973 |
| RandomLP A | 140 | (5,11,14) | 0.276 ± 0.016 | 25.985 ± 5.165 |
| RandomLP B | 180 | (10,7,8) | 0.459 ± 0.021 | 29.022 ± 3.529 |
| RandomLP C | 220 | (5,7,3) | 0.810 ± 0.050 | 28.304 ± 3.920 |
| RandomLP D | 260 | (7,10,5) | 1.197 ± 0.056 | 36.051 ± 5.238 |
| GROW7 | 301 | (19,19,19) | 0.378 ± 0.020 | 60.714 ± 0.634 |
| SC205 | 203 | (1,1,1) | 0.114 ± 0.007 | 14.193 ± 0.578 |

**LP solve time (ms)**

| Instance | Full | Ours | FCNN |
|---|---:|---:|---:|
| Packing | 7.208 ± 0.387 | 3.204 ± 0.136 | 11.249 ± 0.931 |
| MinCostFlow | 12.616 ± 0.497 | 1.890 ± 0.351 | 3.927 ± 1.025 |
| RandomLP A | 18.761 ± 2.068 | 2.789 ± 0.478 | 6.933 ± 0.997 |
| RandomLP B | 31.295 ± 2.158 | 2.774 ± 0.465 | 6.703 ± 1.666 |
| RandomLP C | 48.423 ± 4.036 | 2.765 ± 0.297 | 5.462 ± 0.636 |
| RandomLP D | 67.593 ± 1.726 | 3.433 ± 0.605 | 7.719 ± 2.428 |
| GROW7 | 25.622 ± 1.438 | 6.870 ± 0.431 | 16.727 ± 3.412 |
| SC205 | 7.243 ± 0.369 | 1.638 ± 0.094 | 2.763 ± 0.297 |

On these eight instances, Ours learns **30.1–477.0× faster than FCNN**. Its reduced LP solves faster than both Full and FCNN in every row of Table 2. In Table 1, FCNN has the shortest solve time on SC205.

### Accuracy and dimension plots

The original LP package also provides the following plots and raw results. The scripts retain their original artifact IDs.

| Experiment | Plot | Raw results | Runner ID |
|---|---|---|---|
| Objective ratio versus $\rho$ | [Outside-mass sweep](figures/figure3_fresh_legacy_execution.png) | [CSV](results/fresh/figure3_timing_raw.csv) | `figure3` |
| Objective ratio versus $n$, with $K\in\{10,20,30,50\}$ | [Training-sample comparison](figures/figure4_fresh_legacy_execution.png) | [CSV](results/fresh/figure4_timing_raw.csv) | `figure4` |
| Learned dimension versus $n$ | [Dimension growth](figures/figure2_fresh_rank_growth.png) | [CSV](results/fresh/figure2_rank_growth_raw.csv) | `figure2` |

For the LP plots, objective ratio is the returned objective divided by the Full objective. The code expresses these LPs as maximization problems. The outside-mass plot uses the saved standard errors for its bands. The training-sample plot uses sample standard deviations across held-out LPs. These plots are regenerated outputs of the archived LP package.

## IP results

Each problem uses three physical instances and 120 test costs per instance, giving **360 tests**. The UC model has 144 binary variables and 48 continuous variables. Its compression concerns the 48 on/off variables. The knapsack model has 96 binary variables, all included in the compression.

### Table 3: unit commitment with block-varying costs

Here $n=32$. Each compression baseline uses the same $K$ as Ours for each instance.

| Method | $K$ | Recovery (%) | Objective ratio | Learning time (s) |
|---|---:|---:|---:|---:|
| Full | 48 | 100.00 | 1.000000 | --- |
| Ours | 14–17 | 88.89 | 0.999094 | 0.0165 ± 0.0008 |
| SVM | 14–17 | 18.06 | 0.924027 | 0.130 ± 0.049 |
| Neural Diving | 14–17 | 74.44 | 0.994492 | 236.2 ± 24.8 |
| MLOPT | 14–17 | 20.56 | 0.920463 | 1.066 ± 0.152 |
| Preference Learning | 14–17 | 21.94 | 0.906016 | 662.9 ± 199.4 |

### Table 4: knapsack with conflicting upgrades

Here $n=36$ and $K=32$ for every compressed method.

| Method | $K$ | Recovery (%) | Objective ratio | Learning time (s) |
|---|---:|---:|---:|---:|
| Full | 96 | 100.00 | 1.000000 | --- |
| Ours | 32 | 100.00 | 1.000000 | 0.1146 ± 0.0106 |
| SVM | 32 | 1.11 | 0.886799 | 0.264 ± 0.098 |
| Neural Diving | 32 | 14.44 | 0.960871 | 239.4 ± 18.0 |
| MLOPT | 32 | 9.72 | 0.953409 | 1.538 ± 0.731 |
| Preference Learning | 32 | 14.17 | 0.953540 | 713.1 ± 255.9 |

Ours achieves higher recovery rates and objective ratios than the other compression methods on both problems. It also learns much faster than Neural Diving and Preference Learning.

**Recovery (%)** is the percentage of all 360 tests that return an original-feasible solution within 30 seconds and recover the optimal value within the stated numerical tolerance. **Objective ratio** is Full optimal cost divided by returned cost for UC, and returned profit divided by Full optimal profit for knapsack. Higher is better, with 1 denoting optimality. Ratio means use feasible solutions returned within the time limit.

**Learning time (s)** starts after computing the shared training optima. It includes model fitting, model selection, and refitting. Entries are means ± sample standard deviations across the three instances. The [IP directory](ip/) also contains the additional 2,000-update Neural Diving experiment, frozen checkpoints, and per-test records.

## Running the experiments

### LP code

Install the LP dependencies in a separate environment:

```bash
python -m pip install -r requirements.txt
python reproduce_figures.py --plot-only
```

To rerun an accuracy or dimension experiment, use its runner ID from the table above:

```bash
python reproduce_figures.py --rerun --only figure2
python reproduce_figures.py --rerun --only figure3
python reproduce_figures.py --rerun --only figure4
```

The earlier runtime workloads can also be rerun:

```bash
python benchmark_runtime.py --scope all --repeats 3 --force
```

Their outputs remain in `results/timing/`. The current manuscript timing tables are rebuilt from the separate saved records in `results/paper_lp/`. Full neural retraining can take several hours on a CPU.

### IP code

Install the IP dependencies and evaluate a frozen model:

```bash
python -m pip install -r ip/requirements.txt
python ip/run.py evaluate --problem uc --seed 101 --method neural-diving --limit 1
python ip/run.py evaluate --problem knapsack --seed 1009 --method preference-learning --limit 1
```

Fit a new model from the supplied training optimizers:

```bash
python ip/run.py train --problem uc --seed 101 --method ours
python ip/run.py train --problem knapsack --seed 1009 --method svm
```

New runs are saved under `ip/runs/`. Complete training and evaluation instructions are in [ip/README.md](ip/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `code/core/` | LP compression and projection implementations |
| `code/experiments/` | LP instances, cost distributions, and training protocols |
| `code/runners/` | LP experiment and runtime drivers |
| `data/netlib/selected/` | GROW7 and SC205 benchmark inputs |
| `data/checkpoints/` | Fitted LP neural models |
| `results/paper_lp/` | Current LP timing tables, saved sources, and verifier |
| `results/fresh/`, `figures/` | Archived LP accuracy and dimension results |
| `ip/` | IP code, data, models, results, and documentation |

The software license is in [LICENSE](LICENSE). Benchmark data provenance is documented in [DATA_NOTICE.md](DATA_NOTICE.md).
