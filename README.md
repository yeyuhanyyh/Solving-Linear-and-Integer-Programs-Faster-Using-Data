# Solving Linear and Integer Programs Faster Using Data

Research code and experimental results for the paper by **Yuhan Ye and Omar Bennouna**.

## Problem setting and main results

We study linear programs (LPs) and integer programs (IPs) that are solved repeatedly with the same constraints and different costs:

$$
\min_{x\in\mathcal X} c^\top x.
$$

Here $\mathcal X$ is the fixed, bounded feasible set and $c$ is the cost vector. We learn an affine space $x=x_0+Uz$, where $x_0$ is a reference solution and the columns of $U$ are learned directions. For each new cost, we solve for the reduced variables $z$ while keeping the original constraints, including the integer requirements for IPs.

> **Theorem (Exact compression).**
>
> An affine restriction preserves every optimal solution for every cost in the specified cost set if and only if it contains all solutions that can be optimal for those costs. The smallest such space is their affine hull. Its dimension determines how far the problem can be compressed without losing any optimizer.

<!-- Separate theorem blocks. -->

> **Theorem (Learning guarantee).**
>
> The algorithm starts from a solution that is optimal for an allowed cost, chosen independently of the training data. It adds directions until the affine space contains every optimizer of each training problem. For LPs, let $d^\star$ be the minimum dimension above. After $n$ independent training costs drawn from a distribution on the specified cost set, the algorithm learns at most $d^\star$ directions. With probability at least $1-\delta$ over the training data, its probability of failing to preserve all optimizers for a new independent cost from that distribution is at most
>
> $$
> \frac{4}{n}\left(6d^\star+\log\frac{e}{\delta}\right).
> $$

Thus, the bound decreases as $1/n$ and depends on the dimension needed for optimal decisions. The same guarantee holds for IPs using their corresponding minimum affine dimension. When the optimizer is unique almost surely, learning reduces to independent training solves followed by a basis computation.

The experiments below compare solution quality, learning time, and LP solve time. They cover synthetic LPs, Netlib benchmarks, unit commitment, and knapsack.

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

## Reading the results

| Symbol | Meaning |
|---|---|
| $n$ | Number of training examples |
| $d$ | Number of variables in the original LP after preprocessing |
| $K$ | Number of variables in the reduced LP. For IPs, see the explanation before Tables 3 and 4 |

**Ours** is our affine compression method. **Full** solves the original problem. **OursExact**, used in the supplementary LP experiments, is our method when the set of possible costs is known.

We implement the following baselines based on the cited papers:

| Problem | Method | What it learns or selects | Reference |
|---|---|---|---|
| LP | **DataDrivenProj** | One projection matrix shared across costs | [Sakaue and Oki (2024)](https://proceedings.nips.cc/paper_files/paper/2024/hash/1780db9272f81961459f4906036ebc6f-Abstract-Conference.html) |
| LP | **FCNN** | A neural network that generates a projection for each cost vector | [Iwata and Sakaue (2025)](https://proceedings.mlr.press/v267/iwata25a.html) |
| LP | **Rand** | A random subset of variables | [Akchen and Mišić (2024)](https://doi.org/10.1287/opre.2020.0494) |
| IP | **SVM** | Values for selected integer variables, which are then fixed | [Xavier, Qiu, and Ahmed (2021)](https://doi.org/10.1287/ijoc.2020.0976) |
| IP | **Neural Diving** | Partial integer assignments predicted by a neural network | [Nair et al. (2020)](https://arxiv.org/abs/2012.13349) |
| IP | **MLOPT** | Which strategy from previously solved problems to use | [Bertsimas and Stellato (2022)](https://doi.org/10.1287/ijoc.2022.1181) |
| IP | **Preference Learning** | How to rank reduced formulations by their solution quality and feasibility | [Li et al. (2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34916) |

The [LP timing protocol](results/paper_lp/README.md) and [IP documentation](ip/README.md) explain the implementations, training settings, and changes made to compare methods at the same compression level.

## LP results

The following timings reproduce Tables 1 and 2 of the manuscript. Learning time measures training in seconds. LP solve time measures the time HiGHS spends solving a test LP, in milliseconds. Prediction and preparation times are recorded separately in the [saved results](results/paper_lp/).

### Table 1: learning within a selected cost region

We start with 40 training LPs and learn within a selected region of the cost distribution. The parameter $\rho=0.5$ sets the target probability mass outside that region to one half. For each problem, Ours, DataDrivenProj, and FCNN produce LPs with the same number of variables, shown as $K$. We repeat the timing measurements three times using the same data and report the average ± standard deviation.

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

The recorded tests use 20 LPs for Ours and Full, and 12 for the projection baselines. See the [timing protocol](results/paper_lp/README.md) for details.

### Table 2: comparison at the same number of reduced variables

We run each experiment with three independently sampled datasets, each containing 20 training LPs and 20 test LPs. Within each run, Ours and FCNN use the same data and produce LPs with the same number of variables. The three $K$ values correspond to these three runs. We report the average timing ± standard deviation across the runs.

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

The original LP package also includes the following plots and raw results. Use the runner ID in the commands below to rerun an experiment.

| Experiment | Plot | Raw results | Runner ID |
|---|---|---|---|
| Objective ratio versus $\rho$ | [Outside-mass sweep](figures/figure3_fresh_legacy_execution.png) | [CSV](results/fresh/figure3_timing_raw.csv) | `figure3` |
| Objective ratio versus $n$, with $K\in\{10,20,30,50\}$ | [Training-sample comparison](figures/figure4_fresh_legacy_execution.png) | [CSV](results/fresh/figure4_timing_raw.csv) | `figure4` |
| Learned dimension versus $n$ | [Dimension growth](figures/figure2_fresh_rank_growth.png) | [CSV](results/fresh/figure2_rank_growth_raw.csv) | `figure2` |

For the LP plots, objective ratio is the returned objective divided by the Full objective. The code expresses these LPs as maximization problems. The outside-mass plot uses the saved standard errors for its bands. The training-sample plot uses sample standard deviations across held-out LPs. These plots are regenerated outputs of the archived LP package.

## IP results

We test unit commitment (UC) and knapsack on three problem instances each, using 120 test costs per instance, for **360 tests per problem**. UC has 144 binary variables and 48 continuous variables. We compress its 48 on/off decisions. Knapsack has 96 binary decisions, all of which are compressed.

For Ours, $K$ is the dimension of the learned affine space. We set each baseline to leave $K$ decisions unfixed in the same group of variables. The other decisions in that group are fixed using its predictions.

### Table 3: unit commitment with block-varying costs

Each instance uses 32 training examples. Ours learns dimensions of 14, 14, and 17 for the three instances, and each baseline leaves that many on/off decisions unfixed.

| Method | $K$ | Recovery (%) | Objective ratio | Learning time (s) |
|---|---:|---:|---:|---:|
| Full | 48 | 100.00 | 1.000000 | --- |
| Ours | 14–17 | 88.89 | 0.999094 | 0.0165 ± 0.0008 |
| SVM | 14–17 | 18.06 | 0.924027 | 0.130 ± 0.049 |
| Neural Diving | 14–17 | 74.44 | 0.994492 | 236.2 ± 24.8 |
| MLOPT | 14–17 | 20.56 | 0.920463 | 1.066 ± 0.152 |
| Preference Learning | 14–17 | 21.94 | 0.906016 | 662.9 ± 199.4 |

### Table 4: knapsack with conflicting upgrades

Each instance uses 36 training examples. Ours learns a 32-dimensional affine space, and each baseline leaves 32 item decisions unfixed.

| Method | $K$ | Recovery (%) | Objective ratio | Learning time (s) |
|---|---:|---:|---:|---:|
| Full | 96 | 100.00 | 1.000000 | --- |
| Ours | 32 | 100.00 | 1.000000 | 0.1146 ± 0.0106 |
| SVM | 32 | 1.11 | 0.886799 | 0.264 ± 0.098 |
| Neural Diving | 32 | 14.44 | 0.960871 | 239.4 ± 18.0 |
| MLOPT | 32 | 9.72 | 0.953409 | 1.538 ± 0.731 |
| Preference Learning | 32 | 14.17 | 0.953540 | 713.1 ± 255.9 |

Ours achieves higher recovery rates and objective ratios than the other compression methods on both problems. It also learns much faster than Neural Diving and Preference Learning.

**Recovery (%)** measures how often a method returns the original optimal value. It counts all 360 tests and requires a solution that satisfies the original constraints and is returned within 30 seconds. The numerical tolerance is specified in the [IP documentation](ip/README.md).

**Objective ratio** measures solution quality. It is optimal cost / returned cost for UC and returned profit / optimal profit for knapsack. Higher is better, and 1 means optimal. We average this ratio over feasible solutions returned within the time limit.

**Learning time (s)** measures model fitting, model selection, and refitting after the shared training problems have been solved. We report the average ± standard deviation across the three problem instances. All standard deviations in these tables use the sample standard deviation. The [IP directory](ip/) also contains an experiment with 2,000 additional Neural Diving training updates, saved models, and individual test results.

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
