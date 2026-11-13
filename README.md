# Solving Linear and Integer Programs Faster Using Data

Research code and experimental results for [our paper][paper] by **Yuhan Ye and Omar Bennouna**.

Many applications repeatedly solve linear or integer programs with the same constraints but changing objective coefficients. Our idea is to restrict optimization to an affine space containing the optimal solutions that can arise over a chosen cost region, and reuse this space for future solves. We characterize the smallest such space and develop algorithms that learn a compressed formulation from sampled costs, with guarantees on the probability of preserving the full optimizer set for a new cost. Choosing the cost region lets us balance the size of the compressed problem against the probability of preserving optimality.

## Problem setting and main results

We study linear programs (LPs) with a fixed nonempty bounded feasible polytope $\mathcal X\subseteq\mathbb R^d$ and varying cost vectors $c$ in a nonempty convex set $\mathcal C\subseteq\mathbb R^d$:

$$
\min_{x\in\mathcal X}c^\top x.
$$

Let $\mathcal X^\star(c)$ be the set of optimal solutions for cost $c$, and let $\mathcal X^\star(\mathcal C)$ collect these solutions over all allowed costs:

$$
\begin{aligned}
\mathcal X^\star(c)&:=\mathrm{argmin}_{x\in\mathcal X}c^\top x,\\
\mathcal X^\star(\mathcal C)&:=\bigcup_{c\in\mathcal C}\mathcal X^\star(c),
\end{aligned}
$$

For a set $S\subseteq\mathbb R^d$, define

$$
\begin{aligned}
\mathrm{dir}(S)&:=\mathrm{span}\lbrace x-y:x,y\in S\rbrace,\\
d^\star&:=\dim\mathrm{dir}\bigl(\mathcal X^\star(\mathcal C)\bigr).
\end{aligned}
$$

<a name="theorem-exact-compression"></a>

> #### Exact compression ([Proposition 2 in the paper][paper])
>
> Let $V\subseteq\mathbb R^d$ be an affine space. Then
>
> ```math
> \begin{gathered}
> \left[
> \begin{gathered}
> \mathrm{argmin}_{x\in\mathcal X\cap V}c^\top x =\mathcal X^\star(c)\\
> \forall c\in\mathcal C
> \end{gathered}
> \right]\\
> \iff \mathcal X^\star(\mathcal C)\subseteq V.
> \end{gathered}
> ```
>
> The restriction is exact if and only if
>
> ```math
> \begin{gathered}
> \mathrm{dir}\bigl(\mathcal X^\star(\mathcal C)\bigr)\subseteq\mathrm{dir}(V),\\
> V\cap\mathcal X^\star(\mathcal C)\ne\varnothing.
> \end{gathered}
> ```
>
> Among exact affine restrictions, the minimum dimension is $d^\star$, attained by $V=x_0+\mathrm{dir}(\mathcal X^\star(\mathcal C))$ for any $x_0\in\mathcal X^\star(\mathcal C)$.

Exact compression retains all potentially optimal solutions using only the directions along which they differ.

Write the restriction as $x=x_0+Uz$, where $x_0$ is a reference solution, the columns of $U$ are learned directions, and $z$ contains the reduced variables. Here $\mathrm{range}(U)$ is the column space of $U$, and feasibility requires $x_0+Uz\in\mathcal X$. Define $\mathsf{Exact}(U,x_0,c)$ by $\mathcal X^\star(c)\subseteq x_0+\mathrm{range}(U)$.

[Algorithm 1 in the paper][paper] starts with an empty $U$ and processes each training cost in turn. If an optimal vertex $x$ lies outside $x_0+\mathrm{range}(U)$, it appends $x-x_0$ to $U$. It repeats this step until the space contains all optimizers for that cost, then moves to the next cost.

<a name="theorem-learning-guarantee"></a>

> #### Learning guarantee ([Theorem 5 in the paper][paper])
>
> Let $n\ge1$ and draw $c_1,\ldots,c_n$ independently from a distribution $P_c$ supported on $\mathcal C$. Fix $x_0\in\mathcal X^\star(\mathcal C)$ independently of these samples. With deterministic choices, the learner above returns a matrix $U_n$ after the $n$ training costs, satisfying
>
> ```math
> \begin{gathered}
> \mathrm{range}(U_n)\subseteq\mathrm{dir}\bigl(\mathcal X^\star(\mathcal C)\bigr),\\
> \mathrm{rank}(U_n)\le d^\star,
> \end{gathered}
> ```
>
> ```math
> \displaystyle \mathsf{Exact}(U_n,x_0,c_i)\qquad(i=1,\ldots,n).
> ```
>
> For every $\delta\in(0,1)$, with probability at least $1-\delta$ over the training sample,
>
> ```math
> \begin{gathered}
> \mathbb P_{c\sim P_c}\left[\mathsf{Exact}(U_n,x_0,c)\right]\\
> \ge 1-\frac{4}{n}\left(6d^\star+\log\frac{e}{\delta}\right),
> \end{gathered}
> ```
>
> where the test cost $c\sim P_c$ is independent of the training sample.

The failure bound decreases as $1/n$ and depends on the decision-relevant dimension $d^\star$. When optimizers are unique almost surely, the training solves can run in parallel, followed by a basis computation ([Section 3.2 in the paper][paper]).

**Integer programs (IPs).** Let $\mathcal X_I=\mathcal X\cap\mathbb Z^d\ne\varnothing$. Define $\mathcal X_I^\star(c)$ and $\mathcal X_I^\star(\mathcal C)$ as above, optimizing over $\mathcal X_I$. The corresponding IP results are [Proposition 15 (Appendix C.1)][paper] and [Theorem 16 (Appendix C.2)][paper], with these optimizer sets and $d_I^\star:=\dim\mathrm{dir}(\mathcal X_I^\star(\mathcal C))$. [Algorithm 2 (Appendix C.2)][paper] uses integer optimizers in place of LP vertices, and the compressed problem retains $x_0+Uz\in\mathbb Z^d$.

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
python ip/run.py verify
python ip/validate_results.py
```

These commands use the Python standard library. The repository includes source code, benchmark and synthetic inputs, individual test records, and recorded learning times. Historical fitted checkpoints are also retained.

## Reading the results

| Symbol | Meaning |
|---|---|
| $n$ | Number of training examples |
| $d$ | Number of variables in the original LP after preprocessing |
| $K$ | Number of variables in the reduced LP. For IPs, see the explanation before Tables 3–5 |

Here $K$ is the compression size used in an experiment, while $d^\star$ is the theoretical minimum for exact compression over $\mathcal C$.

**Ours** is our affine compression method. **Full** solves the original problem. **OursExact**, used in the supplementary LP experiments, is our method when the set of possible costs is known.

We implement the following baselines based on the cited papers:

| Problem | Method | What it learns or selects | Reference |
|---|---|---|---|
| LP | **DataDrivenProj** | One projection matrix shared across costs | [Sakaue and Oki (2024)](https://proceedings.nips.cc/paper_files/paper/2024/hash/1780db9272f81961459f4906036ebc6f-Abstract-Conference.html) |
| LP | **FCNN** | A neural network that generates a projection for each cost vector | [Iwata and Sakaue (2025)](https://proceedings.mlr.press/v267/iwata25a.html) |
| LP | **Rand** | A random subset of variables | [Akchen and Mišić (2024)](https://doi.org/10.1287/opre.2020.0494) |
| IP | **SVM** | Values for selected integer variables, which are then fixed | [Xavier, Qiu, and Ahmed (2021)](https://doi.org/10.1287/ijoc.2020.0976) |
| IP | **Neural Diving** | Partial integer assignments predicted by a neural network | [Nair et al. (2020)](https://arxiv.org/abs/2012.13349) |
| IP | **MLOPT** | Which reduced formulation from previously solved problems to use | [Bertsimas and Stellato (2022)](https://doi.org/10.1287/ijoc.2022.1181) |
| IP | **Preference Learning** | How to rank reduced formulations by their solution quality and feasibility | [Li et al. (2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34916) |

The [LP timing protocol](results/paper_lp/README.md) and [IP documentation](ip/README.md) explain the implementations, training settings, and changes made to compare methods at the same compression level.

## LP results

The following timings reproduce [Tables 1 and 2 in Section 4 of the paper][paper]. Learning time measures training in seconds. LP solve time measures the time HiGHS spends solving a test LP, in milliseconds. Prediction and preparation times are recorded separately in the [saved results](results/paper_lp/).

### Table 1: learning within a selected cost region

We start with 40 training LPs and use $\rho=0.5$ as the target probability outside the selected cost region ([Section 3.3 of the paper][paper]). All compressed methods use the same $K$. Timings are means ± sample standard deviations over three repeats on fixed data.

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

Each of three independent runs uses 20 training LPs and 20 test LPs, shared by Ours and FCNN. The methods use the same $K$ within each run. The $K$ triples give the three reduced dimensions. Timings are means ± sample standard deviations across runs.

**Learning time (s)**

| Instance | $d$ | $K$ by run | Ours | FCNN |
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

Archived LP plots and raw results are linked below. Use the runner ID with the commands in [Running the experiments](#running-the-experiments).

| Experiment | Plot | Raw results | Runner ID |
|---|---|---|---|
| Objective ratio versus $\rho$ | [Outside-mass sweep](figures/figure3_fresh_legacy_execution.png) | [CSV](results/fresh/figure3_timing_raw.csv) | `figure3` |
| Objective ratio versus $n$, with $K\in\lbrace10,20,30,50\rbrace$ | [Training-sample comparison](figures/figure4_fresh_legacy_execution.png) | [CSV](results/fresh/figure4_timing_raw.csv) | `figure4` |
| Learned dimension versus $n$ | [Dimension growth](figures/figure2_fresh_rank_growth.png) | [CSV](results/fresh/figure2_rank_growth_raw.csv) | `figure2` |

These plots use maximization objectives, with objective ratio equal to the returned value divided by the Full value. The [LP experiment notes](code/experiments/README.md) describe sampling and error bars.

## IP results

The current IP experiments use a public unit-commitment benchmark, grouped knapsack, and correlated knapsack. The tables below reproduce [Tables 3–5 in Appendix C.3 of the paper][paper]. All methods are calibrated to the same reduced dimension $K$ on the same binary variables. $K$ is the dimension of the added affine restriction, before applying the original constraints. A baseline that only fixes coordinates leaves $K$ of those coordinates unfixed.

UC uses 512 training costs, 64 separate validation costs, and 256 test costs. Grouped knapsack uses three instances with 36 training and 120 test costs each. Correlated knapsack uses three instances with nested training sets of 32, 64, 128, and 256 costs and 100 test costs each. Its three $K$ entries correspond to the three instances in seed order.

### Table 3: Unit commitment

| $n$ | Method | $K$ | Recovery (%) | Relative ratio | Learning time (s) |
|---:|---|---:|---:|---:|---:|
| — | Full | 180 | 100.00 | 1.000000 | — |
| 512 | Ours | 19 | 98.44 | 0.999952 | 0.0088 |
| 512 | SVM | 19 | 97.66 | 0.999818 | 4.18 |
| 512 | MLOPT | 19 | 93.75 | 0.997785 | 47.01 |
| 512 | Neural Diving | 19 | 73.05 | 0.923578 | 86.12 |
| 512 | Preference Learning | 19 | 43.36 | 0.763776 | 172.38 |

### Table 4: Grouped knapsack

| $n$ | Method | $K$ | Recovery (%) | Objective ratio | Learning time (s) |
|---:|---|---:|---:|---:|---:|
| — | Full | 96 | 100.00 | 1.000000 | — |
| 36 | Ours | 32 | 100.00 | 1.000000 | 0.0072 |
| 36 | SVM | 32 | 1.39 | 0.898644 | 0.8526 |
| 36 | MLOPT | 32 | 9.72 | 0.953409 | 0.6562 |
| 36 | Neural Diving | 32 | 10.83 | 0.948165 | 13.17 |
| 36 | Preference Learning | 32 | 15.00 | 0.952625 | 122.00 |

### Table 5: Correlated knapsack

| $n$ | Method | $K$ | Recovery (%) | Relative ratio | Learning time (s) |
|---:|---|---:|---:|---:|---:|
| — | Full | 100 | 100.00 | 1.000000 | — |
| 32 | Ours | (29,1,31) | 94.00 | 0.990033 | 0.0032 |
| 32 | SVM | (29,1,31) | 86.33 | 0.971049 | 0.2103 |
| 32 | MLOPT | (29,1,31) | 89.33 | 0.982914 | 1.10 |
| 32 | Neural Diving | (29,1,31) | 91.67 | 0.983863 | 103.81 |
| 32 | Preference Learning | (29,1,31) | 86.00 | 0.972947 | 86.58 |
| 64 | Ours | (33,1,33) | 96.67 | 0.995254 | 0.0072 |
| 64 | SVM | (33,1,33) | 92.33 | 0.983389 | 0.2653 |
| 64 | MLOPT | (33,1,33) | 96.33 | 0.994779 | 3.07 |
| 64 | Neural Diving | (33,1,33) | 91.67 | 0.897010 | 113.66 |
| 64 | Preference Learning | (33,1,33) | 86.33 | 0.976270 | 86.18 |
| 128 | Ours | (35,1,37) | 98.67 | 0.998102 | 0.0129 |
| 128 | SVM | (35,1,37) | 95.00 | 0.989084 | 0.3336 |
| 128 | MLOPT | (35,1,37) | 98.33 | 0.997152 | 3.35 |
| 128 | Neural Diving | (35,1,37) | 93.33 | 0.964404 | 124.59 |
| 128 | Preference Learning | (35,1,37) | 89.33 | 0.973422 | 132.67 |
| 256 | Ours | (40,1,41) | 99.67 | 0.999525 | 0.0244 |
| 256 | SVM | (40,1,41) | 97.00 | 0.992406 | 0.5422 |
| 256 | MLOPT | (40,1,41) | 99.33 | 0.999051 | 6.58 |
| 256 | Neural Diving | (40,1,41) | 94.67 | 0.966303 | 170.96 |
| 256 | Preference Learning | (40,1,41) | 93.00 | 0.984813 | 240.31 |

**Recovery (%)** is the percentage of all tests that return a feasible solution with the original optimal value, up to numerical tolerance.

For UC and correlated knapsack, **Relative ratio** measures the improvement over a common anchor as `sum(returned value - anchor value) / sum(optimal value - anchor value)`. The anchor is the optimal solution at the first training cost, evaluated at each test cost. For grouped knapsack, **Objective ratio** is the mean returned profit / optimal profit over feasible returns. Higher is better for both metrics. Exact definitions are in the [IP experiment details](ip/EXPERIMENTS.md#metrics-and-learning-time).

Learning time covers the computation used to learn the compression after the training costs and corresponding optimal solutions are provided. For neural methods, this includes choosing model settings and training the network. For our method, it covers [Algorithm 2][paper] excluding the training IP solves. We exclude these solves because all methods use their solutions. Ours, SVM, and MLOPT use one CPU thread. Neural Diving and Preference Learning use an RTX 4090 for UC and an RTX PRO 4500 for both knapsack settings. Timing details are in the [IP experiment notes](ip/EXPERIMENTS.md#metrics-and-learning-time).

Ours has the highest recovery and objective ratios among the compression methods in these experiments. Notably, its learning time is much shorter than that of the neural baselines.

Data, source code, and full-precision results are in [ip/current/](ip/current/). The [IP guide](ip/README.md) explains how to run new experiments.

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

See the [LP experiment notes](code/experiments/README.md) for sampling settings and notes on archived runs.

### IP code

Recompute and check current tables without numerical libraries:

```bash
python ip/run.py verify
```

Run small solver checks using current data:

```bash
python -m pip install -r ip/requirements.txt
python ip/run.py smoke --problem uc --method ours --output ip/runs/uc-smoke
python ip/run.py smoke --problem grouped --method svm --output ip/runs/grouped-smoke
python ip/run.py smoke --problem correlated --seed 41001 --n 32 --method ours --output ip/runs/correlated-smoke
```

Every output directory must be new. Fitting/evaluation commands and baseline adaptations are described in [ip/README.md](ip/README.md).

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

[paper]: https://arxiv.org/abs/2605.25635
