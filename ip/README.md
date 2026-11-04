# Integer-programming experiments

Code and results for unit commitment, grouped knapsack, and correlated knapsack in [Appendix C.3 of our paper][paper]. Numbering follows the current manuscript (November 3, 2026). The linked arXiv preprint is an earlier version.

## Reproduce the tables

Run these commands from the repository root with Python 3.11 or later:

```bash
python ip/run.py verify
python ip/validate_results.py --output ip/runs/recomputed-tables
```

These commands check the saved records and rebuild Tables 3–5 using only the Python standard library. No solver or retraining is required.

## Experiment overview

| Setting | Unit commitment | Grouped knapsack | Correlated knapsack |
|---|---|---|---|
| Physical instances | 1 public benchmark | 3 | 3 |
| Original variables | 2,844, including 540 binary | 96 binary | 100 binary |
| Binary variables compressed, $D$ | 180 on/off variables | 96 item decisions | 100 item decisions |
| Training costs, $n$ | 512 | 36 per instance | 32, 64, 128, 256 per instance |
| Validation | 64 separate costs | About 20% held out during selection | About 20% held out during selection |
| Test costs | 256 | 120 per instance | 100 per instance, shared across sample sizes |
| Learned $K$ | 19 | 32 | See Table 5 |

**Unit commitment.** We use the public `matpower/case14/2017-02-01` benchmark from [UnitCommitment.jl](https://github.com/ANL-CEEESA/UnitCommitment.jl/tree/v0.4.3). The physical system stays fixed, and generator costs vary using the ranges in [MIPLearn](https://anl-ceeesa.github.io/MIPLearn/0.4/guide/problems/#unit-commitment).

**Grouped knapsack.** Each group allows two basic items or one upgrade. We add conflicts between some upgrades ([Pferschy and Schauer, 2009](https://doi.org/10.7155/jgaa.00186)). Upgrade values vary between samples.

**Correlated knapsack.** We adapt the almost-strongly-correlated construction of [Pisinger (2005)](https://doi.org/10.1016/j.cor.2004.03.002) to repeated optimization. Weights and capacity stay fixed while item values vary.

All methods share the training costs and optimal training solutions. They are calibrated to the same reduced dimension $K$ on the same binary variables. Here $K$ is the dimension of the added affine restriction. The original constraints and integrality requirements remain in place.

The [experiment details](EXPERIMENTS.md) give the distributions, seeds, training parameters, and measurement rules.

## Methods

- **Ours** uses [Algorithm 2 (Appendix C.2) in the paper][paper].
- **SVM** predicts binary values from cost coefficients. For UC, it also predicts whether a generator's on/off status stays unchanged between periods ([Xavier et al.](https://doi.org/10.1287/ijoc.2020.0976)).
- **MLOPT** learns to choose a reduced formulation from previously solved problems ([Bertsimas and Stellato](https://arxiv.org/abs/1907.02206)).
- **Neural Diving** predicts a binary assignment and fixes its most confident coordinates ([Nair et al.](https://arxiv.org/abs/2012.13349)). We use one candidate per test.
- **Preference Learning** learns to rank these formulations by solution quality and feasibility ([Li et al.](https://arxiv.org/abs/2501.00307)).

These implementations follow the cited papers, with adaptations to match $K$. The exact rules are described in [Baseline adaptations](EXPERIMENTS.md#baselines-and-matching-the-dimension).

## Metrics and learning time

**Recovery (%)** is the percentage of all tests that return a feasible solution with the original optimal value, up to numerical tolerance. Failed returns count as unsuccessful tests.

For UC and correlated knapsack, **Relative ratio** measures the improvement over a common anchor:

```text
sum(returned value - anchor value) / sum(optimal value - anchor value)
```

The anchor is the optimal solution at the first training cost, evaluated at each test cost. For grouped knapsack, **Objective ratio** is the mean returned profit / optimal profit over feasible returns. Higher is better for both ratios. Exact tolerances and averaging rules are in the [measurement details](EXPERIMENTS.md#metrics-and-learning-time).

Learning time covers the computation used to learn the compression after the training costs and corresponding optimal solutions are provided. For neural methods, this includes choosing model settings and training the network. For our method, it covers [Algorithm 2][paper] excluding the training IP solves. We exclude these solves because all methods use their solutions.

Ours, SVM, and MLOPT use one CPU thread. Neural Diving and Preference Learning use an RTX 4090 for UC and an RTX PRO 4500 for both knapsack settings.

## Run new experiments

Install the dependencies in a separate environment, then try a small run:

```bash
python -m pip install -r ip/requirements.txt
python ip/run.py smoke --problem correlated --n 32 --method ours --output ip/runs/ours-smoke
```

A smoke run fits the method and solves one test problem. To train and evaluate separately:

```bash
python ip/run.py train --problem grouped --seed 1009 --method svm --output ip/runs/grouped-svm-fit
python ip/run.py evaluate --problem grouped --seed 1009 --method svm --checkpoint ip/runs/grouped-svm-fit --limit 120 --output ip/runs/grouped-svm-test
```

To train a neural baseline on a GPU:

```bash
python ip/run.py train --problem correlated --seed 41001 --n 32 --method neural-diving --device cuda --output ip/runs/nd-fit
```

Use a new output directory for each run. Run every instance seed to reproduce a pooled table entry. The [full run instructions](EXPERIMENTS.md#run-new-fits-and-solves) list the methods, seeds, checks, and CPU/GPU options.

The supplied data include training costs, optimal solutions, and test references. Code and records are in [current/](current/). Older experiments remain available through [LEGACY.md](LEGACY.md).

## Results

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

[paper]: https://arxiv.org/abs/2605.25635
