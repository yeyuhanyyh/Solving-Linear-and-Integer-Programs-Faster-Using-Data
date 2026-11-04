# IP experiment details

This document records the instance generators, baseline adaptations, and measurement rules for Tables 3–5. Start with the [IP experiment guide](README.md) for the overview, results, and basic commands.

The tables contain 30 learned-method rows, reconstructed from 9,080 test observations and 80 per-instance fitting summaries. The data, fitted dimensions, and full-precision results are in [current/](current/). References follow the current manuscript (November 3, 2026). The linked arXiv preprint is an earlier version.

## Problem instances

| Setting | Unit commitment | Grouped knapsack | Correlated knapsack |
|---|---|---|---|
| Physical instances | 1 public benchmark | 3 | 3 |
| Original variables | 2,844, including 540 binary | 96 binary | 100 binary |
| Binary variables compressed, $D$ | 180 on/off variables | 96 item decisions | 100 item decisions |
| Training costs, $n$ | 512 | 36 per instance | 32, 64, 128, 256 per instance |
| Validation | 64 separate costs | About 20% held out during selection | About 20% held out during selection |
| Test costs | 256 | 120 per instance | 100 per instance, shared across sample sizes |
| Learned $K$ | 19 | 32 | See Table 5 |
| Instance seeds | 86001 | 1009, 1013, 1019 | 41001, 41002, 41003 |

**Unit commitment.** We use `matpower/case14/2017-02-01` from [UnitCommitment.jl v0.4.3](https://github.com/ANL-CEEESA/UnitCommitment.jl/tree/v0.4.3). It has five generators, 14 buses, 20 lines, and 36 hourly periods. The feasible set retains power balance, output and ramp limits, minimum on/off durations, reserves, network constraints, and benchmark penalty variables. Exact elimination of redundant auxiliary variables gives 2,844 variables and 2,544 constraints. Only the 180 on/off variables are compressed. For each generator and cost sample, the on, fuel, and startup cost coefficients are drawn independently from Uniform[0,1000], Uniform[0,50], and Uniform[0,10000], respectively, then rounded to two decimals. They stay constant over the horizon. These ranges follow the [MIPLearn v0.4 UC generator](https://anl-ceeesa.github.io/MIPLearn/0.4/guide/problems/#unit-commitment). Benchmark penalty costs stay fixed.

**Grouped knapsack.** There are 32 groups with two basic items and one upgrade each. A group's capacity permits the two basic items together or the upgrade alone. Basic weights are uniform integers from 10 through 20. The capacity is their sum, and the upgrade weight is an integer strictly between the larger basic weight and that capacity. A fixed graph excludes selected pairs of upgrades. It contains a cycle and the edge between groups 1 and 3, and each remaining edge is included with probability 0.15. This uses the conflict-constraint idea of [Pferschy and Schauer (2009)](https://doi.org/10.7155/jgaa.00186). Each basic item has value 0.25, and upgrade values are independently Uniform[0.1,2.5]. This is a separate structured example, not a claim that all knapsack instances have these relations.

**Correlated knapsack.** Following the coefficient construction studied by [Pisinger (2005)](https://doi.org/10.1016/j.cor.2004.03.002), each instance has 100 weights drawn independently from the integers 1 through 1000 and capacity `floor(sum(weights)/2)`. We keep these weights and capacity fixed. Each sampled profit is `weight + 100 + epsilon`, where `epsilon` is independently uniform on {-2,-1,0,1,2}. This adapts the almost-strongly-correlated benchmark distribution to repeated optimization with varying profits. Training sets are nested. The same 100 fresh test costs per instance are used at all four sample sizes.

## Baselines and matching the dimension

Ours uses [Algorithm 2 (Appendix C.2) in the paper][paper]. All methods use the same training costs and supplied optimal solutions. For each instance, all compression methods are calibrated to the same reduced dimension $K$ for the same binary variables. $K$ is the dimension of the added affine restriction. The original constraints and integrality requirements remain in the reduced problem.

* **SVM** follows the prediction approach of [Xavier, Qiu and Ahmed (2021)](https://doi.org/10.1287/ijoc.2020.0976). Linear classifiers predict binary values from cost features. For UC, they also predict whether consecutive on/off variables are equal. Five-fold predictions estimate the reliability of each proposed fixing or equality. Features are standardized within each fold, and the linear SVC uses `C=1`. Unanimous relations and proposals meeting the paper's acceptance thresholds rank first. We add consistent, independent equations until their rank is $D-K$. Lower-ranked proposals are used if needed to reach that dimension. This last step is our dimension-matching adaptation of the paper's acceptance policy.
* **MLOPT** follows [Bertsimas and Stellato](https://arxiv.org/abs/1907.02206). It trains a classifier to predict strategies extracted from training optima. A strategy records integer values and active constraints. Its MLP has three ReLU layers of width 32 or 64, dropout 0.1, batch size 32, and Adam with learning rate 0.001 or 0.01 for 50 or 150 epochs. It considers up to ten predicted strategies. A one-strategy bank uses a constant classifier.
* **Neural Diving** follows [Nair et al.](https://arxiv.org/abs/2012.13349). A six-layer bipartite graph network predicts binary values using formulation and root-LP features. Width is 32 or 64. Training uses binary cross-entropy, Adam at 0.001, batch size 16, and early stopping within 150 epochs. For correlated knapsack, training-only validation can select up to 2,000 additional updates. The main comparison uses **one** most-likely assignment and fixes its most confident coordinates. The earlier ten-candidate control is not included in these tables.
* **Preference Learning** follows the ranking objective of [Li et al. (2025)](https://arxiv.org/abs/2501.00307). It learns scores for training-derived strategies using objective values and constraint violations. The network has two attention blocks of width 128, uses AdamW with learning rate 0.0001 or 0.001, and runs for the recorded local budget of 1,200 epochs. This is a local training schedule, not a claim that the original paper prescribes or requires that many epochs. Validation chooses the number of candidates, and Eq. (21) selects using normalized infeasibility.

MLOPT and Preference Learning are adapted to the same $K$. For UC, each candidate receives its own dimension-matched restriction before solving. MLOPT chooses the best feasible objective, and Preference Learning retains its Eq. (21) selection rule. For knapsack, the chosen strategy supplies an assignment, and softmax-weighted agreement ranks the coordinates to fix. These are paper-based implementations with explicit adaptations, not official author code or exact reproductions of every original-paper setting.

For UC, Preference Learning uses the saved decoder correction in `current/source/uc/baselines/neural/tie_api.py`. If a certified optimal reduced-LP return violates the original model, it may use the fixed training representative of that strategy only when the representative satisfies both formulations and has the same objective within tolerance. This changes only a tie choice and adds no optimization solve. The rule was fixed using training diagnostics.

## Metrics and learning time

Recovery counts every planned test, including failures. A recovered solution must satisfy the original constraints and match the original optimum within `1e-6 + 1e-9*abs(optimum)`. The complete-return budget is 30 seconds for UC and grouped knapsack and 20 seconds for correlated knapsack.

For UC and correlated knapsack, the common anchor is the complete optimal solution at the first training cost. Let its value on test $i$ be $v_i^a$, the method's returned value be $v_i^m$, and the original optimum be $v_i^\star$. The relative objective ratio is

$$\frac{\sum_i(v_i^m-v_i^a)}{\sum_i(v_i^\star-v_i^a)}.$$

Values are total costs for UC and profits for knapsack. For UC, numerator and denominator both reverse sign relative to the corresponding cost reductions. We retain negative method improvements and zero-headroom tests. We do not clip ratios or average per-test relative ratios. All current UC and correlated-knapsack rows have valid returns on the entire test cohort.

For grouped knapsack, we report the mean `returned profit / optimal profit` over valid returns. Recovery still uses all 360 tests. The current valid-return counts are 360 for Ours, MLOPT, and Preference Learning, 355 for SVM, and 282 for Neural Diving. The saved observations include every failure.

Learning time covers the computation used to learn the compression after the training costs and corresponding optimal solutions are provided. For neural methods, this includes choosing model settings and training the network. For our method, it covers [Algorithm 2][paper] excluding the training IP solves. We exclude these solves because all methods use their solutions.

Ours, SVM, and MLOPT use one CPU thread. Neural Diving and Preference Learning use an RTX 4090 for UC and an RTX PRO 4500 for both knapsack settings. CPU timing uses three fresh fits per instance. Neural fitting uses the saved GPU fit per instance. Tables report means across instances after averaging CPU repeats. These are recorded fitting times on those devices, not estimates of the minimum time each algorithm must require. The exact component arithmetic is retained in `current/timing/` and `current/learning_records.json`.

Data loading, shared training-optimum generation, final export, and test prediction/solving are outside the reported fitting clocks. Neural Diving's reported clock covers architecture/epoch selection and refitting after graph/root-LP preparation. Separately timed UC validation solves/decodes are excluded. Method-specific strategy and reward preparation within the fitting routine remains included. No IP solve-time or speedup column is reported.

## Run new fits and solves

```bash
python -m venv .venv-ip
# Activate .venv-ip, then install:
python -m pip install -r ip/requirements.txt
python ip/current/check_inputs.py
python ip/run.py check-data --problem uc
python ip/run.py check-data --problem grouped --seed 1009
python ip/run.py check-data --problem correlated --seed 41001 --n 32

# A smoke run fits the selected method and evaluates one held-out cost.
python ip/run.py smoke --problem correlated --n 32 --method ours --output ip/runs/ours-smoke

# Fit from the supplied optimal training solutions, then evaluate independently.
python ip/run.py train --problem grouped --seed 1009 --method svm --output ip/runs/grouped-svm-fit
python ip/run.py evaluate --problem grouped --seed 1009 --method svm --checkpoint ip/runs/grouped-svm-fit --limit 120 --output ip/runs/grouped-svm-test

# Explicit GPU training for either neural method:
python ip/run.py train --problem correlated --seed 41001 --n 32 --method neural-diving --device cuda --output ip/runs/nd-fit
python ip/run.py train --problem uc --method preference-learning --device cuda --output ip/runs/uc-preference-fit
```

Methods are `full`, `ours`, `svm`, `mlopt`, `neural-diving`, and `preference-learning`. Full needs no fit or checkpoint. Use a separate command/process for every setting and method. Every output directory must be new. Evaluate all seeds to obtain the displayed pooled results. Fresh timing depends on the machine and dependency versions. Fresh fits and solver returns are saved separately and never replace the paper records. A CPU neural rerun must explicitly use `--device cpu`; it is not a reproduction of the paper's GPU timing.

The numerical banks avoid requiring Julia or regenerating training labels. They retain the original model, costs, optimal solutions, and test references. The portable runner loads these banks and calls the frozen scientific modules under `current/source/`. The original standalone grouped runner is also retained in `current/grouped/`. Path relocation and command dispatch are separate from the algorithms. [PORTABILITY.md](PORTABILITY.md) records the release boundary.

[paper]: https://arxiv.org/abs/2605.25635
