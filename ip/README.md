# IP experiments

This directory contains the source, synthetic training/test data, frozen fitted models, raw returns and learning-time records for Tables 3 and 4 of **Solving Linear and Integer Programs Faster Using Data**. It also includes the fixed additional 2,000-update Neural Diving control. The LP release is in the repository root.

## Reproduce the archived tables without training

From the repository root:

```bash
python ip/run.py verify-artifacts
python ip/validate_results.py
```

These commands use Python's standard library. The first checks every archived file against `artifact-manifest.json`. The second recomputes the reported metrics from the individual test records, verifies the returned objectives and original constraints, and writes `ip/results/table3.csv`, `table4.csv` and `summary.json`. No solver, model fitting or cloud account is needed for this arithmetic reproduction.

## Run the solvers and fitted predictors

Use a separate virtual environment for the IP code:

```bash
python -m venv .venv-ip
# Linux/macOS:
source .venv-ip/bin/activate
# Windows PowerShell instead: .venv-ip/Scripts/Activate.ps1
python -m pip install -r ip/requirements.txt
python ip/run.py smoke --problem uc --seed 101 --method ours
python ip/run.py smoke --problem knapsack --seed 1009 --method neural-diving
python ip/run.py evaluate --problem uc --seed 101 --method preference-learning --limit 1
python ip/run.py evaluate --problem knapsack --seed 1009 --method svm --limit 120
```

Methods are `full`, `ours`, `svm`, `mlopt`, `neural-diving`, `preference-learning` and `nd-continued`. Run a new process for each problem/seed/method, because the archived runtimes use stable Python module names. UC seeds are 101, 211 and 307. Knapsack seeds are 1009, 1013 and 1019. `smoke` solves one training example. `evaluate` uses the first `--limit` entries of the frozen test bank, with 120 giving the complete bank for that seed. Run all three seeds to obtain 360 tests per method/problem. The default outer return budget is 30 seconds per example, including prediction, decoding, restricted solving and final feasibility checks.

New outputs go into a new timestamped directory under `ip/runs/`. An explicit `--output` must also name a new directory. Archived results are never overwritten. Re-solving may differ near numerical or time thresholds across machines and dependency versions. Recomputing the archived tables is deterministic and does not replace the original records with reruns.

The original recorded CPU environment includes Python 3.14.4, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.11.0+cpu, HiGHS 1.15.1 and scikit-learn 1.9.1. Detailed per-fit runtime metadata is retained with the checkpoints. For strict checkpoint compatibility, use those versions when available. Requirements give minimum versions for a fresh environment. The trusted included pickle and PyTorch files must only be loaded after archive verification.

## Fit new models from the supplied training optimizers

```bash
python ip/run.py train --problem uc --seed 101 --method ours
python ip/run.py train --problem knapsack --seed 1009 --method svm
python ip/run.py train --problem uc --seed 101 --method neural-diving --device cpu
python ip/run.py train --problem knapsack --seed 1009 --method preference-learning --device cpu
```

`mlopt` is also supported. Neural fitting executes the original complete selection/refitting schedules and can take several minutes per seed. No training starts during import, verification, smoke or evaluation. The shared certified training optimizers are supplied, so these fitting commands do not regenerate their labels. Fresh fitting uses new completion markers and source/data guards. A rerun can load a fresh artifact using `evaluate --checkpoint PATH`: pass the `.pkl` or `.pt` file, or the fitted `li` directory for Preference Learning. `nd-continued` evaluates the archived fixed-control checkpoint. The original continuation algorithms and their complete training trajectories are retained under `snapshot/`; the portable fitting wrapper does not launch a new continuation control.

## Problems and data

| Current experiment | UC (Table 3) | Knapsack (Table 4) |
|---|---|---|
| Physical instances | 3 | 3 |
| Variables | 192: 144 binary and 48 continuous | 96 binary |
| Shared compression scope | 48 commitment variables; startup/shutdown remain binary and dispatch remains continuous | All 96 item decisions |
| Fitted training examples per instance | 32 | 36 |
| Fresh test examples per instance | 120 | 120 |
| Learned affine dimension K | 14, 14, 17 | 32 on each instance |
| Cost features | 12 generator/time-block multipliers | 32 upgrade values |
| Cost distribution | Independent LogUniform[1/3,3] multipliers | Independent Uniform[0.1,2.5] upgrade values; basic-item values 0.25 |

UC has four generators and twelve periods. Each generator has one multiplier for each four-period block, shared across its four cost components. The demand range is 0.30–0.50 of total capacity. Startup cost is capacity times a uniform rate in [22,60]. All original UC constraints and variable domains remain in the reduced problem. The source generator and the exact frozen matrices, physical parameters and costs are included.

Knapsack has 32 groups, each with two basic items and one upgrade. Basic weights are independent uniform integer draws from 10 through 20, and the local capacity is their sum. The upgrade weight is a uniform integer strictly between the larger basic weight and the capacity. An upgrade cannot be combined with a basic item in its group. A graph excludes selected pairs of upgrades: a cycle and an edge between groups 1 and 3 are fixed, and each remaining edge is added independently with probability 0.15. Internally the maximization is solved by minimizing negative values. Reported ratios use the original positive gross values, with no objective shift.

These settings were developed through exploratory experiments before the retained fresh test banks. The training prefixes came from exploratory pools of 64 UC labels and 69 knapsack labels. The knapsack prefix was selected to reach rank 32 on the three physical instances. The retained records distinguish the original fits, independent test streams and later supplemental baselines. Preference Learning was added after those shared test banks had already been generated. Its models were frozen before their own evaluation and its tuning uses only the training prefix.

## Baselines and matching K

These are this project's paper-based implementations and dimension-matched adaptations, not the original authors' official software. The displayed names refer to the underlying method families. Adaptations are explicit in the source and metadata.

* **SVM** follows the variable-prediction approach of [Xavier, Qiu and Ahmed (2021)](https://doi.org/10.1287/ijoc.2020.0976). Nonconstant coordinates use StandardScaler followed by a linear SVC with C=1 and training-only probability calibration. Constant labels are handled exactly. To match K, the D−K common coordinates with the highest MAP confidence are fixed. The original paper's acceptance gates are replaced by this fixed coordinate budget.
* **MLOPT** follows [Bertsimas and Stellato](https://arxiv.org/abs/1907.02206). It predicts training-extracted strategies using a three-hidden-layer MLP. The grid contains widths 32 or 64, learning rates 0.001 or 0.01, and 50 or 150 epochs, followed by an all-training-data refit. Dropout is 0.1 and batch size is 32. The native top-ten feasible decoded candidate with the best objective supplies the anchor. Coordinates are ranked by agreement with that anchor across retained strategies, weighted by predicted scores, and D−K are fixed. This is a coordinate-budget adaptation of strategy learning.
* **Neural Diving** follows [Nair et al. (2020)](https://arxiv.org/abs/2012.13349). The complete binary predictor is a bipartite graph network with six message-passing layers and selected width 32 or 64. It uses formulation and root-LP features, singleton certified-optimum labels, NLL training, Adam, batch size 16, learning rate 0.001 and gradient clipping at 5. The training-only tiny-fit gate and width/epoch selection are retained. To match K, complete-predictor MAP confidence fixes D−K common coordinates instead of learning separate coverage-mask models.
* **Preference Learning** follows [Li et al. (2025)](https://arxiv.org/abs/2501.00307), using a width-128 network with two attention layers. The original grid uses learning rates 0.0001/0.001, preference parameters 0.8/0.9 and 0.1/0.2, 1,200 epochs per nondegenerate fit, and batch size 128. The requested top-k grid is 1/3/5/10/30, clipped to the available development strategy count. Its native top-k/Eq.21 strategy selection supplies the integer-pattern anchor. Weighted agreement then selects D−K fixed coordinates. The exact final attention block, optimizer/grid and strategy adaptations are documented in the source.

For UC, D in the matching rule is the 48 commitment variables, rather than the full 192-variable formulation. For knapsack it is all 96 binary variables. K always comes from the affine span of the same supplied training optimizers. Ours imposes its learned affine equations. The baseline adaptations leave exactly K common coordinates unfixed, which matches the coordinate budget rather than asserting that different restrictions define identical feasible sets.

The extra Neural Diving control starts from each original selected checkpoint, resets Adam and performs exactly 2,000 additional updates on all training labels. Batches are 16/16 for UC and 16/16/4 for knapsack. The learning rate is 0.001 for updates 1–1,000 and 0.0001 for updates 1,001–2,000. Only the final update is evaluated. Original and continued checkpoints and results are both retained.

## Metrics and archive layout

**Recovery (%)** uses all 360 tests as its denominator. A recovered result must be feasible in the original formulation, returned within the 30-second budget, and match the certified original optimum within `1e-6 + 1e-9*abs(original optimum)`. **Objective ratio** is original/returned objective for UC minimization, and returned/original gross value for knapsack maximization. It is at most one, up to the stated numerical tolerance. Means use timely original-feasible returned solutions; the CSV/JSON audit reports their denominators and retains failed cases in the raw records. Recovery and ratio therefore measure different things.

**Learning time (s)** is the recorded fitting interval after the shared training optima were available. It includes each method's model selection and refitting. The archive keeps mean and sample standard deviation across the three physical instances. Additional 2,000-update control time is separate from the original fit. These are concurrent single-thread CPU fitting records, not isolated hardware benchmarks. Knapsack's original training-label generation time was not recorded and is not replaced with a later certification time. The manuscript does not currently report IP solve-time/speedup columns.

* `artifact-manifest.json`: every archived file's byte count, SHA-256 and role.
* `snapshot/`: original relative experiment/source layout, with original bytes preserved. Historical directory names remain so local scientific imports and provenance can be traced.
* `provenance/`: the current table-cell and learning-time audits.
* `results/`: portable CSV/JSON tables and independent verification report.
* `verification.json`: release smoke checks on one frozen cost for each of seven method/checkpoint variants on both problems. Independent comparisons match the archived objectives, common dimensions and fixed-coordinate patterns. Fresh affine/SVM/MLOPT fitting and checkpoint reload were also checked. Full neural retraining was not repeated for this release.
* `run.py`: path-relocation and execution layer. It never rewrites the original snapshots. UC absolute path keys are canonicalized only for guard identity, after real source bytes are checked. Knapsack uses a path-relocated selection manifest saved with the new run. All original file digests and schema checks remain active. Both constructors validate the supplied matrices, domains, training data, affine equations and model bindings. Checkpoint loaders also verify the training/matrix fingerprints and tensor/strategy identity.

Cloud credentials, API/SSH access, billing controls, manuscript sources and dependency caches are not part of this release. All supplied optimization data are synthetic.
