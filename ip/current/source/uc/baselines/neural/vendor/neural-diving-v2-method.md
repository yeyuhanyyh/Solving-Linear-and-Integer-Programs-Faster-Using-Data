# Neural Diving v2 method and experiment protocol

Status following the 2026-10-10 UTC training-only repairs: Attempt1 stopped before confirmation generation. Attempt2 completed the first UC case but stopped at both knapsack tiny-fit gates. Their source, checkpoints and diagnostics are preserved under audit/remote-attempt1 and audit/remote-attempt2. Training-only paired probes motivated certified root-LP features, best-state restoration, the fixed1200-step schedule and uniform graph row-max scaling described below. No complete v2 cohort or confirmation performance is claimed. V1 source, checkpoints and raw results remain untouched.

## Source and implementation identity

The reference is Nair et al., *Solving Mixed Integer Programs Using Neural Networks*, immutable [arXiv v3, 29 July 2021](https://arxiv.org/html/2012.13349v3), Sections 3.1–3.3 and 6.1–6.2, Eqs.11–17 and Algorithm1. The relevant components are a bipartite graph network, energy-weighted feasible labels, independent binary prediction, an acceptance head with coverage loss, sampled partial assignments and classical subproblem completion.

The [official neural_mip_solving directory](https://github.com/google-deepmind/deepmind-research/tree/master/neural_mip_solving) contains a dataset README, not a released complete implementation of this algorithm. This code is our source-disclosed implementation and HiGHS port. It is not author code, an exact reproduction of the authors' experimental pipeline, or evidence that performance matches their tuned solver.

## Two separately labelled methods

**ND-native** predicts every original integer coordinate: 432 UC variables in the u/v/w blocks, or 200 knapsack x variables. Each coverage model independently samples its binary values and acceptance mask. One partial assignment per coverage is solved, in seeded random order, under one shared 30-second budget. Continuous UC dispatch p is represented in the graph, but is never predicted or fixed.

**ND-matched-K MAP-confidence variant** uses the separate complete-assignment predictor. It fixes exactly D-K highest-confidence common coordinates, using confidence max(p,1-p), MAP value p>=1/2, and original-column order for confidence ties. D is 144 UC commitment u coordinates or 200 knapsack x coordinates. UC v/w/p remain for the solver. This deterministic cardinality policy is an explicit variant of native Algorithm1, not the original acceptance sampling rule. It uses the existing training-derived Ours K without choosing K from test outcomes.

The matched count is the dimension of the affine space imposed by these explicit coordinate fixings. Original constraints, integrality and presolve can restrict it further. Neither 432-minus-native-fixings nor a solver's presolved variable count is equated with Ours' affine rank on u.

## Graph and features

Six coefficient-weighted graph layers replace v1's three. Relevant UC cross-generator cost information follows four-to-six-edge paths through output limits and demand balance: other-generator fuel prices can require four edges, and other-generator commitment costs can require six. Six layers make these paths available; this does not imply that every cost feature needs six layers or that the trained network will use the information effectively.

For every case, graph construction divides row r and both its bounds by s_r=max(1,max_j|A_rj|). The symmetric variable/constraint adjacency uses these signed coefficients together with unscaled self loops, an MLP before aggregation, LayerNorm and concatenated layer outputs. This positive row scaling preserves the underlying interval constraints. It is fixed uniformly for UC and knapsack, without per-case selection.

Nair Section3.3 specifies original coefficient-weighted adjacency. Our row-max scaling is a disclosed equivalent-instance preprocessing adaptation motivated by training conditioning; it is not identified as an error in the paper or as author code. It changes graph coefficients and row bounds, including structural bound-feature scaling. The shared signed-log bound scale is recomputed from those graph bounds, so finite variable-bound channels can also change. The original matrix and bounds passed to root-LP extraction, subproblem solves and feasibility checks remain unchanged. Raw graph inputs remain available explicitly for source checks and archived ablations.

There are 13 formulation features: node type; cost coefficient; integer indicator; finite variable-bound values/flags; finite row-bound values/flags; and degree. Cost uses a single RMS scale computed on the selection-training costs, then recomputed on all training costs for final refit. No test costs, variable index, known solution, or global cost vector are node inputs. Bounds use signed log scaling.

Nair Section3.1 explicitly uses the solver features from [Gasse's official extraction code](https://raw.githubusercontent.com/ds4dm/learn2branch/master/utilities.py). The repaired implementation adds15 root-LP channels: scaled primal values, integer fractionality, normalized reduced costs, bound activity and five basis indicators at variables; normalized duals, scaled slack, lower/upper tightness and objective cosine at rows. The original formulation features remain, for28 channels in total.

The LP is the unrestricted continuous relaxation of the original matrix, bounds and current cost. HiGHS simplex is fixed to one thread and a recorded seed. Optimal status, valid primal/dual/basis, original continuous feasibility, dual stationarity/sign support and primal-dual objective agreement are required. Fractional values at originally integer columns are legitimate root-LP features. A failed certificate stops feature extraction; an integer optimizer or training label is never substituted.

This is a documented HiGHS subset port of the solver feature stack. It omits incumbent values, historical solver ages and SCIP's presolve/cut state. Interval rows stay unsplit; primal values use fixed original bound scales; row dual/slack use original model units and row norms. Graph row-max scaling does not change that root-feature convention. No variable-ID or time-position feature was added.

## Training labels and exact energy contract

All original feasible pool assignments are retained after deduplication by the full integer assignment. Duplicate integer assignments use their best continuous completion. The archived certified optimal training solution must be present in every pool.

For UC, the portable data adapter must solve the conditional LP for every retained integer assignment, or use the already certified original optimizer's conditional optimality. It records original incumbent energy, completed energy, LP status, extra time and a pool-level conditional_lp_completion_certified flag. The learner rejects uncertified UC pools. Knapsack has no continuous coordinates.

Weights are softmax(-original minimization objective), using a stable common shift and no temperature or objective-unit rescaling. A large pool can consequently have effective size close to one; pool sizes and effective sizes are reported separately.

Warmup minimizes the weighted full-assignment binary negative log likelihood, divided by the fixed integer-coordinate count. This is a constant scaling of Eq.11, not a selective loss. Coverage models clone the warmup weights and then minimize accepted-coordinate weighted BCE divided by the sum of acceptance probabilities, plus 32*max(C-mean(q),0)^2. Differentiable acceptance probabilities q are used during optimization; actual masks are Bernoulli draws during native inference.

The staged complete-predictor warmup is a disclosed optimization adaptation. It ensures the matched policy does not take values from coordinates that a selective model rarely trained. The complete predictor and the selective heads are stored independently.

## Frozen candidates and validation

Training uses the same 120 archived cost instances. The formal runner's recorded case-resolved seed reserves24 for selection and uses96 for fitting. The standalone module/developer-probe default is1909; it is not substituted for the formal seed. No old test instance participates in selection. Width candidates are32 and64, graph depth is six, and coverage targets are0.1,0.25,0.5,0.75 and0.9. Adam uses learning rate0.001, batch size16 and gradient clipping5.

The complete predictor has at most150 warmup epochs. Validation full NLL chooses its epoch, with patience20. Each selective model has at most200 additional epochs. Epoch0 (the warmup-only weights) is an admissible coverage policy. Every25 epochs, each model is evaluated by solving subproblems on the same seeded eight of the 24 validation costs, with a three-second subproblem limit.

Coverage checkpoints are selected lexicographically by recovered original optimum count, feasible count, shorter elapsed time and lower selective loss. Recovery uses the same strict tolerance as the final benchmark: 1e-6+1e-9*abs(reference objective). These are archived training reference optima, never confirmation labels.

A width's selected native ensemble is evaluated on those same eight validation costs under the complete shared30-second budget. Width selection prioritizes recovery count, feasible count, shorter elapsed time, lower full NLL and smaller width. At least half those validation costs must have a feasible native result. All validation indices, quality records, fixed maps and selected epoch counts are saved. This finite search is not a claim of exhaustive hyperparameter optimization.

After selection, the complete predictor is reinitialized and trained from scratch on all120 costs for the selected number of warmup epochs. Every coverage network clones that all120 predictor and receives its selected additional epoch count on all120 costs. No selection continues during this refit. The final complete-predictor full NLL, entropy excess and expected MAP bit error are recorded. Final-model validation reported in metadata refers explicitly to the pre-refit selection model; it is not presented as an independent assessment of the all120 fit.

## Sanity gate before confirmation testing

Each width first fits a separate network to eight of the96 fitting costs for1200 gradient steps. Steps1–400 use the configured base learning rate0.001; steps401–1200 use0.1 times that rate. The best diagnosed model and matching Adam state are retained and restored before checking the gate and small subproblems. Per-u/v/w losses and errors, pre-clip gradient norms, learning rates, terminal diagnostics and restored diagnostics are recorded.

The entropy floor is computed per cost and per original integer coordinate from that cost's weighted label marginal. Full NLL minus this floor equals mean Bernoulli KL up to numerical precision. The required excess remains at most0.03. The entropy calculation retains exact zero/one endpoints and clips only possible float64 summation roundoff outside[0,1].

This floor is the unrestricted conditionally independent optimum. A finite shared GCN need not attain it. This small-subset gate checks whether the configured implementation can fit a simple training task; it does not claim the final120-cost model has the same fit quality.

Four of those training costs also receive small subproblem checks: fix the most confident10% of common u/x coordinates, with five seconds per subproblem, and require at least50% feasible completions. Recovery is reported but is not a gate. These are integration checks, not full matched-K performance targets. In particular, the gate does not require every training instance to recover its optimum under a difficult final restriction.

A width that fails the small-subset gate is ineligible. If no width passes both this gate and the native validation feasibility gate, fit raises TrainingSanityError with diagnostics and no completed checkpoint is returned. Model.training_metadata['sanity_gate']['passed'] must be true before the runner allows confirmation testing. A gate failure stops the experiment; it is not permission to tune on previously known test outcomes.

## Runtime, timing and confirmation split

GPU fitting is supported with device='cuda'. CPU is the default. CUBLAS_WORKSPACE_CONFIG=:4096:8 is set before importing Torch and recorded. CUDA sparse reductions use fixed RNG seeds with deterministic-algorithm warnings recorded. Repeated GPU executions and CPU/GPU comparisons are not promised bitwise identical; small numerical differences can change later optimization trajectories.

Root-LP features are cached offline only for the supplied120 archived training costs; cost/matrix digests and every certificate/time are recorded. This preprocessing is included in learning time. Training/gradient validation may use the cache. Every timed solve_cost call independently recomputes its current root LP inside the outer timer, including construction and feature assembly. One native ensemble shares that result across its five heads. Native and matched calls for the same cost cannot share a cached online LP. Solver-based validation follows this same fresh-feature path.

Final benchmark inference is run after model.to('cpu'), with one PyTorch thread and serial, one-thread HiGHS solves. root_lp_seconds is included in inference_seconds and total seconds; subproblem time is separately recorded. Feature extraction and all subproblems consume the same30-second budget. If the root LP fails or exhausts its allocation, an explicit invalid root_lp_feature_failure is returned with its diagnostics, without fallback. Temporary online features are removed after every call. Saved checkpoints contain no confirmation-cost feature cache; format3 rejects the earlier13-channel format.

The total30-second native budget is shared across all five subproblems, with each pending model allocated its share of the remaining time. Solver creation/checking overhead can make elapsed wall time slightly exceed the solver limit; actual elapsed time is always retained. Matched mode makes one subproblem call under the same total cap.

There is no full-original-problem fallback, unfixing after infeasibility, or access to confirmation optima inside solve_cost. Every returned point is checked against the original matrix, bounds, all integer domains and the winning fixed assignment. best_fixed and best_submip_index identify the actual winning restriction, including certificate upgrades on an equal objective. A subproblem optimum is not a certificate of original-problem optimality: globally_certified is always false.

The experiment runner creates fresh confirmation costs only after all case/method models and the protocol have been frozen. It solves independent original reference problems and evaluates recovery separately. Model code never reads case.test_costs, case.test_solutions or old result curves. Source review and lightweight tests do not certify future baseline performance.

## API

- fit(case,pools,config=None,validation=None,solve_fn=None); validation must be None under the internal-split protocol.
- model.save(path), load(case,path,device='cpu'), save_metadata(model,path), model.to('cpu').
- solve_cost(model,case,cost,seed,total_time_limit=30,solve_fn=None,mode='native').
- solve_cost(...,mode='matched',K=case.affine['rank']) or solve_matched_cost(...,K=...).
- Result fields include x, objective, valid, certified (local only), globally_certified, status, seconds, inference_seconds, root_lp_seconds, root_lp_metadata, solver_seconds, submip_records, best_fixed and best_submip_index.
- Matched results additionally include scope_indices, target_K and remaining_scope_count.
- Fit metadata includes sanity_gate, width candidates, selection/refit times, selected epoch counts, complete-predictor diagnostics, pool provenance and input digests.

test_neural_diving_interfaces.py checks entropy/loss identities, graph paths and raw adjacency, native replay, common-coordinate cardinality, winning restrictions, pool-completion rejection and CPU checkpoint roundtrips on a synthetic miniature. It performs no real-case training or MIP solver call.

test_nd_root_lp_features.py separately checks fractional root solutions, signed row/column dual certificates, equalities, zero norms and overflow on tiny synthetic LPs. These are genuine synthetic LP solves, reported separately from real-case solves.

## Training-only repair evidence and scope

The first frozen formal case used its case-resolved seed and tiny costs[2,3,5,8,9,10,11,12]. Width32/64 terminal excess NLLs were0.150/0.143; their best logged values were0.124/0.117. Hence terminal-only checkpointing caused regressions, but restoring the best point alone would not meet0.03. Exact weighted graph color refinement found no six-layer mixed-label equivalence classes, so an unavoidable feature symmetry was not established. Its four successful small subproblems fixed only14 constant u coordinates, so that result was integration evidence rather than proof of full/native learning quality.

Paired developer probes use seed1909 and tiny costs[0,1,3,4,5,7,8,9]. These differ from the formal case's tiny subset, so comparisons against the formal gate cannot isolate step count. On the paired subset, formulation-only raw/scaled graph probes with2000 steps and rate decay reached best excess0.108/0.096, still above0.03. Adding the certified root-LP features on the same developer subset reduced400-step excess to0.03649 and1200-step excess to0.00469, with MAP bit error0.00203. The original threshold was retained; the added information and stable optimization account for the development repair.

Attempt2 then passed the first UC case's two tiny-fit gates (excess0.00643/0.00736), but knapsack_seed101_a0.05 failed with best excess0.04445/0.04051 for widths32/64. Its marginal entropy floor0.001378 was verified independently; every item's initial cost feature was distinct, so an exact input collision did not explain the failure. Analytic fractional-knapsack rounding had weighted bit error0.012065, close to but higher than the failed networks. Most differences from integer labels were at LP variables already equal to0 or1, showing the role of integer repacking beyond rounding the lone fractional item.

A paired training-only width64 probe used the formal knapsack seed584051720 and the same tiny costs[2,3,4,5,6,8,9,10]. Both groups used the same initialization and rate schedule, with1200 steps followed by a diagnostic extension to4000 at learning rate0.0001. Raw inputs reached best excess0.03361 at3800 steps. Uniform row-max scaling reached0.01628 at1200 and0.00716 at4000, with final expected MAP bit error0.00394 versus raw0.01063 and LP rounding0.012065. These results support input conditioning as a practical concern, not a hard representability bound or a bug in Nair's raw adjacency.

The raw probe's initial diagnostics and steps25/50 exactly matched the earlier formal width64 run. At step75 the NLL differed by about3e-8, and subsequent trajectories diverged; the probe's1200 excess was0.04379 rather than the earlier best0.04051. Its seed, subset and schedule were matched, but it was not a bitwise replay. The two probe groups remain a controlled comparison within that execution.

Because normalized inputs passed the existing gate within1200 steps, the formal schedule remains1200 with the unchanged0.03 threshold. The4000-step extension is diagnostic evidence, not a new formal training budget. All cases now use the same row-max preprocessing and start from scratch under a new source/protocol freeze. No conclusion is drawn that width32 or every other case will pass; their actual tiny-fit and native-validation gates remain mandatory.

These are training diagnostics, not held-out results or certification that all cases will pass. Width selection, native feasibility validation and all120 refits remain required; fresh confirmation costs are generated only after all final models are frozen. Earlier attempts are preserved, not overwritten or omitted.

