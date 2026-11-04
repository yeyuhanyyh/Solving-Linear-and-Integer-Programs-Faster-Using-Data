# Li-only representative tie correction, v2

This is a separate implementation revision. No scientific training, LP/IP solve or cloud action was run while preparing it. ND, MLOPT, the physical model, cost distribution, labels, K, architecture, optimizer and Li parameter grid are unchanged.

`TieBank` subclasses the frozen `NativeLPStrategyBank`. It constructs exactly the same reduced inequality models and calls the original `adapter.solve` exactly once per decode, with the same objective, seed, LP relaxation and time limit. It does not use an equality-KKT shortcut, another strategy, a second LP, a full formulation or a warm start.

The correction applies only when that solve is optimal, reduced-feasible and certified, but its returned point violates the original formulation. Its primal objective, reported objective and reported dual bound must agree within `1e-7 + 1e-10 * abs(primal objective)`. The bound and certificate are those returned by the existing adapter; this code does not independently reconstruct a dual solution.

The sole alternative point is the fixed first training representative already recorded by the extracted strategy. Its index, point hash and strategy signature are retained. Original-model feasibility, all reduced rows and bounds, and every fixed integer coordinate are checked. Its objective is recomputed under the current cost and must agree with both the raw optimum and bound under the same strict tolerance. This selects another point on the same optimal face. It does not accept a more expensive feasible point or search a training library for a replacement. A raw original-feasible return is never changed. Any failed check preserves the raw result.

Every decode retains raw status/objective, the adapter's actual `certified` flag and bound, and tie evidence. For attempted ties, the raw point hash is also retained. `original_certified` remains false: solving a strategy-restricted LP does not certify global optimality of the original IP. The inherited legacy `reduced_certified` field is not a substitute for the explicitly preserved `reduced_solver_certified` and `reduced_certificate` fields.

Counters are allocated at bank construction, so the frozen pair-data routine can safely subtract its pre-decode snapshot. They count raw original feasibility, attempted/accepted ties, and certificate/feasibility/objective rejection. There is no per-pair filesystem log. Certificate details are returned in memory, and native query output retains the selected/candidate evidence.

API contracts:

- `fit_tie(api, case, validation, output, device)` returns the original `api.NeuralLearner` interface.
- `load_tie(api, case, output)` restores the explicit tie subclass and identical factorized network. It must replace the original Li loader for this revision.
- `TieExternalValidationPreference._bank` supplies `TieBank` to both development and final training. The original fit, loss, SetCover and validation rules are inherited unchanged. Native query ranking and Eq.(21) selection remain unchanged.
- `source_fingerprint()` returns hashes for `tie_bank.py` and `tie_api.py`.

Pair caches may only be created in a fresh v2 fit directory. Existing or failed directories are rejected. The original internal cache key is preserved and additionally bound to the new decoder version/source. A distinct checkpoint implementation version prevents the old loader from silently using the old decoder. All source, training, validation and artifact bindings are checked before loading weights. Model-specific construction and tie checks are inside the existing learning-time spans.

Before deployment, the parent package must pin these files, route only Li fit/load/probe through this API, and run the already defined training-only own-strategy audit. This preparation does not itself establish the correction's behavior on the real bank.
