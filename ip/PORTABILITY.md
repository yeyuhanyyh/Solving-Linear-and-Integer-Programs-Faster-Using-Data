# Source and execution correspondence

`snapshot/` is a byte-preserved export. `artifact-manifest.json` records the source root used by the original provenance and the SHA-256 of each included file. The portable wrapper verifies those actual bytes before importing a scientific runtime or opening a checkpoint. No snapshot source is edited to remove or update a frozen digest.

## Constructors and path relocation

| Portable operation | Original implementation retained | Portability change |
|---|---|---|
| UC constructor | `confirmation-v1/confirm_n32.py:load_case` and `baseline_helpers.py:_runtime`, `_validate_training`, `_case_binding` | The helper's returned source-path dictionary keys are canonicalized to the recorded original root. Numerical arrays, source hashes, dataset fingerprints, model receipt guards and serialized models are unchanged. The helper still checks every relocated source's bytes. |
| Knapsack constructor | `adapter/generic-binary-v2/generic_binary_pilot.py:load_case`, `validate_selection`, `assert_selected_case_schema` | A copy of the selection manifest with paths relocated to `snapshot/` is saved in the new run directory. Its original file SHA values, dimensions, physical/schema declarations and selection fields are preserved. The case retains the original selection-manifest identity after construction. |
| Exact geometry audit | Original affine metadata plus an independent exact rational rank/normal check in `run.py:validate_geometry` | Recompute rank from the training binary optimizers, check the first optimizer anchor, independent normals and exact containment. No tolerance-based rank substitution. |
| Cost mapping | Original explicit offset/coefficient tensors | Bind `offset + features @ coefficients.T`, as in the original UC ND/Preference and generic knapsack adapters. Check against every training cost and loaded test bank. |

The UC constructor continues to verify the original matrix/domain ordering (144 integer and 48 continuous variables), training objectives, UC physics, saved SVM/MLOPT source/training binding, model artifact hashes and training-optimum certificates. Knapsack retains its signed-cost, binary-domain, original-row, source-NPZ, training-feasibility and exact-rank checks. Source-location identity is the only normalized UC binding field. No model tensor, source digest or training-data equality is silently skipped.

## Inference and new fitting

| Method | Predictor/restriction and inference | New fitting |
|---|---|---|
| Full | Original `experiment_core.solve` without an affine restriction | No fit |
| Ours | Original `experiment_core.solve` with the verified affine restriction on the common coordinates | Original `experiment_core.fit_affine` on the same training binary optimizer rows |
| SVM (UC) | Original saved pickle, original `baseline_helpers.predict_svm` and `svm_match` | Original `fast_baselines.fit_coordinate_svm_small`; attach the helper's original source/data binding |
| SVM (knapsack) | Original guarded `svm_mlopt_components.load_component` and `propose_svm`, then original `adapter.solve` | Original `svm_mlopt_components.fit_svm` and `save_component` |
| MLOPT (UC) | Original saved pickle, original `baseline_helpers.predict_mlopt`, native decoded top-ten anchor and matching rule | Original `mlopt_baseline.fit` with the exact eight-configuration grid from `baseline_helpers.fit_baselines` |
| MLOPT (knapsack) | Original guarded component loader, `propose_mlopt`, then original restricted solve | Original `svm_mlopt_components.fit_mlopt` and `save_component` |
| Neural Diving | Original `train_uc_baselines.load_nd_complete` and `neural_diving_baseline.solve_matched_cost` | Original `train_uc_baselines.train_one` / `fit_nd_complete` with the original width, feature, readiness-gate and epoch-selection schedules |
| Preference Learning | Original `preference_baseline.load_fitted` and `train_uc_baselines.propose_li_native_anchor`, then original restricted solve | Original `train_uc_baselines.train_one`, including the native grid, pair generation, all-label refit and readiness checks |
| Additional ND control | Original continued predictor loaded by `load_nd_complete`; same matched solve | No new continuation launched by the wrapper. Original control source, protocol, trajectories and checkpoints are included. |

ND loading recomputes the training cost digest and graph/matrix digest, checks the original sanity gate and loads the original state tensors. Preference loading recomputes the training/model hashes, rebuilds the strategy bank, compares exact strategy signatures and pair-cache source/configuration identity, and restores the original state dictionary. The knapsack SVM/MLOPT loader checks original arrays, model, objective mapping, source hashes and component binding. UC SVM/MLOPT use the original constructor's equivalent binding checks.

Fresh fitting never reuses an archived `complete.json`. It writes new case/method directories under the requested new output, with a release-specific guard containing the actual release source hash, archive manifest identity, data fingerprint, original source hashes and K. The neural trainer's guard callback is changed for this new publication identity only. Its fitting code, seeds, architecture, grids, optimizer and schedules are unchanged. UC ND retains its hashed supplement seed; UC Preference Learning and MLOPT retain the physical seed; knapsack retains the seed specified in the original case metadata.

The fresh wrapper records its own wall/CPU interval separately from the original method's fitting metadata. It does not substitute these rerun times for the archived paper cells. Existing artifacts, source seals and raw results remain read-only. The independent table validator does not use this execution layer or the model pickles.
