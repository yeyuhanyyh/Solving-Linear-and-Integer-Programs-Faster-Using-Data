The independent second stream preserves an objective-ratio advantage after the fixed additional neural-network training budget. Ours returns the original optimum on all 360 costs. Original Neural Diving has a pooled ratio of 0.960871; its fixed +2000-step control has 0.956640. All three original-ND seed means remain at least 0.0302 below Ours. These are results for this synthetic problem family, not a claim about every IP family.

There are 32 coupled bundles, 96 original binary variables, and matched K=32. Every fit uses the same 36 certified training costs for a physical seed; coordinate-fixing methods fix exactly 64 original variables. The common prefix 36 was chosen by training-rank coverage; 69 exploratory labels had previously been collected per seed. Each seed receives 120 new independent costs from the same training law. All 15 model roles and their sources were frozen before this stream was generated. The original gross-profit objective is used throughout, without subtracting the base-project returns.

| Method | Recovery (%) | Objective ratio | Ratio N / 360 |
|---|---:|---:|---:|
| Ours | 100.00 | 1.000000 | 360 / 360 |
| Neural Diving (original) | 14.44 | 0.960871 | 359 / 360 |
| Neural Diving (+2000 steps) | 11.39 | 0.956640 | 347 / 360 |
| SVM | 1.11 | 0.886799 | 358 / 360 |
| MLOPT | 9.72 | 0.953409 | 360 / 360 |

Recovery uses all 360 costs. The objective-ratio mean uses the N valid, timely returns shown above; failures are retained, and none exceeded the time budget. The pooled mean gives equal weight to each valid return.

Each cell below is **returned ratio / canonical-completion diagnostic**. Ours is 1 / 1 for all three seeds.

| Seed | Original ND | ND +2000 | SVM | MLOPT |
|---|---:|---:|---:|---:|
| 1009 | 0.955515 / 0.958608 | 0.955892 / 0.968069 | 0.908306 / 0.908380 | 0.967069 / 0.967069 |
| 1013 | 0.957275 / 0.958678 | 0.957460 / 0.961254 | 0.855211 / 0.856430 | 0.939546 / 0.939546 |
| 1019 | 0.969778 / 0.971122 | 0.956548 / 0.958165 | 0.896701 / 0.898184 | 0.953612 / 0.953612 |

For a returned upgrade decision C, canonical completion adds both basic projects wherever C=0. Its gross profit is 0.5G + sum_g (xi_g − 0.5) C_g. This diagnostic preserves the returned upgrade choices and is feasible in the original problem, but can violate the method's predicted fixings. It does not replace the returned point or the primary ratio. The pooled canonical ratios are 0.962814 for original ND, 0.962460 for continued ND, 0.887723 for SVM, and 0.953409 for MLOPT. Thus, missing basic-project returns explain only part of the loss. On the 346 costs where both ND versions return valid solutions, their means are 0.961690 and 0.956717, respectively.

| Seed | Width / parameters | Original fit (s) | Extra fit (s) | Combined (s) | Train bit error, before → after |
|---|---:|---:|---:|---:|---:|
| 1009 | 32 / 41,858 | 218.84 | 79.23 | 298.07 | 16.580% → 0.405% |
| 1013 | 64 / 151,298 | 252.24 | 135.61 | 387.85 | 16.059% → 0.116% |
| 1019 | 64 / 151,298 | 247.11 | 134.52 | 381.63 | 16.493% → 1.476% |

The original ND fit retains the complete predictor, six GCN layers, 28 formulation/root-LP features, both width candidates 32/64, the 1200-step small-sample sanity gates, and the original training/selection/refit schedule. The same-label-budget adaptation uses singleton certified-optimum pools and the deterministic matched-K proposal; it does not train unused selective-mask networks. The control preserves the selected architecture and initial weights, resets Adam explicitly, and trains only the 36 labels for exactly 2000 updates (batch16,16,4; learning rate 0.001 for 1000 updates, then 0.0001). Update 2000 is the sole final checkpoint; no validation or test outcome selects it.

Extra optimization substantially lowers training error but does not improve the pooled independent objective ratio. This does not establish convergence of all possible network fits or a generalization bound. Recorded fit times exclude training-label generation and were collected with concurrent single-thread processes; they are not isolated speedup measurements. SVM model-fit times are 0.319/0.151/0.321 s; MLOPT's complete eight-grid/validation/refit times are 1.962/0.694/1.957 s. Their import/wrapper overheads are separately recorded.

Evidence: [independent check (360 references, 1800 returns, zero discrepancies)](<D:/C盘迁移归档/桌面大文件夹/research/ip-paper-baselines-v2-20261009/ip-followup-20261025/objective-gap-followup-20261025/uniform-gross-stream2-training-budget-control/independent-lightweight-check.json>), [training-control receipt](<D:/C盘迁移归档/桌面大文件夹/research/ip-paper-baselines-v2-20261009/ip-followup-20261025/objective-gap-designs-20261025/coupled-bundle/training-budget-control-v1/continuation-training-summary.json>), [architecture and complete training protocol](<D:/C盘迁移归档/桌面大文件夹/research/ip-paper-baselines-v2-20261009/ip-followup-20261025/objective-gap-designs-20261025/coupled-bundle/training-budget-control-v1/control-protocol.json>), [baseline fit-time records](<D:/C盘迁移归档/桌面大文件夹/research/ip-paper-baselines-v2-20261009/ip-followup-20261025/objective-gap-followup-20261025/extra-baselines-coupled-v1/learning-times.json>).
