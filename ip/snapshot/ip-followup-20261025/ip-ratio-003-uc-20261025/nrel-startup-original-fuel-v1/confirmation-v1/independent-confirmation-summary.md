# Independent UC confirmation review

**PASS, zero discrepancies.** Independent recomputation checked all 360 Full references and 1,080 method returns, including the original objective, ranged-row feasibility, separately coded UC physics, compression restrictions, K and the 30-second return budget. No model was loaded and no reference IP was solved again.

| Method | Recovery (%) | Original cost ratio | Ratio N | Recovery N |
|---|---:|---:|---:|---:|
| Ours | 88.8889 | 0.999094 | 360 | 360 |
| SVM | 18.0556 | 0.924027 | 341 | 360 |
| MLOPT | 20.5556 | 0.920463 | 360 | 360 |

The ratio is the Full optimal total cost divided by the returned total cost, averaged over valid timely returns. No objective constant is subtracted. On the same valid queries, the mean advantages of Ours are **0.075135 over SVM** and **0.078631 over MLOPT**. The MLOPT gap exceeds 0.0596 on each of the three fixed seeds. SVM has 19 invalid returns; they remain in its Recovery denominator. No returns exceed the time budget.

Each synthetic instance has four generators and twelve periods: 192 original variables, comprising 144 binary and 48 continuous variables. All methods fit the same 32 training costs per seed. K is 14, 14 and 17 within the 48 commitment coordinates; startup/shutdown variables remain binary and dispatch is reoptimized. The preliminary exploration had already collected 64 labels per seed, so total label acquisition must not be described as only 32. Global exactness of the learned affine space is not established.

Fuel coefficients retain their original range of 22–45. Capacity-scaled startup charges use a synthetic rate of 22–60 per MW. NREL provides motivation for the scale of a capital/maintenance lower-bound component; the construction is neither calibrated plant data nor a full startup-fuel estimate. The comparisons use the declared K adaptations of coordinate SVM and native-top-ten MLOPT. This confirmation does not compare Neural Diving or Li. Timing speedups require the separate controlled timing experiment.

The companion JSON gives exact values and source hashes. The full arithmetic audit is `independent-confirmation-review.json` (SHA256 `784e2754a485ba5cfa68f49eacd4d5d32040e9046fc0b201b15156ee190cfb91`).
