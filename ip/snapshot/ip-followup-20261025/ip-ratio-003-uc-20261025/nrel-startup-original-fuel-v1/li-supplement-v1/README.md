# Full Li supplement for the new n32 UC setting

This new sibling directory contains only a thin driver. It imports the
existing `train_uc_baselines.train_one(..., "li")` and unchanged native Li
code. Old UC generators, sources, models and outcome records stay read-only.

All three cases use the same32 training costs/labels as the current NREL-scale
startup-charge robustness setting. Common coordinates are48 commitment `u`
variables; K is14/14/17 at physical seeds101/211/307. Startup/shutdown `v,w`
remain binary, and48 dispatch `p` variables remain continuous. Matching fixes
only48-K `u` coordinates. Costs use the explicit192-by12 generator/time-block
affine map, including all four original objective components; no old
generator-only multiplier shortcut is used.

The Li architecture and training are unchanged: hidden width128, two residual
attention layers,1200 epochs per nondegenerate fit, AdamW, original two
learning rates, both lambda grids, internal training-only80/20 validation,
and two all-n final LR fits. The existing native runner retains requested
top-k candidates1/3/5/10/30, clipping to development SetCover size. No external
cost or outcome selects parameters, epochs, states, or strategies at fitting.

The one-certified-training-query smoke passed. It reproduced all32 original
cost vectors and checked original UC physics and integer/continuous domains.
No neural update and no confirmation-bank read occurred in that smoke.

## Commands

Use `D:/C盘迁移归档/桌面大文件夹/research/ip-comparison-20260924/.venv/Scripts/python.exe`.

After the root/adapter releases a worker slot:

```text
python uc_li_supplement.py --fit --seeds 101 211 307 --device cpu
python uc_li_supplement.py --freeze-all
python evaluate_same_bank_v2.py
```

Each seed is an independent job if root releases more than one slot. At all
times the combined Li/ND/other experiment worker count must stay at most4,
with one computational thread per process. The initial rough CPU estimate is
5–20 minutes per physical case; actual metadata records fitting and its phases.
No architecture or epoch reduction is used to accelerate the comparison.

All three Li checkpoints must pass the native readiness gate and the new
source/training guards before `li-model-freeze.json` is created. Only after
that freeze can the driver read the existing120-cost-per-seed bank. Original
reference certificates and all bank hashes are rechecked. The bank existed
before these Li fits; output explicitly labels this a later baseline
supplement on the same existing bank, rather than a new pre-frozen experiment.
UC objective ratio is original minimum cost divided by returned total cost.
There is no cost shift, fresh sampling, test tuning, or cross-setting reuse.

Outputs are `run-v1/models/<case_id>/li/` and
`same-bank-results-v2/<case_id>/test*.json`. Raw values, invalid/timeout returns,
selected fixings, model metadata and learning times remain fully recorded.

The already-running fit command also contains the initial evaluation-v1
operation. That immutable source and any v1 outputs are preserved, but v1 is
superseded. The separately sealed v2 clock stops only after original CSR
constraints, UC physics, fixing residuals and cost-vector multiplication have
been checked; all those operations count toward the same30second outer budget.
The final report uses v2 for every one of the360 fixed costs, with no selection
between versions and no change to models or test banks.
