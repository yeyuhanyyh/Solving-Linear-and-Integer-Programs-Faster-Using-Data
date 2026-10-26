# Coupled-bundle SVM/MLOPT supplement

This is a new sibling folder. Existing source, training files, fitted models
and results are unchanged. `svm_mlopt_components.py` and its `extra-runtime`
dependency are byte-for-byte copies of the finalized hard-knapsack components.
They already accept the generic binary `FrozenCase`, arbitrary original rows,
and an explicit signed affine objective map; no family restriction is imposed.

The fixed cohort is G32/n36 at all three seeds1009/1013/1019, followed by
G16/n37 at the same three seeds. The SVM hyperparameters and MLOPT full eight
configurations, native top-ten objective winner and all-n refit are unchanged.
Both methods fix D-K original binary columns to match the learned affine K.

## Commands

Use the scientific Python executable:

`D:/C盘迁移归档/桌面大文件夹/research/ip-comparison-20260924/.venv/Scripts/python.exe`

One case, after root confirms that a worker slot is available:

```text
python coupled_svm_mlopt_driver.py --manifest <training-only-manifest.json> --case-dir <case-directory> --methods svm mlopt --device cpu --fit --out results
```

All six cases serially, with G32 prioritized:

```text
python run_six_serial.py --cpu-slot-released-receipt <prior-UC-run-v2-completion.json>
```

The queue uses one process at a time. Together with the separately scheduled
three continuation workers this respects root's four-worker limit. Its
completion file freezes all twelve checkpoint roles before any development
file is opened. The existing fresh stream was generated before these fits;
these models are never represented as frozen before that earlier stream.

After all twelve models are frozen, diagnostics on the unchanged old
development banks can run with:

```text
python evaluate_existing_development.py
```

This does not tune epochs or parameters, create a fresh stream, or read the
previous confirmation bank. Stream2 is managed separately by root and must
freeze all intended roles before new RNG.

## API for root's stream2 wrapper

```python
import coupled_svm_mlopt_driver as extra
extra.enable()
case = extra.generic.load_case(case_dir, manifest_path)
folder, complete = extra.validate_checkpoint(case, "svm", out, manifest_path)
model = extra.load_model(case, "svm", out, manifest_path)
fixed, diagnostics = extra.components.propose_svm(model, case, cost, feature)
# MLOPT: method="mlopt", propose_mlopt(...)
```

The output is `results/models/<case_id>/{svm,mlopt}/`, with `complete.json`,
`metadata.json`, and `model.pkl`. The complete marker contains an immutable
training/source/manifest guard and all artifact hashes. `propose_mlopt` can
return `None` if no native feasible anchor exists; there is no full-IP fallback.

## Smoke and rough runtime

The one-training-cost interface smoke passed on G32/n36/seed1009: both
methods propose64 fixes and retain K32. Real SVM fitting took0.495seconds.
MLOPT used its original bank/pruning/decoder with deterministic zero logits
only for the interface smoke; **no neural network was fitted by this smoke**
and those probe weights are never saved as experiment models. No development
or confirmation arrays were accessed. See `smoke-results/training-interface-smoke.json`.

The copied original MLOPT full-grid synthetic n8 check took about3seconds.
For these n36/n37 cases, a rough CPU estimate is5–30seconds per case plus
Python/Torch startup; actual timing is recorded separately for all eight grid
fits, validation, final refit and total. This is an estimate, not a result.
