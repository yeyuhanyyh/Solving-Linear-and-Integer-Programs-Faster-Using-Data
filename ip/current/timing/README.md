# Recorded fitting measurements

The JSON files are gzip-compressed to keep the full grid/epoch traces without
making source diffs unreadable. Decompression recovers the original JSON bytes.
Python can read one with `json.load(gzip.open(path, "rt", encoding="utf-8"))`.

`mlopt/` contains the three CPU fits for each current UC/correlated condition.
`grouped/` contains CPU repetitions and the current GPU neural fits.
`correlated/` contains the original GPU neural measurements.
`correlated_cpu.json.gz` holds the Ours/SVM CPU repetitions.
`uc_svm.json.gz` holds the current UC SVM CPU repetitions.
`uc_derivation.json.gz` explains the UC neural clock subtraction.

The compact `../learning_records.json` lists per-instance fitting summaries.
The verifier checks them against these detailed measurements before rebuilding
the table averages. No training-IP generation time is included.
