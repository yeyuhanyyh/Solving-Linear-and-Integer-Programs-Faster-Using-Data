# Current IP release and historical archive

`current/settings.json` records the current problems and manuscript revision. `current/results.csv` holds 30 full-precision rows. `current/observations.jsonl` normalizes all 9,080 saved query records, preserving values, failures, recovery flags, dimensions, and anchors. `current/provenance.json` records original experiment-relative paths and hashes.

`current/verify.py` independently aggregates observations and fitting records and checks every table cell. It uses no checkpoint and invokes no solver. It checks saved feasibility flags, without claiming to re-prove feasibility from these scalar observations. Completed original-domain audits underpin those flags.

`current/data/`, `current/testing/`, and `current/grouped/` contain frozen numerical inputs. `current/source/` contains scientific implementations. `current/run.py` supplies portable paths, converts the banks into existing case interfaces, and dispatches fitting/evaluation. The copied MLOPT loader only changes the directory alias `source/random` to `source/correlated`. Original source modules retain historical internal names and comments.

New runs require a new output directory outside `current/`. Training reads only supplied training/validation arrays. Tests open for evaluation after fitting. No cloud credentials, launch scripts, manuscript files or account data are included. Published numbers come from existing experiments, not new smoke tests.

The October snapshot, checkpoints and artifact manifest remain unchanged. Use `legacy_run.py`, `legacy_validate_results.py`, [LEGACY.md](LEGACY.md) and [LEGACY_PORTABILITY.md](LEGACY_PORTABILITY.md) for that protocol. Its UC setting and timings are not the current Tables 3–5.
