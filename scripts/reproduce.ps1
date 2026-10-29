# Rebuild everything from scratch (Windows PowerShell, run from the project folder).
# Existing data files are never overwritten; delete them first if you really want to regenerate.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

& $py scripts\download_data.py          # UCI 498, tickets, Loghub 2k samples (+ data/provenance.json)
& $py generate_incidents.py             # data/processed/sim_incidents.csv (seed 42)
& $py data_loader.py                    # ticket splits + ServiceNow resolution times
& $py train_triage.py                   # artifacts/triage.joblib, results/triage_val.json
& $py -m rag.store --rebuild            # artifacts/kb_frozen.joblib (train data only)
& $py -m eval.calibrate                 # artifacts/guardrail.joblib (calibrator: train OOF, thresholds: val)
& $py -m pytest -q                      # automated tests
& $py -m eval.run_all                   # TEST split, once: results/eval_results.json + eval_tables.md + plots
& $py -m demos.run_demos                # three acceptance demos
