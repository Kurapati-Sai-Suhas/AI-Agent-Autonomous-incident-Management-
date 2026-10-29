# Datasets and provenance

All data lives in `data/` (git-ignored). Raw downloads are never modified; derived files go to
`data/processed/`. `python scripts/download_data.py` re-downloads anything missing and writes
`data/provenance.json` (URL, file, size, SHA-256, download date, license). It never overwrites a file.

Download date for every file below: **2026-10-29** (see `data/provenance.json` for checksums).

| Dataset | Source | License | Local files | Used for |
|---|---|---|---|---|
| Incident management process enriched event log (UCI id 498) | https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log (zip: `https://archive.ics.uci.edu/static/public/498/incident+management+process+enriched+event+log.zip`) | CC BY 4.0 | `data/servicenow/uci_498_incident_event_log.zip` (2.4 MB), `incident_event_log.csv` (141,712 event rows) | **Only** the manual-process MTTR baseline (resolution-time distribution). It has structured fields and **no free-text descriptions**, so it is not used for text classification or retrieval. |
| Customer support tickets (Tobi-Bueck) | https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets | CC BY-NC 4.0 (non-commercial) | `data/tickets/aa_dataset-tickets-multi-lang-5-2-50-version.csv` (26 MB) | Queue + priority triage models (English rows only); incident index (train split only). The `answer` column is an agent reply, used as a resolution *hint*, not a verified fix. |
| Loghub 2k samples: Apache, Hadoop, HDFS, OpenStack, Proxifier | https://github.com/logpai/loghub (raw files from the repository) | Loghub research license; cite Zhu et al. 2023 and He et al. 2020 | `data/loghub/<System>/<System>_2k.log`, `_structured.csv`, `_templates.csv` | Drain parsing check (grouping accuracy vs. ground-truth templates) and the log retrieval index. |
| Simulator incidents (generated) | `python generate_incidents.py` (seed 42) | project-generated | `data/processed/sim_incidents.csv` | Fault classification, diagnosis, guardrail calibration/tuning, end-to-end tests, MTTR benchmark. |

## Notes and deviations from the brief

* The brief mentions full Loghub files from Zenodo (e.g. `HDFS_v1` with 575,061 blocks). These multi-GB
  archives were **not** downloaded; the 2,000-line samples with ground-truth templates are enough for the
  parser check and a log index, and keep the project light. Nothing here reports results on the full files.
* ServiceNow (UCI 498) figures reproduced from the raw file: 24,918 incidents; 23,362 have a resolution
  time; median 22.1 h, mean 178.2 h, 75th percentile 148.5 h, max 8,070.2 h (matches the brief).
  Method: per incident, first `opened_at` to the last non-missing `resolved_at` (ordered by `sys_updated_at`).
* Tickets: 16,338 English rows, 0 exact duplicates after whitespace/case normalisation; split 70/15/15
  stratified by queue with seed 42 -> 11,436 / 2,451 / 2,451 (matches the brief). Queue models use only the
  three IT queues (Technical Support, IT Support, Service Outages and Maintenance).
* Near-duplicate check: exact normalised-text overlap between ticket splits is tested (`tests/`) and is zero;
  fuzzy near-duplicates were not removed.

## Citations

* *Incident management process enriched event log*, UCI Machine Learning Repository, dataset id 498
  (use the citation and source paper listed on the dataset page).
* Bueck, T. *Customer Support Tickets* dataset, Hugging Face, doi:10.57967/hf/6184.
* Zhu, J., He, S., He, P., Liu, J., Lyu, M. R. *Loghub: A Large Collection of System Log Datasets for AI-driven
  Log Analytics.* ISSRE 2023. Repository: https://github.com/logpai/loghub
* He, P., Zhu, J., Zheng, Z., Lyu, M. R. *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* ICWS 2017.
  (Implementation used: `drain3`.)

Verify the exact author lists/titles against the publishers before putting them in the final report.
