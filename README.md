# AI-Agent Based Autonomous Incident Management System

Capstone project (B.Tech CSE AI & ML). A multi-agent pipeline that triages an IT incident, retrieves similar
past incidents / logs / runbooks, diagnoses the likely root cause with cited evidence, applies **only a
whitelisted fix on a simulator**, verifies recovery, and otherwise escalates to a human with an explainable
handoff report.

> **Safety:** every remediation runs against `sim/service.py` (a Python object). No shell commands are ever
> executed on your computer or any real system. Unknown actions are rejected at three independent layers.

* Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* Datasets and provenance: [docs/DATA.md](docs/DATA.md)
* Evaluation report (results, limitations): [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md)
* Raw result tables: [results/eval_tables.md](results/eval_tables.md)

## Five agents + supporting parts

| Agent / part | What it does |
|---|---|
| Triage | TF-IDF + class-balanced logistic regression: queue, priority (real tickets) and fault (simulator). Returns label, confidence, runner-up, margin, evidence words. |
| Diagnosis | Hybrid RAG (BM25 + LSA vectors, RRF) over incidents, logs (Drain-parsed Loghub + simulator) and runbooks; ranked hypotheses with cited ids. Works without any LLM. |
| Resolution | Picks the lowest-risk whitelisted runbook whose preconditions hold, records rejected candidates, or reports `no_matching_runbook`. |
| Guardrail | Calibrated confidence (from triage margin, retrieval similarity, hypothesis gap, top score — never an LLM self-report) + action risk + direct-evidence policy → execute or escalate. |
| Validation | Re-checks simulator health and fresh logs; failure → rollback → escalate (`fix_not_verified`). |
| Escalation | Structured handoff built from the trace: stuck_at, per-agent confidence, why_escalated, hypotheses + evidence, actions taken, start_here. |
| Orchestration | LangGraph (`graph.py`) with conditional edges and a human-in-the-loop interrupt before `human_review`. |
| Learning loop | Human fixes are appended to a live index only; the evaluation index stays frozen. |
| Dashboard | Streamlit (`app.py`). |

## Setup (Windows PowerShell, VS Code)

Python 3.12 or 3.13 (developed and tested on 3.12.10; every dependency has 3.13 wheels).

```powershell
cd incedentagent
py -3.12 -m venv .venv            # or: py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env            # optional; defaults run fully offline, no API key needed
```

In VS Code: *Python: Select Interpreter* → `.venv\Scripts\python.exe`.

## Rebuild data, models and results

```powershell
.\scripts\reproduce.ps1
```

or step by step:

```powershell
python scripts\download_data.py   # UCI 498 + tickets + Loghub 2k samples; writes data\provenance.json
python generate_incidents.py      # 1,200 simulator incidents, 70/15/15 split, seed 42
python data_loader.py             # ticket splits + ServiceNow resolution times
python train_triage.py            # triage models (train) + validation metrics
python -m rag.store --rebuild     # frozen knowledge base (train data only)
python -m eval.calibrate          # calibrator (train OOF) + guardrail thresholds (val only)
python -m pytest -q               # tests
python -m eval.run_all            # final evaluation on the TEST split
```

Pause OneDrive sync while the data downloads (about 30 MB) are written.

## Run the demos

```powershell
python -m demos.run_demos                         # all three, with PASS/FAIL acceptance checks
python -m demos.run_demos --demo 1 --fault disk_full
python -m demos.run_demos --demo 2 --unsupported cert_expired
python -m demos.run_demos --demo 3
```

* **Demo 1 – supported fault** (`db_pool_exhausted`, `disk_full`, `memory_leak`, `service_crash`): diagnosed,
  matching low-risk runbook dry-run then executed on the simulator, validation confirms health, incident closed.
* **Demo 2 – unsupported fault** (`network_partition`, `cert_expired`): no runbook is invented; escalation
  report with stuck point `resolution / no_matching_runbook`; zero simulator actions.
* **Demo 3 – ambiguous incident** (vague alert + one log clue shared by two faults): the guardrail abstains
  (`low_confidence`, `insufficient_evidence`), shows retrieved evidence and an ordered `start_here` list.

## Dashboard

```powershell
streamlit run app.py
```

Tabs: run an incident (inject a fault, the ambiguous demo, a test-split incident, or your own text/logs; optional
"fix will not hold" to see rollback) · escalations & learning loop (record the human's fix) · evaluation & MTTR ·
runbooks.

## Optional LLM

All LLM calls go through `llm.py`, configured only by `.env` (`LLM_PROVIDER=none|anthropic|openai`,
`LLM_MODEL`, `LLM_API_KEY`). The LLM only re-words hypotheses and the handoff's `why_escalated`; sentences that
cite ids not in the retrieved evidence are dropped. Triage, confidence and action selection never use an LLM.
The default is `none` and every test runs offline. The LLM path has **not** been exercised with a real key.

## Tests

`python -m pytest -q` covers: unknown/invalid actions rejected at every layer; health derived from state;
rollback restores config; low confidence / high risk / no evidence cannot bypass the guardrail; all three demos;
failed validation → rollback → escalation; citations refer to real records; frozen index contains train data only;
thresholds tuned on val and calibration never reads test; ticket splits disjoint; LangGraph HITL interrupt +
learning loop; offline operation without an LLM key.

## Project layout

```
schema.py  graph.py  guardrail.py  executor.py  learning.py  llm.py  app.py
generate_incidents.py  data_loader.py  train_triage.py
sim/service.py          agents/{triage,diagnosis,resolution,validation,escalation}.py
rag/{index,logparse,store}.py   runbooks/__init__.py
eval/{calibrate,run_all,metrics}.py   demos/run_demos.py   tests/   scripts/   docs/   results/
data/ (git-ignored)     artifacts/ (git-ignored, rebuilt by the scripts)
```

## Honest scope

Simulator results show the pipeline's behaviour on template-generated incidents; they are **not** evidence of
production performance. See the limitations section of the evaluation report.
