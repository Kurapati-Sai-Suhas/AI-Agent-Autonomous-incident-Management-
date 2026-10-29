# Evaluation report

Generated from `results/eval_results.json` (`python -m eval.run_all`, run on 2026-10-30). Full tables:
[`results/eval_tables.md`](../results/eval_tables.md). Plots: `results/plots/`.

**How to read the numbers**

| Label | Meaning |
|---|---|
| **Measured (real data)** | computed on a held-out split of a public dataset (tickets, Loghub) |
| **Measured (simulator)** | computed by running the real pipeline against `FakeService` on held-out *generated* incidents |
| **Simulated** | uses assumed step durations and/or sampled human times (MTTR) |
| **Not run** | planned; no number is reported |

Protocol: fixed splits with seed 42. Models trained on train; calibrator fit on train (5-fold out-of-fold,
leave-self-out retrieval); thresholds tuned on val only; knowledge base frozen from train only before
evaluation (manifest id-hashes `bf62326d…/7a6cef47…/ddf1b7f1…`); test used for the numbers below.

Test split was evaluated **twice**. The first run's citation-faithfulness checker misread the `[cause]` labels in
`start_here` as citations; labels were changed to `(cause)` and the run repeated. Before that, one earlier run was
stopped *before any result was read* when Demo 3 (a hand-written scenario, not test data) exposed that the guardrail
could act with no direct log evidence; the `insufficient_evidence` policy was added and val tuning re-run. No model,
threshold or rule was changed based on test results.

## 1. Triage — measured (real data for queue/priority; simulator for fault)

| model | split | n | accuracy | macro-F1 | majority baseline | brief's val reference |
|---|---|---|---|---|---|---|
| queue (3 IT queues) | val | 1,101 | 0.788 | 0.759 | 0.646 | 0.752 / 0.716 |
| queue (3 IT queues) | **test** | 1,102 | **0.814** | **0.790** | 0.644 | – |
| priority | val | 2,451 | 0.668 | 0.660 | 0.396 | 0.624 / 0.616 |
| priority | **test** | 2,451 | **0.651** | **0.637** | 0.419 | – |
| fault (all) | test | 180 | 0.922 | 0.922 | 0.172 | – |
| fault (clear incidents) | test | 152 | 1.000 | 1.000 | – | – |
| fault (ambiguous incidents) | test | 28 | 0.500 | 0.441 | – | – |

The TF-IDF (uni+bigram, sublinear tf) + class-balanced LR baseline matches or beats the brief's reference
numbers on val. Fault classification is perfect on clear simulator incidents because their text is templated;
the ambiguous subset is the meaningful number. No transformer/LLM triage upgrade was trained (**not run**).

## 2. End-to-end pipeline — measured (simulator), 180 test incidents

| metric | full system |
|---|---|
| diagnosis accuracy (top hypothesis = injected fault) | 0.922 (clear 1.000, ambiguous 0.500) |
| auto-resolution rate, all incidents | 0.567 (102 / 180) |
| auto-resolution rate, supported faults | 0.843 (102 / 121) |
| verified resolution (claimed resolved **and** simulator healthy afterwards) | 0.567 — 0 false success claims |
| **false-action rate** (a fix that does not resolve the true fault was executed) | **0.000** (0 wrong actions, 102 executed) |
| escalation rate | 0.433 |
| escalation precision / recall vs. "should escalate" (= unsupported or ambiguous) | 1.000 / 1.000 (78 TP, 0 FP, 0 FN) |
| unsupported faults escalated | 59 / 59 |
| ambiguous incidents escalated | 28 / 28 |
| citation faithfulness (every cited id exists in retrieved set / incident logs) | 1.000 |
| trace completeness (all stages + close/escalation present) | 1.000 |
| escalation reasons (counts, multi-label) | no_matching_runbook 58 · insufficient_evidence 28 · low_confidence 26 · conflicting_evidence 20 · triage_ambiguous 18 |
| stuck point of escalations | resolution 50 · triage 18 · diagnosis 10 |

Escalation is triggered for every ambiguous incident partly **by construction**: ambiguous simulator incidents
contain no "strong" fault signature, and the direct-evidence policy requires one. The escalation P/R of 1.0 is
therefore a property of this simulator, not a general result.

### Stress set — supported test faults whose fix does not hold (121 incidents)

The simulator was told to let the correct runbook run but keep the fault active.

| | full system | no validation (ablation) |
|---|---|---|
| incidents claimed resolved | 0 | 102 |
| false success claims (claimed resolved, simulator still unhealthy) | **0** | **102** |
| escalated with `fix_not_verified` after rollback | 102 | 0 |
| config restored by rollback | yes (tested) | – |

This is the clearest evidence for the validation + rollback design: without it, every non-holding fix is
silently reported as success.

## 3. Ablations — measured (simulator, test)

| config | auto-resolve | wrong actions | escalation P / R | notes |
|---|---|---|---|---|
| full system | 0.567 | 0 | 1.00 / 1.00 | |
| no RAG (guardrail unchanged) | 0.000 | 0 | 0.43 / 1.00 | guardrail refuses to act without retrieval support → escalates everything |
| no RAG, retrieval gate removed | 0.567 | 0 | 1.00 / 1.00 | diagnosis accuracy unchanged (0.922) |
| no validation | 0.567 | 0 | 1.00 / 1.00 | identical on normal faults; **102 false successes** on the stress set |
| no calibration (raw score, default thresholds) | 0.567 | 0 | 1.00 / 1.00 | |
| no XAI trace / handoff | **not run** | | | needs the human usefulness study (below) |

Honest reading: on this simulator, RAG and calibration do not change the decisions, because clear incidents are
easy and the direct-evidence policy already blocks the ambiguous ones. RAG's measured contribution here is the
evidence it supplies (citable similar incidents and log windows), not accuracy. A harder simulator or real
incidents are needed to separate these components.

## 4. Calibration — measured (simulator)

| | ECE |
|---|---|
| val, raw hypothesis score | 0.044 |
| val, calibrated (LR on 4 signals) | 0.037 |
| test, calibrated | 0.024 |

Reliability curve: `results/plots/reliability_val.png`. Risk/coverage over the confidence threshold alone
(test, runbook-matched incidents, other guardrail rules ignored; `results/plots/risk_coverage.png`):

| confidence threshold | coverage | wrong-action share |
|---|---|---|
| 0.0 (act on everything) | 1.000 | 0.090 |
| 0.5 | 0.992 | 0.083 |
| 0.9 (tuned on val) | 0.836 | 0.000 |

So the calibrated confidence on its own already separates the wrong actions; the extra guardrail rules are a
second layer. With only 180 val / 180 test incidents and
~93 % accuracy, most bins are sparse; treat the ECE values as rough.

## 5. Retrieval — measured (simulator queries over the frozen 12,276-document incident index)

Relevant = a resolved **train** simulator incident with the same fault (tickets act as distractors).

| mode | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | 0.933 | 0.983 | 0.989 | 1.000 | 0.958 |
| vector (LSA, 128-d) | 0.900 | 0.933 | 0.967 | 0.989 | 0.926 |
| hybrid (RRF) | 0.911 | 0.967 | 0.978 | 0.989 | 0.940 |

BM25 alone is best here, as expected with template text that shares exact error strings. Vectors are LSA
(TF-IDF→SVD), **not** a neural sentence-transformer model (optional, not run). Cross-encoder rerank: **not run**.
Retrieval on Loghub logs has no relevance labels and was not scored.

## 6. Log parsing — measured (real data, Loghub 2k samples)

Drain (drain3, depth 4, similarity 0.4), grouping accuracy against Loghub ground-truth templates:

| system | lines | templates | GA no masking | GA generic masking |
|---|---|---|---|---|
| Apache | 2,000 | 6 | 1.000 | 1.000 |
| Hadoop | 2,000 | 114 | 0.954 | 0.963 |
| HDFS | 2,000 | 14 | 0.998 | 0.998 |
| OpenStack | 2,000 | 43 | 0.310 | 0.299 |
| Proxifier | 2,000 | 8 | 0.026 | 0.017 |

OpenStack and Proxifier are poor with this generic configuration; published Drain results use dataset-specific
pre-processing and thresholds, which were not tuned here. Parsing quality only affects template-id metadata, not
the retrieval text.

## 7. MTTR — simulated

| | median | mean | p75 | p90 |
|---|---|---|---|---|
| manual baseline (one ServiceNow time sampled per test incident) | 15.9 h | 170.7 h | 152.0 h | 342.0 h |
| agent pipeline | 0.038 h (2.3 min) | 66.6 h | 0.88 h | 125.4 h |
| auto-resolved incidents only (n = 102) | 1.9 min | | | |

Real ServiceNow distribution (23,362 incidents with a resolution time): median 22.1 h, mean 178.2 h, p75 148.5 h,
max 8,070 h. Method: escalated incidents get agent time **plus the same sampled human time** (no handoff speed-up
assumed), so the improvement comes only from auto-resolved cases. Agent step durations are assumptions
(`graph.STEP_SECONDS`, `executor.ACTION_SECONDS`). This compares a simulated environment with a human-process
baseline: it shows the *scale* of a possible improvement, **not** a production MTTR. Plot: `results/plots/mttr_cdf.png`.

## 8. Not done / planned

* Handoff usefulness study (teammates rate sample reports 1–5) and the "no XAI" ablation — **not run**.
* Transformer / LLM-few-shot triage comparison — **not run**.
* Neural embeddings (sentence-transformers) and cross-encoder rerank — implemented as an optional switch for
  embeddings only; **not run**.
* LLM wording path — code and citation filter exist and are unit-tested offline; **never called with a real key**.
* Agreement across repeated runs as a confidence signal — not implemented.
* Full Loghub archives (HDFS_v1, Hadoop labels, OpenStack anomaly labels) and log-anomaly evaluation — not run.

## 9. Limitations

* The simulator generates **template text**; perfect accuracy on clear incidents says nothing about real
  incidents. Ambiguity is defined by the generator (one shared clue), which the direct-evidence policy detects
  by construction.
* Loghub logs come from different systems than the simulator; they test parsing and indexing only.
* Ticket priority/queue labels are from a synthetic-style public ticket dataset (CC BY-NC); the `answer` field is
  a reply, not a verified fix.
* MTTR is simulated; human time is sampled from a different organisation's process.
* Small evaluation sets (180 test incidents): differences of a few incidents are within noise.
* Developed on Python 3.12 (3.13 was not installed on the build machine); dependencies all provide 3.13 wheels but
  the suite was not executed on 3.13.
