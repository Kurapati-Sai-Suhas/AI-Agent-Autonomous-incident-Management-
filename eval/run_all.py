"""Final evaluation on the TEST split (run once, after calibration/tuning on val).

Everything below is MEASURED on the simulator / public data, except the MTTR
comparison, which is SIMULATED: agent time comes from fixed per-step durations and
human time is sampled from the real ServiceNow resolution-time distribution.

Run:  python -m eval.run_all          -> results/eval_results.json, results/*.md, results/plots/*.png
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agents.triage import TriageAgent, incident_text
from data_loader import (IT_QUEUES, describe_hours, load_sim_incidents, prepare_tickets,
                         servicenow_resolution_hours)
from eval.metrics import ece, prf, reliability
from graph import Runtime, build_graph, run_incident
from guardrail import Guardrail
from rag.logparse import SYSTEMS, grouping_accuracy, load_structured, parse_contents
from rag.store import load_frozen
from schema import Incident
from sim.service import RESOLVES, FakeService

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PLOTS = RES / "plots"
SEED = 42
STAGES = ["ingestion", "triage", "diagnosis", "resolution", "guardrail"]
_ID = re.compile(r"\[([^\]\s]+)\]")


# ------------------------------------------------------------------ triage
def eval_triage(agent: TriageAgent, sim_test: pd.DataFrame) -> dict:
    t = prepare_tickets()["test"]
    tq = t[t.queue.isin(IT_QUEUES)]

    def sc(model, x, y):
        p = model.predict(x)
        return {"n": int(len(y)), "accuracy": round(accuracy_score(y, p), 4),
                "macro_f1": round(f1_score(y, p, average="macro"), 4),
                "majority_baseline_acc": round(float(y.value_counts(normalize=True).iloc[0]), 4)}
    x = [incident_text(a, b) for a, b in zip(sim_test.text, sim_test.logs)]
    sim_test = sim_test.assign(x=x)
    amb = sim_test.ambiguous
    m = agent.models
    return {"queue": sc(m["queue"], tq.text, tq.queue), "priority": sc(m["priority"], t.text, t.priority),
            "fault_all": sc(m["fault"], sim_test.x, sim_test.fault),
            "fault_clear": sc(m["fault"], sim_test[~amb].x, sim_test[~amb].fault),
            "fault_ambiguous": sc(m["fault"], sim_test[amb].x, sim_test[amb].fault)}


# ------------------------------------------------------------------ pipeline
def run_pipeline(rt: Runtime, df: pd.DataFrame, resistant: bool = False) -> list[dict]:
    g = build_graph(rt)
    rows = []
    for r in df.itertuples():
        svc = FakeService(seed=int(r.id.split("-")[1]))
        svc.inject_fault(r.fault, resistant=resistant)
        inc = Incident(id=r.id, source="alert", text=r.text, logs=r.logs)
        s = run_incident(rt, inc, svc, graph=g)
        executed = [e["detail"]["runbook"] for e in s["trace"]
                    if e["agent"] == "executor" and e["detail"].get("status") == "executed"]
        wrong = [a for a in executed if a not in RESOLVES[r.fault]]
        healthy = svc.health_check()["healthy"]
        stuck = (s.get("handoff") or {}).get("stuck_at")
        rows.append({
            "id": r.id, "fault": r.fault, "ambiguous": bool(r.ambiguous), "supported": bool(r.supported),
            "status": s["status"], "pred_fault": s["diagnosis"]["top_cause"],
            "diag_correct": s["diagnosis"]["top_cause"] == r.fault,
            "confidence": s["guardrail"]["confidence"], "runbook": s["resolution"]["runbook"],
            "executed": executed, "wrong_action": bool(wrong),
            "resolved_claimed": s["status"] == "resolved", "healthy_after": healthy,
            "escalated": s["status"] != "resolved",
            "should_escalate": (not r.supported) or bool(r.ambiguous),
            "reasons": sorted(set(s.get("escalation_reasons", []))),
            "stuck_agent": stuck["agent"] if stuck else None,
            "elapsed_s": float(s["elapsed_s"]),
            "faithful": citations_faithful(s),
            "trace_complete": all(any(e["agent"] == st for e in s["trace"]) for st in STAGES)
                              and (s["status"] == "resolved" or any(e["agent"] == "escalation" for e in s["trace"])),
        })
    return rows


def citations_faithful(s: dict) -> bool:
    """Every evidence id cited by diagnosis or handoff exists in the retrieved set / incident logs."""
    inc, d = s["incident"], s["diagnosis"]
    allowed = {f"{inc['id']}:log:{i}" for i in range(len(inc["logs"]))}
    for group in d["retrieved"].values():
        allowed.update(h["id"] for h in group)
    cited = [e for h in d["hypotheses"] for e in h["supporting_evidence"]]
    cited += _ID.findall(" ".join(h["explanation"] for h in d["hypotheses"]))
    if s.get("handoff"):
        ho = s["handoff"]
        cited += [e for h in ho["hypotheses"] for e in h["evidence"]]
        cited += [e["id"] for e in ho["evidence"]]
        cited += _ID.findall(" ".join(ho["start_here"]))
    return all(c in allowed for c in cited)


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    n = len(df)
    sup = df[df.supported]
    return {
        "n": n,
        "diagnosis_accuracy": round(df.diag_correct.mean(), 4),
        "diagnosis_accuracy_clear": round(df[~df.ambiguous].diag_correct.mean(), 4),
        "diagnosis_accuracy_ambiguous": round(df[df.ambiguous].diag_correct.mean(), 4),
        "auto_resolution_rate_all": round(df.resolved_claimed.mean(), 4),
        "auto_resolution_rate_supported": round(sup.resolved_claimed.mean(), 4) if len(sup) else None,
        "verified_resolution_rate_all": round((df.resolved_claimed & df.healthy_after).mean(), 4),
        "false_success_claims": int((df.resolved_claimed & ~df.healthy_after).sum()),
        "false_action_rate": round(df.wrong_action.mean(), 4),
        "wrong_actions": int(df.wrong_action.sum()),
        "actions_executed": int(df.executed.map(len).sum()),
        "escalation_rate": round(df.escalated.mean(), 4),
        "escalation_vs_should_escalate": prf(df.escalated.tolist(), df.should_escalate.tolist()),
        "unsupported_escalated": f"{int(df[~df.supported].escalated.sum())}/{int((~df.supported).sum())}",
        "ambiguous_escalated": f"{int(df[df.ambiguous].escalated.sum())}/{int(df.ambiguous.sum())}",
        "citation_faithfulness": round(df.faithful.mean(), 4),
        "trace_complete_rate": round(df.trace_complete.mean(), 4),
        "escalation_reason_counts": pd.Series([r for rs in df.reasons for r in rs]).value_counts().to_dict(),
        "stuck_agent_counts": df[df.escalated].stuck_agent.value_counts().to_dict(),
    }


# ------------------------------------------------------------------ retrieval
def eval_retrieval(kb, test: pd.DataFrame) -> dict:
    out = {}
    for mode in ["bm25", "vector", "hybrid"]:
        hits = {1: [], 3: [], 5: [], 10: []}
        rr = []
        for r in test.itertuples():
            res = kb.incidents.search(incident_text(r.text, r.logs), k=10, mode=mode)
            rel = [h["meta"].get("fault") == r.fault for h in res]
            first = next((i + 1 for i, x in enumerate(rel) if x), None)
            rr.append(1 / first if first else 0.0)
            for k in hits:
                hits[k].append(any(rel[:k]))
        out[mode] = {f"recall@{k}": round(float(np.mean(v)), 4) for k, v in hits.items()} | \
                    {"mrr@10": round(float(np.mean(rr)), 4)}
    out["note"] = ("Relevant = a resolved TRAIN simulator incident with the same fault, inside the full "
                   "incident index (12k docs incl. tickets as distractors). Cross-encoder rerank: not run.")
    return out


def eval_parsing() -> dict:
    out = {}
    for s in SYSTEMS:
        df = load_structured(s)
        content, truth = df.Content.astype(str).tolist(), df.EventId.tolist()
        out[s] = {"lines": int(len(df)), "true_templates": int(df.EventId.nunique()),
                  "ga_no_masking": round(grouping_accuracy(truth, parse_contents(content, mask=False)), 4),
                  "ga_generic_masking": round(grouping_accuracy(truth, parse_contents(content, mask=True)), 4)}
    return out


# ------------------------------------------------------------------ MTTR
def eval_mttr(rows: list[dict]) -> dict:
    hours = servicenow_resolution_hours().to_numpy()
    rng = np.random.default_rng(SEED)
    human = rng.choice(hours, size=len(rows), replace=True)        # paired draw per incident
    agent = np.array([r["elapsed_s"] / 3600 + (h if r["escalated"] else 0.0)
                      for r, h in zip(rows, human)])
    stats = lambda a: {"median_h": round(float(np.median(a)), 4), "mean_h": round(float(a.mean()), 2),
                       "p75_h": round(float(np.percentile(a, 75)), 2), "p90_h": round(float(np.percentile(a, 90)), 2)}
    auto = np.array([r["elapsed_s"] / 60 for r in rows if not r["escalated"]])
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for a, lab, c in [(human, "manual baseline (ServiceNow-sampled)", "#d08c2a"), (agent, "agent pipeline", "#2a6fd0")]:
        x = np.sort(np.maximum(a, 1e-3))
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=lab, color=c)
    ax.set_xscale("log"); ax.set_xlabel("time to resolve (hours, log scale)"); ax.set_ylabel("fraction resolved")
    ax.set_title("Simulated MTTR (test incidents)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "mttr_cdf.png", dpi=130); plt.close(fig)
    return {"servicenow_distribution": describe_hours(pd.Series(hours)),
            "manual_baseline": stats(human), "agent_pipeline": stats(agent),
            "auto_resolved_minutes": {"n": int(len(auto)), "median_min": round(float(np.median(auto)), 2) if len(auto) else None},
            "method": ("SIMULATED. Manual = one human resolution time per incident sampled from the 23,362 real "
                       "ServiceNow resolution times. Agent = simulated step durations; escalated incidents add the "
                       "SAME sampled human time (no speed-up from the handoff is assumed). Not a production figure.")}


def risk_coverage(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    df = df[df.runbook.notna()]
    out = []
    for tau in np.linspace(0, 1, 21):
        sel = df[df.confidence >= tau]
        wrong = sum(rb not in RESOLVES[f] for rb, f in zip(sel.runbook, sel.fault))
        out.append({"threshold": round(float(tau), 2), "coverage": round(len(sel) / max(len(df), 1), 4),
                    "risk_wrong_action": round(wrong / len(sel), 4) if len(sel) else 0.0})
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.plot([o["coverage"] for o in out], [o["risk_wrong_action"] for o in out], "o-", color="#2a6fd0")
    ax.set_xlabel("coverage (share of runbook-matched incidents acted on)"); ax.set_ylabel("risk (wrong-action share)")
    ax.set_title("Risk vs coverage (test)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(PLOTS / "risk_coverage.png", dpi=130); plt.close(fig)
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    t0 = time.time()
    kb = load_frozen()
    assert kb.manifest["frozen"] and kb.manifest["n_learned"] == 0, "evaluation needs the frozen KB"
    sim = load_sim_incidents()
    test = sim[sim.split == "test"]
    supported_test = test[test.supported]
    agent = TriageAgent()
    res = {"split": "test", "kb_manifest": kb.manifest}

    print("triage ..."); res["triage"] = eval_triage(agent, test)

    configs = {
        "full_system": dict(),
        "no_rag": dict(use_rag=False),
        # same, but the guardrail no longer demands retrieval support and uses the raw score
        "no_rag_no_retrieval_gate": dict(use_rag=False, guardrail=Guardrail(None, Guardrail.load().thresholds,
                                                                             use_retrieval_check=False)),
        "no_validation": dict(use_validation=False),
        "no_calibration": dict(guardrail=Guardrail(None, None)),
    }
    full_rows = None
    res["pipeline"], res["stress_persistent_faults"] = {}, {}
    for name, kw in configs.items():
        print(f"pipeline: {name} ...")
        rt = Runtime(kb=kb, triage_agent=agent, guardrail=kw.pop("guardrail", Guardrail.load()), **kw)
        rows = run_pipeline(rt, test)
        res["pipeline"][name] = summarize(rows)
        # stress: supported faults whose fixes do not hold (tests validation + rollback)
        res["stress_persistent_faults"][name] = summarize(run_pipeline(rt, supported_test, resistant=True))
        if name == "full_system":
            full_rows = rows
            pd.DataFrame(full_rows).to_csv(RES / "pipeline_test_rows.csv", index=False)

    df = pd.DataFrame(full_rows)
    conf, corr = df.confidence.to_numpy(), df.diag_correct.to_numpy()
    res["calibration_test"] = {"ece": round(ece(conf, corr), 4), "reliability": reliability(conf, corr)}
    res["risk_coverage_test"] = risk_coverage(full_rows)
    print("retrieval ..."); res["retrieval_test"] = eval_retrieval(kb, test)
    print("log parsing ..."); res["log_parsing_drain"] = eval_parsing()
    print("mttr ..."); res["mttr"] = eval_mttr(full_rows)
    res["runtime_s"] = round(time.time() - t0, 1)
    RES.mkdir(exist_ok=True)
    (RES / "eval_results.json").write_text(json.dumps(res, indent=2, default=str))
    write_markdown(res)
    print(json.dumps({k: res["pipeline"]["full_system"][k] for k in
                      ["diagnosis_accuracy", "auto_resolution_rate_all", "false_action_rate",
                       "escalation_vs_should_escalate", "citation_faithfulness"]}, indent=2))
    print(f"done in {res['runtime_s']} s -> results/eval_results.json, results/eval_tables.md")
    return 0


def write_markdown(res: dict) -> None:
    L = ["# Evaluation tables (test split, generated by `python -m eval.run_all`)", ""]
    t = res["triage"]
    L += ["## Triage (measured)", "", "| model | n | accuracy | macro-F1 | majority baseline |", "|---|---|---|---|---|"]
    for k, v in t.items():
        L.append(f"| {k} | {v['n']} | {v['accuracy']} | {v['macro_f1']} | {v['majority_baseline_acc']} |")
    keys = ["diagnosis_accuracy", "diagnosis_accuracy_clear", "diagnosis_accuracy_ambiguous",
            "auto_resolution_rate_all", "auto_resolution_rate_supported", "verified_resolution_rate_all",
            "false_success_claims", "false_action_rate", "wrong_actions", "escalation_rate",
            "unsupported_escalated", "ambiguous_escalated", "citation_faithfulness", "trace_complete_rate"]
    for title, block in [("Pipeline on simulator test incidents (measured on simulator)", res["pipeline"]),
                         ("Stress set: supported faults whose fix does not hold (measured on simulator)",
                          res["stress_persistent_faults"])]:
        names = list(block)
        L += ["", f"## {title}", "", "| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(str(block[n][k]) for n in names) + " |")
        L.append("| escalation precision / recall | " + " | ".join(
            f"{block[n]['escalation_vs_should_escalate']['precision']} / {block[n]['escalation_vs_should_escalate']['recall']}"
            for n in names) + " |")
    r = res["retrieval_test"]
    L += ["", "## Retrieval (incident index, test queries)", "", "| mode | R@1 | R@3 | R@5 | R@10 | MRR@10 |", "|---|---|---|---|---|---|"]
    for m in ["bm25", "vector", "hybrid"]:
        L.append(f"| {m} | " + " | ".join(str(r[m][k]) for k in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"]) + " |")
    L += ["", r["note"], "", "## Drain log parsing (Loghub 2k samples, grouping accuracy)", "",
          "| system | lines | true templates | GA (no masking) | GA (generic masking, used) |", "|---|---|---|---|---|"]
    for s, v in res["log_parsing_drain"].items():
        L.append(f"| {s} | {v['lines']} | {v['true_templates']} | {v['ga_no_masking']} | {v['ga_generic_masking']} |")
    m = res["mttr"]
    L += ["", "## MTTR (SIMULATED)", "", "| | median h | mean h | p75 h | p90 h |", "|---|---|---|---|---|"]
    for k in ["manual_baseline", "agent_pipeline"]:
        L.append(f"| {k} | {m[k]['median_h']} | {m[k]['mean_h']} | {m[k]['p75_h']} | {m[k]['p90_h']} |")
    L += ["", m["method"], "", f"Calibration on test: ECE = {res['calibration_test']['ece']}"]
    (RES / "eval_tables.md").write_text("\n".join(L) + "\n", encoding="utf8")


if __name__ == "__main__":
    sys.exit(main())
