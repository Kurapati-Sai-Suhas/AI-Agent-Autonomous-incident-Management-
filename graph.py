r"""LangGraph wiring of the full pipeline.

ingest -> triage -> diagnosis -> resolution -> guardrail
   guardrail --execute--> executor --executed--> validation --fixed--> close -> END
                                    \--rejected--> escalate        \--not fixed--> rollback -> escalate
   guardrail --escalate--> escalate -> human_review (HITL interrupt) -> END

Shared state holds the incident, trace, triage, hypotheses, per-stage confidence,
selected runbook, risk, escalation reasons and simulated elapsed time. The
simulator object and the knowledge base live in the Runtime (not in the state),
so the state stays JSON-serialisable for the LangGraph checkpointer.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import executor as ex
from agents import diagnosis as dg
from agents import escalation as es
from agents import validation as va
from agents.resolution import select_runbook
from agents.triage import TriageAgent
from guardrail import Guardrail
from schema import FAILED, LOW_CONFIDENCE, OK, UNSAFE, Incident, TraceEntry
from sim.service import FakeService

# simulated seconds per agent step (MTTR benchmark)
STEP_SECONDS = {"ingest": 1, "triage": 2, "diagnosis": 8, "resolution": 1, "guardrail": 1,
                "escalate": 5}


class State(TypedDict, total=False):
    incident: dict
    trace: Annotated[list, operator.add]
    elapsed_s: Annotated[float, operator.add]
    triage: dict
    diagnosis: dict
    resolution: dict
    guardrail: dict
    execution: dict
    validation: dict
    escalation_reasons: Annotated[list, operator.add]
    handoff: dict
    human_resolution: dict
    status: str


@dataclass
class Runtime:
    """Everything the nodes need that is not plain data."""
    kb: Any
    triage_agent: TriageAgent
    guardrail: Guardrail
    services: dict[str, FakeService] = field(default_factory=dict)
    use_rag: bool = True
    use_validation: bool = True


def _t(agent, action, evidence, confidence, outcome, **detail) -> list[dict]:
    return [TraceEntry(agent, action, list(evidence), confidence, outcome, detail).to_dict()]


def make_nodes(rt: Runtime) -> dict:
    th = rt.guardrail.thresholds

    def ingest(s: State):
        inc = s["incident"]
        missing = [k for k in ("id", "text", "logs") if k not in inc]
        if missing:
            raise ValueError(f"incident missing fields {missing}")
        return {"trace": _t("ingestion", "normalize incident", [inc["id"]], None, OK,
                            source=inc.get("source"), n_log_lines=len(inc["logs"])),
                "elapsed_s": STEP_SECONDS["ingest"], "status": "triaging"}

    def triage(s: State):
        inc = s["incident"]
        out = rt.triage_agent.run(inc["text"], inc["logs"])
        f = out["fault"]
        outcome = OK if f["margin"] >= th["margin"] else LOW_CONFIDENCE
        return {"triage": out, "elapsed_s": STEP_SECONDS["triage"],
                "trace": _t("triage", f"predict fault={f['label']} queue={out['queue']['label']} "
                            f"priority={out['priority']['label']}", f["evidence_words"], f["confidence"],
                            outcome, margin=f["margin"], runner_up=f["runner_up"],
                            reason=None if outcome == OK else
                            f"fault margin {f['margin']:.2f} < {th['margin']:.2f} ({f['label']} vs {f['runner_up']})")}

    def diagnosis(s: State):
        inc = Incident(**s["incident"])
        out = dg.diagnose(inc, s["triage"], rt.kb, use_rag=rt.use_rag)
        weak = []
        if rt.use_rag and out["top_retrieval_similarity"] < th["top_sim"]:
            weak.append(f"top similarity {out['top_retrieval_similarity']:.2f} < {th['top_sim']:.2f}")
        if out["score_gap"] < th["gap"]:
            weak.append(f"hypothesis gap {out['score_gap']:.2f} < {th['gap']:.2f}")
        if out["top_direct_evidence"] == 0:
            weak.append(f"no incident log line directly supports {out['top_cause']}")
        top = out["hypotheses"][0]
        return {"diagnosis": out, "elapsed_s": STEP_SECONDS["diagnosis"],
                "trace": _t("diagnosis", f"rank hypotheses; top={top['cause']}", top["supporting_evidence"],
                            top["score"], LOW_CONFIDENCE if weak else OK,
                            hypotheses=[(h["cause"], h["score"]) for h in out["hypotheses"]],
                            reason="; ".join(weak) or None)}

    def resolution(s: State):
        cause = s["diagnosis"]["top_cause"]
        out = select_runbook(cause, rt.services.get(s["incident"]["id"]))
        if out["runbook"] is None:
            outcome, reason = FAILED, out["matched_because"]
        elif out["risk"] == "high":
            outcome, reason = UNSAFE, f"{out['runbook']} is high risk"
        else:
            outcome, reason = OK, None
        return {"resolution": out, "elapsed_s": STEP_SECONDS["resolution"],
                "trace": _t("resolution", f"select runbook -> {out['runbook'] or out['status']}",
                            out["evidence"], None, outcome, runbook=out["runbook"], risk=out["risk"],
                            rejected=out["rejected"], reason=reason)}

    def guardrail(s: State):
        out = rt.guardrail.decide(s["triage"], s["diagnosis"], s["resolution"])
        if out["decision"] == "execute":
            outcome = OK
        else:
            outcome = UNSAFE if "action_unsafe" in out["reasons"] else LOW_CONFIDENCE
        return {"guardrail": out, "elapsed_s": STEP_SECONDS["guardrail"],
                "escalation_reasons": out["reasons"],
                "trace": _t("guardrail", f"decision={out['decision']}", list(out["signals"]),
                            out["confidence"], outcome, reasons=out["reasons"],
                            reason=", ".join(out["reasons"]) or None)}

    def executor(s: State):
        name = s["resolution"]["runbook"]
        svc = rt.services[s["incident"]["id"]]
        out = ex.execute(name, svc)
        ok = out["status"] == "executed"
        return {"execution": {k: v for k, v in out.items()}, "elapsed_s": out.get("seconds", ex.DRY_RUN_SECONDS),
                "escalation_reasons": [] if ok else ["action_rejected"],
                "trace": _t("executor", f"dry-run then run {name} on simulator", [f"RB:{name}"], None,
                            OK if ok else FAILED, runbook=name, status=out["status"],
                            dry_run=out["dry_run"], reason=None if ok else out["dry_run"].get("reason"))}

    def validation(s: State):
        name = s["resolution"]["runbook"]
        svc = rt.services[s["incident"]["id"]]
        if not rt.use_validation:   # ablation: trust the action blindly
            return {"validation": {"fixed": True, "skipped": True}, "elapsed_s": 0,
                    "trace": _t("validation", "SKIPPED (ablation)", [], None, OK, fixed=True)}
        out = va.validate(name, svc)
        return {"validation": out, "elapsed_s": out["seconds"],
                "escalation_reasons": [] if out["fixed"] else ["fix_not_verified"],
                "trace": _t("validation", "re-check health and logs", out["evidence"], None,
                            OK if out["fixed"] else FAILED, fixed=out["fixed"],
                            still_wrong=out["still_wrong"],
                            reason=None if out["fixed"] else f"still failing: {out['still_wrong']}")}

    def rollback(s: State):
        name = s["resolution"]["runbook"]
        out = ex.rollback(name, rt.services[s["incident"]["id"]], s["execution"]["snapshot"])
        return {"elapsed_s": out["seconds"],
                "trace": _t("executor", f"rollback {name}", [f"RB:{name}"], None, OK,
                            runbook=name, status="rolled_back", restored_config=out["restored_config"])}

    def escalate(s: State):
        report = es.build_handoff(dict(s))
        return {"handoff": report, "status": "escalated", "elapsed_s": STEP_SECONDS["escalate"],
                "trace": _t("escalation", "build handoff report",
                            [e["id"] for e in report["evidence"][:5]], None, OK,
                            stuck_at=report["stuck_at"], reasons=report["why_escalated"]["reasons"])}

    def human_review(s: State):
        hr = s.get("human_resolution")
        if not hr:
            return {"status": "awaiting_human"}
        import learning
        case = learning.record_human_resolution(s["incident"], hr["root_cause"], hr["fix"], hr.get("notes", ""))
        return {"status": "resolved_by_human",
                "trace": _t("human", f"resolved: {hr['root_cause']} via {hr['fix']}", [case["id"]], None, OK,
                            learned_case=case["id"])}

    def close(s: State):
        return {"status": "resolved",
                "trace": _t("close", "incident closed (verified fixed)", [], None, OK,
                            mttr_s=s.get("elapsed_s", 0))}

    return {"ingest": ingest, "triage": triage, "diagnosis": diagnosis, "resolution": resolution,
            "guardrail": guardrail, "executor": executor, "validation": validation,
            "rollback": rollback, "escalate": escalate, "human_review": human_review, "close": close}


# ------------------------------------------------------------------ routing
def after_guardrail(s): return "executor" if s["guardrail"]["decision"] == "execute" else "escalate"
def after_executor(s): return "validation" if s["execution"]["status"] == "executed" else "escalate"
def after_validation(s): return "close" if s["validation"]["fixed"] else "rollback"


def build_graph(rt: Runtime, hitl: bool = False):
    from langgraph.graph import END, START, StateGraph
    nodes = make_nodes(rt)
    g = StateGraph(State)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    for a, b in [("ingest", "triage"), ("triage", "diagnosis"), ("diagnosis", "resolution"),
                 ("resolution", "guardrail"), ("rollback", "escalate"), ("escalate", "human_review"),
                 ("human_review", END), ("close", END)]:
        g.add_edge(a, b)
    g.add_conditional_edges("guardrail", after_guardrail, ["executor", "escalate"])
    g.add_conditional_edges("executor", after_executor, ["validation", "escalate"])
    g.add_conditional_edges("validation", after_validation, ["close", "rollback"])
    if hitl:
        from langgraph.checkpoint.memory import InMemorySaver
        return g.compile(checkpointer=InMemorySaver(), interrupt_before=["human_review"])
    return g.compile()


def run_incident(rt: Runtime, inc: Incident, svc: FakeService | None, graph=None,
                 thread_id: str | None = None) -> dict:
    """Run one incident end to end. Returns the final state."""
    if svc is not None:
        rt.services[inc.id] = svc
    graph = graph or build_graph(rt)
    cfg = {"configurable": {"thread_id": thread_id or inc.id}}
    try:
        return graph.invoke({"incident": inc.to_dict(), "trace": [], "elapsed_s": 0.0,
                             "escalation_reasons": []}, cfg)
    finally:
        if thread_id is None:
            rt.services.pop(inc.id, None)


def load_runtime(live: bool = False, **kw) -> Runtime:
    from rag.store import load_frozen, load_live
    return Runtime(kb=load_live() if live else load_frozen(), triage_agent=TriageAgent(),
                   guardrail=Guardrail.load(), **kw)
