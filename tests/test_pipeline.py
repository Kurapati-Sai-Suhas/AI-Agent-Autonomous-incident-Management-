"""End-to-end acceptance tests on the simulator (the three demos + rollback path)."""
import pytest

from conftest import needs_artifacts
from demos.run_demos import ambiguous_incident, live_incident
from graph import run_incident
from schema import Incident
from sim.service import RESOLVES, SUPPORTED_FAULTS, FakeService

pytestmark = needs_artifacts
STAGES = ["ingestion", "triage", "diagnosis", "resolution", "guardrail"]


def _run(runtime, graph, fault, inc_id, resistant=False, seed=7):
    svc = FakeService(seed=seed)
    svc.inject_fault(fault, resistant=resistant)
    return run_incident(runtime, live_incident(svc, inc_id), svc, graph=graph), svc


def _stages(state):
    return {e["agent"] for e in state["trace"]}


@pytest.mark.parametrize("fault", SUPPORTED_FAULTS)
def test_demo1_supported_fault_is_fixed_and_verified(runtime, graph, fault):
    s, svc = _run(runtime, graph, fault, f"T1-{fault}")
    assert s["diagnosis"]["top_cause"] == fault
    assert s["status"] == "resolved"
    assert svc.health_check()["healthy"]                      # simulator really recovered
    assert all(a["action"] in RESOLVES[fault] for a in svc.action_history)
    assert s["validation"]["fixed"] and not s["validation"].get("skipped")
    assert set(STAGES) <= _stages(s) and "validation" in _stages(s)


@pytest.mark.parametrize("fault", ["network_partition", "cert_expired"])
def test_demo2_unsupported_fault_escalates_without_inventing_fix(runtime, graph, fault):
    s, svc = _run(runtime, graph, fault, f"T2-{fault}")
    assert s["status"] == "awaiting_human"
    assert svc.action_history == []                           # nothing executed
    h = s["handoff"]
    for key in ["incident_id", "summary", "stuck_at", "per_agent_confidence", "why_escalated",
                "hypotheses", "actions_taken", "start_here", "evidence"]:
        assert key in h
    assert h["why_escalated"]["reasons"] and h["start_here"]
    assert set(STAGES) <= _stages(s) and "escalation" in _stages(s)


def test_demo3_ambiguous_incident_triggers_guardrail(runtime, graph):
    svc = FakeService(seed=303)
    svc.inject_fault("db_pool_exhausted")
    s = run_incident(runtime, ambiguous_incident(svc), svc, graph=graph)
    assert s["status"] == "awaiting_human"
    assert svc.action_history == []
    assert s["guardrail"]["decision"] == "escalate" and s["guardrail"]["reasons"]
    h = s["handoff"]
    assert h["stuck_at"] is not None and h["evidence"] and h["start_here"]


def test_failed_validation_rolls_back_and_escalates(runtime, graph):
    s, svc = _run(runtime, graph, "db_pool_exhausted", "T4", resistant=True)
    agents = [(e["agent"], e["action"]) for e in s["trace"]]
    assert any(a == "validation" for a, _ in agents)
    assert any(a == "executor" and act.startswith("rollback") for a, act in agents)
    assert "fix_not_verified" in s["escalation_reasons"]
    assert s["status"] == "awaiting_human"
    assert s["handoff"]["stuck_at"]["agent"] == "validation"
    assert svc.config == FakeService.DEFAULT_CONFIG           # rollback restored config


def test_citations_refer_to_real_records(runtime, graph):
    s, _ = _run(runtime, graph, "network_partition", "T5")
    inc = s["incident"]
    own = {f"{inc['id']}:log:{i}" for i in range(len(inc["logs"]))}
    retrieved = {h["id"] for g in s["diagnosis"]["retrieved"].values() for h in g}
    cited = [e for h in s["diagnosis"]["hypotheses"] for e in h["supporting_evidence"]]
    cited += [e["id"] for e in s["handoff"]["evidence"]]
    import re
    cited += re.findall(r"\[([^\]\s]+)\]", " ".join(s["handoff"]["start_here"]))   # every [x] must be an id
    assert cited
    for c in cited:
        assert c in own or (c in retrieved and runtime.kb.get(c) is not None), c


def test_human_in_the_loop_interrupt_and_learning(runtime, tmp_path, monkeypatch):
    from graph import build_graph
    from rag import store
    monkeypatch.setattr(store, "LEARNED", tmp_path / "learned.jsonl")
    g = build_graph(runtime, hitl=True)
    svc = FakeService(seed=9)
    svc.inject_fault("cert_expired")
    inc = Incident(id="T6", source="alert", text=svc.alert_text(), logs=svc.get_logs())
    cfg = {"configurable": {"thread_id": "T6"}}
    s = run_incident(runtime, inc, svc, graph=g, thread_id="T6")
    assert g.get_state(cfg).next == ("human_review",)          # paused for the human
    g.update_state(cfg, {"human_resolution": {"root_cause": "cert_expired", "fix": "manual cert rotation"}})
    final = g.invoke(None, cfg)
    assert final["status"] == "resolved_by_human"
    learned = store.load_learned()
    assert learned and learned[0]["id"] == "LEARNED:T6"
    runtime.services.pop("T6", None)
