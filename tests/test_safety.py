"""Safety properties: whitelist, rejection of unknown actions, rollback, guardrail."""
import pytest

import executor as ex
from guardrail import Guardrail
from runbooks import RUNBOOKS, get_runbook
from sim.service import FAULTS, RESOLVES, WHITELIST, FakeService, UnknownActionError


def test_whitelist_matches_runbooks():
    assert set(WHITELIST) == set(RUNBOOKS)
    assert 10 <= len(RUNBOOKS) <= 15


@pytest.mark.parametrize("bad", ["rm -rf /", "shutdown", "format_disk", "restart_service; reboot", ""])
def test_unknown_actions_rejected_everywhere(bad):
    svc = FakeService(seed=1)
    svc.inject_fault("disk_full")
    with pytest.raises(UnknownActionError):
        svc.apply_fix(bad)
    with pytest.raises(KeyError):
        get_runbook(bad)
    out = ex.execute(bad, svc)
    assert out["status"] == "rejected"
    assert svc.action_history == []          # nothing ran
    assert svc.fault == "disk_full"


def test_health_reflects_state_not_return_value():
    for fault in FAULTS:
        svc = FakeService(seed=2)
        svc.inject_fault(fault)
        assert not svc.health_check()["healthy"]
        for action in WHITELIST:
            s2 = FakeService(seed=2)
            s2.inject_fault(fault)
            s2.apply_fix(action)
            assert s2.health_check()["healthy"] == (action in RESOLVES[fault])


def test_rollback_restores_config():
    svc = FakeService(seed=3)
    svc.inject_fault("db_pool_exhausted", resistant=True)
    before = svc.snapshot()
    out = ex.execute("increase_db_pool_size", svc)
    assert out["status"] == "executed" and svc.config["db_pool_size"] == 100
    ex.rollback("increase_db_pool_size", svc, out["snapshot"])
    assert svc.config == before


def _inputs(margin=0.9, top_sim=0.9, gap=0.6, top_score=0.95, direct=2):
    tri = {"fault": {"margin": margin}}
    diag = {"top_retrieval_similarity": top_sim, "score_gap": gap, "top_score": top_score,
            "top_direct_evidence": direct}
    return tri, diag


def test_confident_but_no_direct_log_evidence_escalates():
    g = Guardrail(None, None, use_calibration=False)
    tri, diag = _inputs(direct=0)
    out = g.decide(tri, diag, {"runbook": "reset_db_pool", "risk": "low"})
    assert out["decision"] == "escalate" and "insufficient_evidence" in out["reasons"]


def test_low_confidence_cannot_bypass_guardrail():
    g = Guardrail(None, {"margin": 0.2, "top_sim": 0.3, "gap": 0.1, "confidence": 0.6}, use_calibration=False)
    tri, diag = _inputs(margin=0.05, top_score=0.4, gap=0.02)
    low_risk_match = {"runbook": "reset_db_pool", "risk": "low"}
    out = g.decide(tri, diag, low_risk_match)
    assert out["decision"] == "escalate"
    assert {"triage_ambiguous", "conflicting_evidence", "low_confidence"} <= set(out["reasons"])


def test_high_risk_runbook_is_never_auto_executed():
    g = Guardrail(None, None, use_calibration=False)
    tri, diag = _inputs()
    out = g.decide(tri, diag, {"runbook": "rollback_last_deploy", "risk": "high"})
    assert out["decision"] == "escalate" and "action_unsafe" in out["reasons"]


def test_no_runbook_escalates_and_confident_low_risk_executes():
    g = Guardrail(None, None, use_calibration=False)
    tri, diag = _inputs()
    assert "no_matching_runbook" in g.decide(tri, diag, {"runbook": None, "risk": None})["reasons"]
    assert g.decide(tri, diag, {"runbook": "reset_db_pool", "risk": "low"})["decision"] == "execute"


def test_works_without_llm_key(monkeypatch):
    import llm
    monkeypatch.setenv("LLM_PROVIDER", "openai")     # provider set but no key -> offline
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert llm.provider() == "none" and llm.complete("hi") is None


def test_keep_cited_drops_uncited_and_unknown_ids():
    from agents.diagnosis import keep_cited
    allowed = {"INC:SIM-00001", "DEMO-1:log:2"}
    text = ("Pool exhausted [INC:SIM-00001]. The database is on fire. "
            "Also see [INC:SIM-99999]. Log shows it [DEMO-1:log:2].")
    out = keep_cited(text, allowed)
    assert "fire" not in out and "SIM-99999" not in out
    assert "[INC:SIM-00001]" in out and "[DEMO-1:log:2]" in out
