"""Escalation agent: builds the structured human handoff report FROM THE TRACE.

Nothing in the report is invented: every field is copied or derived from the trace
and the pipeline state. An optional LLM may re-word `why_escalated` only; its output
is kept only if every id it cites exists in the evidence (see diagnosis.keep_cited).
"""
from __future__ import annotations

import llm
from agents.diagnosis import keep_cited
from schema import stuck_point

REASON_TEXT = {
    "triage_ambiguous": "the triage model could not clearly separate the top two fault categories",
    "low_retrieval_support": "no sufficiently similar resolved incident was found in the knowledge base",
    "conflicting_evidence": "the top two root-cause hypotheses scored too close to each other",
    "low_confidence": "the calibrated confidence that the diagnosis is correct is below the threshold",
    "insufficient_evidence": "no log line of this incident directly supports the most likely cause",
    "no_matching_runbook": "no whitelisted runbook addresses the most likely cause",
    "action_unsafe": "the only matching runbook is too risky to run without human approval",
    "fix_not_verified": "a runbook was executed but validation failed, so it was rolled back",
}

CAUSE_CHECKS = {
    "db_pool_exhausted": ["Check DB pool metrics (active / max / waiting) for checkout-api",
                          "Look for long-running queries or connection leaks on the primary DB"],
    "disk_full": ["Check /var usage on the checkout host and the largest log directories",
                  "Confirm log rotation is working"],
    "memory_leak": ["Check heap / RSS trend of checkout-api since the last deploy",
                    "Capture a heap dump before restarting if the leak repeats"],
    "service_crash": ["Read the last crash reason (exit code / signal) of checkout.service",
                      "Check restart count and recent deploys"],
    "network_partition": ["Verify connectivity from checkout to payments-svc (ping / traceroute)",
                          "Check recent network, firewall or cluster-membership changes",
                          "No automated runbook exists for this cause: involve the network team"],
    "cert_expired": ["Inspect the certificate served by payments.internal (expiry, CN)",
                     "Rotate the certificate through the PKI process (manual, not automated)"],
}


def build_handoff(state: dict) -> dict:
    trace = state["trace"]
    inc = state["incident"]
    triage, diag = state.get("triage") or {}, state.get("diagnosis") or {}
    res, guard = state.get("resolution") or {}, state.get("guardrail") or {}
    reasons = list(state.get("escalation_reasons") or [])
    stuck = stuck_point(trace)
    hyps = diag.get("hypotheses", [])

    start_here = []
    if "fix_not_verified" in reasons:
        v = state.get("validation") or {}
        start_here.append(f"Runbook '{res.get('runbook')}' ran and was rolled back; still failing: "
                          f"{', '.join(v.get('still_wrong', [])) or 'unknown'} - verify the diagnosis first")
    if "triage_ambiguous" in reasons or "conflicting_evidence" in reasons:
        if len(hyps) >= 2:
            start_here.append(f"Decide between '{hyps[0]['cause']}' and '{hyps[1]['cause']}': "
                              f"compare their evidence {_cites(hyps[0])} vs {_cites(hyps[1])}")
    if "low_retrieval_support" in reasons:
        start_here.append("No close past incident: treat as new and check service dashboards first")
    if "insufficient_evidence" in reasons and hyps:
        start_here.append(f"Logs do not directly show '{hyps[0]['cause']}': pull a wider log window and "
                          f"the service metrics before acting")
    if "action_unsafe" in reasons:
        start_here.append(f"Runbook '{res.get('runbook')}' is {res.get('risk')}-risk: approve or reject it manually")
    # cause-specific checks for the top hypothesis and any close runner-up
    for h in [x for x in hyps[:2] if hyps and x["score"] >= 0.25 * hyps[0]["score"]]:
        for check in CAUSE_CHECKS.get(h["cause"], []):
            # square brackets are reserved for evidence ids, so the cause label uses parentheses
            start_here.append(f"({h['cause']}) {check} {_cites(h)}".strip())
    start_here = list(dict.fromkeys(start_here))[:7]

    actions = [{"agent": e["agent"], "action": e["action"], "outcome": e["outcome"],
                "detail": {k: v for k, v in e["detail"].items() if k in
                           ("runbook", "status", "fixed", "still_wrong", "restored_config")}}
               for e in trace if e["agent"] in ("executor", "validation")]
    evidence = []
    for group in ("incidents", "logs", "runbooks"):
        for h in (diag.get("retrieved") or {}).get(group, []):
            evidence.append({"id": h["id"], "type": group, "similarity": h["similarity"],
                             "label": h["meta"].get("fault") or h["meta"].get("queue") or h["meta"].get("service"),
                             "snippet": h.get("snippet", "")[:160]})

    why = "; ".join(REASON_TEXT.get(r, r) for r in reasons) or "escalated by request"
    report = {
        "incident_id": inc["id"],
        "summary": f"{inc['text'][:160]} | predicted fault: {triage.get('fault', {}).get('label')} "
                   f"| queue: {triage.get('queue', {}).get('label')} | priority: {triage.get('priority', {}).get('label')}",
        "stuck_at": None if stuck is None else {"agent": stuck["agent"], "action": stuck["action"],
                                                "outcome": stuck["outcome"],
                                                "reason": stuck["detail"].get("reason", stuck["outcome"])},
        "per_agent_confidence": {
            "triage_fault_confidence": triage.get("fault", {}).get("confidence"),
            "triage_fault_margin": triage.get("fault", {}).get("margin"),
            "diagnosis_top_score": diag.get("top_score"),
            "diagnosis_score_gap": diag.get("score_gap"),
            "retrieval_top_similarity": diag.get("top_retrieval_similarity"),
            "guardrail_calibrated_confidence": guard.get("confidence"),
        },
        "why_escalated": {"reasons": reasons, "text": why},
        "hypotheses": [{"cause": h["cause"], "score": h["score"], "evidence": h["supporting_evidence"],
                        "missing_or_contradicting": h["missing_or_contradicting"]} for h in hyps],
        "actions_taken": actions,
        "evidence": evidence,
        "incident_log_lines": [{"id": i, "line": l} for i, l in zip(_log_ids(inc), inc["logs"])],
        "start_here": start_here,
        "worded_by": "template",
    }
    if llm.available():
        allowed = {e["id"] for e in evidence} | set(_log_ids(inc))
        text = llm.complete("Explain in 2-3 sentences why this incident was escalated, citing evidence ids "
                            f"in [brackets] exactly as given. Facts: reasons={reasons}; stuck_at={report['stuck_at']}; "
                            f"hypotheses={report['hypotheses'][:2]}",
                            system="Restate only the given facts. Never invent ids or actions.")
        cited = keep_cited(text or "", allowed)
        if cited:
            report["why_escalated"]["text_llm"] = cited
            report["worded_by"] = "llm (citation-checked)"
    return report


def _cites(h: dict) -> str:
    return " ".join(f"[{e}]" for e in h["supporting_evidence"][:2])


def _log_ids(inc: dict) -> list[str]:
    return [f"{inc['id']}:log:{i}" for i in range(len(inc["logs"]))]
