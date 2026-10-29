"""Resolution agent: maps the best hypothesis to ONE whitelisted runbook, or none.

It never writes commands. It records why the chosen runbook matched and why every
other candidate was rejected. If nothing matches it reports no_matching_runbook.
"""
from __future__ import annotations

from runbooks import RISK_ORDER, RUNBOOKS
from sim.service import FakeService


def select_runbook(cause: str, svc: FakeService | None) -> dict:
    candidates, rejected = [], []
    for rb in RUNBOOKS.values():
        if cause not in rb.handles:
            continue
        ok, why = rb.precondition(svc) if svc is not None else (True, "no service state")
        if not ok:
            rejected.append({"runbook": rb.name, "reason": f"precondition failed: {why}"})
            continue
        candidates.append(rb)
    candidates.sort(key=lambda rb: (RISK_ORDER[rb.risk], rb.name))
    if not candidates:
        return {"runbook": None, "status": "no_matching_runbook", "risk": None,
                "matched_because": f"no whitelisted runbook handles '{cause}'",
                "rejected": rejected, "evidence": []}
    best, rest = candidates[0], candidates[1:]
    rejected += [{"runbook": rb.name, "reason": f"higher or equal risk ({rb.risk}) than {best.name} ({best.risk})"}
                 for rb in rest]
    return {"runbook": best.name, "status": "matched", "risk": best.risk,
            "matched_because": f"'{cause}' is listed in {best.name}.handles; lowest-risk candidate "
                               f"whose preconditions hold",
            "expected_effect": best.expected_effect, "rejected": rejected,
            "evidence": [f"RB:{best.name}"]}
