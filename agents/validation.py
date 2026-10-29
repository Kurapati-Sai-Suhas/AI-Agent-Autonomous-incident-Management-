"""Validation agent: verifies recovery from the simulator's health and fresh logs.

It does NOT look at the runbook's return value. "fixed" requires:
  * every health check passing, including the runbook's expected checks, and
  * no ERROR/CRIT lines in a fresh log window.
It probes up to `probes` times (a simple retry for slow recoveries).
"""
from __future__ import annotations

import re

from runbooks import get_runbook
from sim.service import FakeService

PROBE_SECONDS = 30
_BAD = re.compile(r"\b(ERROR|CRIT|FATAL)\b")


def validate(name: str, svc: FakeService, probes: int = 2) -> dict:
    rb = get_runbook(name)
    history = []
    for attempt in range(1, probes + 1):
        health = svc.health_check()
        logs = svc.get_logs(n=6)
        bad_lines = [l for l in logs if _BAD.search(l)]
        failing = [k for k, v in health["checks"].items() if not v]
        expected_failing = [k for k in rb.expected_checks if not health["checks"].get(k, False)]
        fixed = health["healthy"] and not bad_lines and not expected_failing
        history.append({"attempt": attempt, "healthy": health["healthy"], "failing_checks": failing,
                        "error_lines": bad_lines[:3]})
        if fixed:
            break
    return {"fixed": fixed, "checked": ["health_check (all checks)", f"expected: {rb.expected_checks}",
                                        "fresh log window for ERROR/CRIT lines"],
            "still_wrong": [] if fixed else (failing or ["error lines in logs"]),
            "probes": history, "seconds": PROBE_SECONDS * len(history),
            "evidence": [f"health:{svc.name}:probe{h['attempt']}" for h in history]}
