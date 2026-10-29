"""Shared data structures: the Incident object and the explanation-trace entry."""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any

# Trace outcomes. The "stuck point" is the first trace entry whose outcome is not "ok".
OK, LOW_CONFIDENCE, FAILED, UNSAFE = "ok", "low_confidence", "failed", "unsafe"
OUTCOMES = (OK, LOW_CONFIDENCE, FAILED, UNSAFE)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


@dataclass
class Incident:
    """Normalized incident (an alert, a ticket or a group of log lines)."""
    id: str
    source: str                      # "alert" | "ticket" | "log"
    text: str
    logs: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=now_iso)
    category: str | None = None
    priority: str | None = None
    status: str = "open"

    def to_dict(self) -> dict:
        return asdict(self)

    def log_ids(self) -> list[str]:
        """Stable ids for the incident's own log lines, so they can be cited."""
        return [f"{self.id}:log:{i}" for i in range(len(self.logs))]


@dataclass
class TraceEntry:
    """One step of the explanation trace."""
    agent: str
    action: str
    evidence: list[str]
    confidence: float | None
    outcome: str
    detail: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=now_iso)

    def __post_init__(self):
        if self.outcome not in OUTCOMES:
            raise ValueError(f"bad outcome {self.outcome!r}; expected one of {OUTCOMES}")

    def to_dict(self) -> dict:
        return asdict(self)


def stuck_point(trace: list[dict]) -> dict | None:
    """First trace entry whose outcome is not ok (None if every step was ok)."""
    for entry in trace:
        if entry["outcome"] != OK:
            return entry
    return None
