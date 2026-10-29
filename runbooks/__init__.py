"""Hand-written runbooks. Each one acts ONLY on the FakeService simulator.

A runbook = name, description, causes it addresses, preconditions, risk level,
action function, rollback function and expected effect. The Resolution agent can
only choose from RUNBOOKS; it never writes commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sim.service import WHITELIST, FakeService

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class Runbook:
    name: str
    description: str
    handles: list[str]                         # root causes this runbook addresses
    risk: str                                  # low | medium | high
    expected_effect: str
    expected_checks: list[str]                 # health_check keys that must be True afterwards
    precondition: Callable[[FakeService], tuple[bool, str]] = field(
        default=lambda s: (True, "no precondition"))

    def action(self, svc: FakeService) -> bool:
        return svc.apply_fix(self.name)        # whitelisted action on the simulator only

    def rollback(self, svc: FakeService, snapshot: dict) -> None:
        svc.restore(snapshot)                  # undo config side effects

    def to_dict(self) -> dict:
        return {"id": f"RB:{self.name}", "name": self.name, "description": self.description,
                "handles": self.handles, "risk": self.risk, "expected_effect": self.expected_effect,
                "expected_checks": self.expected_checks}


def _pool_below_max(s):
    return (s.config["db_pool_size"] < 200, f"db_pool_size={s.config['db_pool_size']} (max 200)")


def _has_replica_capacity(s):
    return (s.config["replicas"] < 6, f"replicas={s.config['replicas']} (max 6)")


def _retention_reducible(s):
    return (s.config["log_retention_days"] > 3, f"log_retention_days={s.config['log_retention_days']}")


def _on_primary(s):
    return (s.config["db_primary"].startswith("db-primary"), f"db_primary={s.config['db_primary']}")


RUNBOOKS: dict[str, Runbook] = {rb.name: rb for rb in [
    Runbook("restart_service", "Graceful restart of the checkout-api process.",
            ["memory_leak", "service_crash"], "low",
            "process up, heap back to baseline", ["process_up", "heap_ok", "error_rate_ok"]),
    Runbook("reset_db_pool", "Drop and re-create the DB connection pool.",
            ["db_pool_exhausted"], "low",
            "db pool usage below 90%", ["db_pool_ok", "error_rate_ok"]),
    Runbook("clear_disk_logs", "Delete rotated application logs older than 1 day on /var/log/checkout.",
            ["disk_full"], "low", "disk usage below 90%", ["disk_ok", "error_rate_ok"]),
    Runbook("compress_old_logs", "Halve log retention and compress archived logs.",
            ["disk_full"], "low", "disk usage below 90%", ["disk_ok", "error_rate_ok"],
            _retention_reducible),
    Runbook("increase_db_pool_size", "Double the DB pool size (max 200).",
            ["db_pool_exhausted"], "medium", "db pool usage below 90%",
            ["db_pool_ok", "error_rate_ok"], _pool_below_max),
    Runbook("kill_long_running_queries", "Terminate queries running longer than 5 minutes.",
            ["db_pool_exhausted"], "medium", "connections released",
            ["db_pool_ok", "error_rate_ok"]),
    Runbook("drain_and_restart_node", "Drain traffic from the node, restart, re-add.",
            ["service_crash", "memory_leak"], "medium", "process up and serving",
            ["process_up", "heap_ok", "error_rate_ok"]),
    Runbook("scale_out_replicas", "Add one replica (mitigation for load/memory pressure).",
            ["memory_leak", "high_load"], "medium", "load spread over more replicas",
            ["heap_ok", "error_rate_ok"], _has_replica_capacity),
    Runbook("flush_cache", "Flush and re-enable the application cache.",
            ["stale_cache"], "low", "cache consistent", ["error_rate_ok"]),
    Runbook("enable_rate_limiting", "Enable a 200 rps rate limit on checkout (mitigation).",
            ["high_load", "db_pool_exhausted"], "medium", "load reduced",
            ["error_rate_ok"]),
    Runbook("rollback_last_deploy", "Roll back to the previous release.",
            ["service_crash", "memory_leak"], "high", "previous version running",
            ["process_up", "heap_ok", "error_rate_ok"]),
    Runbook("failover_db_to_replica", "Promote the DB replica to primary.",
            ["db_pool_exhausted"], "high", "traffic served by replica",
            ["db_pool_ok", "error_rate_ok"], _on_primary),
]}

assert set(RUNBOOKS) == set(WHITELIST), "runbook registry and simulator whitelist must match"


def get_runbook(name: str) -> Runbook:
    """Look up a runbook; unknown names are rejected (never executed)."""
    if name not in RUNBOOKS:
        raise KeyError(f"no runbook named {name!r} (not whitelisted)")
    return RUNBOOKS[name]
