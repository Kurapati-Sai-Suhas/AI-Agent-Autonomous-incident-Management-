"""Executor: dry run first, then run ONE whitelisted runbook on the simulator.

Safety layers (any one is enough to block an action):
  1. the runbook name must be in the registry (runbooks.get_runbook)
  2. the action must be in the simulator whitelist (FakeService raises otherwise)
  3. preconditions are checked in the dry run
A config snapshot is taken before execution so rollback is always possible.
Nothing here runs a shell command or touches the real machine.
"""
from __future__ import annotations

from runbooks import get_runbook
from sim.service import WHITELIST, FakeService, UnknownActionError

# Simulated durations (seconds) used for the MTTR benchmark.
ACTION_SECONDS = {"restart_service": 90, "reset_db_pool": 30, "clear_disk_logs": 45,
                  "compress_old_logs": 120, "increase_db_pool_size": 60,
                  "kill_long_running_queries": 40, "drain_and_restart_node": 180,
                  "scale_out_replicas": 240, "flush_cache": 20, "enable_rate_limiting": 30,
                  "rollback_last_deploy": 600, "failover_db_to_replica": 300}
DRY_RUN_SECONDS, ROLLBACK_SECONDS = 5, 60


def dry_run(name: str, svc: FakeService) -> dict:
    if name not in WHITELIST:
        return {"ok": False, "reason": f"{name!r} is not a whitelisted action"}
    try:
        rb = get_runbook(name)
    except KeyError as e:
        return {"ok": False, "reason": str(e)}
    ok, why = rb.precondition(svc)
    return {"ok": ok, "reason": why, "would_run": f"{rb.name} on simulator '{svc.name}'",
            "risk": rb.risk, "rollback_available": True}


def execute(name: str, svc: FakeService) -> dict:
    """Run a runbook after a successful dry run. Never trusts the return value as success."""
    plan = dry_run(name, svc)
    if not plan["ok"]:
        return {"status": "rejected", "dry_run": plan, "snapshot": None}
    rb = get_runbook(name)
    snap = svc.snapshot()
    try:
        returned = rb.action(svc)
    except UnknownActionError as e:      # defence in depth
        return {"status": "rejected", "dry_run": plan, "error": str(e), "snapshot": snap}
    return {"status": "executed", "dry_run": plan, "snapshot": snap,
            "action_returned": returned, "seconds": DRY_RUN_SECONDS + ACTION_SECONDS[name]}


def rollback(name: str, svc: FakeService, snapshot: dict) -> dict:
    get_runbook(name).rollback(svc, snapshot)
    return {"status": "rolled_back", "restored_config": snapshot, "seconds": ROLLBACK_SECONDS}
