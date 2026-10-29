"""FakeService: a simulated checkout API used for every remediation in this project.

Nothing in this module touches the real machine. "Actions" only change fields of a
Python object. Only actions in WHITELIST are accepted; anything else raises
UnknownActionError.

Health is derived from the service state (active fault + config), NOT from the
return value of apply_fix, so the Validation agent can genuinely verify recovery.
"""
from __future__ import annotations

import copy
import datetime as dt
import random

SUPPORTED_FAULTS = ["db_pool_exhausted", "disk_full", "memory_leak", "service_crash"]
UNSUPPORTED_FAULTS = ["network_partition", "cert_expired"]   # must be escalated
FAULTS = SUPPORTED_FAULTS + UNSUPPORTED_FAULTS

# Every action the simulator understands (the runbook whitelist).
WHITELIST = [
    "restart_service", "reset_db_pool", "clear_disk_logs",            # original three
    "compress_old_logs", "increase_db_pool_size", "kill_long_running_queries",
    "drain_and_restart_node", "scale_out_replicas", "flush_cache",
    "enable_rate_limiting", "rollback_last_deploy", "failover_db_to_replica",
]

# Ground truth: which actions actually clear which fault inside the simulator.
RESOLVES = {
    "db_pool_exhausted": {"reset_db_pool", "increase_db_pool_size",
                          "kill_long_running_queries", "failover_db_to_replica"},
    "disk_full": {"clear_disk_logs", "compress_old_logs"},
    "memory_leak": {"restart_service", "drain_and_restart_node", "rollback_last_deploy"},
    "service_crash": {"restart_service", "drain_and_restart_node", "rollback_last_deploy"},
    "network_partition": set(),
    "cert_expired": set(),
}
# Canonical (lowest-risk) fix stored as the incident label.
CORRECT_FIX = {"db_pool_exhausted": "reset_db_pool", "disk_full": "clear_disk_logs",
               "memory_leak": "restart_service", "service_crash": "restart_service",
               "network_partition": None, "cert_expired": None}


class UnknownActionError(ValueError):
    """Raised when an action outside the whitelist is requested."""


# ---------------------------------------------------------------- log vocabulary
# Strong clues identify one fault. Weak clues are shared between faults (ambiguity).
STRONG = {
    "db_pool_exhausted": [
        "ERROR db.pool: connection pool exhausted (active={n}/{n}, waiting={w})",
        "ERROR checkout.repo: HikariPool-1 - Connection is not available, request timed out after 30000ms",
        "WARN db.pool: max_connections reached for user checkout_svc on db-primary-{h}",
    ],
    "disk_full": [
        "ERROR fs: No space left on device (errno=28) writing /var/log/checkout/app.log",
        "CRIT disk.monitor: /var usage at {p}% on host chk-{h} (threshold 90%)",
        "ERROR checkout.audit: failed to write audit record: disk quota exceeded",
    ],
    "memory_leak": [
        "WARN jvm.gc: GC overhead limit approaching, heap used {p}%",
        "ERROR checkout.api: java.lang.OutOfMemoryError: Java heap space",
        "WARN mem.monitor: RSS of checkout-api grew from {a}MB to {b}MB in 30 min",
    ],
    "service_crash": [
        "CRIT systemd: checkout.service: Main process exited, code=killed, status=11/SEGV",
        "ERROR healthcheck: checkout-api not responding on :8080 (connection refused)",
        "WARN supervisor: checkout-api restarted {w} times in 5 minutes",
    ],
    "network_partition": [
        "ERROR net: no route to host 10.0.{h}.{w} (payments-svc)",
        "WARN raft: lost quorum, peers unreachable: [node-2, node-3]",
        "ERROR checkout.client: dial tcp 10.0.{h}.{w}:443: i/o timeout",
    ],
    "cert_expired": [
        "ERROR tls: x509: certificate has expired or is not yet valid (CN=payments.internal)",
        "ERROR checkout.client: SSL handshake failed with payments.internal: certificate expired",
        "WARN cert.monitor: certificate payments.internal expired {w} days ago",
    ],
}
W_TIMEOUT = "ERROR checkout.api: request timed out after {t}ms"
W_SLOW = "WARN checkout.api: upstream call slow (latency_ms={t})"
W_502 = "ERROR lb: upstream checkout-api returned 502 Bad Gateway"
W_PERSIST = "ERROR checkout.api: internal error while persisting order {o}"
W_P99 = "WARN slo: checkout p99 latency above objective ({t} ms)"
W_PAYMENTS = "ERROR checkout.client: call to payments.internal failed"
WEAK = {
    "db_pool_exhausted": [W_TIMEOUT, W_SLOW],
    "disk_full": [W_PERSIST, W_P99],
    "memory_leak": [W_SLOW, W_P99],
    "service_crash": [W_502, W_PERSIST],
    "network_partition": [W_TIMEOUT, W_502, W_PAYMENTS],
    "cert_expired": [W_PAYMENTS],
}
NOISE = [
    "INFO checkout.api: GET /health 200 {s}ms",
    "INFO checkout.api: POST /orders 201 {t}ms",
    "DEBUG cache: hit ratio 0.9{s}",
    "INFO scheduler: job cleanup_sessions finished in {s}s",
    "INFO checkout.api: GET /cart/{o} 200 {s}ms",
    "INFO auth: token refreshed for client web-frontend",
    "DEBUG metrics: exported 42 series to prometheus",
    "INFO deploy: config reload complete (version v1.4.2)",
]
HEALTHY_AFTER_FIX = [
    "INFO checkout.api: GET /health 200 {s}ms",
    "INFO checkout.api: POST /orders 201 {t}ms",
    "INFO checkout.api: all dependency checks passing",
]


class FakeService:
    """A fake checkout API. Holds an active fault and a small config dict."""

    DEFAULT_CONFIG = {"db_pool_size": 50, "replicas": 2, "log_retention_days": 14,
                      "cache_enabled": True, "rate_limit_rps": None,
                      "deployed_version": "v1.4.2", "db_primary": "db-primary-1"}

    def __init__(self, seed: int | None = None, name: str = "checkout-api"):
        self.name = name
        self.rng = random.Random(seed)
        self.fault: str | None = None
        self.resistant = False          # if True, fixes "apply" but the fault comes back
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.action_history: list[dict] = []
        self.clock = dt.datetime(2026, 1, 1, 12, 0, 0)

    # ------------------------------------------------------------ fault control
    def inject_fault(self, fault: str, resistant: bool = False) -> None:
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}")
        self.fault, self.resistant = fault, resistant

    def alert_text(self) -> str:
        if self.fault is None:
            return "No active alerts for checkout-api"
        return {
            "db_pool_exhausted": "Checkout API errors: database connections unavailable",
            "disk_full": "Disk usage critical on checkout host",
            "memory_leak": "Checkout API memory usage growing, latency rising",
            "service_crash": "Checkout API down: health check failing",
            "network_partition": "Checkout cannot reach payments service",
            "cert_expired": "TLS errors calling payments service",
        }[self.fault]

    # ------------------------------------------------------------ observation
    def _fill(self, template: str) -> str:
        r = self.rng
        return template.format(n=self.config["db_pool_size"], w=r.randint(2, 40),
                               h=r.randint(1, 9), p=r.randint(91, 99), a=r.randint(600, 900),
                               b=r.randint(1800, 3900), t=r.randint(1200, 30000),
                               s=r.randint(1, 9), o=r.randint(10000, 99999))

    def _stamp(self, line: str) -> str:
        self.clock += dt.timedelta(seconds=self.rng.randint(1, 20))
        return f"{self.clock:%Y-%m-%d %H:%M:%S} {line}"

    def get_logs(self, n: int = 8) -> list[str]:
        """Current log window: fault clues mixed with normal noise lines."""
        lines: list[str] = []
        if self.fault is not None:
            lines += [self._fill(t) for t in self.rng.sample(STRONG[self.fault], 2)]
            lines.append(self._fill(self.rng.choice(WEAK[self.fault])))
            pool = NOISE
        else:
            pool = HEALTHY_AFTER_FIX + NOISE
        while len(lines) < n:
            lines.append(self._fill(self.rng.choice(pool)))
        self.rng.shuffle(lines)
        return [self._stamp(line) for line in lines]

    def metrics(self) -> dict:
        """Metrics derived from the true state (what a monitoring system would see)."""
        f = self.fault
        return {
            "process_up": f != "service_crash",
            "db_pool_used_pct": 100 if f == "db_pool_exhausted" else 35,
            "disk_used_pct": 97 if f == "disk_full" else 61,
            "heap_used_pct": 96 if f == "memory_leak" else 48,
            "payments_reachable": f != "network_partition",
            "tls_valid": f != "cert_expired",
            "error_rate_pct": 0.2 if f is None else 23.0,
        }

    def health_check(self) -> dict:
        m = self.metrics()
        checks = {
            "process_up": m["process_up"],
            "db_pool_ok": m["db_pool_used_pct"] < 90,
            "disk_ok": m["disk_used_pct"] < 90,
            "heap_ok": m["heap_used_pct"] < 90,
            "payments_reachable": m["payments_reachable"],
            "tls_valid": m["tls_valid"],
            "error_rate_ok": m["error_rate_pct"] < 1.0,
        }
        return {"healthy": all(checks.values()), "checks": checks, "metrics": m}

    # ------------------------------------------------------------ actions
    def snapshot(self) -> dict:
        return copy.deepcopy(self.config)

    def restore(self, snap: dict) -> None:
        self.config = copy.deepcopy(snap)

    def apply_fix(self, action: str) -> bool:
        """Apply a whitelisted action. Returns True only if the active fault was cleared.

        Callers must still verify with health_check(); the return value is not trusted.
        """
        if action not in WHITELIST:
            raise UnknownActionError(f"action {action!r} is not whitelisted")
        # Side effects on config (so rollback has something to undo).
        c = self.config
        if action == "increase_db_pool_size":
            c["db_pool_size"] = min(c["db_pool_size"] * 2, 200)
        elif action == "scale_out_replicas":
            c["replicas"] += 1
        elif action == "compress_old_logs":
            c["log_retention_days"] = max(3, c["log_retention_days"] // 2)
        elif action == "flush_cache":
            c["cache_enabled"] = True
        elif action == "enable_rate_limiting":
            c["rate_limit_rps"] = 200
        elif action == "rollback_last_deploy":
            c["deployed_version"] = "v1.4.1"
        elif action == "failover_db_to_replica":
            c["db_primary"] = "db-replica-1"

        cleared = self.fault is not None and action in RESOLVES[self.fault] and not self.resistant
        self.action_history.append({"action": action, "fault_before": self.fault,
                                    "cleared": cleared})
        if cleared:
            self.fault = None
        return cleared
