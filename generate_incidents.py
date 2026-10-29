"""Generate 1,200 labeled simulator incidents (200 per fault).

* 15% of incidents are ambiguous: a generic alert and only ONE clue in the logs,
  and that clue is a "weak" line shared by two or more faults.
* Clear incidents get a fault-specific (or sometimes generic) alert, 2-3 strong
  clues, sometimes a weak clue, and normal noise lines.
* Split 70/15/15 (train/val/test), stratified by fault+ambiguous, seed 42.

Output: data/processed/sim_incidents.csv (refuses to overwrite unless --force).
Run:    python generate_incidents.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sim.service import CORRECT_FIX, FAULTS, NOISE, STRONG, SUPPORTED_FAULTS, WEAK, FakeService

SEED = 42
N_PER_FAULT = 200
AMBIGUOUS_FRAC = 0.15
OUT = Path(__file__).parent / "data" / "processed" / "sim_incidents.csv"

ALERTS = {
    "db_pool_exhausted": [
        "Checkout API errors: database connections unavailable",
        "High error rate on /orders, DB connection timeouts",
        "checkout-db connection pool saturation alert",
        "Orders failing: cannot acquire database connection",
        "Spike in 5xx on checkout, db pool at limit",
    ],
    "disk_full": [
        "Disk usage critical on checkout host",
        "Checkout host filesystem /var almost full",
        "Write failures on checkout node: no space left",
        "Disk space alert for chk host, log volume growing",
        "Audit logging failing on checkout due to storage",
    ],
    "memory_leak": [
        "Checkout API memory usage growing, latency rising",
        "Heap usage high on checkout-api, frequent GC pauses",
        "Checkout-api RSS climbing steadily since deploy",
        "OutOfMemory warnings on checkout service",
        "Checkout getting slower over hours, memory alarm",
    ],
    "service_crash": [
        "Checkout API down: health check failing",
        "checkout-api process keeps restarting",
        "Checkout service crashed, 502 from load balancer",
        "Health probe failing for checkout pods",
        "Checkout unavailable, process exited unexpectedly",
    ],
    "network_partition": [
        "Checkout cannot reach payments service",
        "Network errors between checkout and payments cluster",
        "Payments calls timing out, hosts unreachable",
        "Cluster quorum lost, peers unreachable from checkout",
        "Connectivity alarm: checkout to payments-svc",
    ],
    "cert_expired": [
        "TLS errors calling payments service",
        "Certificate alarm for payments.internal",
        "SSL handshake failures from checkout",
        "Payments integration failing with certificate error",
        "x509 errors in checkout client logs",
    ],
}
GENERIC_ALERTS = [
    "Checkout API degraded",
    "Elevated error rate on checkout",
    "Customer reports: checkout not working properly",
    "SLO burn alert: checkout",
    "Checkout service unhealthy, investigate",
]
BASE_PRIORITY = {"db_pool_exhausted": "high", "disk_full": "medium", "memory_leak": "medium",
                 "service_crash": "high", "network_partition": "high", "cert_expired": "high"}


def make_incident(i: int, fault: str, ambiguous: bool, rng: random.Random, svc: FakeService) -> dict:
    t0 = dt.datetime(2026, 3, 1) + dt.timedelta(minutes=rng.randint(0, 60 * 24 * 60))
    if ambiguous:
        text = rng.choice(GENERIC_ALERTS)
        clues = [rng.choice(WEAK[fault])]
    else:
        text = rng.choice(GENERIC_ALERTS) if rng.random() < 0.15 else rng.choice(ALERTS[fault])
        clues = rng.sample(STRONG[fault], rng.choice([2, 3]))
        if rng.random() < 0.5:
            clues.append(rng.choice(WEAK[fault]))
    lines = [svc._fill(c) for c in clues]
    lines += [svc._fill(rng.choice(NOISE)) for _ in range(rng.randint(4, 8))]
    rng.shuffle(lines)
    stamps, logs = t0, []
    for line in lines:
        stamps += dt.timedelta(seconds=rng.randint(1, 30))
        logs.append(f"{stamps:%Y-%m-%d %H:%M:%S} {line}")
    prio = BASE_PRIORITY[fault]
    if rng.random() < 0.2:
        prio = rng.choice(["low", "medium", "high"])
    return {"id": f"SIM-{i:05d}", "fault": fault, "text": text, "logs": json.dumps(logs),
            "priority": prio, "correct_fix": CORRECT_FIX[fault] or "",
            "supported": fault in SUPPORTED_FAULTS, "ambiguous": ambiguous,
            "timestamp": t0.isoformat()}


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    svc = FakeService(seed=seed)
    rows, i = [], 0
    n_amb = int(N_PER_FAULT * AMBIGUOUS_FRAC)
    for fault in FAULTS:
        for k in range(N_PER_FAULT):
            rows.append(make_incident(i, fault, k < n_amb, rng, svc))
            i += 1
    df = pd.DataFrame(rows)
    strat = df["fault"] + "|" + df["ambiguous"].astype(str)
    train, rest = train_test_split(df, test_size=0.30, random_state=seed, stratify=strat)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed,
                                 stratify=strat.loc[rest.index])
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT} exists; not overwriting (use --force).")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} incidents -> {OUT}")
    print(pd.crosstab(df["fault"], df["split"]))
    print("ambiguous:", int(df["ambiguous"].sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
