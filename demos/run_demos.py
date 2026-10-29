"""The three required demo scenarios (simulator only; no real commands are run).

  Demo 1  supported fault    -> diagnose, pick whitelisted fix, run on FakeService, verify recovery
  Demo 2  unsupported fault  -> no invented fix, structured escalation report
  Demo 3  ambiguous incident -> guardrail abstains, shows why, evidence and where to start

Run:  python -m demos.run_demos            (all three)
      python -m demos.run_demos --demo 2   (one)
      python -m demos.run_demos --fault memory_leak   (choose the Demo 1 fault)
Outputs: printed report + results/demos/demo<N>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph import build_graph, load_runtime, run_incident
from schema import Incident
from sim.service import NOISE, W_TIMEOUT, FakeService

OUT = Path(__file__).resolve().parents[1] / "results" / "demos"


def live_incident(svc: FakeService, inc_id: str) -> Incident:
    """Ingest an incident the way monitoring would: alert text + current log window."""
    return Incident(id=inc_id, source="alert", text=svc.alert_text(), logs=svc.get_logs(n=8))


def ambiguous_incident(svc: FakeService) -> Incident:
    """Vague alert + a single clue ('request timed out') shared by two different faults."""
    lines = [svc._stamp(svc._fill(t)) for t in [NOISE[0], NOISE[1], W_TIMEOUT, NOISE[3], NOISE[5]]]
    return Incident(id="DEMO-3", source="alert", text="Checkout API degraded, users report slowness",
                    logs=lines)


def show(state: dict, svc: FakeService) -> None:
    inc = state["incident"]
    print(f"\nIncident {inc['id']}: {inc['text']}")
    for line in inc["logs"]:
        print(f"   | {line}")
    print("\nExplanation trace:")
    for e in state["trace"]:
        conf = "" if e["confidence"] is None else f" conf={e['confidence']:.2f}"
        print(f"  [{e['outcome']:>14}] {e['agent']:<10} {e['action']}{conf}")
        if e["detail"].get("reason"):
            print(f"  {'':>16} reason: {e['detail']['reason']}")
    d = state["diagnosis"]
    print("\nHypotheses:")
    for h in d["hypotheses"][:3]:
        print(f"  - {h['explanation']}")
    g = state["guardrail"]
    print(f"\nGuardrail: {g['decision']}  calibrated confidence={g['confidence']}  reasons={g['reasons']}")
    print(f"Status: {state['status']}  | simulated elapsed: {state['elapsed_s']:.0f} s"
          f"  | simulator healthy now: {svc.health_check()['healthy']}")
    if state.get("handoff"):
        h = state["handoff"]
        print("\n=== HUMAN HANDOFF REPORT ===")
        print(f"summary      : {h['summary']}")
        print(f"stuck_at     : {h['stuck_at']}")
        print(f"why_escalated: {h['why_escalated']['text']}  {h['why_escalated']['reasons']}")
        print(f"confidence   : {h['per_agent_confidence']}")
        print("actions taken:", h["actions_taken"] or "none (no action was executed)")
        print("evidence retrieved:")
        for e in h["evidence"][:6]:
            print(f"   {e['id']:<28} sim={e['similarity']:.2f} label={e['label']}")
        print("start_here:")
        for i, s in enumerate(h["start_here"], 1):
            print(f"   {i}. {s}")


def demo1(rt, g, fault: str):
    print("=" * 78 + f"\nDEMO 1 - supported fault: {fault}\n" + "=" * 78)
    svc = FakeService(seed=101)
    svc.inject_fault(fault)
    print("health before:", svc.health_check()["checks"])
    s = run_incident(rt, live_incident(svc, "DEMO-1"), svc, graph=g)
    show(s, svc)
    print("health after :", svc.health_check()["checks"])
    return s, svc


def demo2(rt, g, fault: str = "network_partition"):
    print("=" * 78 + f"\nDEMO 2 - unsupported fault: {fault}\n" + "=" * 78)
    svc = FakeService(seed=202)
    svc.inject_fault(fault)
    s = run_incident(rt, live_incident(svc, "DEMO-2"), svc, graph=g)
    show(s, svc)
    print("simulator actions executed:", svc.action_history or "none")
    return s, svc


def demo3(rt, g):
    print("=" * 78 + "\nDEMO 3 - ambiguous incident with insufficient evidence\n" + "=" * 78)
    svc = FakeService(seed=303)
    svc.inject_fault("db_pool_exhausted")   # hidden truth; the incident only shows one shared clue
    s = run_incident(rt, ambiguous_incident(svc), svc, graph=g)
    show(s, svc)
    print("simulator actions executed:", svc.action_history or "none")
    return s, svc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, choices=[1, 2, 3])
    ap.add_argument("--fault", default="db_pool_exhausted",
                    choices=["db_pool_exhausted", "disk_full", "memory_leak", "service_crash"])
    ap.add_argument("--unsupported", default="network_partition", choices=["network_partition", "cert_expired"])
    args = ap.parse_args()
    rt = load_runtime()
    g = build_graph(rt)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {1: lambda: demo1(rt, g, args.fault), 2: lambda: demo2(rt, g, args.unsupported), 3: lambda: demo3(rt, g)}
    ok = True
    for n in ([args.demo] if args.demo else [1, 2, 3]):
        s, svc = runs[n]()
        (OUT / f"demo{n}.json").write_text(json.dumps(s, indent=2, default=str))
        expected = {1: s["status"] == "resolved" and svc.health_check()["healthy"],
                    2: s["status"] != "resolved" and not svc.action_history,
                    3: s["status"] != "resolved" and not svc.action_history}[n]
        ok &= expected
        print(f"\n>>> Demo {n} acceptance check: {'PASS' if expected else 'FAIL'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
