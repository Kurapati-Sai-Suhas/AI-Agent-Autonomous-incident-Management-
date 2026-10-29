"""Knowledge base = three hybrid indexes + the learning-loop store.

  (a) incident index : real tickets (train split) + resolved simulator incidents (train split)
                       [+ human-resolved cases from the learning loop, LIVE mode only]
  (b) log index      : Drain-parsed Loghub chunks + simulator log windows (train split)
  (c) runbook index  : the whitelisted runbooks

Evaluation uses the FROZEN knowledge base: train data only, no learned cases, and a
manifest hash that tests check before any evaluation run.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import joblib

from agents.triage import incident_text
from rag.index import HybridIndex
from rag.logparse import loghub_chunks
from runbooks import RUNBOOKS

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FROZEN = ART / "kb_frozen.joblib"
LEARNED = ART / "learned_cases.jsonl"


class KnowledgeBase:
    def __init__(self, incidents: HybridIndex, logs: HybridIndex, runbooks: HybridIndex,
                 manifest: dict):
        self.incidents, self.logs, self.runbooks, self.manifest = incidents, logs, runbooks, manifest

    def get(self, doc_id: str) -> dict | None:
        """Resolve any citable id across the three indexes."""
        for idx in (self.incidents, self.logs, self.runbooks):
            d = idx.get(doc_id)
            if d is not None:
                return d
        return None


def _ids_hash(idx: HybridIndex) -> str:
    return hashlib.sha256("\n".join(sorted(d["id"] for d in idx.docs)).encode()).hexdigest()[:16]


def incident_docs(sim, tickets, learned: list[dict] | None = None) -> list[dict]:
    docs = []
    for r in tickets.itertuples():
        docs.append({"id": f"INC:{r.ticket_id}", "text": str(r.text),
                     "meta": {"kind": "ticket", "queue": r.queue, "priority": r.priority,
                              "resolution_hint": str(r.answer)[:400] if isinstance(r.answer, str) else "",
                              "split": "train"}})
    for r in sim.itertuples():
        docs.append({"id": f"INC:{r.id}", "text": incident_text(r.text, r.logs),
                     "meta": {"kind": "sim_incident", "fault": r.fault, "root_cause": r.fault,
                              "fix": r.correct_fix or "escalated (no runbook)",
                              "supported": bool(r.supported), "split": "train"}})
    for c in learned or []:
        docs.append({"id": c["id"], "text": incident_text(c["text"], c["logs"]),
                     "meta": {"kind": "learned", "fault": c["root_cause"], "root_cause": c["root_cause"],
                              "fix": c["fix"], "notes": c.get("notes", ""), "split": "learned"}})
    return docs


def log_docs(sim) -> list[dict]:
    docs = loghub_chunks()
    for r in sim.itertuples():
        docs.append({"id": f"LOG:sim:{r.id}", "text": incident_text("", r.logs).strip(),
                     "meta": {"kind": "sim_logs", "service": "checkout-api", "fault": r.fault,
                              "split": "train"}})
    return docs


def runbook_docs() -> list[dict]:
    return [{"id": f"RB:{rb.name}",
             "text": f"{rb.name.replace('_', ' ')}. {rb.description} Handles: "
                     f"{', '.join(h.replace('_', ' ') for h in rb.handles)}. Effect: {rb.expected_effect}",
             "meta": {"kind": "runbook", **rb.to_dict()}} for rb in RUNBOOKS.values()]


def load_learned() -> list[dict]:
    if not LEARNED.exists():
        return []
    return [json.loads(line) for line in LEARNED.read_text(encoding="utf8").splitlines() if line.strip()]


def build(include_learned: bool = False) -> KnowledgeBase:
    from data_loader import load_sim_incidents, prepare_tickets
    sim = load_sim_incidents()
    sim_train = sim[sim.split == "train"]          # never val/test
    tickets_train = prepare_tickets()["train"]
    learned = load_learned() if include_learned else []
    inc = HybridIndex(incident_docs(sim_train, tickets_train, learned), "incidents")
    logs = HybridIndex(log_docs(sim_train), "logs")
    rbs = HybridIndex(runbook_docs(), "runbooks")
    manifest = {
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "frozen": not include_learned, "splits_used": ["train"] + (["learned"] if learned else []),
        "n_docs": {"incidents": len(inc), "logs": len(logs), "runbooks": len(rbs)},
        "n_learned": len(learned), "embedder": inc.embedder.name if hasattr(inc.embedder, "name") else "?",
        "ids_sha256": {"incidents": _ids_hash(inc), "logs": _ids_hash(logs), "runbooks": _ids_hash(rbs)},
    }
    return KnowledgeBase(inc, logs, rbs, manifest)


def load_frozen(rebuild: bool = False) -> KnowledgeBase:
    """The evaluation KB. Built once from train data, then reused unchanged."""
    if FROZEN.exists() and not rebuild:
        return joblib.load(FROZEN)
    kb = build(include_learned=False)
    ART.mkdir(exist_ok=True)
    joblib.dump(kb, FROZEN)
    (ART / "kb_frozen_manifest.json").write_text(json.dumps(kb.manifest, indent=2))
    return kb


def load_live() -> KnowledgeBase:
    """Frozen KB plus human-resolved cases (used by the dashboard / demos, never by eval)."""
    learned = load_learned()
    return build(include_learned=True) if learned else load_frozen()


if __name__ == "__main__":
    # import via the package so pickled classes are rag.store.*, not __main__.*
    from rag.store import load_frozen as _load_frozen
    kb = _load_frozen(rebuild="--rebuild" in sys.argv)
    print(json.dumps(kb.manifest, indent=2))
