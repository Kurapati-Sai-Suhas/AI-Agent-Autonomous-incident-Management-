"""Learning loop: human-resolved escalations are added to the LIVE incident index.

Cases go to artifacts/learned_cases.jsonl. The FROZEN evaluation index never loads
this file (rag.store.load_frozen), so learned cases cannot leak into evaluation.
"""
from __future__ import annotations

import datetime as dt
import json

from rag import store


def record_human_resolution(incident: dict, root_cause: str, fix: str, notes: str = "") -> dict:
    existing = {c["id"] for c in store.load_learned()}
    case_id = f"LEARNED:{incident['id']}"
    n = 1
    while case_id in existing:            # never overwrite an earlier record
        n += 1
        case_id = f"LEARNED:{incident['id']}#{n}"
    case = {"id": case_id, "incident_id": incident["id"], "text": incident["text"],
            "logs": incident["logs"], "root_cause": root_cause, "fix": fix, "notes": notes,
            "recorded_at": dt.datetime.now().isoformat(timespec="seconds")}
    store.LEARNED.parent.mkdir(exist_ok=True)
    with open(store.LEARNED, "a", encoding="utf8") as f:
        f.write(json.dumps(case) + "\n")
    return case
