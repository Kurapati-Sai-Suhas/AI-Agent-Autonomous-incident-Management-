"""Train the three triage models on TRAIN splits only; report VALIDATION metrics.

Run:  python train_triage.py          (writes artifacts/triage.joblib, results/triage_val.json)
The test split is untouched here; eval/run_all.py reports test metrics once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score

from agents.triage import ARTIFACT, incident_text, make_model
from data_loader import IT_QUEUES, load_sim_incidents, prepare_tickets

RESULTS = Path(__file__).parent / "results"


def ticket_frames():
    s = prepare_tickets()
    q = {k: v[v["queue"].isin(IT_QUEUES)] for k, v in s.items()}
    return s, q


def main() -> int:
    tickets, it_tickets = ticket_frames()
    sim = load_sim_incidents()
    sim["x"] = [incident_text(t, l) for t, l in zip(sim["text"], sim["logs"])]
    tr, va = sim[sim.split == "train"], sim[sim.split == "val"]

    models = {
        "queue": make_model().fit(it_tickets["train"]["text"], it_tickets["train"]["queue"]),
        "priority": make_model().fit(tickets["train"]["text"], tickets["train"]["priority"]),
        "fault": make_model(C=2.0).fit(tr["x"], tr["fault"]),
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump(models, ARTIFACT)

    def score(model, x, y):
        p = model.predict(x)
        return {"n": int(len(y)), "accuracy": round(accuracy_score(y, p), 4),
                "macro_f1": round(f1_score(y, p, average="macro"), 4),
                "majority_baseline_acc": round(float(y.value_counts(normalize=True).iloc[0]), 4)}

    res = {
        "split": "val",
        "queue": score(models["queue"], it_tickets["val"]["text"], it_tickets["val"]["queue"]),
        "priority": score(models["priority"], tickets["val"]["text"], tickets["val"]["priority"]),
        "fault_all": score(models["fault"], va["x"], va["fault"]),
        "fault_clear": score(models["fault"], va[~va.ambiguous]["x"], va[~va.ambiguous]["fault"]),
        "fault_ambiguous": score(models["fault"], va[va.ambiguous]["x"], va[va.ambiguous]["fault"]),
        "reference_from_brief_val": {"queue": {"accuracy": 0.752, "macro_f1": 0.716},
                                     "priority": {"accuracy": 0.624, "macro_f1": 0.616}},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "triage_val.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
