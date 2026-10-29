"""Fit the confidence calibrator (TRAIN, out-of-fold) and tune thresholds (VAL only).

1. 5-fold out-of-fold triage probabilities on the train split, and leave-self-out
   retrieval (the incident's own index entries are excluded), so the calibrator sees
   realistic, not memorised, signals.
2. Logistic regression: [margin, top_sim, gap, top_score] -> P(top hypothesis correct).
3. On VAL: ECE + reliability curve (calibrated vs raw score), then a grid search of the
   guardrail thresholds. Objective = correct auto-fixes - 10 x wrong actions
   (ties -> the more conservative setting). The test split is never read here.

Run:  python -m eval.calibrate      -> artifacts/guardrail.joblib, results/calibration_val.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agents.diagnosis import diagnose
from agents.resolution import select_runbook
from agents.triage import TriageAgent, incident_text, make_model
from data_loader import load_sim_incidents
from eval.metrics import ece, reliability
from guardrail import ART, FEATURES, Guardrail, signals
from rag.store import load_frozen
from schema import Incident
from sim.service import RESOLVES, FakeService

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42


def triage_from_probs(classes, p) -> dict:
    order = np.argsort(p)[::-1]
    return {"fault": {"label": classes[order[0]], "confidence": float(p[order[0]]),
                      "runner_up": classes[order[1]], "margin": float(p[order[0]] - p[order[1]]),
                      "probs": {c: float(x) for c, x in zip(classes, p)}, "evidence_words": []}}


def collect(df, kb, triage_fn, leave_self_out: bool) -> list[dict]:
    rows = []
    for r, tri in zip(df.itertuples(), triage_fn):
        inc = Incident(id=r.id, source="alert", text=r.text, logs=r.logs)
        excl = {f"INC:{r.id}", f"LOG:sim:{r.id}"} if leave_self_out else None
        d = diagnose(inc, tri, kb, exclude_ids=excl)
        rb = select_runbook(d["top_cause"], FakeService())
        rows.append({"id": r.id, "fault": r.fault, "ambiguous": bool(r.ambiguous),
                     "supported": bool(r.supported), "sig": signals(tri, d), "triage": tri,
                     "diag": d, "resolution": rb, "correct": d["top_cause"] == r.fault})
    return rows


def main() -> int:
    df = load_sim_incidents()
    df["x"] = [incident_text(t, l) for t, l in zip(df["text"], df["logs"])]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    kb = load_frozen()
    assert kb.manifest["frozen"] and kb.manifest["n_learned"] == 0

    # 1) out-of-fold triage on TRAIN
    model = make_model(C=2.0)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, tr["x"], tr["fault"], cv=cv, method="predict_proba")
    classes = sorted(tr["fault"].unique())
    train_rows = collect(tr, kb, [triage_from_probs(classes, p) for p in oof], leave_self_out=True)

    # 2) calibrator
    X = np.array([[r["sig"][f] for f in FEATURES] for r in train_rows])
    y = np.array([r["correct"] for r in train_rows])
    cal = LogisticRegression(max_iter=1000).fit(X, y)

    # 3) VAL: real triage model (trained on full train), frozen KB
    agent = TriageAgent()
    val_tri = [{"fault": agent.run(r.text, r.logs)["fault"]} for r in va.itertuples()]
    val_rows = collect(va, kb, val_tri, leave_self_out=False)
    Xv = np.array([[r["sig"][f] for f in FEATURES] for r in val_rows])
    yv = np.array([r["correct"] for r in val_rows])
    conf_cal = cal.predict_proba(Xv)[:, 1]
    conf_raw = Xv[:, FEATURES.index("top_score")]
    calib = {"val_n": int(len(yv)), "val_diag_accuracy": float(yv.mean()),
             "ece_calibrated": round(ece(conf_cal, yv), 4), "ece_raw_score": round(ece(conf_raw, yv), 4),
             "reliability_calibrated": reliability(conf_cal, yv),
             "reliability_raw": reliability(conf_raw, yv),
             "train_oof_diag_accuracy": float(y.mean()),
             "calibrator_coef": dict(zip(FEATURES, cal.coef_[0].round(4).tolist())),
             "calibrator_intercept": round(float(cal.intercept_[0]), 4)}

    # 4) threshold grid on VAL
    grid = {"margin": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], "top_sim": [0.0, 0.2, 0.3, 0.4, 0.5],
            "gap": [0.0, 0.05, 0.1, 0.15, 0.2], "confidence": [0.5, 0.6, 0.7, 0.8, 0.9]}
    best = None
    for vals in itertools.product(*grid.values()):
        th = dict(zip(grid, vals))
        g = Guardrail(cal, th)
        correct = wrong = esc = 0
        for r in val_rows:
            dec = g.decide(r["triage"], r["diag"], r["resolution"])
            if dec["decision"] == "execute":
                if r["resolution"]["runbook"] in RESOLVES[r["fault"]]:
                    correct += 1
                else:
                    wrong += 1
            else:
                esc += 1
        utility = correct - 10 * wrong
        key = (utility, sum(vals))            # tie -> more conservative (larger thresholds)
        if best is None or key > best[0]:
            best = (key, th, {"auto_correct": correct, "wrong_actions": wrong, "escalated": esc})
    _, th, stats = best
    joblib.dump({"calibrator": cal, "thresholds": th, "fitted_on": "train (5-fold OOF, leave-self-out retrieval)",
                 "tuned_on": "val"}, ART)
    out = {"calibration": calib, "chosen_thresholds": th, "val_outcome_at_chosen": stats,
           "objective": "auto_correct - 10*wrong_actions; ties -> larger thresholds"}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibration_val.json").write_text(json.dumps(out, indent=2))
    plot_reliability(conf_cal, conf_raw, yv, RESULTS / "plots" / "reliability_val.png")
    print(json.dumps({k: v for k, v in calib.items() if not k.startswith("reliability")}, indent=2))
    print("chosen thresholds:", th, stats)
    return 0


def plot_reliability(conf_cal, conf_raw, y, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect")
    for conf, lab, c in [(conf_raw, f"raw score (ECE {ece(conf_raw, y):.3f})", "#d08c2a"),
                         (conf_cal, f"calibrated (ECE {ece(conf_cal, y):.3f})", "#2a6fd0")]:
        rel = reliability(conf, y)
        ax.plot([r["mean_conf"] for r in rel], [r["accuracy"] for r in rel], "o-", color=c, label=lab)
    ax.set_xlabel("confidence"); ax.set_ylabel("diagnosis accuracy")
    ax.set_title("Reliability (validation split)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
