"""Confidence + risk guardrail: decides between executing a runbook and escalating.

Confidence is NEVER an LLM self-report. It is computed from:
  margin   - triage fault margin (top prob - second prob)
  top_sim  - top cosine similarity of a labelled similar incident
  gap      - score gap between the top two diagnosis hypotheses
  top_score- score of the best hypothesis
A logistic-regression calibrator maps these signals to P(diagnosis correct). It is
fit on out-of-fold TRAIN predictions; ECE is checked on VAL; thresholds are tuned on
VAL only (eval/calibrate.py). Test data is never used here.

Escalation reasons: triage_ambiguous, low_retrieval_support, conflicting_evidence,
low_confidence, insufficient_evidence, no_matching_runbook, action_unsafe
(and fix_not_verified after validation).

insufficient_evidence is a fixed safety POLICY, not a tuned threshold: the system never
acts unless at least one log line of the incident itself directly matches a signature
of the diagnosed cause (similar past incidents alone are not enough to act on).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

ART = Path(__file__).parent / "artifacts" / "guardrail.joblib"
FEATURES = ["margin", "top_sim", "gap", "top_score"]
DEFAULT_THRESHOLDS = {"margin": 0.20, "top_sim": 0.30, "gap": 0.10, "confidence": 0.60}
MEDIUM_RISK_EXTRA = 0.10   # medium-risk runbooks need this much more confidence
MIN_DIRECT_EVIDENCE = 1    # policy: >= 1 incident log line must support the diagnosed cause


def signals(triage: dict, diag: dict) -> dict:
    return {"margin": float(triage["fault"]["margin"]),
            "top_sim": float(diag["top_retrieval_similarity"]),
            "gap": float(diag["score_gap"]),
            "top_score": float(diag["top_score"]),
            "direct_evidence": float(diag.get("top_direct_evidence", 0))}


class Guardrail:
    def __init__(self, calibrator=None, thresholds: dict | None = None, use_calibration: bool = True,
                 use_retrieval_check: bool = True):
        self.calibrator = calibrator
        self.thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        self.use_calibration = use_calibration and calibrator is not None
        self.use_retrieval_check = use_retrieval_check

    @classmethod
    def load(cls, path: Path = ART, **kw) -> "Guardrail":
        if not path.exists():
            print("[guardrail] no tuned artifact; using default thresholds, uncalibrated")
            return cls(None, None, **kw)
        blob = joblib.load(path)
        return cls(blob["calibrator"], blob["thresholds"], **kw)

    def confidence(self, sig: dict) -> float:
        if self.use_calibration:
            x = np.array([[sig[f] for f in FEATURES]])
            return float(self.calibrator.predict_proba(x)[0, 1])
        return float(sig["top_score"])   # uncalibrated: raw hypothesis score

    def decide(self, triage: dict, diag: dict, resolution: dict) -> dict:
        sig = signals(triage, diag)
        conf = self.confidence(sig)
        t = self.thresholds
        reasons = []
        if sig["margin"] < t["margin"]:
            reasons.append("triage_ambiguous")
        if self.use_retrieval_check and sig["top_sim"] < t["top_sim"]:
            reasons.append("low_retrieval_support")
        if sig["gap"] < t["gap"]:
            reasons.append("conflicting_evidence")
        if conf < t["confidence"]:
            reasons.append("low_confidence")
        if sig["direct_evidence"] < MIN_DIRECT_EVIDENCE:
            reasons.append("insufficient_evidence")
        if resolution["runbook"] is None:
            reasons.append("no_matching_runbook")
        elif resolution["risk"] == "high":
            reasons.append("action_unsafe")
        elif resolution["risk"] == "medium" and conf < t["confidence"] + MEDIUM_RISK_EXTRA:
            reasons.append("action_unsafe")
        return {"decision": "escalate" if reasons else "execute", "reasons": reasons,
                "confidence": round(conf, 4), "signals": {k: round(v, 4) for k, v in sig.items()},
                "thresholds": t, "calibrated": self.use_calibration}
