"""Triage agent: predicts queue, priority and fault category.

Baseline model (per the brief): TF-IDF + class-balanced logistic regression.
  * queue    - trained on real tickets, limited to the three IT queues
  * priority - trained on real tickets (low / medium / high)
  * fault    - trained on simulator incidents (alert text + log lines)
Every prediction returns label, confidence, runner-up, margin and evidence words.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "triage.joblib"
_TS = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ")


def incident_text(text: str, logs: list[str]) -> str:
    """Model input for the fault classifier: alert text + log lines without timestamps."""
    return " ".join([text] + [_TS.sub("", line) for line in logs])


def make_model(C: float = 4.0, max_features: int = 50000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  max_features=max_features, lowercase=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=C)),
    ])


def explain(model: Pipeline, text: str, top_k: int = 5) -> dict:
    """Prediction with confidence, runner-up, margin and the words that pushed it."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    x = vec.transform([text])
    proba = clf.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    top, second = order[0], order[1]
    # contribution of each present term to the winning class score
    coef = clf.coef_[top] if clf.coef_.shape[0] > 1 else clf.coef_[0] * (1 if top == 1 else -1)
    contrib = x.toarray()[0] * coef
    vocab = vec.get_feature_names_out()
    idx = [i for i in np.argsort(contrib)[::-1][:top_k] if contrib[i] > 0]
    return {
        "label": str(clf.classes_[top]),
        "confidence": float(proba[top]),
        "runner_up": str(clf.classes_[second]),
        "margin": float(proba[top] - proba[second]),
        "evidence_words": [str(vocab[i]) for i in idx],
        "probs": {str(c): float(p) for c, p in zip(clf.classes_, proba)},
    }


class TriageAgent:
    """Loads the three trained models and triages an incident."""

    def __init__(self, models: dict[str, Pipeline] | None = None, path: Path = ARTIFACT):
        if models is None:
            if not path.exists():
                raise FileNotFoundError(f"{path} missing - run: python train_triage.py")
            models = joblib.load(path)
        self.models = models

    def run(self, text: str, logs: list[str]) -> dict:
        return {
            "queue": explain(self.models["queue"], text),
            "priority": explain(self.models["priority"], text),
            "fault": explain(self.models["fault"], incident_text(text, logs)),
        }
