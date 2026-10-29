"""Split integrity, index freezing and no-leakage checks."""
import re
from pathlib import Path

import pytest

from conftest import ROOT, needs_artifacts


@pytest.fixture(scope="module")
def sim():
    from data_loader import load_sim_incidents
    return load_sim_incidents()


def test_sim_dataset_shape(sim):
    assert len(sim) == 1200
    assert sim.groupby("fault").size().eq(200).all()
    assert abs(sim.ambiguous.mean() - 0.15) < 1e-9
    assert sim.split.value_counts().to_dict() == {"train": 840, "val": 180, "test": 180}
    assert sim.id.is_unique


def test_ticket_splits_disjoint():
    from data_loader import _norm, prepare_tickets
    s = prepare_tickets()
    ids = [set(d.ticket_id) for d in s.values()]
    texts = [set(d.text.map(_norm)) for d in s.values()]
    for a in range(3):
        for b in range(a + 1, 3):
            assert not ids[a] & ids[b]
            assert not texts[a] & texts[b]


@needs_artifacts
def test_frozen_index_contains_train_only(sim):
    from rag.store import load_frozen
    kb = load_frozen()
    assert kb.manifest["frozen"] and kb.manifest["n_learned"] == 0
    heldout = set(sim[sim.split != "train"].id)
    indexed = {d["id"].split(":")[-1] for idx in (kb.incidents, kb.logs) for d in idx.docs
               if d["meta"].get("kind") in ("sim_incident", "sim_logs")}
    assert indexed and not (indexed & heldout)
    assert not any(d["id"].startswith("LEARNED:") for d in kb.incidents.docs)
    from data_loader import prepare_tickets
    test_tickets = set(prepare_tickets()["test"].ticket_id) | set(prepare_tickets()["val"].ticket_id)
    assert not any(d["id"][4:] in test_tickets for d in kb.incidents.docs if d["meta"]["kind"] == "ticket")


@needs_artifacts
def test_thresholds_tuned_on_val_and_calibration_never_reads_test():
    import joblib
    blob = joblib.load(ROOT / "artifacts" / "guardrail.joblib")
    assert blob["tuned_on"] == "val" and blob["fitted_on"].startswith("train")
    src = (ROOT / "eval" / "calibrate.py").read_text()
    assert not re.search(r"""split\s*==\s*["']test["']""", src)
    src = (ROOT / "train_triage.py").read_text()
    assert not re.search(r"""["']test["']\]""", src)


def test_hybrid_search_keeps_exact_identifiers():
    from rag.index import HybridIndex
    docs = [{"id": "A", "text": "ERROR fs: No space left on device (errno=28)", "meta": {}},
            {"id": "B", "text": "WARN jvm.gc: GC overhead limit approaching", "meta": {}},
            {"id": "C", "text": "INFO checkout.api: GET /health 200", "meta": {}}]
    idx = HybridIndex(docs)
    for mode in ("bm25", "hybrid"):
        assert idx.search("errno=28", k=1, mode=mode)[0]["id"] == "A"
    assert idx.search("x", k=3, exclude_ids={"A", "B", "C"}) == []


def test_drain_grouping_accuracy_sanity():
    from rag.logparse import grouping_accuracy
    assert grouping_accuracy(["E1", "E1", "E2"], [1, 1, 2]) == 1.0
    assert grouping_accuracy(["E1", "E1", "E2"], [1, 1, 1]) == 0.0
