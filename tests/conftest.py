import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACTS_READY = all((ROOT / "artifacts" / f).exists()
                      for f in ["triage.joblib", "kb_frozen.joblib", "guardrail.joblib"])
needs_artifacts = pytest.mark.skipif(
    not ARTIFACTS_READY, reason="run train_triage.py, rag.store and eval.calibrate first")


@pytest.fixture(scope="session")
def runtime():
    from graph import load_runtime
    return load_runtime()


@pytest.fixture(scope="session")
def graph(runtime):
    from graph import build_graph
    return build_graph(runtime)


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Tests always run offline (no LLM key)."""
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
