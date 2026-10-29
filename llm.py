"""Single wrapper for every LLM call in the project.

Provider settings come from .env (never hard-coded):
    LLM_PROVIDER = none | anthropic | openai      (default: none -> offline, deterministic)
    LLM_MODEL    = model name for the provider
    LLM_API_KEY  = key (read from the environment only; never printed or logged)

LLMs are used ONLY to word hypotheses and handoff text from structured, retrieved
material. Confidence, triage and action selection never depend on an LLM.
If no provider is configured, or a call fails, callers fall back to templates.
"""
from __future__ import annotations

import json
import os
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional at import time
    pass

TIMEOUT_S = 30


def provider() -> str:
    p = os.getenv("LLM_PROVIDER", "none").strip().lower()
    return p if p in {"anthropic", "openai"} and os.getenv("LLM_API_KEY") else "none"


def available() -> bool:
    return provider() != "none"


def complete(prompt: str, system: str = "", max_tokens: int = 400) -> str | None:
    """Return the model's text, or None if no LLM is configured or the call failed."""
    p = provider()
    if p == "none":
        return None
    key, model = os.environ["LLM_API_KEY"], os.getenv("LLM_MODEL", "")
    try:
        if p == "anthropic":
            body = {"model": model or "claude-haiku-5-5", "max_tokens": max_tokens,
                    "system": system, "messages": [{"role": "user", "content": prompt}]}
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                data = json.load(r)
            return "".join(b.get("text", "") for b in data.get("content", []))
        if p == "openai":
            body = {"model": model or "gpt-4o-mini", "max_tokens": max_tokens,
                    "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}]}
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                data = json.load(r)
            return data["choices"][0]["message"]["content"]
    except Exception as e:  # network/auth problems -> deterministic fallback
        print(f"[llm] call failed ({type(e).__name__}); using deterministic fallback")
    return None
