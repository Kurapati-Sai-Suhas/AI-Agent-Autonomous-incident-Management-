"""Diagnosis agent: ranked root-cause hypotheses with traceable evidence ids.

Deterministic (no-LLM) method = retrieval + rules:
  score(cause) = W_TRIAGE * triage probability
               + W_RETRIEVAL * similarity-weighted vote of similar resolved incidents
               + W_RULES * share of log-signature rules matched in the incident's own logs
Every supporting item is an id that exists in the retrieved set or in the incident's
own log lines, so citations can be checked. An optional LLM may only re-word the
hypotheses; sentences that cite unknown ids are dropped (see `keep_cited`).
"""
from __future__ import annotations

import re

import llm
from agents.triage import incident_text
from schema import Incident

W_TRIAGE, W_RETRIEVAL, W_RULES = 0.4, 0.3, 0.3
K_INCIDENTS, K_LOGS, K_RUNBOOKS = 8, 4, 3

# Generic operational error signatures (domain knowledge, not simulator templates).
SIGNATURES = {
    "db_pool_exhausted": [r"connection pool|pool exhausted|max_connections|connection is not available"],
    "disk_full": [r"no space left|errno=28|disk quota|usage at 9\d%"],
    "memory_leak": [r"outofmemory|heap (space|used)|gc overhead|rss .* grew"],
    "service_crash": [r"segv|main process exited|connection refused|restarted \d+ times"],
    "network_partition": [r"no route to host|lost quorum|unreachable|i/o timeout"],
    "cert_expired": [r"x509|certificate (has )?expired|ssl handshake failed"],
}
_ID = re.compile(r"\[([A-Za-z]+:[^\]\s]+|[A-Za-z0-9-]+:log:\d+)\]")


def match_rules(inc: Incident) -> dict[str, list[str]]:
    """cause -> list of incident log-line ids that match one of its signatures."""
    hits: dict[str, list[str]] = {c: [] for c in SIGNATURES}
    for lid, line in zip(inc.log_ids(), inc.logs):
        low = line.lower()
        for cause, pats in SIGNATURES.items():
            if any(re.search(p, low) for p in pats):
                hits[cause].append(lid)
    return hits


def retrieve(inc: Incident, kb, use_rag: bool = True, exclude_ids: set[str] | None = None) -> dict:
    if not use_rag:
        return {"incidents": [], "logs": [], "runbooks": []}
    q = incident_text(inc.text, inc.logs)
    err = " ".join(l for l in inc.logs if re.search(r"ERROR|CRIT|WARN", l)) or q
    strip = lambda hits: [{k: v for k, v in h.items() if k != "text"} | {"snippet": h["text"][:240]}
                          for h in hits]
    return {
        "incidents": strip(kb.incidents.search(q, k=K_INCIDENTS, exclude_ids=exclude_ids)),
        "logs": strip(kb.logs.search(err, k=K_LOGS, exclude_ids=exclude_ids)),
        "runbooks": strip(kb.runbooks.search(inc.text + " " + err, k=K_RUNBOOKS)),
    }


def diagnose(inc: Incident, triage: dict, kb, use_rag: bool = True,
             exclude_ids: set[str] | None = None) -> dict:
    retrieved = retrieve(inc, kb, use_rag, exclude_ids)
    rules = match_rules(inc)
    probs = triage["fault"]["probs"]

    # similarity-weighted vote of labelled neighbours (simulator + learned cases)
    labelled = [h for h in retrieved["incidents"] if h["meta"].get("fault")]
    total = sum(h["similarity"] for h in labelled) or 1.0
    vote = {c: sum(h["similarity"] for h in labelled if h["meta"]["fault"] == c) / total for c in probs}
    w_t, w_r, w_l = (W_TRIAGE, W_RETRIEVAL, W_RULES) if use_rag else (0.6, 0.0, 0.4)

    hyps = []
    for cause in probs:
        rule_score = min(1.0, len(rules.get(cause, [])) / 2)
        score = w_t * probs[cause] + w_r * vote.get(cause, 0.0) + w_l * rule_score
        support = rules.get(cause, [])[:3]
        support += [h["id"] for h in labelled if h["meta"]["fault"] == cause][:3]
        support += [h["id"] for h in retrieved["logs"] if h["meta"].get("fault") == cause][:2]
        missing = []
        if not rules.get(cause):
            missing.append(f"no log line matches a {cause} signature")
        if use_rag and vote.get(cause, 0) == 0:
            missing.append("no similar resolved incident had this cause")
        hyps.append({"cause": cause, "score": round(score, 4),
                     "components": {"triage": round(probs[cause], 4), "retrieval": round(vote.get(cause, 0.0), 4),
                                    "rules": round(rule_score, 4)},
                     "supporting_evidence": support, "missing_or_contradicting": missing})
    hyps.sort(key=lambda h: -h["score"])
    for h in hyps:
        h["explanation"] = template_explanation(h, retrieved)

    top_sim = max([h["similarity"] for h in labelled], default=0.0)
    out = {
        "hypotheses": hyps[:4],
        "top_cause": hyps[0]["cause"],
        "top_score": hyps[0]["score"],
        "score_gap": round(hyps[0]["score"] - hyps[1]["score"], 4),
        "top_retrieval_similarity": round(top_sim, 4),
        # log lines of THIS incident that directly match a signature of the top cause
        "top_direct_evidence": len(rules.get(hyps[0]["cause"], [])),
        "retrieved": retrieved,
        "rule_hits": {c: v for c, v in rules.items() if v},
        "llm_used": False,
    }
    if llm.available():
        _llm_reword(inc, out)
    return out


def template_explanation(h: dict, retrieved: dict) -> str:
    c = h["components"]
    parts = [f"{h['cause']} (score {h['score']:.2f}): triage p={c['triage']:.2f}, "
             f"neighbour vote={c['retrieval']:.2f}, rule match={c['rules']:.2f}."]
    if h["supporting_evidence"]:
        parts.append("Evidence: " + ", ".join(f"[{e}]" for e in h["supporting_evidence"]) + ".")
    if h["missing_or_contradicting"]:
        parts.append("Gaps: " + "; ".join(h["missing_or_contradicting"]) + ".")
    return " ".join(parts)


def allowed_ids(inc: Incident, retrieved: dict) -> set[str]:
    ids = set(inc.log_ids())
    for group in retrieved.values():
        ids.update(h["id"] for h in group)
    return ids


def keep_cited(text: str, allowed: set[str]) -> str:
    """Keep only sentences that cite at least one id, and only ids we actually retrieved."""
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        cited = _ID.findall(sent)
        if cited and all(c in allowed for c in cited):
            kept.append(sent)
    return " ".join(kept)


def _llm_reword(inc: Incident, out: dict) -> None:
    allowed = allowed_ids(inc, out["retrieved"])
    for h in out["hypotheses"][:2]:
        prompt = ("Rewrite this root-cause hypothesis in 2 sentences for an on-call engineer. "
                  "Use ONLY the facts below and cite evidence ids in square brackets exactly as given.\n"
                  f"{h['explanation']}")
        text = llm.complete(prompt, system="You only restate given facts. Never invent ids.")
        if text:
            cited = keep_cited(text, allowed)
            if cited:
                h["explanation_llm"] = cited
                out["llm_used"] = True
