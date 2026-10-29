"""Streamlit dashboard for the incident-management agents.

Run:  streamlit run app.py
Everything acts on the FakeService simulator only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import learning
from data_loader import load_sim_incidents
from demos.run_demos import ambiguous_incident, live_incident
from graph import build_graph, load_runtime, run_incident
from runbooks import RUNBOOKS
from schema import Incident
from sim.service import FAULTS, FakeService

ROOT = Path(__file__).parent
RES = ROOT / "results"
st.set_page_config(page_title="Incident Agents", layout="wide")


@st.cache_resource(show_spinner="Loading models and knowledge base ...")
def runtime_and_graph():
    rt = load_runtime(live=True)          # frozen KB + human-resolved cases
    return rt, build_graph(rt)


@st.cache_data
def sim_incidents():
    return load_sim_incidents()


rt, graph = runtime_and_graph()
st.session_state.setdefault("runs", [])
OUTCOME_ICON = {"ok": "✅", "low_confidence": "⚠️", "failed": "❌", "unsafe": "⛔"}

st.title("AI-agent incident management (simulator)")
st.caption("Triage → Diagnosis (hybrid RAG) → Resolution (whitelisted runbooks) → Guardrail → "
           "Executor + Validation → close or Escalation handoff. No real commands are ever run.")

tab_run, tab_queue, tab_eval, tab_rb = st.tabs(["Run an incident", "Escalations & learning loop",
                                                "Evaluation & MTTR", "Runbooks"])

# ---------------------------------------------------------------- run
with tab_run:
    c1, c2 = st.columns([1, 2])
    with c1:
        mode = st.radio("Incident source", ["Inject fault into simulator", "Demo 3: ambiguous incident",
                                            "Test-split incident", "Write my own"])
        fault, resistant, inc, svc = None, False, None, None
        if mode == "Inject fault into simulator":
            fault = st.selectbox("Fault", FAULTS)
            resistant = st.checkbox("Fix will not hold (exercise validation + rollback)")
        elif mode == "Test-split incident":
            df = sim_incidents()
            test = df[df.split == "test"]
            pick = st.selectbox("Incident", test.id, format_func=lambda i: f"{i} - {test.set_index('id').loc[i, 'fault']}"
                                + (" (ambiguous)" if test.set_index('id').loc[i, 'ambiguous'] else ""))
        elif mode == "Write my own":
            own_text = st.text_input("Alert text", "Checkout API errors: database connections unavailable")
            own_logs = st.text_area("Log lines (one per line)",
                                    "2026-01-01 12:00:01 ERROR db.pool: connection pool exhausted (active=50/50, waiting=12)")
            own_fault = st.selectbox("Hidden true fault in the simulator", FAULTS)
        go = st.button("Run pipeline", type="primary")

    if go:
        n = len(st.session_state.runs) + 1
        if mode == "Inject fault into simulator":
            svc = FakeService(seed=1000 + n)
            svc.inject_fault(fault, resistant=resistant)
            inc = live_incident(svc, f"LIVE-{n:03d}")
        elif mode == "Demo 3: ambiguous incident":
            svc = FakeService(seed=303)
            svc.inject_fault("db_pool_exhausted")
            inc = ambiguous_incident(svc)
            inc.id = f"LIVE-{n:03d}"
        elif mode == "Test-split incident":
            r = test.set_index("id").loc[pick]
            svc = FakeService(seed=int(pick.split("-")[1]))
            svc.inject_fault(r.fault)
            inc = Incident(id=f"{pick}-run{n}", source="alert", text=r.text, logs=list(r.logs))
        else:
            svc = FakeService(seed=2000 + n)
            svc.inject_fault(own_fault)
            inc = Incident(id=f"LIVE-{n:03d}", source="alert", text=own_text,
                           logs=[l for l in own_logs.splitlines() if l.strip()])
        state = run_incident(rt, inc, svc, graph=graph)
        state["_healthy_after"] = svc.health_check()["healthy"]
        state["_actions"] = svc.action_history
        st.session_state.runs.append(state)

    if st.session_state.runs:
        s = st.session_state.runs[-1]
        with c2:
            status = s["status"]
            color = "green" if status == "resolved" else "orange"
            st.subheader(f"{s['incident']['id']}: :{color}[{status}]")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted fault", s["triage"]["fault"]["label"])
            m2.metric("Calibrated confidence", f"{s['guardrail']['confidence']:.2f}")
            m3.metric("Simulated time", f"{s['elapsed_s']:.0f} s")
            m4.metric("Simulator healthy", "yes" if s["_healthy_after"] else "no")
            st.markdown(f"**Alert:** {s['incident']['text']}")
            st.code("\n".join(f"[{i}] {l}" for i, l in enumerate(s["incident"]["logs"])), language="log")

        st.markdown("#### Explanation trace")
        st.dataframe(pd.DataFrame([{
            "": OUTCOME_ICON.get(e["outcome"], ""), "agent": e["agent"], "action": e["action"],
            "confidence": None if e["confidence"] is None else round(e["confidence"], 3),
            "outcome": e["outcome"], "reason": e["detail"].get("reason") or "",
            "evidence": ", ".join(map(str, e["evidence"][:6]))} for e in s["trace"]]),
            hide_index=True, use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.markdown("#### Triage")
            st.dataframe(pd.DataFrame([{"task": k, "label": v["label"], "confidence": round(v["confidence"], 3),
                                        "runner-up": v["runner_up"], "margin": round(v["margin"], 3),
                                        "evidence words": ", ".join(v["evidence_words"])}
                                       for k, v in s["triage"].items()]), hide_index=True)
            st.markdown("#### Guardrail")
            g = s["guardrail"]
            st.write({"decision": g["decision"], "reasons": g["reasons"], "signals": g["signals"],
                      "thresholds (tuned on val)": g["thresholds"]})
            st.markdown("#### Resolution")
            st.write({k: s["resolution"][k] for k in ("runbook", "status", "risk", "matched_because", "rejected")})
            if s.get("validation"):
                st.markdown("#### Validation")
                st.write({k: s["validation"].get(k) for k in ("fixed", "checked", "still_wrong", "probes")})
        with b:
            st.markdown("#### Ranked hypotheses")
            for h in s["diagnosis"]["hypotheses"]:
                with st.expander(f"{h['cause']} — score {h['score']:.2f}", expanded=h is s["diagnosis"]["hypotheses"][0]):
                    st.write(h.get("explanation_llm") or h["explanation"])
                    for eid in h["supporting_evidence"]:
                        doc = rt.kb.get(eid)
                        if doc is not None:
                            st.caption(f"`{eid}` — {doc['text'][:220]}")
                        elif ":log:" in eid:
                            st.caption(f"`{eid}` — {s['incident']['logs'][int(eid.rsplit(':', 1)[1])]}")
        if s.get("handoff"):
            h = s["handoff"]
            st.markdown("### Human handoff report")
            st.error(f"**Why escalated:** {h['why_escalated'].get('text_llm') or h['why_escalated']['text']}  "
                     f"(`{', '.join(h['why_escalated']['reasons'])}`)")
            if h["stuck_at"]:
                st.warning(f"**Stuck at:** {h['stuck_at']['agent']} → {h['stuck_at']['reason']}")
            st.markdown("**Start here:**\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(h["start_here"], 1)))
            st.markdown("**Actions already taken:** " + (json.dumps(h["actions_taken"]) if h["actions_taken"] else "none"))
            st.dataframe(pd.DataFrame(h["evidence"]), hide_index=True, use_container_width=True)
            with st.expander("Full JSON report"):
                st.json(h)

# ---------------------------------------------------------------- escalations / learning
with tab_queue:
    esc = [s for s in st.session_state.runs if s["status"] != "resolved"]
    done = [s for s in st.session_state.runs if s["status"] == "resolved"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Runs this session", len(st.session_state.runs))
    k2.metric("Auto-resolved (verified)", sum(1 for s in done if s["_healthy_after"]))
    k3.metric("Escalated to human", len(esc))
    if esc:
        pick = st.selectbox("Escalated incident", [s["incident"]["id"] for s in esc])
        s = next(x for x in esc if x["incident"]["id"] == pick)
        st.write(s["handoff"]["why_escalated"])
        st.markdown("**Record the human's final resolution (adds it to the LIVE incident index; "
                    "the frozen evaluation index is never changed).**")
        with st.form("resolve"):
            rc = st.selectbox("Root cause", FAULTS + ["other"])
            fix = st.text_input("Final fix applied by the human", "manual certificate rotation")
            notes = st.text_area("Notes", "")
            if st.form_submit_button("Save human resolution"):
                case = learning.record_human_resolution(s["incident"], rc, fix, notes)
                st.success(f"Saved {case['id']}. Rebuilding the live index ...")
                runtime_and_graph.clear()
                st.rerun()
    else:
        st.info("No escalations yet in this session. Inject network_partition or cert_expired to see one.")
    st.markdown("#### Learned cases (live index only)")
    from rag.store import load_learned
    learned = load_learned()
    st.dataframe(pd.DataFrame([{k: c[k] for k in ("id", "root_cause", "fix", "recorded_at")} for c in learned])
                 if learned else pd.DataFrame(), hide_index=True)
    st.caption(f"Live KB manifest: {rt.kb.manifest['n_docs']}  learned={rt.kb.manifest['n_learned']}")

# ---------------------------------------------------------------- evaluation
with tab_eval:
    p = RES / "eval_results.json"
    if not p.exists():
        st.info("Run `python -m eval.run_all` to generate evaluation results.")
    else:
        r = json.loads(p.read_text())
        st.caption("Measured on the simulator test split / public data. MTTR is SIMULATED (see method).")
        st.markdown("#### Pipeline (full system and ablations)")
        keys = ["diagnosis_accuracy", "auto_resolution_rate_all", "verified_resolution_rate_all",
                "false_action_rate", "false_success_claims", "escalation_rate", "citation_faithfulness"]
        st.dataframe(pd.DataFrame({n: {k: v[k] for k in keys} for n, v in r["pipeline"].items()}))
        st.markdown("#### Stress set: fixes that do not hold")
        st.dataframe(pd.DataFrame({n: {k: v[k] for k in keys} for n, v in r["stress_persistent_faults"].items()}))
        st.markdown("#### Triage")
        st.dataframe(pd.DataFrame(r["triage"]).T)
        st.markdown("#### Retrieval")
        st.dataframe(pd.DataFrame({k: v for k, v in r["retrieval_test"].items() if k != "note"}).T)
        st.markdown("#### MTTR (simulated)")
        st.dataframe(pd.DataFrame({k: r["mttr"][k] for k in ("manual_baseline", "agent_pipeline")}).T)
        st.caption(r["mttr"]["method"])
        cols = st.columns(3)
        for col, img in zip(cols, ["mttr_cdf.png", "reliability_val.png", "risk_coverage.png"]):
            if (RES / "plots" / img).exists():
                col.image(str(RES / "plots" / img))

# ---------------------------------------------------------------- runbooks
with tab_rb:
    st.markdown("Only these whitelisted runbooks can run, and only against the simulator.")
    st.dataframe(pd.DataFrame([rb.to_dict() for rb in RUNBOOKS.values()]), hide_index=True, use_container_width=True)
