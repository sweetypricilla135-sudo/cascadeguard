"""CascadeGuard dashboard.   Run:  streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from cascadeguard.explain import DEFAULT_HOST, DEFAULT_MODEL, build_facts, explain, ollama_status
from cascadeguard.pipeline import BASELINE_END, run_pipeline
from cascadeguard.recovery import EFFECT
from cascadeguard.simulator import FAULTS, Scenario
from cascadeguard.topology import SERVICES
from cascadeguard.viz import graph_dot

st.set_page_config(page_title="CascadeGuard", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner="Simulating, detecting, predicting...")
def cached_run(root, fault, seed, minutes_after, horizon, n_sims):
    scn = Scenario(root=root, fault_type=fault, seed=seed)
    return run_pipeline(scn, now=scn.t0 + minutes_after, horizon=horizon, n_sims=n_sims)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Scenario")
    root = st.selectbox("Service to break (hidden from the AI)", list(SERVICES), index=list(SERVICES).index("orders-db"))
    fault = st.selectbox("Fault type", list(FAULTS), format_func=lambda k: FAULTS[k]["label"])
    minutes_after = st.slider("Observe N minutes after the fault starts", 1, 30, 6,
                              help="Early = the cascade has not happened yet, so the system must PREDICT it.")
    horizon = st.slider("Prediction horizon (min)", 5, 30, 15)
    n_sims = st.select_slider("Monte-Carlo simulations", [200, 500, 1000, 2000], value=1000)
    seed = st.number_input("Random seed", 0, 9999, 7)
    st.divider()
    st.header("Explanation engine")
    engine = st.radio("Engine", ["auto", "template", "ollama"], help="auto = Ollama if available, else built-in template")
    model = st.text_input("Ollama model", DEFAULT_MODEL)
    ok, _, msg = ollama_status(DEFAULT_HOST, model)
    st.caption(("🟢 " if ok else "⚪ ") + msg)

res = cached_run(root, fault, int(seed), minutes_after, horizon, n_sims)

st.title("🛡️ CascadeGuard")
st.caption("Predict cascading software failures, find the root cause, and choose the best recovery - before the whole system crashes.")

if res.cause is None:
    st.success("No anomaly detected - all services look normal. Increase 'minutes after the fault starts'.")
    st.stop()

cause, pred, rec = res.cause, res.prediction, res.recovery
at_risk = [s for s in pred.fail_prob.index if s not in res.already_failed and pred.fail_prob[s] >= 0.5]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unhealthy services now", int(res.detection.anomalous.sum()))
c2.metric("Likely root cause", cause["service"])
c3.metric(f"At risk in next {horizon} min", len(at_risk))
c4.metric("Best recovery", rec.iloc[0]["action"].split(" (")[0])

tabs = st.tabs(["1 Overview", "2 Root cause", "3 Prediction", "4 Recovery", "5 Explanation", "6 Validation"])

# ------------------------------------------------------------------ 1 overview
with tabs[0]:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Service health right now")
        st.caption("Colour = current stress (green healthy → red failing). Thick border = suspected root cause. Dashed = soft dependency.")
        st.graphviz_chart(graph_dot(res.G, res.detection["stress"].to_dict(), cause["service"]))
    with right:
        st.subheader("Latency vs. normal")
        m = res.sim.metrics[res.sim.metrics.time <= res.now]
        piv = m.pivot(index="time", columns="service", values="latency_ms")
        ratio = piv / piv.loc[:BASELINE_END - 1].mean()
        default = list(res.detection.sort_values("stress", ascending=False).index[:4])
        picked = st.multiselect("Services", list(SERVICES), default=default)
        if picked:
            st.line_chart(ratio[picked])
        st.caption("1.0 = normal latency. The fault starts at minute 40.")
        st.subheader("Latest error logs")
        errs = res.sim.logs[(res.sim.logs.time <= res.now) & (res.sim.logs.level == "ERROR")].tail(12)
        st.dataframe(errs[["time", "service", "message"]].iloc[::-1], hide_index=True)

# ------------------------------------------------------------------ 2 root cause
with tabs[1]:
    st.subheader(f"Root cause: `{cause['service']}`")
    st.markdown(f"**Diagnosed fault:** {cause['fault_label']}  (log-based confidence {cause['fault_confidence']:.0%})")
    st.markdown("**Why this service?**\n" + "\n".join(f"- {r}" for r in cause["reasons"]))
    if cause["evidence"]:
        st.markdown("**Log evidence from the culprit:**")
        st.code("\n".join(cause["evidence"]))
    st.markdown("**Ranking of all unhealthy services** (higher = more likely the origin)")
    st.dataframe(res.ranking.round(2))
    st.caption("score = 30% earliness + 35% coverage of the other sick services + 20% healthy dependencies + 15% own (non-upstream) errors")

# ------------------------------------------------------------------ 3 prediction
with tabs[2]:
    st.subheader(f"Which services will fail in the next {horizon} minutes?")
    left, right = st.columns([1, 1])
    with left:
        st.graphviz_chart(graph_dot(res.G, pred.fail_prob.to_dict(), cause["service"]))
        st.caption("Colour = probability of failing (Monte-Carlo cascade simulation).")
    with right:
        table = pd.DataFrame({"failure probability": pred.fail_prob, "expected time to failure (min)": pred.mean_ttf})
        table["status"] = ["failing now" if s in res.already_failed else "at risk" if p >= 0.5 else "watch" if p >= 0.2 else "safe"
                           for s, p in pred.fail_prob.items()]
        st.dataframe(table.sort_values("failure probability", ascending=False).round(2))
        st.line_chart(pd.Series(pred.failed_per_step, name="expected # failed services (if we do nothing)"))

# ------------------------------------------------------------------ 4 recovery
with tabs[3]:
    best = rec.iloc[0]
    st.subheader("Recovery strategies compared")
    st.success(f"**Recommended: {best['action']}** - cuts modelled damage by {best['improvement_pct']:.0f}% vs. doing nothing.")
    show = rec[["action", "downtime", "side_effect", "total_cost", "improvement_pct"]].rename(columns={
        "downtime": "expected downtime (service-min)", "side_effect": "side-effect cost", "total_cost": "total cost", "improvement_pct": "% better than nothing"})
    st.dataframe(show.round(1), hide_index=True)
    st.bar_chart(rec.set_index("action")["total_cost"])
    with st.expander("Assumptions behind the strategy model (editable priors, not measured data)"):
        pri = pd.DataFrame({k: v[1] for k, v in EFFECT.items()}).T
        pri.insert(0, "minutes to take effect", [EFFECT[k][0] for k in pri.index])
        st.dataframe(pri)
        st.caption("Cell = probability the strategy fixes that fault type. Isolation cuts the callers' dependency on the broken service; load-shedding halves cascade speed at the cost of dropped traffic.")

# ------------------------------------------------------------------ 5 explanation
with tabs[4]:
    st.subheader("Incident briefing")
    facts = build_facts(res)
    key = json.dumps([facts, engine, model], sort_keys=True, default=str)
    uses_llm = engine == "ollama" or (engine == "auto" and ok)
    if uses_llm and st.session_state.get("exp_key") != key:
        if st.button("Generate briefing with Ollama"):
            with st.spinner("Asking the local LLM..."):
                st.session_state.exp = explain(facts, engine, model)
                st.session_state.exp_key = key
    elif not uses_llm:
        st.session_state.exp, st.session_state.exp_key = explain(facts, "template"), key
    if st.session_state.get("exp_key") == key:
        text, used, note = st.session_state.exp
        st.markdown(text)
        st.caption(f"Written by: **{used}**" + (f" - {note}" if note else ""))
    st.info("The LLM only *narrates* results computed by the pipeline. It never decides the root cause, the risk or the recommendation - so without Ollama the built-in template gives the same content.")
    with st.expander("Facts handed to the explainer"):
        st.json(facts)

# ------------------------------------------------------------------ 6 validation
with tabs[5]:
    st.subheader("How did we do? (hidden ground truth revealed)")
    s = res.summary
    a, b, c = st.columns(3)
    a.metric("True root cause", res.scenario.root, "✅ found" if s["root_correct"] else "❌ missed")
    b.metric("True fault type", res.scenario.fault_type, "✅ found" if s["fault_correct"] else "❌ missed")
    c.metric("Cascade prediction", f"precision {s['precision']:.0%}", f"recall {s['recall']:.0%}")
    v = res.validation.copy()
    v["eta_min"] = v["eta_min"].round(1)
    st.dataframe(v.round(2), hide_index=True)
    st.caption("predicted_prob came from observations up to the current minute only; 'actually_failed' is read from the simulator's hidden future. Data is simulated, so this checks internal consistency, not real-world accuracy.")
