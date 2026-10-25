"""Step 6 - EXPLAIN in plain English.

DESIGN RULE: the LLM never makes a decision. Detection, root cause, prediction
and recommendation are all computed by the code above. The LLM only narrates
those structured facts. That is why a deterministic TEMPLATE can replace it
with identical content when Ollama is not installed (e.g. on a judge's laptop).

Engines:  "auto"      -> Ollama if reachable, otherwise template
          "ollama"    -> Ollama only (falls back to template with a warning on error)
          "template"  -> always the built-in template
"""
from __future__ import annotations

import json
import os

import requests

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def build_facts(res) -> dict:
    """Everything the explanation may talk about, as plain data."""
    cause, pred, rec = res.cause, res.prediction, res.recovery
    risky = [(s, float(pred.fail_prob[s]), pred.mean_ttf[s]) for s in pred.fail_prob.index if s not in res.already_failed]
    risky = sorted(risky, key=lambda x: -x[1])
    best, base = rec.iloc[0], rec[rec.strategy == "do_nothing"].iloc[0]
    runner = rec.iloc[1]
    return dict(
        observation_minute=res.now,
        horizon_min=res.horizon,
        failing_now=res.already_failed,
        unhealthy_services=[s for s in res.detection.index[res.detection.anomalous]],
        root_cause=dict(service=cause["service"], fault=cause["fault_label"], fault_confidence=round(cause["fault_confidence"], 2),
                        reasons=cause["reasons"], log_evidence=cause["evidence"]),
        next_to_fail=[dict(service=s, probability=round(p, 2), eta_min=None if (e != e) else round(float(e), 1)) for s, p, e in risky if p >= 0.2][:5],
        recommended=dict(action=best["action"], total_cost=round(float(best.total_cost), 1), improvement_pct=round(float(best.improvement_pct), 0)),
        runner_up=dict(action=runner["action"], total_cost=round(float(runner.total_cost), 1)),
        do_nothing_cost=round(float(base.total_cost), 1),
    )


def template_explanation(f: dict) -> str:
    rc = f["root_cause"]
    nxt = f["next_to_fail"]
    lines = [
        f"**What is happening.** At minute {f['observation_minute']}, {len(f['unhealthy_services'])} services look abnormal "
        f"({', '.join(f['unhealthy_services'])}). {len(f['failing_now'])} are already failing.",
        "",
        f"**Root cause.** `{rc['service']}` is the likely origin, diagnosed as *{rc['fault']}* "
        f"(log-based confidence {rc['fault_confidence']:.0%}). Why: " + "; ".join(rc["reasons"]) + ".",
    ]
    if rc["log_evidence"]:
        lines.append("Log evidence: " + " | ".join(f"`{m}`" for m in rc["log_evidence"][:3]))
    lines.append("")
    if nxt:
        lines.append(f"**What fails next** (next {f['horizon_min']} min): " + ", ".join(
            f"`{n['service']}` ({n['probability']:.0%}" + (f", ~{n['eta_min']:.0f} min" if n["eta_min"] is not None else "") + ")" for n in nxt) + ".")
    else:
        lines.append("**What fails next.** No additional service is at significant risk in this window.")
    lines += [
        "",
        f"**Recommended action.** {f['recommended']['action']} - modelled damage drops from {f['do_nothing_cost']} to "
        f"{f['recommended']['total_cost']} service-minutes ({f['recommended']['improvement_pct']:.0f}% better than doing nothing). "
        f"Runner-up: {f['runner_up']['action']} ({f['runner_up']['total_cost']}).",
    ]
    return "\n".join(lines)


def _prompt(facts: dict) -> str:
    return (
        "You are an on-call SRE assistant. Using ONLY the JSON facts below, write a short incident briefing "
        "with exactly four bold-labelled parts: What is happening, Root cause, What fails next, Recommended action. "
        "Be concrete, max 150 words, plain English, do not invent services, numbers or causes.\n\nFACTS:\n"
        + json.dumps(facts, indent=2, default=str)
    )


def ollama_status(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL):
    """Return (available, model_to_use, message)."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=1.5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return False, None, "Ollama not reachable - using built-in template explanation."
    if not names:
        return False, None, "Ollama is running but has no models installed - using template."
    for n in names:
        if n == model or n.split(":")[0] == model.split(":")[0]:
            return True, n, f"Ollama ready ({n})."
    return True, names[0], f"Model '{model}' not installed; using '{names[0]}'."


def explain(facts: dict, engine: str = "auto", model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: int = 60):
    """Return (text, engine_used, note)."""
    if engine == "template":
        return template_explanation(facts), "template", ""
    ok, use_model, msg = ollama_status(host, model)
    if not ok:
        return template_explanation(facts), "template", msg
    try:
        r = requests.post(f"{host}/api/generate", timeout=timeout,
                          json=dict(model=use_model, prompt=_prompt(facts), stream=False, options=dict(temperature=0.2)))
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        if text:
            return text, f"ollama:{use_model}", msg
    except Exception as e:  # network error, timeout, model error ...
        return template_explanation(facts), "template", f"Ollama call failed ({type(e).__name__}) - used template."
    return template_explanation(facts), "template", "Ollama returned nothing - used template."
