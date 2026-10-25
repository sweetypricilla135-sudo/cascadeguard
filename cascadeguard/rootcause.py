"""Step 3 - ROOT CAUSE: of all the services that look sick, which one started it?

Every anomalous service is scored on four easy-to-explain clues:

  earliness   (30%)  did it go wrong before the others?
  coverage    (35%)  would a failure HERE explain the other sick services?
                     (are they all callers/ancestors of this service in the graph)
  dep_health  (20%)  are its own dependencies healthy?  If a dependency is also
                     sick, this service is probably a victim, not the cause.
  own_errors  (15%)  share of its error logs that are NOT "timeout calling X"
                     (victims log upstream timeouts, the culprit logs its own errors)

We then read the culprit's logs/metrics to classify WHICH kind of fault it is.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from .simulator import FAULTS, UPSTREAM_PREFIXES

W = {"earliness": 0.30, "coverage": 0.35, "dep_health": 0.20, "own_errors": 0.15}


def _own_error_fraction(logs: pd.DataFrame, service: str, t_from: int, t_to: int) -> float:
    l = logs[(logs.service == service) & (logs.level == "ERROR") & (logs.time >= t_from) & (logs.time <= t_to)]
    if l.empty:
        return 0.0
    upstream = l.message.str.startswith(UPSTREAM_PREFIXES)
    return float((~upstream).mean())


def classify_fault(logs: pd.DataFrame, det: pd.DataFrame, service: str, t_from: int, t_to: int):
    """Return (fault_type, confidence 0..1, evidence log lines)."""
    l = logs[(logs.service == service) & (logs.level == "ERROR") & (logs.time >= t_from) & (logs.time <= t_to)]
    own = l[~l.message.str.startswith(UPSTREAM_PREFIXES)]
    lowered = own.message.str.lower()
    hits = {ft: int(lowered.apply(lambda m: any(k in m for k in spec["keywords"])).sum()) for ft, spec in FAULTS.items()}
    total = sum(hits.values())
    kw_all = [k for spec in FAULTS.values() for k in spec["keywords"]]
    specific = own[lowered.apply(lambda m: any(k in m for k in kw_all))]
    evidence = specific.drop_duplicates("message").tail(4).message.tolist()
    if total > 0:
        best = max(hits, key=hits.get)
        return best, hits[best] / total, evidence
    row = det.loc[service]
    sig = {"cpu_saturation": row.cpu_z, "memory_leak": row.mem_z, "disk_io": row.disk_z}
    best = max(sig, key=sig.get)
    if sig[best] > 4:
        return best, 0.5, evidence
    return "db_slowdown", 0.2, evidence


def rank_root_causes(G: nx.DiGraph, det: pd.DataFrame, logs: pd.DataFrame, now: int):
    sick = det[det.anomalous]
    if sick.empty:
        return pd.DataFrame(), None
    sick_set = set(sick.index)
    t_min, t_max = sick.onset.min(), sick.onset.max()
    rows = {}
    for c in sick.index:
        onset = sick.loc[c, "onset"]
        earliness = 1.0 if t_max == t_min else 1 - (onset - t_min) / (t_max - t_min)
        others = sick_set - {c}
        coverage = 1.0 if not others else len(others & nx.ancestors(G, c)) / len(others)
        deps = list(G.successors(c))
        dep_health = 1.0 if not deps else 1 - sum(d in sick_set for d in deps) / len(deps)
        own = _own_error_fraction(logs, c, int(onset), now)
        score = W["earliness"] * earliness + W["coverage"] * coverage + W["dep_health"] * dep_health + W["own_errors"] * own
        rows[c] = dict(score=score, earliness=earliness, coverage=coverage, dep_health=dep_health, own_errors=own, onset=int(onset))
    ranking = pd.DataFrame.from_dict(rows, orient="index").sort_values("score", ascending=False)

    top = ranking.index[0]
    ftype, fconf, evidence = classify_fault(logs, det, top, int(ranking.loc[top, "onset"]), now)
    margin = float(ranking.score.iloc[0] - (ranking.score.iloc[1] if len(ranking) > 1 else 0))
    top_row = ranking.iloc[0]
    reasons = [
        f"first to go wrong (onset minute {int(top_row.onset)})" if top_row.earliness >= 0.99 else f"went wrong early (onset minute {int(top_row.onset)})",
        f"explains {top_row.coverage:.0%} of the other unhealthy services through the dependency graph",
        "its own dependencies are healthy" if top_row.dep_health >= 0.99 else f"{top_row.dep_health:.0%} of its dependencies are healthy",
        f"{top_row.own_errors:.0%} of its errors are its own (not upstream timeouts)",
    ]
    info = dict(service=top, fault_type=ftype, fault_label=FAULTS[ftype]["label"], fault_confidence=fconf,
                evidence=evidence, margin=margin, reasons=reasons)
    return ranking, info
