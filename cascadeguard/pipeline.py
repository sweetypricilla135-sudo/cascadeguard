"""Glue: run the whole chain  simulate -> detect -> root cause -> predict -> recommend."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .detection import detect
from .propagation import MCResult, monte_carlo
from .recovery import evaluate_strategies
from .rootcause import rank_root_causes
from .simulator import Scenario, SimResult, simulate
from .topology import build_graph

BASELINE_END = 30      # first 30 minutes = "normal behaviour" the detector learns from
FAIL_STRESS = 0.6      # stress above this = service counts as already failing


@dataclass
class PipelineResult:
    scenario: Scenario
    now: int
    horizon: int
    sim: SimResult
    G: object
    detection: pd.DataFrame
    ranking: pd.DataFrame
    cause: dict | None
    already_failed: list
    prediction: MCResult | None
    recovery: pd.DataFrame | None
    validation: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)


def run_pipeline(scn: Scenario, now: int | None = None, horizon: int = 15, n_sims: int = 1000, mc_seed: int = 0) -> PipelineResult:
    sim = simulate(scn)
    G = build_graph()
    now = scn.t0 + 6 if now is None else now
    now = max(BASELINE_END + 2, min(now, scn.n_steps - 1 - horizon))

    det = detect(sim.metrics, now, baseline_end=min(BASELINE_END, scn.t0 - 5))
    ranking, cause = rank_root_causes(G, det, sim.logs, now)
    if cause is None:
        return PipelineResult(scn, now, horizon, sim, G, det, ranking, None, [], None, None)

    already_failed = sorted(set(det.index[det.stress >= FAIL_STRESS]) | {cause["service"]})
    pred = monte_carlo(G, already_failed, horizon, n_sims, intrinsic={cause["service"]}, seed=mc_seed)
    rec = evaluate_strategies(G, cause["service"], cause["fault_type"], already_failed, horizon, n_sims, seed=mc_seed)

    # --- validation against the hidden ground truth (never used by the predictor) ---
    future = sim.truth.loc[now + 1: now + horizon].max()
    rows = []
    for s in G.nodes:
        if s in already_failed:
            continue
        rows.append(dict(service=s, predicted_prob=float(pred.fail_prob[s]), eta_min=pred.mean_ttf[s],
                         actually_failed=bool(future[s] >= FAIL_STRESS)))
    val = pd.DataFrame(rows).sort_values("predicted_prob", ascending=False).reset_index(drop=True)

    flagged = val.predicted_prob >= 0.5
    tp = int((flagged & val.actually_failed).sum())
    summary = dict(
        root_correct=cause["service"] == scn.root,
        fault_correct=cause["fault_type"] == scn.fault_type,
        top3_contains_root=scn.root in list(ranking.index[:3]),
        precision=tp / max(1, int(flagged.sum())),
        recall=tp / max(1, int(val.actually_failed.sum())),
        best_action=rec.iloc[0]["action"],
    )
    return PipelineResult(scn, now, horizon, sim, G, det, ranking, cause, already_failed, pred, rec, val, summary)
