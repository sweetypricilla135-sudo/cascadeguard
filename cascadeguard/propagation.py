"""Step 4 - PREDICT: which services will fail next, and how soon?

A Monte-Carlo cascade simulation on the dependency graph:
  * start from the services that are already failing
  * every minute, each healthy service may be "infected" by each failed
    dependency with probability  gain * weight^2  (hard dependencies spread
    fast, soft ones like a cache spread slowly)
  * services heal a couple of minutes after all their dependencies are healthy
  * repeat 1000x -> probability of failure + average time-to-failure per service

The SAME engine is reused in recovery.py to test "what if we restart / isolate ...?".
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class MCResult:
    fail_prob: pd.Series         # P(service fails at some point within horizon)
    mean_ttf: pd.Series          # average minutes until it fails (NaN if never)
    expected_failed_services: float
    expected_failed_steps: float  # "downtime": failed-service-minutes
    failed_per_step: np.ndarray


def monte_carlo(G: nx.DiGraph, initial_failed, horizon: int = 15, n_sims: int = 1000, edge_scale=None,
                recovery=None, intrinsic=None, heal_lag: int = 2, gain: float = 0.4, seed: int = 0) -> MCResult:
    """
    edge_scale : {(caller, callee): factor}  multiply infection probability on that edge
    recovery   : {service: (minutes_until_fixed, success_probability)} for services with their own fault
    intrinsic  : services whose fault does not heal by itself (the root cause)
    """
    rng = np.random.default_rng(seed)
    names = list(G.nodes)
    idx = {n: i for i, n in enumerate(names)}
    S = len(names)
    edge_scale = edge_scale or {}
    A, P = np.zeros((S, S)), np.zeros((S, S))
    for u, v, d in G.edges(data=True):
        A[idx[u], idx[v]] = 0 if edge_scale.get((u, v), 1.0) < 0.3 else 1   # cut edge (circuit breaker) = no longer a dependency
        P[idx[u], idx[v]] = min(1.0, gain * d["weight"] ** 2 * edge_scale.get((u, v), 1.0))

    failed = np.zeros((n_sims, S), bool)
    for s in initial_failed:
        failed[:, idx[s]] = True
    intr = np.zeros(S, bool)
    for s in intrinsic or []:
        intr[idx[s]] = True
    rec = np.full((n_sims, S), np.inf)
    for s, (steps, p) in (recovery or {}).items():
        rec[rng.random(n_sims) < p, idx[s]] = steps

    ever = failed.copy()
    first = np.where(failed, 0.0, np.nan)
    nofail = np.zeros((n_sims, S))
    steps_failed = failed.sum(1).astype(float)
    per_step = [failed.sum(1).mean()]

    for step in range(1, horizon + 1):
        fm = failed.astype(float)
        no_infection = np.prod(1 - fm[:, None, :] * P[None, :, :], axis=2)
        infected = rng.random((n_sims, S)) > no_infection
        has_failed_dep = (fm @ A.T) > 0
        nofail = np.where(has_failed_dep, 0, nofail + 1)
        recovered = failed & intr[None, :] & (step >= rec)          # root cause fixed
        healed = failed & ~intr[None, :] & (nofail >= heal_lag)     # victims recover after deps do
        newly = infected & ~failed
        failed = (failed & ~recovered & ~healed) | newly
        first = np.where(np.isnan(first) & newly, step, first)
        ever |= failed
        steps_failed += failed.sum(1)
        per_step.append(failed.sum(1).mean())

    return MCResult(
        fail_prob=pd.Series(ever.mean(0), index=names),
        mean_ttf=pd.DataFrame(first, columns=names).mean(),
        expected_failed_services=float(ever.sum(1).mean()),
        expected_failed_steps=float(steps_failed.mean()),
        failed_per_step=np.array(per_step),
    )
