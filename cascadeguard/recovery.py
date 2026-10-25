"""Step 5 - RECOMMEND: compare recovery strategies by simulating each one.

For every strategy we re-run the cascade simulation with the strategy's effect
applied, then compare the expected damage:

  total cost = expected failed service-minutes ("downtime") + side-effect cost

IMPORTANT (be honest in the pitch): the effectiveness numbers below are
expert-style PRIORS we encoded, not measured data. They are easy to edit and
are shown to the user in the dashboard. What is computed from data is the
diagnosed fault type, the root cause and the cascade.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from .propagation import monte_carlo

# strategy -> (minutes until it takes effect, {fault_type: probability it fixes the root cause})
EFFECT = {
    "restart":   (2, {"db_slowdown": 0.60, "memory_leak": 0.90, "cpu_saturation": 0.50, "bad_deploy": 0.20, "disk_io": 0.30}),
    "failover":  (3, {"db_slowdown": 0.85, "memory_leak": 0.70, "cpu_saturation": 0.70, "bad_deploy": 0.10, "disk_io": 0.85}),
    "scale_out": (4, {"db_slowdown": 0.20, "memory_leak": 0.10, "cpu_saturation": 0.90, "bad_deploy": 0.00, "disk_io": 0.10}),
    "rollback":  (2, {"db_slowdown": 0.05, "memory_leak": 0.05, "cpu_saturation": 0.05, "bad_deploy": 0.95, "disk_io": 0.05}),
}
SIDE_COST = {"restart": 1.0, "failover": 1.0, "scale_out": 0.5, "rollback": 1.0}

LABELS = {
    "do_nothing": "Do nothing",
    "restart": "Restart the root-cause service",
    "failover": "Fail over to a healthy replica",
    "scale_out": "Scale out (add instances)",
    "rollback": "Roll back the last deployment",
    "isolate": "Isolate it (circuit-break its callers)",
    "shed_load": "Shed load at the gateway (rate-limit)",
}


def evaluate_strategies(G: nx.DiGraph, root: str, fault_type: str, initial_failed, horizon: int = 15,
                        n_sims: int = 1000, seed: int = 0) -> pd.DataFrame:
    callers = list(G.predecessors(root))
    rows = []
    for key in LABELS:
        edge_scale, recovery, side = {}, {}, 0.0
        if key in EFFECT:
            minutes, probs = EFFECT[key]
            recovery = {root: (minutes, probs.get(fault_type, 0.05))}
            side = SIDE_COST[key]
        elif key == "isolate":      # callers stop calling the broken service (they degrade gracefully)
            edge_scale = {(c, root): 0.15 for c in callers}
            side = 0.25 * horizon * len(callers)
        elif key == "shed_load":    # drop part of the traffic so retries stop piling up
            edge_scale = {(u, v): 0.5 for u, v in G.edges}
            side = 0.4 * horizon
        res = monte_carlo(G, initial_failed, horizon, n_sims, edge_scale, recovery, intrinsic={root}, seed=seed)
        rows.append(dict(strategy=key, action=LABELS[key],
                         expected_failed_services=res.expected_failed_services,
                         downtime=res.expected_failed_steps, side_effect=side,
                         total_cost=res.expected_failed_steps + side))
    df = pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)
    base = float(df.loc[df.strategy == "do_nothing", "total_cost"].iloc[0])
    df["improvement_pct"] = (base - df.total_cost) / base * 100 if base > 0 else 0.0
    return df
