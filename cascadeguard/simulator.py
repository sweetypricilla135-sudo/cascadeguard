"""Step 1 - SIMULATE: generate realistic metrics + logs and inject one fault.

The problem statement explicitly allows simulated logs, so this module is our
"data source". A single root service is broken at time t0. Its damage then
spreads to every service that depends on it, with a delay, exactly like a real
cascading failure. We keep the hidden ground truth (`truth`) so the dashboard
can later check whether our predictions were right.

One time step = one minute.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .topology import build_graph

START = pd.Timestamp("2026-10-25 10:00:00")

# fault type -> human label + the log lines a broken service would print
FAULTS = {
    "db_slowdown": {
        "label": "Connection-pool exhaustion / slow queries",
        "messages": [
            "connection pool exhausted (active=100/100)",
            "slow query detected: 8420ms",
            "lock wait timeout exceeded",
        ],
        "keywords": ["connection pool", "slow query", "lock wait"],
    },
    "memory_leak": {
        "label": "Memory leak",
        "messages": [
            "heap usage at 97%, GC overhead limit exceeded",
            "OutOfMemoryError: Java heap space",
            "container memory near limit",
        ],
        "keywords": ["heap", "outofmemory", "memory near limit"],
    },
    "cpu_saturation": {
        "label": "CPU saturation",
        "messages": [
            "CPU throttling detected (usage 98%)",
            "worker threads starved, request queue growing",
            "event loop lag 2300ms",
        ],
        "keywords": ["cpu throttling", "worker threads", "event loop"],
    },
    "disk_io": {
        "label": "Disk I/O saturation",
        "messages": [
            "disk I/O wait 74%, write latency high",
            "fsync took 5200ms",
            "no space left on device warning (volume 96% full)",
        ],
        "keywords": ["disk i/o", "fsync", "no space left"],
    },
    "bad_deploy": {
        "label": "Bad deployment / release regression",
        "messages": [
            "NullPointerException in handler after deploy v2.3.1",
            "unhandled exception: config key missing (release v2.3.1)",
            "health check failing since deploy v2.3.1",
        ],
        "keywords": ["after deploy", "release v", "since deploy"],
    },
}

BASE_LATENCY = {"gateway": 40.0, "service": 70.0, "database": 18.0, "cache": 3.0, "queue": 10.0}

UPSTREAM_MSGS = [
    "timeout calling {dep} after 5000ms",
    "retrying request to {dep} (attempt 3/3)",
    "circuit breaker half-open for {dep}",
    "503 Service Unavailable from {dep}",
]
UPSTREAM_PREFIXES = ("timeout calling", "retrying request", "circuit breaker", "503 Service")
NORMAL_MSGS = ["request handled in {ms:.0f}ms", "health check ok", "GET /api 200 in {ms:.0f}ms", "cache lookup ok"]


@dataclass(frozen=True)
class Scenario:
    root: str = "orders-db"
    fault_type: str = "db_slowdown"
    t0: int = 40           # minute at which the fault starts
    n_steps: int = 120
    ramp: int = 8          # minutes for the fault to reach full strength
    seed: int = 7
    lines_per_step: int = 3


@dataclass
class SimResult:
    scenario: Scenario
    metrics: pd.DataFrame   # long format: time, service, latency_ms, error_rate, cpu, mem, disk_io
    logs: pd.DataFrame      # time, timestamp, service, level, message
    truth: pd.DataFrame     # HIDDEN ground-truth degradation 0..1 (rows=time, cols=service)


def simulate(scn: Scenario) -> SimResult:
    rng = np.random.default_rng(scn.seed)
    G = build_graph()
    names = list(G.nodes)
    idx = {n: i for i, n in enumerate(names)}
    T, S = scn.n_steps, len(names)
    deg = np.zeros((T, S))   # how degraded each service is (0 healthy .. 1 dead)
    fault = np.zeros((T, S))  # intrinsic fault strength (only the root has one)

    for t in range(T):
        if t >= scn.t0:
            fault[t, idx[scn.root]] = min(1.0, (t - scn.t0 + 1) / scn.ramp)
        for n in names:
            i = idx[n]
            propagated = 0.0
            if t > 0:
                for d in G.successors(n):
                    # a caller only suffers once its dependency is clearly unhealthy
                    propagated += G[n][d]["weight"] * max(0.0, deg[t - 1, idx[d]] - 0.25)
            target = min(1.0, fault[t, i] + 1.3 * propagated)
            prev = deg[t - 1, i] if t > 0 else 0.0
            deg[t, i] = 0.55 * prev + 0.45 * target   # smoothing = realistic delay

    metric_rows, log_rows = [], []
    for t in range(T):
        ts = START + pd.Timedelta(minutes=t)
        for n in names:
            i = idx[n]
            kind = G.nodes[n]["kind"]
            dg, ft = deg[t, i], fault[t, i]
            is_root = n == scn.root
            base = BASE_LATENCY[kind]
            lat = base * (1 + 8 * dg ** 1.3) * rng.lognormal(0, 0.05)
            err = float(np.clip(0.005 + 0.75 * dg ** 1.4 + rng.normal(0, 0.004), 0, 1))
            cpu = float(np.clip(0.30 + 0.25 * dg + (0.55 * ft if is_root and scn.fault_type == "cpu_saturation" else 0) + rng.normal(0, 0.015), 0, 1))
            mem = float(np.clip(0.45 + 0.05 * dg + (0.50 * ft if is_root and scn.fault_type == "memory_leak" else 0) + rng.normal(0, 0.01), 0, 1))
            disk = float(np.clip(0.10 + 0.05 * dg + (0.70 * ft if is_root and scn.fault_type == "disk_io" else 0) + rng.normal(0, 0.01), 0, 1))
            metric_rows.append((t, n, lat, err, cpu, mem, disk))

            bad_deps = [(deg[t, idx[d]], d) for d in G.successors(n) if deg[t, idx[d]] > 0.25]
            p_err = min(0.95, err * 1.3)
            for _ in range(scn.lines_per_step):
                if rng.random() < p_err:
                    level = "ERROR"
                    if is_root and ft > 0.15 and rng.random() < 0.7:
                        msg = str(rng.choice(FAULTS[scn.fault_type]["messages"]))
                    elif bad_deps:
                        worst = max(bad_deps)[1]
                        msg = str(rng.choice(UPSTREAM_MSGS)).format(dep=worst)
                    else:
                        msg = "request failed with status 500"
                elif dg > 0.15 and rng.random() < 0.5:
                    level, msg = "WARN", f"latency above SLO: {lat:.0f}ms"
                else:
                    level, msg = "INFO", str(rng.choice(NORMAL_MSGS)).format(ms=lat)
                log_rows.append((t, ts, n, level, msg))

    metrics = pd.DataFrame(metric_rows, columns=["time", "service", "latency_ms", "error_rate", "cpu", "mem", "disk_io"])
    logs = pd.DataFrame(log_rows, columns=["time", "timestamp", "service", "level", "message"])
    truth = pd.DataFrame(deg, columns=names)
    truth.index.name = "time"
    return SimResult(scn, metrics, logs, truth)
