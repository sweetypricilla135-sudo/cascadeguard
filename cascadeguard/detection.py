"""Step 2 - DETECT: which services look abnormal right now, and since when?

Method (simple on purpose, easy to explain to judges):
  * learn each service's normal latency / error rate from a quiet baseline window
  * z-score every new minute against that baseline
  * a service is ANOMALOUS once it is >= z_thresh sigmas off for `persist`
    minutes in a row; its `onset` is the first minute of that run
  * `stress` (0..1) is a plain health number used later by the predictor
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect(metrics: pd.DataFrame, now: int, baseline_end: int, z_thresh: float = 4.0, persist: int = 2) -> pd.DataFrame:
    m = metrics[metrics["time"] <= now]
    piv = {c: m.pivot(index="time", columns="service", values=c) for c in ["latency_ms", "error_rate", "cpu", "mem", "disk_io"]}
    base = {c: p.loc[: baseline_end - 1] for c, p in piv.items()}

    def z(col, floor_rel=0.0, floor_abs=0.0):
        mu = base[col].mean()
        sd = np.maximum(base[col].std(), np.maximum(floor_rel * mu, floor_abs))
        return (piv[col] - mu) / sd, mu

    z_lat, mu_lat = z("latency_ms", floor_rel=0.08)
    z_err, _ = z("error_rate", floor_abs=0.01)
    z_cpu, _ = z("cpu", floor_abs=0.03)
    z_mem, _ = z("mem", floor_abs=0.03)
    z_disk, _ = z("disk_io", floor_abs=0.03)

    flag = (z_lat >= z_thresh) | (z_err >= z_thresh)
    post = flag.loc[baseline_end:]
    run = post.rolling(persist).sum() == persist

    last2 = lambda df: df.iloc[-2:].mean()
    lat_ratio = last2(piv["latency_ms"]) / mu_lat
    stress = np.clip(np.maximum(last2(piv["error_rate"]) / 0.5, (lat_ratio - 1) / 6), 0, 1)

    rows = {}
    for s in flag.columns:
        hit = run[s]
        onset = int(hit.idxmax()) - (persist - 1) if hit.any() else None
        rows[s] = {
            "anomalous": bool(hit.any()),
            "onset": onset,
            "latency_z": float(last2(z_lat)[s]),
            "error_z": float(last2(z_err)[s]),
            "cpu_z": float(last2(z_cpu)[s]),
            "mem_z": float(last2(z_mem)[s]),
            "disk_z": float(last2(z_disk)[s]),
            "stress": float(stress[s]),
        }
    det = pd.DataFrame.from_dict(rows, orient="index")
    det["status"] = pd.cut(det["stress"], [-0.01, 0.3, 0.6, 1.01], labels=["healthy", "degraded", "failing"])
    return det
