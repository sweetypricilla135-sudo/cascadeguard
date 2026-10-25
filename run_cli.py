"""Terminal version of the demo (no Streamlit needed):  python run_cli.py --root orders-db --fault db_slowdown"""
import argparse

from cascadeguard.explain import build_facts, explain
from cascadeguard.pipeline import run_pipeline
from cascadeguard.simulator import FAULTS, Scenario
from cascadeguard.topology import SERVICES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="orders-db", choices=list(SERVICES))
    ap.add_argument("--fault", default="db_slowdown", choices=list(FAULTS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--minutes-after", type=int, default=6, help="how long after the fault starts we look")
    ap.add_argument("--engine", default="auto", choices=["auto", "template", "ollama"])
    a = ap.parse_args()

    scn = Scenario(root=a.root, fault_type=a.fault, seed=a.seed)
    res = run_pipeline(scn, now=scn.t0 + a.minutes_after)
    if res.cause is None:
        print("No anomaly detected yet."); return
    print(f"\n=== Observation at minute {res.now} (fault injected at {scn.t0}) ===")
    print("\nRoot-cause ranking:\n", res.ranking.round(2).head(5).to_string())
    print("\nFailure risk (next %d min):" % res.horizon)
    print(res.validation[["service", "predicted_prob", "eta_min", "actually_failed"]].round(2).head(6).to_string(index=False))
    print("\nRecovery options:\n", res.recovery[["action", "downtime", "side_effect", "total_cost", "improvement_pct"]].round(1).to_string(index=False))
    text, used, note = explain(build_facts(res), a.engine)
    print(f"\n--- Explanation [{used}] {note}\n{text}")
    print("\nValidation vs hidden truth:", res.summary)


if __name__ == "__main__":
    main()
