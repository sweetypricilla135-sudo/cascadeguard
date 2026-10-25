import itertools

import pytest

from cascadeguard.explain import build_facts, explain, template_explanation
from cascadeguard.pipeline import run_pipeline
from cascadeguard.propagation import monte_carlo
from cascadeguard.simulator import FAULTS, Scenario
from cascadeguard.topology import SERVICES, build_graph


def test_graph_is_acyclic():
    import networkx as nx
    assert nx.is_directed_acyclic_graph(build_graph())


@pytest.mark.parametrize("root,fault", list(itertools.product(["orders-db", "user-db", "cache", "payments"], FAULTS)))
def test_root_cause_and_fault_found(root, fault):
    r = run_pipeline(Scenario(root=root, fault_type=fault, seed=3), n_sims=200)
    assert r.cause["service"] == root
    assert r.cause["fault_type"] == fault


def test_no_anomaly_before_fault():
    assert run_pipeline(Scenario(), now=35).cause is None


def test_prediction_flags_dependents():
    r = run_pipeline(Scenario(root="orders-db"), n_sims=500)
    assert r.prediction.fail_prob["orders"] > 0.8
    assert r.prediction.fail_prob["user-db"] < 0.05


def test_recommendation_matches_fault():
    best = lambda ft: run_pipeline(Scenario(fault_type=ft), n_sims=500).recovery.iloc[0]["strategy"]
    assert best("bad_deploy") == "rollback"
    assert best("cpu_saturation") == "scale_out"
    assert best("memory_leak") == "restart"


def test_fixing_never_worse_than_nothing_when_effective():
    G = build_graph()
    base = monte_carlo(G, ["orders-db"], intrinsic={"orders-db"}, seed=1)
    fixed = monte_carlo(G, ["orders-db"], intrinsic={"orders-db"}, recovery={"orders-db": (2, 1.0)}, seed=1)
    assert fixed.expected_failed_steps < base.expected_failed_steps


def test_explainer_falls_back_without_ollama():
    f = build_facts(run_pipeline(Scenario(), n_sims=200))
    text, used, _ = explain(f, "auto", host="http://localhost:1")   # nothing listens there
    assert used == "template" and text == template_explanation(f)
