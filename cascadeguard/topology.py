"""Step 0 - the system we are protecting: a microservice dependency graph.

An edge  caller -> callee  means "caller needs callee to work".
`weight` is how HARD that dependency is (0.9 = caller breaks without it,
0.3 = soft dependency such as a cache the caller can live without).
A failure therefore travels in the OPPOSITE direction of the arrows:
callee -> caller.
"""
import networkx as nx

# name -> kind (kind only affects baseline latency and how the node is drawn)
SERVICES = {
    "gateway": "gateway",
    "auth": "service",
    "orders": "service",
    "payments": "service",
    "inventory": "service",
    "notifications": "service",
    "user-db": "database",
    "orders-db": "database",
    "payments-db": "database",
    "inventory-db": "database",
    "cache": "cache",
    "queue": "queue",
}

# (caller, callee, dependency strength)
EDGES = [
    ("gateway", "auth", 0.85),
    ("gateway", "orders", 0.85),
    ("gateway", "payments", 0.60),
    ("gateway", "inventory", 0.70),
    ("auth", "user-db", 0.90),
    ("auth", "cache", 0.30),
    ("orders", "orders-db", 0.90),
    ("orders", "payments", 0.80),
    ("orders", "queue", 0.50),
    ("orders", "cache", 0.30),
    ("orders", "notifications", 0.25),
    ("payments", "payments-db", 0.90),
    ("payments", "orders-db", 0.60),
    ("inventory", "inventory-db", 0.90),
    ("inventory", "cache", 0.30),
    ("notifications", "queue", 0.80),
]


def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for name, kind in SERVICES.items():
        G.add_node(name, kind=kind)
    for caller, callee, w in EDGES:
        G.add_edge(caller, callee, weight=w)
    return G


def blast_radius(G: nx.DiGraph, service: str) -> set:
    """Every service that (directly or indirectly) depends on `service`."""
    return nx.ancestors(G, service)
