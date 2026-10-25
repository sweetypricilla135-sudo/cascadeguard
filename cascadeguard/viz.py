"""Draw the dependency graph as Graphviz DOT (rendered by st.graphviz_chart, no extra install)."""
from __future__ import annotations

SHAPES = {"gateway": "doubleoctagon", "service": "box", "database": "cylinder", "cache": "component", "queue": "parallelogram"}


def risk_color(v: float) -> str:
    """0 -> green, 0.5 -> yellow, 1 -> red."""
    v = max(0.0, min(1.0, float(v)))
    g, y, r = (183, 228, 199), (255, 224, 102), (255, 107, 107)
    a, b, t = (g, y, v * 2) if v < 0.5 else (y, r, (v - 0.5) * 2)
    c = [int(a[i] + (b[i] - a[i]) * t) for i in range(3)]
    return "#%02x%02x%02x" % tuple(c)


def graph_dot(G, values: dict, root: str | None = None) -> str:
    out = ["digraph G {", "rankdir=TB;", 'bgcolor="transparent";', "nodesep=0.35; ranksep=0.5;",
           'node [style=filled, fontname="Helvetica", fontsize=11, fontcolor="#111111"];',
           'edge [color="#8a8a8a", arrowsize=0.6];']
    for n, d in G.nodes(data=True):
        v = values.get(n, 0.0)
        is_root = n == root
        out.append(f'"{n}" [label="{n}\\n{v:.0%}", shape={SHAPES[d["kind"]]}, fillcolor="{risk_color(v)}", '
                   f'color="{"#000000" if is_root else "#555555"}", penwidth={4 if is_root else 1}];')
    for u, v, d in G.edges(data=True):
        out.append(f'"{u}" -> "{v}" [style={"dashed" if d["weight"] < 0.4 else "solid"}];')
    out.append("}")
    return "\n".join(out)
