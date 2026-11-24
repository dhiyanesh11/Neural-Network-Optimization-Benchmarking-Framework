"""Interactive Plotly Pareto charts."""

from collections.abc import Iterable
from typing import Any

from nnbench.analysis.pareto import pareto_front


def plot_pareto(rows: Iterable[dict[str, Any]], x: str = "latency_mean_ms", y: str = "accuracy"):
    """Build a Plotly scatter figure for accuracy against latency or model size."""
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("Pareto plotting requires the 'analytics' extra.") from exc
    materialized = list(rows)
    if not materialized:
        import plotly.graph_objects as go

        return go.Figure().update_layout(title=f"Accuracy vs. {x.replace('_', ' ')}")
    plot_rows = []
    for row in materialized:
        plot_rows.append({**row, **row.get("metrics", {})})
    figure = px.scatter(
        plot_rows,
        x=x,
        y=y,
        color="optimizer" if plot_rows and "optimizer" in plot_rows[0] else None,
        hover_data=(
            [key for key in ("model", "backend", "optimizer") if key in plot_rows[0]]
            if plot_rows
            else None
        ),
        title=f"Accuracy vs. {x.replace('_', ' ')}",
    )
    front = pareto_front(
        materialized,
        latency_key=x if "latency" in x else "latency_mean_ms",
        size_key=x if "size" in x else "model_size_mb",
    )
    front_rows = [{**row, **row.get("metrics", {})} for row in front]
    if front_rows:
        figure.add_scatter(
            x=[row[x] for row in front_rows],
            y=[row[y] for row in front_rows],
            mode="lines+markers",
            name="Pareto front",
        )
    return figure
