"""Streamlit dashboard for saved nnbench runs."""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from nnbench.analysis.plots import plot_pareto
from nnbench.analysis.report import flatten_result
from nnbench.store import ResultStore

st.set_page_config(page_title="nnbench results", layout="wide")
st.title("Neural Network Optimization Benchmarks")
db_path = Path(os.environ.get("NNBENCH_DB", "artifacts/nnbench.sqlite"))
rows = ResultStore(db_path).query_results()
if not rows:
    st.info(
        f"No runs are stored in {db_path}. Run `nnbench run experiment=demo --synthetic` first."
    )
    st.stop()

models = sorted({row["model"] for row in rows})
selected_model = st.sidebar.selectbox("Model", ["All", *models])
filtered = (
    rows if selected_model == "All" else [row for row in rows if row["model"] == selected_model]
)
flat_rows = [flatten_result(row) for row in filtered]
frame = pd.DataFrame(flat_rows)
tabs = st.tabs(["Results", "Pareto", "Latency", "Compare", "Hardware"])

with tabs[0]:
    st.subheader("Benchmark runs")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="nnbench-results.csv",
        mime="text/csv",
    )

with tabs[1]:
    axis = st.selectbox("Trade-off axis", ["latency_mean_ms", "model_size_mb"])
    st.plotly_chart(plot_pareto(filtered, x=axis), use_container_width=True)

with tabs[2]:
    st.plotly_chart(
        px.histogram(frame, x="latency_mean_ms", color="optimizer", nbins=30),
        use_container_width=True,
    )

with tabs[3]:
    choices = sorted({row["optimizer"] for row in filtered})
    selected = st.multiselect(
        "Optimizers to compare", choices, default=choices[: min(3, len(choices))]
    )
    st.dataframe(
        frame[frame["optimizer"].isin(selected)].sort_values("optimizer"),
        use_container_width=True,
        hide_index=True,
    )

with tabs[4]:
    st.subheader("Hardware and runtime")
    st.json(filtered[-1].get("hardware", {}))
