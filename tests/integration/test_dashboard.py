"""Streamlit AppTest smoke check for dashboard rendering."""

from streamlit.testing.v1 import AppTest

from nnbench.store import ResultStore


def test_dashboard_renders_sample_database_without_exception(monkeypatch, tmp_path) -> None:
    database = tmp_path / "sample.sqlite"
    ResultStore(database).save_result(
        {
            "accuracy": 0.9,
            "latency_mean_ms": 5.0,
            "latency_p50_ms": 4.8,
            "model_size_mb": 1.2,
            "throughput_samples_per_second": 200.0,
            "parameter_count": 100,
            "device": "cpu",
        },
        experiment_name="app-test",
        config={"seed": 42},
        metadata={"model": "toy", "optimizer": "baseline", "hardware": {"cpu_count": 4}},
    )
    monkeypatch.setenv("NNBENCH_DB", str(database))
    app = AppTest.from_file("dashboard/app.py", default_timeout=20).run()
    assert not app.exception
    assert any("Benchmark runs" in item.value for item in app.subheader)
