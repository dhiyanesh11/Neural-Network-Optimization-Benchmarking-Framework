"""CLI command behavior tests."""

from nnbench.cli import main


def test_list_commands_exit_success(capsys) -> None:
    assert main(["list-models"]) == 0
    assert "mobilenet_v2" in capsys.readouterr().out
    assert main(["list-optimizers"]) == 0
    assert "prune_l1" in capsys.readouterr().out


def test_invalid_config_has_clear_error(capsys, caplog, tmp_path) -> None:
    exit_code = main(["run", str(tmp_path / "missing.yaml"), "--db", str(tmp_path / "x.sqlite")])
    assert exit_code == 2
    assert "Cannot read config" in caplog.text


def test_run_and_report_commands_work_with_synthetic_data(tmp_path, capsys) -> None:
    config = tmp_path / "smoke.yaml"
    config.write_text(
        "seed: 7\nmodel: mobilenet_v2\ndataset: cifar10\nbatch_size: 2\n"
        "synthetic_samples: 4\nmax_eval_batches: 1\nartifact_dir: '"
        + str(tmp_path / "artifacts")
        + "'\n"
        "benchmark:\n  warmup_iters: 1\n  measure_iters: 2\n"
        "optimizers:\n  - name: ptq_dynamic_int8\n",
        encoding="utf-8",
    )
    database = tmp_path / "runs.sqlite"
    assert main(["run", str(config), "--db", str(database), "--synthetic"]) == 0
    assert '"optimizer": "ptq_dynamic_int8"' in capsys.readouterr().out
    report_dir = tmp_path / "report"
    assert main(["report", "--db", str(database), "--out", str(report_dir)]) == 0
    assert (report_dir / "results.csv").exists()
