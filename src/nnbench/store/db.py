"""SQLite persistence and query APIs for experiment runs."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nnbench.store.metadata import config_hash as make_config_hash
from nnbench.store.metadata import git_commit as get_git_commit
from nnbench.store.metadata import hardware_info as collect_hardware

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class ResultStore:
    """SQLite-backed store that creates its schema on first use."""

    def __init__(self, path: str | Path = "artifacts/nnbench.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def connect(self) -> sqlite3.Connection:
        """Open a row-enabled SQLite connection with foreign keys enabled."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save_result(
        self,
        result: dict[str, Any],
        *,
        experiment_name: str = "default",
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Save a benchmark dictionary and return the created run ID."""
        metadata = metadata or {}
        config_value = config or {}
        digest = metadata.get("config_hash") or make_config_hash(config_value)
        now = datetime.now(UTC).isoformat()
        model = str(metadata.get("model", result.get("model", "unknown")))
        optimizer = str(metadata.get("optimizer", result.get("optimizer", "baseline")))
        params = metadata.get("params", config_value.get("optimizer_params", {}))
        backend = str(metadata.get("backend", "pytorch"))
        hardware = metadata.get("hardware", collect_hardware())
        timestamp = str(metadata.get("timestamp", now))
        commit = metadata.get("git_commit", get_git_commit())
        metrics = result.get("metrics", result)

        with closing(self.connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO experiments(name, config_json, config_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    experiment_name,
                    json.dumps(config_value, sort_keys=True, default=str),
                    digest,
                    now,
                ),
            )
            experiment = connection.execute(
                "SELECT id FROM experiments WHERE name=? AND config_hash=?",
                (experiment_name, digest),
            ).fetchone()
            cursor = connection.execute(
                "INSERT INTO runs(experiment_id, model, optimizer, params_json, backend, "
                "hardware_json, timestamp, git_commit, config_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    experiment["id"],
                    model,
                    optimizer,
                    json.dumps(params, sort_keys=True, default=str),
                    backend,
                    json.dumps(hardware, sort_keys=True, default=str),
                    timestamp,
                    commit,
                    digest,
                ),
            )
            run_id = int(cursor.lastrowid)
            for key, value in metrics.items():
                if key in {"model", "optimizer"}:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    connection.execute(
                        "INSERT INTO metrics(run_id, name, value_num) VALUES (?, ?, ?)",
                        (run_id, key, float(value)),
                    )
                elif value is not None:
                    connection.execute(
                        "INSERT INTO metrics(run_id, name, value_text) VALUES (?, ?, ?)",
                        (run_id, key, json.dumps(value, sort_keys=True, default=str)),
                    )
        return run_id

    def query_results(
        self,
        *,
        model: str | None = None,
        optimizer: str | None = None,
        backend: str | None = None,
        experiment_name: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return run rows joined with their metric values."""
        clauses: list[str] = []
        params: list[Any] = []
        for field, value in (
            ("r.model", model),
            ("r.optimizer", optimizer),
            ("r.backend", backend),
        ):
            if value is not None:
                clauses.append(f"{field}=?")
                params.append(value)
        if experiment_name is not None:
            clauses.append("e.name=?")
            params.append(experiment_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(limit)
        with closing(self.connect()) as connection, connection:
            rows = connection.execute(
                "SELECT r.*, e.name AS experiment_name FROM runs r "
                "JOIN experiments e ON e.id=r.experiment_id "
                f"{where} ORDER BY r.id{limit_sql}",
                params,
            ).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result["params"] = json.loads(result.pop("params_json"))
                result["hardware"] = json.loads(result.pop("hardware_json"))
                result["metrics"] = {}
                for metric in connection.execute(
                    "SELECT name, value_num, value_text FROM metrics WHERE run_id=?", (row["id"],)
                ):
                    result["metrics"][metric["name"]] = (
                        metric["value_num"]
                        if metric["value_num"] is not None
                        else json.loads(metric["value_text"])
                    )
                results.append(result)
        return results


def save_result(
    db_path: str | Path,
    result: dict[str, Any],
    *,
    experiment_name: str = "default",
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Convenience wrapper to persist one benchmark result."""
    return ResultStore(db_path).save_result(
        result, experiment_name=experiment_name, config=config, metadata=metadata
    )


def query_results(db_path: str | Path, **filters: Any) -> list[dict[str, Any]]:
    """Convenience wrapper to query stored benchmark results."""
    return ResultStore(db_path).query_results(**filters)
