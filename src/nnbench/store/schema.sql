PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(name, config_hash)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    optimizer TEXT NOT NULL,
    params_json TEXT NOT NULL,
    backend TEXT NOT NULL,
    hardware_json TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    git_commit TEXT,
    config_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value_num REAL,
    value_text TEXT,
    PRIMARY KEY(run_id, name)
);

CREATE INDEX IF NOT EXISTS idx_runs_model_optimizer ON runs(model, optimizer);
CREATE INDEX IF NOT EXISTS idx_runs_backend ON runs(backend);
