"""Loading and validation for experiment configuration files."""

from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = {"seed", "model", "dataset", "optimizers"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config and validate its required top-level fields."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ValueError(f"Cannot read config '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config '{config_path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping at its top level.")
    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(missing)}")
    if not isinstance(config["seed"], int):
        raise ValueError("Config key 'seed' must be an integer.")
    for key in ("model", "dataset"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Config key '{key}' must be a non-empty string.")
    if not isinstance(config["optimizers"], list):
        raise ValueError("Config key 'optimizers' must be a list.")
    return config
