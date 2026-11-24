"""Experiment hardware and source revision metadata."""

import hashlib
import json
import platform
import subprocess
from typing import Any

import psutil
import torch


def config_hash(config: Any) -> str:
    """Return a stable SHA-256 for a JSON-serializable config value."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hardware_info() -> dict[str, Any]:
    """Collect core host, memory, and optional GPU details."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_mb": round(psutil.virtual_memory().total / (1024**2), 2),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def git_commit() -> str | None:
    """Return the current git commit when this process runs in a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
