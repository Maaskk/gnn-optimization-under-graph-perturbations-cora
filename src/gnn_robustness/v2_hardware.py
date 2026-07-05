from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch


def capture_environment_metadata(path: str | Path) -> Path:
    """Write local hardware/software metadata for reproducibility."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    try:
        import torch_geometric

        metadata["torch_geometric"] = torch_geometric.__version__
    except Exception as exc:  # pragma: no cover - metadata should never fail runs
        metadata["torch_geometric_error"] = str(exc)
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        metadata["git_commit"] = "unknown"
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
