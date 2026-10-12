"""Reproducibility metadata helpers that do not import heavyweight ASR packages."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_metadata() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "transformers_version": package_version("transformers"),
        "torch_version": package_version("torch"),
        "torchaudio_version": package_version("torchaudio"),
        "cuda_version": None,
    }
