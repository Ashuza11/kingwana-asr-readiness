"""Configuration-backed model registry for later concrete adapters."""

from pathlib import Path
from typing import Any

from kingwana_asr.config import ConfigurationError, load_yaml


def load_model_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_yaml(path)
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ConfigurationError("models.yaml must contain a non-empty 'models' mapping")
    return models
