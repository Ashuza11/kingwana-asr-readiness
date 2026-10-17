"""Configuration-backed model registry for later concrete adapters."""

from pathlib import Path
from typing import Any

from kingwana_asr.config import ConfigurationError, load_yaml
from kingwana_asr.models.base import ASRModel


def load_model_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_yaml(path)
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ConfigurationError("models.yaml must contain a non-empty 'models' mapping")
    return models


def create_model(
    model_key: str,
    registry: dict[str, dict[str, Any]],
    *,
    device: str = "auto",
) -> ASRModel:
    """Construct an implemented model adapter without loading model weights."""
    try:
        config = registry[model_key]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ConfigurationError(f"Unknown model {model_key!r}; choose one of: {choices}") from exc

    if model_key == "whisper_large_v3":
        from kingwana_asr.models.whisper import WhisperASRModel

        return WhisperASRModel(model_key, config, device=device)
    if model_key == "sunflower_asr_51":
        from kingwana_asr.models.sunflower import SunflowerASRModel

        return SunflowerASRModel(model_key, config, device=device)
    if model_key == "mms_1b_all":
        from kingwana_asr.models.mms import MMSASRModel

        return MMSASRModel(model_key, config, device=device)
    if model_key == "omnilingual_ctc_1b":
        from kingwana_asr.models.omnilingual import OmnilingualASRModel

        return OmnilingualASRModel(model_key, config, device=device)
    if model_key == "xlsr_swahili":
        from kingwana_asr.models.xlsr import XLSRASRModel

        return XLSRASRModel(model_key, config, device=device)
    raise ConfigurationError(f"Model adapter is not implemented yet: {model_key}")
