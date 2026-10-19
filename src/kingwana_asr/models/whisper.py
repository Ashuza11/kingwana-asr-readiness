"""Hugging Face Whisper Large v3 adapter."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from kingwana_asr.models.base import ASRModel
from kingwana_asr.utils.device import DeviceSpec, select_device


class ModelDependencyError(RuntimeError):
    """Raised when optional ASR packages are not installed."""


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise ModelDependencyError(
            "Whisper dependencies are not installed; run `uv sync --extra asr`"
        ) from exc
    return torch, (AutoModelForSpeechSeq2Seq, AutoProcessor), pipeline


def _audio_duration_seconds(audio_path: Path) -> float:
    try:
        import soundfile
    except ImportError as exc:
        raise ModelDependencyError("SoundFile is required for Whisper duration detection") from exc
    info = soundfile.info(audio_path)
    return float(info.frames) / float(info.samplerate)


class WhisperASRModel(ASRModel):
    """Single-clip adapter with explicit language and transcription task."""

    def __init__(self, model_id: str, config: dict[str, Any], *, device: str = "auto") -> None:
        self.model_id = model_id
        self.display_name = str(config["display_name"])
        self.hf_id = str(config["hf_id"])
        self.language = str(config["language"])
        self.task = str(config.get("task", "transcribe"))
        self.decoding = dict(config.get("decoding", {}))
        self.long_form_threshold_seconds = float(config.get("long_form_threshold_seconds", 30.0))
        if self.long_form_threshold_seconds <= 0:
            raise ValueError("Whisper long_form_threshold_seconds must be positive")
        self.configured_revision = config.get("revision")
        self.requested_device = device
        self.model_revision: str | None = None
        self.device_spec: DeviceSpec | None = None
        self._torch: Any | None = None
        self._model: Any | None = None
        self._pipeline: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def language_config(self) -> str:
        return (
            f"language={self.language};task={self.task};"
            f"long_form_timestamps_over_seconds={self.long_form_threshold_seconds:g}"
        )

    def load(self) -> None:
        if self.is_loaded:
            return
        torch, model_classes, pipeline_factory = _load_dependencies()
        model_class, processor_class = model_classes
        device_spec = select_device(self.requested_device, torch_module=torch)
        torch_dtype = getattr(torch, device_spec.dtype)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        configured_revision = self.configured_revision
        if configured_revision:
            load_kwargs["revision"] = configured_revision

        model = model_class.from_pretrained(self.hf_id, **load_kwargs)
        model.to(device_spec.device)
        processor = processor_class.from_pretrained(
            self.hf_id,
            **({"revision": configured_revision} if configured_revision else {}),
        )
        inference_pipeline = pipeline_factory(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device_spec.device,
        )

        self._torch = torch
        self._model = model
        self._pipeline = inference_pipeline
        self.device_spec = device_spec
        self.model_revision = configured_revision or getattr(model.config, "_commit_hash", None)

    def transcribe(self, audio_path: Path) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before transcription")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        generate_kwargs = {
            "language": self.language,
            "task": self.task,
            **self.decoding,
        }
        pipeline_kwargs: dict[str, Any] = {"generate_kwargs": generate_kwargs}
        if _audio_duration_seconds(audio_path) > self.long_form_threshold_seconds:
            pipeline_kwargs["return_timestamps"] = True
        result = self._pipeline(str(audio_path), **pipeline_kwargs)
        if not isinstance(result, dict) or "text" not in result:
            raise RuntimeError("Whisper pipeline returned an invalid result")
        return str(result["text"]).strip()

    def unload(self) -> None:
        self._pipeline = None
        self._model = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
