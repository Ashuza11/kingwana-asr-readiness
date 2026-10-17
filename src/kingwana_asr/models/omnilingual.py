"""Meta Omnilingual ASR CTC adapter using the official inference package."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from kingwana_asr.models.base import ASRModel
from kingwana_asr.models.whisper import ModelDependencyError
from kingwana_asr.utils.device import DeviceSpec, select_device


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from huggingface_hub import model_info
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except ImportError as exc:
        raise ModelDependencyError(
            "Omnilingual dependencies could not be imported; run "
            "`uv sync --extra asr --extra omnilingual`"
        ) from exc
    return torch, ASRInferencePipeline, model_info


class OmnilingualASRModel(ASRModel):
    """Decode with OmniASR CTC, which officially has no language conditioning."""

    def __init__(self, model_id: str, config: dict[str, Any], *, device: str = "auto") -> None:
        self.model_id = model_id
        self.display_name = str(config["display_name"])
        self.hf_id = str(config["hf_id"])
        self.model_card = str(config["model_card"])
        self.evaluation_language = str(config["evaluation_language"])
        self.closest_supported_language = str(config["closest_supported_language"])
        self.language_conditioning = str(config["language_conditioning"])
        self.decoding = dict(config.get("decoding", {}))
        self.requested_device = device
        if self.evaluation_language != "swc":
            raise ValueError("OmniASR evaluation language must be Kingwana (swc)")
        if self.closest_supported_language != "swh_Latn":
            raise ValueError("OmniASR closest supported label must be swh_Latn")
        if self.language_conditioning != "none":
            raise ValueError("OmniASR CTC does not support language conditioning")
        if self.decoding.get("strategy") != "ctc_greedy":
            raise ValueError("OmniASR CTC requires deterministic CTC greedy decoding")
        self.model_revision: str | None = None
        self.device_spec: DeviceSpec | None = None
        self._torch: Any | None = None
        self._pipeline: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def language_config(self) -> str:
        return (
            f"evaluation_language={self.evaluation_language};"
            f"closest_supported_language={self.closest_supported_language};"
            "language_conditioning=none;decoding=ctc_greedy"
        )

    def load(self) -> None:
        if self.is_loaded:
            return
        torch, pipeline_class, model_info = _load_dependencies()
        selected = select_device(self.requested_device, torch_module=torch)
        # Meta reports and documents OmniASR GPU inference in bfloat16. CPU remains float32.
        dtype_name = "bfloat16" if selected.device.startswith("cuda") else "float32"
        device_spec = DeviceSpec(
            requested=selected.requested,
            device=selected.device,
            dtype=dtype_name,
            accelerator_name=selected.accelerator_name,
            total_memory_bytes=selected.total_memory_bytes,
        )
        pipeline = pipeline_class(
            model_card=self.model_card,
            device=device_spec.device,
            dtype=getattr(torch, dtype_name),
        )
        try:
            revision = model_info(self.hf_id).sha
        except Exception:  # model loading remains usable if provenance lookup is unavailable
            revision = None
        self._torch = torch
        self._pipeline = pipeline
        self.device_spec = device_spec
        self.model_revision = revision

    def transcribe(self, audio_path: Path) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before transcription")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        # Do not pass `lang`: Meta's CTC pipeline explicitly ignores it.
        results = self._pipeline.transcribe([str(audio_path)], batch_size=1)
        if len(results) != 1:
            raise RuntimeError("OmniASR returned an invalid number of results")
        return str(results[0]).strip()

    def unload(self) -> None:
        self._pipeline = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
