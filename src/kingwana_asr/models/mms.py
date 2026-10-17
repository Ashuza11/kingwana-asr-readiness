"""Meta MMS-1B-All adapter with an explicit Swahili language adapter."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from kingwana_asr.models.base import ASRModel
from kingwana_asr.models.whisper import ModelDependencyError
from kingwana_asr.utils.device import DeviceSpec, select_device

SAMPLE_RATE_HZ = 16_000
SWAHILI_ADAPTER = "swh"


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import librosa
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
    except ImportError as exc:
        raise ModelDependencyError(
            "MMS dependencies are not installed; run `uv sync --extra asr --extra audio`"
        ) from exc
    return torch, Wav2Vec2ForCTC, AutoProcessor, librosa


class MMSASRModel(ASRModel):
    """Decode one clip using Meta's explicitly selected `swh` adapter."""

    def __init__(self, model_id: str, config: dict[str, Any], *, device: str = "auto") -> None:
        self.model_id = model_id
        self.display_name = str(config["display_name"])
        self.hf_id = str(config["hf_id"])
        self.language = str(config["language"])
        self.adapter = str(config["adapter"])
        self.decoding = dict(config.get("decoding", {}))
        self.configured_revision = config.get("revision")
        self.requested_device = device
        if self.language != SWAHILI_ADAPTER or self.adapter != SWAHILI_ADAPTER:
            raise ValueError("MMS Swahili must use language=swh and adapter=swh")
        if self.decoding.get("strategy") != "ctc_greedy":
            raise ValueError("MMS currently supports only deterministic CTC greedy decoding")
        self.model_revision: str | None = None
        self.device_spec: DeviceSpec | None = None
        self._torch: Any | None = None
        self._librosa: Any | None = None
        self._torch_dtype: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def language_config(self) -> str:
        return f"language={self.language};adapter={self.adapter};decoding=ctc_greedy"

    def load(self) -> None:
        if self.is_loaded:
            return
        torch, model_class, processor_class, librosa = _load_dependencies()
        device_spec = select_device(self.requested_device, torch_module=torch)
        torch_dtype = getattr(torch, device_spec.dtype)
        revision_kwargs = {"revision": self.configured_revision} if self.configured_revision else {}
        processor = processor_class.from_pretrained(self.hf_id, **revision_kwargs)
        model = model_class.from_pretrained(
            self.hf_id,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **revision_kwargs,
        )

        # Both calls are required: the tokenizer vocabulary and CTC adapter head
        # must target the same language. Otherwise MMS silently remains English.
        processor.tokenizer.set_target_lang(self.adapter)
        model.load_adapter(self.adapter)
        model.to(device_spec.device)
        model.eval()

        self._torch = torch
        self._librosa = librosa
        self._torch_dtype = torch_dtype
        self._model = model
        self._processor = processor
        self.device_spec = device_spec
        self.model_revision = self.configured_revision or getattr(
            model.config, "_commit_hash", None
        )

    def transcribe(self, audio_path: Path) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before transcription")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        audio, _ = self._librosa.load(audio_path, sr=SAMPLE_RATE_HZ, mono=True)
        inputs = self._processor(
            audio,
            sampling_rate=SAMPLE_RATE_HZ,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_inputs: dict[str, Any] = {}
        for name, value in inputs.items():
            if name == "input_values":
                value = value.to(device=self.device_spec.device, dtype=self._torch_dtype)
            else:
                value = value.to(device=self.device_spec.device)
            model_inputs[name] = value
        with self._torch.no_grad():
            logits = self._model(**model_inputs).logits
        predicted_ids = self._torch.argmax(logits, dim=-1)[0]
        return str(self._processor.decode(predicted_ids)).strip()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self._librosa = None
        self._torch_dtype = None
