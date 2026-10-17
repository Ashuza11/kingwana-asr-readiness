"""SunflowerASR-51 adapter with the model card's forced Swahili token."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from kingwana_asr.models.base import ASRModel
from kingwana_asr.models.whisper import ModelDependencyError
from kingwana_asr.utils.device import DeviceSpec, select_device

SAMPLE_RATE_HZ = 16_000
SWAHILI_LANGUAGE_TOKEN_ID = 50_318


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import librosa
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise ModelDependencyError(
            "Sunflower dependencies are not installed; run `uv sync --extra asr --extra audio`"
        ) from exc
    return torch, WhisperForConditionalGeneration, WhisperProcessor, librosa


class SunflowerASRModel(ASRModel):
    """Decode one clip using the explicit `swa` token documented by Sunbird AI."""

    def __init__(self, model_id: str, config: dict[str, Any], *, device: str = "auto") -> None:
        self.model_id = model_id
        self.display_name = str(config["display_name"])
        self.hf_id = str(config["hf_id"])
        self.language = str(config["language"])
        self.language_token_id = int(config["language_token_id"])
        self.task = str(config.get("task", "transcribe"))
        self.decoding = dict(config.get("decoding", {}))
        self.configured_revision = config.get("revision")
        self.requested_device = device
        if self.language != "swa" or self.language_token_id != SWAHILI_LANGUAGE_TOKEN_ID:
            raise ValueError("Sunflower Swahili must use language=swa and token ID 50318")
        self.model_revision: str | None = None
        self.device_spec: DeviceSpec | None = None
        self._torch: Any | None = None
        self._librosa: Any | None = None
        self._torch_dtype: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._forced_decoder_ids: list[tuple[int, int]] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def language_config(self) -> str:
        return (
            f"language={self.language};language_token_id={self.language_token_id};task={self.task}"
        )

    def load(self) -> None:
        if self.is_loaded:
            return
        torch, model_class, processor_class, librosa = _load_dependencies()
        device_spec = select_device(self.requested_device, torch_module=torch)
        torch_dtype = getattr(torch, device_spec.dtype)
        load_kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if self.configured_revision:
            load_kwargs["revision"] = self.configured_revision
        model = model_class.from_pretrained(self.hf_id, **load_kwargs)
        model.to(device_spec.device)
        processor = processor_class.from_pretrained(
            self.hf_id,
            use_fast=True,
            extra_special_tokens={},
            **({"revision": self.configured_revision} if self.configured_revision else {}),
        )
        transcribe_token = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        no_timestamps_token = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self._forced_decoder_ids = [
            (1, self.language_token_id),
            (2, int(transcribe_token)),
            (3, int(no_timestamps_token)),
        ]
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
        if not self.is_loaded or self._forced_decoder_ids is None:
            raise RuntimeError("Model must be loaded before transcription")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        audio, _ = self._librosa.load(audio_path, sr=SAMPLE_RATE_HZ, mono=True)
        inputs = self._processor(
            audio,
            sampling_rate=SAMPLE_RATE_HZ,
            do_normalize=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        features = inputs.input_features.to(
            device=self.device_spec.device,
            dtype=self._torch_dtype,
        )
        attention_mask = inputs.attention_mask.to(device=self.device_spec.device)
        generate_kwargs = {
            "forced_decoder_ids": self._forced_decoder_ids,
            "attention_mask": attention_mask,
            "do_sample": False,
            **self.decoding,
        }
        predicted_ids = self._model.generate(features, **generate_kwargs)
        results = self._processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not results:
            raise RuntimeError("Sunflower returned an invalid result")
        return str(results[0]).strip()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._forced_decoder_ids = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self._librosa = None
        self._torch_dtype = None
