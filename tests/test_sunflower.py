from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kingwana_asr.models.sunflower as sunflower_module
from kingwana_asr.models.sunflower import SunflowerASRModel


class FakeFeatures:
    def __init__(self) -> None:
        self.to_kwargs: dict[str, object] = {}

    def to(self, **kwargs: object) -> FakeFeatures:
        self.to_kwargs = kwargs
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda token: {
                "<|transcribe|>": 50_358,
                "<|notimestamps|>": 50_363,
            }[token]
        )
        self.features = FakeFeatures()
        self.attention_mask = FakeFeatures()
        self.call_kwargs: dict[str, object] = {}

    def __call__(self, audio: object, **kwargs: object) -> SimpleNamespace:
        self.call_kwargs = kwargs
        return SimpleNamespace(
            input_features=self.features,
            attention_mask=self.attention_mask,
        )

    def batch_decode(self, ids: object, **kwargs: object) -> list[str]:
        return ["  niko hapa  "]


class FakeModel:
    config = SimpleNamespace(_commit_hash="sunflower-revision")

    def __init__(self) -> None:
        self.generate = Mock(return_value=[[1, 2, 3]])

    def to(self, device: str) -> None:
        self.device = device


def config() -> dict[str, object]:
    return {
        "display_name": "SunflowerASR-51",
        "hf_id": "Sunbird/SunflowerASR-51-african-languages",
        "language": "swa",
        "language_token_id": 50_318,
        "task": "transcribe",
        "decoding": {"num_beams": 1},
    }


def fake_dependencies() -> tuple[SimpleNamespace, Mock, Mock, SimpleNamespace]:
    model = FakeModel()
    model_class = Mock()
    model_class.from_pretrained.return_value = model
    processor = FakeProcessor()
    processor_class = Mock()
    processor_class.from_pretrained.return_value = processor
    torch = SimpleNamespace(
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    librosa = SimpleNamespace(load=Mock(return_value=([0.0], 16_000)))
    return torch, model_class, processor_class, librosa


def test_sunflower_forces_official_swahili_decoder_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependencies = fake_dependencies()
    monkeypatch.setattr(sunflower_module, "_load_dependencies", lambda: dependencies)
    audio = tmp_path / "synthetic.wav"
    audio.touch()
    adapter = SunflowerASRModel("sunflower_asr_51", config(), device="cpu")
    adapter.load()
    prediction = adapter.transcribe(audio)
    assert prediction == "niko hapa"
    dependencies[2].from_pretrained.assert_called_once_with(
        "Sunbird/SunflowerASR-51-african-languages",
        use_fast=True,
        extra_special_tokens={},
    )
    model = dependencies[1].from_pretrained.return_value
    model.generate.assert_called_once_with(
        dependencies[2].from_pretrained.return_value.features,
        forced_decoder_ids=[(1, 50_318), (2, 50_358), (3, 50_363)],
        attention_mask=dependencies[2].from_pretrained.return_value.attention_mask,
        do_sample=False,
        num_beams=1,
    )
    assert dependencies[2].from_pretrained.return_value.call_kwargs == {
        "sampling_rate": 16_000,
        "do_normalize": True,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }
    assert adapter.language_config == "language=swa;language_token_id=50318;task=transcribe"
    assert adapter.model_revision == "sunflower-revision"


def test_sunflower_rejects_wrong_language_configuration() -> None:
    values = config()
    values["language_token_id"] = 1
    with pytest.raises(ValueError, match="token ID 50318"):
        SunflowerASRModel("sunflower_asr_51", values)
