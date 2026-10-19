from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kingwana_asr.models.whisper as whisper_module
from kingwana_asr.models.whisper import WhisperASRModel


class FakeModel:
    config = SimpleNamespace(_commit_hash="synthetic-revision")

    def __init__(self) -> None:
        self.device = None

    def to(self, device: str) -> None:
        self.device = device


def fake_dependencies(result: object) -> tuple[SimpleNamespace, tuple[Mock, Mock], Mock]:
    model = FakeModel()
    model_class = Mock()
    model_class.from_pretrained.return_value = model
    processor_class = Mock()
    processor_class.from_pretrained.return_value = SimpleNamespace(
        tokenizer="tokenizer", feature_extractor="feature_extractor"
    )
    inference_pipeline = Mock(return_value=result)
    pipeline_factory = Mock(return_value=inference_pipeline)
    torch = SimpleNamespace(
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    return torch, (model_class, processor_class), pipeline_factory


def config() -> dict[str, object]:
    return {
        "display_name": "Whisper Large v3",
        "hf_id": "openai/whisper-large-v3",
        "language": "swahili",
        "task": "transcribe",
        "long_form_threshold_seconds": 30.0,
        "decoding": {"num_beams": 1},
    }


def test_whisper_uses_explicit_language_and_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependencies = fake_dependencies({"text": "  niko hapa  "})
    monkeypatch.setattr(whisper_module, "_load_dependencies", lambda: dependencies)
    monkeypatch.setattr(whisper_module, "_audio_duration_seconds", lambda _path: 1.0)
    audio = tmp_path / "synthetic.audio"
    audio.touch()
    adapter = WhisperASRModel("whisper_large_v3", config())

    adapter.load()
    prediction = adapter.transcribe(audio)

    assert prediction == "niko hapa"
    inference_pipeline = dependencies[2].return_value
    inference_pipeline.assert_called_once_with(
        str(audio),
        generate_kwargs={"language": "swahili", "task": "transcribe", "num_beams": 1},
    )
    assert adapter.model_revision == "synthetic-revision"
    assert adapter.device_spec is not None
    assert adapter.device_spec.device == "cpu"


def test_whisper_requires_load_before_transcription(tmp_path: Path) -> None:
    adapter = WhisperASRModel("whisper_large_v3", config())
    with pytest.raises(RuntimeError, match="must be loaded"):
        adapter.transcribe(tmp_path / "missing.audio")


def test_whisper_rejects_invalid_pipeline_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependencies = fake_dependencies({})
    monkeypatch.setattr(whisper_module, "_load_dependencies", lambda: dependencies)
    monkeypatch.setattr(whisper_module, "_audio_duration_seconds", lambda _path: 1.0)
    audio = tmp_path / "synthetic.audio"
    audio.touch()
    adapter = WhisperASRModel("whisper_large_v3", config())
    adapter.load()
    with pytest.raises(RuntimeError, match="invalid result"):
        adapter.transcribe(audio)


def test_whisper_enables_timestamps_for_long_form_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependencies = fake_dependencies({"text": "maandishi marefu"})
    monkeypatch.setattr(whisper_module, "_load_dependencies", lambda: dependencies)
    monkeypatch.setattr(whisper_module, "_audio_duration_seconds", lambda _path: 30.1)
    audio = tmp_path / "long.wav"
    audio.touch()
    adapter = WhisperASRModel("whisper_large_v3", config())

    adapter.load()
    prediction = adapter.transcribe(audio)

    assert prediction == "maandishi marefu"
    dependencies[2].return_value.assert_called_once_with(
        str(audio),
        generate_kwargs={"language": "swahili", "task": "transcribe", "num_beams": 1},
        return_timestamps=True,
    )
