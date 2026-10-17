from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kingwana_asr.models.xlsr as xlsr_module
from kingwana_asr.models.xlsr import XLSRASRModel


class FakeTensor:
    def __init__(self) -> None:
        self.to_kwargs: dict[str, object] = {}

    def to(self, **kwargs: object) -> FakeTensor:
        self.to_kwargs = kwargs
        return self

    def __getitem__(self, index: int) -> FakeTensor:
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.input_values = FakeTensor()
        self.attention_mask = FakeTensor()
        self.call_kwargs: dict[str, object] = {}
        self.decode = Mock(return_value="  niko hapa  ")

    def __call__(self, audio: object, **kwargs: object) -> dict[str, FakeTensor]:
        self.call_kwargs = kwargs
        return {
            "input_values": self.input_values,
            "attention_mask": self.attention_mask,
        }


class FakeModel:
    config = SimpleNamespace(_commit_hash="xlsr-revision")

    def __init__(self) -> None:
        self.eval = Mock()
        self.logits = FakeTensor()

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.call_kwargs = kwargs
        return SimpleNamespace(logits=self.logits)

    def to(self, device: str) -> None:
        self.device = device


def config() -> dict[str, object]:
    return {
        "display_name": "XLS-R Swahili",
        "hf_id": "thinkKenya/wav2vec2-large-xls-r-300m-sw",
        "language": "sw",
        "evaluation_language": "swc",
        "decoding": {"strategy": "ctc_greedy"},
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
        argmax=Mock(return_value=FakeTensor()),
        no_grad=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    librosa = SimpleNamespace(load=Mock(return_value=([0.0], 16_000)))
    return torch, model_class, processor_class, librosa


def test_xlsr_uses_ctc_greedy_without_language_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependencies = fake_dependencies()
    monkeypatch.setattr(xlsr_module, "_load_dependencies", lambda: dependencies)
    audio = tmp_path / "synthetic.wav"
    audio.touch()
    adapter = XLSRASRModel("xlsr_swahili", config(), device="cpu")

    adapter.load()
    prediction = adapter.transcribe(audio)

    assert prediction == "niko hapa"
    processor = dependencies[2].from_pretrained.return_value
    model = dependencies[1].from_pretrained.return_value
    model.eval.assert_called_once_with()
    assert processor.call_kwargs == {
        "sampling_rate": 16_000,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }
    assert processor.input_values.to_kwargs == {"device": "cpu", "dtype": "float32"}
    assert processor.attention_mask.to_kwargs == {"device": "cpu"}
    assert adapter.language_config == (
        "evaluation_language=swc;checkpoint_language=sw;decoding=ctc_greedy"
    )
    assert adapter.model_revision == "xlsr-revision"


@pytest.mark.parametrize(
    ("field", "value"),
    [("language", "en"), ("evaluation_language", "swh")],
)
def test_xlsr_rejects_wrong_language_metadata(field: str, value: str) -> None:
    values = config()
    values[field] = value
    with pytest.raises(ValueError, match="checkpoint_language=sw"):
        XLSRASRModel("xlsr_swahili", values)
