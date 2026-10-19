from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kingwana_asr.models.omnilingual as omnilingual_module
from kingwana_asr.models.omnilingual import OmnilingualASRModel


def config() -> dict[str, object]:
    return {
        "display_name": "Meta Omnilingual ASR CTC 1B",
        "hf_id": "facebook/omniASR-CTC-1B",
        "model_card": "omniASR_CTC_1B",
        "evaluation_language": "swc",
        "closest_supported_language": "swh_Latn",
        "language_conditioning": "none",
        "decoding": {"strategy": "ctc_greedy", "chunk_seconds": 30.0},
    }


def fake_dependencies() -> tuple[SimpleNamespace, Mock, Mock]:
    pipeline = SimpleNamespace(transcribe=Mock(return_value=["  niko hapa  "]))
    pipeline_class = Mock(return_value=pipeline)
    torch = SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    model_info = Mock(return_value=SimpleNamespace(sha="omni-revision"))
    return torch, pipeline_class, model_info


def test_omnilingual_ctc_uses_no_language_conditioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependencies = fake_dependencies()
    monkeypatch.setattr(omnilingual_module, "_load_dependencies", lambda: dependencies)
    soundfile = SimpleNamespace(
        info=Mock(return_value=SimpleNamespace(samplerate=16_000, frames=1))
    )
    monkeypatch.setattr(
        omnilingual_module,
        "_load_audio_dependencies",
        lambda: (SimpleNamespace(mean=lambda audio, axis: audio), soundfile),
    )
    audio = tmp_path / "synthetic.wav"
    audio.touch()
    adapter = OmnilingualASRModel("omnilingual_ctc_1b", config(), device="cpu")

    adapter.load()
    prediction = adapter.transcribe(audio)

    assert prediction == "niko hapa"
    dependencies[1].assert_called_once_with(
        model_card="omniASR_CTC_1B", device="cpu", dtype="float32"
    )
    pipeline = dependencies[1].return_value
    pipeline.transcribe.assert_called_once_with([str(audio)], batch_size=1)
    assert adapter.language_config == (
        "evaluation_language=swc;closest_supported_language=swh_Latn;"
        "language_conditioning=none;decoding=ctc_greedy;chunk_seconds=30"
    )
    assert adapter.model_revision == "omni-revision"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evaluation_language", "swh", "Kingwana"),
        ("closest_supported_language", "swc_Latn", "swh_Latn"),
        ("language_conditioning", "swh_Latn", "does not support"),
    ],
)
def test_omnilingual_rejects_misleading_language_configuration(
    field: str, value: str, message: str
) -> None:
    values = config()
    values[field] = value
    with pytest.raises(ValueError, match=message):
        OmnilingualASRModel("omnilingual_ctc_1b", values)


def test_omnilingual_chunks_audio_above_package_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependencies = fake_dependencies()
    pipeline = dependencies[1].return_value
    pipeline.transcribe.return_value = ["sehemu moja", "sehemu mbili"]
    monkeypatch.setattr(omnilingual_module, "_load_dependencies", lambda: dependencies)
    audio = tmp_path / "long.wav"
    audio.touch()

    def write(path: Path, *_args: object, **_kwargs: object) -> None:
        Path(path).touch()

    soundfile = SimpleNamespace(
        info=Mock(return_value=SimpleNamespace(samplerate=2, frames=70)),
        read=Mock(return_value=([0.0] * 70, 2)),
        write=Mock(side_effect=write),
    )
    monkeypatch.setattr(
        omnilingual_module,
        "_load_audio_dependencies",
        lambda: (SimpleNamespace(mean=lambda audio, axis: audio), soundfile),
    )
    adapter = OmnilingualASRModel("omnilingual_ctc_1b", config(), device="cpu")

    adapter.load()
    prediction = adapter.transcribe(audio)

    assert prediction == "sehemu moja sehemu mbili"
    paths = pipeline.transcribe.call_args.args[0]
    assert len(paths) == 2
    assert pipeline.transcribe.call_args.kwargs == {"batch_size": 1}
