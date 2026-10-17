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
        "decoding": {"strategy": "ctc_greedy"},
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
        "language_conditioning=none;decoding=ctc_greedy"
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
