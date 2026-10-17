from pathlib import Path

import pytest

from kingwana_asr.config import ConfigurationError
from kingwana_asr.models.mms import MMSASRModel
from kingwana_asr.models.omnilingual import OmnilingualASRModel
from kingwana_asr.models.registry import create_model, load_model_registry
from kingwana_asr.models.sunflower import SunflowerASRModel
from kingwana_asr.models.whisper import WhisperASRModel
from kingwana_asr.models.xlsr import XLSRASRModel

EXPECTED_MODELS = {
    "whisper_large_v3": ("openai/whisper-large-v3", "whisper", "swahili"),
    "sunflower_asr_51": (
        "Sunbird/SunflowerASR-51-african-languages",
        "whisper",
        "swa",
    ),
    "mms_1b_all": ("facebook/mms-1b-all", "mms", "swh"),
    "omnilingual_ctc_1b": ("facebook/omniASR-CTC-1B", "omnilingual", None),
    "xlsr_swahili": (
        "thinkKenya/wav2vec2-large-xls-r-300m-sw",
        "wav2vec2",
        "sw",
    ),
}


def test_primary_model_registry_is_exact_and_explicit() -> None:
    registry = load_model_registry(Path("configs/models.yaml"))
    assert set(registry) == set(EXPECTED_MODELS)
    for key, (hf_id, family, language) in EXPECTED_MODELS.items():
        assert registry[key]["hf_id"] == hf_id
        assert registry[key]["family"] == family
        if language is not None:
            assert registry[key]["language"] == language
        assert registry[key]["decoding"]

    omnilingual = registry["omnilingual_ctc_1b"]
    assert omnilingual["evaluation_language"] == "swc"
    assert omnilingual["closest_supported_language"] == "swh_Latn"
    assert omnilingual["language_conditioning"] == "none"


def test_sunflower_uses_documented_forced_swahili_token() -> None:
    registry = load_model_registry(Path("configs/models.yaml"))
    assert registry["sunflower_asr_51"]["language_token_id"] == 50318


def test_registry_constructs_implemented_adapters() -> None:
    registry = load_model_registry(Path("configs/models.yaml"))
    model = create_model("whisper_large_v3", registry, device="cpu")
    assert isinstance(model, WhisperASRModel)
    sunflower = create_model("sunflower_asr_51", registry, device="cpu")
    assert isinstance(sunflower, SunflowerASRModel)
    mms = create_model("mms_1b_all", registry, device="cpu")
    assert isinstance(mms, MMSASRModel)
    omnilingual = create_model("omnilingual_ctc_1b", registry, device="cpu")
    assert isinstance(omnilingual, OmnilingualASRModel)
    xlsr = create_model("xlsr_swahili", registry, device="cpu")
    assert isinstance(xlsr, XLSRASRModel)


def test_registry_rejects_unknown_model() -> None:
    registry = load_model_registry(Path("configs/models.yaml"))
    with pytest.raises(ConfigurationError, match="Unknown model"):
        create_model("not_a_model", registry)
