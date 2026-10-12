from pathlib import Path

from kingwana_asr.models.registry import load_model_registry

EXPECTED_MODELS = {
    "whisper_large_v3": ("openai/whisper-large-v3", "whisper", "swahili"),
    "sunflower_asr_51": (
        "Sunbird/SunflowerASR-51-african-languages",
        "whisper",
        "swa",
    ),
    "mms_1b_all": ("facebook/mms-1b-all", "mms", "swh"),
    "omnilingual_ctc_1b": (
        "facebook/omniASR-CTC-1B",
        "omnilingual",
        "swh_Latn",
    ),
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
        assert registry[key]["language"] == language
        assert registry[key]["decoding"]


def test_sunflower_uses_documented_forced_swahili_token() -> None:
    registry = load_model_registry(Path("configs/models.yaml"))
    assert registry["sunflower_asr_51"]["language_token_id"] == 50318
