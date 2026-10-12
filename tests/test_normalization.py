import unicodedata

from kingwana_asr.evaluation.normalization import normalize_text


def test_normalization_is_deterministic_and_preserves_lexical_forms() -> None:
    source = "  NÍko—na PESA, mingi!  "
    assert normalize_text(source) == "níko na pesa mingi"
    assert normalize_text(source) == normalize_text(source)


def test_normalization_applies_nfc() -> None:
    decomposed = "Cafe\u0301"
    result = normalize_text(decomposed)
    assert result == "café"
    assert unicodedata.is_normalized("NFC", result)


def test_punctuation_becomes_separator_without_translation() -> None:
    assert normalize_text("Kingwana/Swahili: lipa-sasa") == "kingwana swahili lipa sasa"
