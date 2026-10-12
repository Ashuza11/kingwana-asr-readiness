import pytest

from kingwana_asr.evaluation.metrics import calculate_metrics


def test_exact_match_has_zero_error() -> None:
    result = calculate_metrics(["Niko na pesa"], ["niko na pesa"])
    assert result.wer == 0.0
    assert result.cer == 0.0
    assert result.reference_words == 3


def test_one_word_substitution() -> None:
    result = calculate_metrics(["niko na pesa"], ["niko na simu"])
    assert result.wer == pytest.approx(1 / 3)
    assert result.word_substitutions == 1
    assert result.cer > 0


def test_corpus_micro_wer_uses_total_words() -> None:
    result = calculate_metrics(["moja", "mbili tatu nne"], ["tofauti", "mbili tatu nne"])
    assert result.wer == pytest.approx(1 / 4)


def test_mismatched_lengths_fail() -> None:
    with pytest.raises(ValueError, match="same length"):
        calculate_metrics(["moja"], [])


def test_known_word_deletion_and_character_error_values() -> None:
    result = calculate_metrics(["mimi nina deni"], ["mimi deni"])
    assert result.wer == pytest.approx(1 / 3)
    assert result.cer == pytest.approx(5 / 14)
    assert result.word_deletions == 1


def test_empty_hypothesis_scores_as_complete_deletion() -> None:
    result = calculate_metrics(["mimi nina deni"], [""])
    assert result.wer == 1.0
    assert result.cer == 1.0


def test_empty_normalized_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty after normalization"):
        calculate_metrics(["!!!"], [""])
