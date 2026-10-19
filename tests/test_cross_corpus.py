import pytest

from kingwana_asr.evaluation.cross_corpus import compare_model_summaries


def test_cross_corpus_difference_is_business_minus_comparator() -> None:
    rows = compare_model_summaries(
        [{"model_id": "model_a", "wer": "0.75", "cer": "0.40"}],
        [{"model_id": "model_a", "wer": "0.25", "cer": "0.10"}],
    )

    assert rows[0]["wer_cross_corpus_difference_pp"] == "50.000000"
    assert rows[0]["cer_cross_corpus_difference_pp"] == "30.000000"


def test_cross_corpus_requires_identical_model_sets() -> None:
    with pytest.raises(ValueError, match="same model IDs"):
        compare_model_summaries(
            [{"model_id": "model_a", "wer": "0.75", "cer": "0.40"}],
            [{"model_id": "model_b", "wer": "0.25", "cer": "0.10"}],
        )
