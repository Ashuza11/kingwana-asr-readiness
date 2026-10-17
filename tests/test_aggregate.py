import pytest

from kingwana_asr.evaluation.aggregate import aggregate_predictions


def prediction(
    clip_id: str, speaker_id: str, reference: str, hypothesis: str, scenario: str
) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "speaker_id": speaker_id,
        "reference": reference,
        "prediction": hypothesis,
        "scenario": scenario,
        "code_switched": False,
        "status": "success",
    }


def test_aggregation_by_speaker_and_speaker_macro() -> None:
    report = aggregate_predictions(
        [
            prediction("c1", "S1", "moja", "tofauti", "sale"),
            prediction("c2", "S2", "moja mbili tatu", "moja mbili tatu", "sale"),
        ]
    )
    assert report["global"]["wer"] == pytest.approx(1 / 4)
    assert report["per_speaker"]["S1"]["wer"] == 1.0
    assert report["per_speaker"]["S2"]["wer"] == 0.0
    assert report["speaker_macro_wer"] == pytest.approx(0.5)


def test_failed_predictions_are_not_silently_scored() -> None:
    record = prediction("c1", "S1", "moja", "", "sale")
    record["status"] = "error"
    with pytest.raises(ValueError, match="Cannot score failed"):
        aggregate_predictions([record])


def test_string_false_code_switch_value_is_not_treated_as_true() -> None:
    record = prediction("c1", "S1", "moja", "moja", "sale")
    record["code_switched"] = "false"
    report = aggregate_predictions([record])
    assert set(report["by_code_switch"]) == {"not_code_switched"}


@pytest.mark.parametrize("value", ["unknown", "ambiguous", None])
def test_unavailable_code_switch_value_is_excluded(value: object) -> None:
    record = prediction("c1", "S1", "moja", "moja", "sale")
    record["code_switched"] = value
    report = aggregate_predictions([record])
    assert report["by_code_switch"] == {}


def test_invalid_code_switch_value_fails_loudly() -> None:
    record = prediction("c1", "S1", "moja", "moja", "sale")
    record["code_switched"] = "sometimes"
    with pytest.raises(ValueError, match="Invalid code_switched"):
        aggregate_predictions([record])


def test_optional_annotation_strata_are_aggregated() -> None:
    record = prediction("c1", "S1", "moja", "moja", "sale")
    record.update(
        code_switch_status="ambiguous",
        languages="swc+fr",
        noise_level="medium",
        quality_rating="2",
    )
    report = aggregate_predictions([record])
    assert set(report["by_code_switch_status"]) == {"ambiguous"}
    assert set(report["by_language_mix"]) == {"swc+fr"}
    assert set(report["by_noise_level"]) == {"medium"}
    assert set(report["by_quality_rating"]) == {"2"}
