from collections import Counter

import pytest

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.manual_review import (
    MANUAL_REVIEW_FIELDS,
    build_manual_review_rows,
    select_error_stratified_sample,
    select_speaker_scenario_sample,
    summarize_completed_manual_review,
    summarize_manual_review_edits,
    validate_manual_sample,
)


def _candidates() -> list[dict[str, str]]:
    return [
        {
            "clip_id": f"clip-{speaker}-{scenario}-{repeat}",
            "speaker_id": speaker,
            "scenario": scenario,
            "reference_verbatim": "mimi nina deni",
        }
        for speaker in ("s1", "s2")
        for scenario in ("sale", "debt")
        for repeat in range(3)
    ]


def test_manual_sample_covers_every_speaker_scenario_cell() -> None:
    selected = select_speaker_scenario_sample(_candidates(), size=4, seed=42)
    validate_manual_sample(selected, expected_size=4)
    assert Counter(row["speaker_id"] for row in selected) == {"s1": 2, "s2": 2}
    assert Counter(row["scenario"] for row in selected) == {"sale": 2, "debt": 2}
    assert len({(row["speaker_id"], row["scenario"]) for row in selected}) == 4


def test_manual_sample_fails_when_a_cell_is_missing() -> None:
    rows = [
        row
        for row in _candidates()
        if not (row["speaker_id"] == "s2" and row["scenario"] == "debt")
    ]
    with pytest.raises(ManifestError, match="has 0 clips"):
        select_speaker_scenario_sample(rows, size=4, seed=42)


def test_manual_worksheet_contains_metrics_and_blank_human_fields() -> None:
    selected = [_candidates()[0]]
    predictions = {
        "model-a": [
            {
                "clip_id": selected[0]["clip_id"],
                "prediction": "mimi deni",
                "status": "success",
            }
        ]
    }
    rows = build_manual_review_rows(selected, predictions, model_order=["model-a"], seed=42)
    assert len(rows) == 1
    assert rows[0]["word_deletions"] == "1"
    assert rows[0]["manual_review_status"] == "pending"
    assert all(rows[0][field] == "" for field in MANUAL_REVIEW_FIELDS)


def test_error_stratified_sample_keeps_outlier_and_balances_margins() -> None:
    candidates = _candidates()
    predictions = [
        {
            "clip_id": row["clip_id"],
            "prediction": (
                "mimi nina deni " + "ziada " * 60
                if row["clip_id"] == "clip-s1-sale-0"
                else "mimi nina deni"
            ),
            "status": "success",
        }
        for row in candidates
    ]

    selected = select_error_stratified_sample(
        candidates, predictions, size=8, seed=42, insertion_threshold=50
    )

    assert len(selected) == 8
    assert Counter(row["speaker_id"] for row in selected) == {"s1": 4, "s2": 4}
    assert Counter(row["scenario"] for row in selected) == {"sale": 4, "debt": 4}
    outlier = next(row for row in selected if row["clip_id"] == "clip-s1-sale-0")
    assert outlier["sample_stratum"] == "whisper_high_insertion"


def test_manual_summary_rejects_pending_and_summarizes_complete_rows() -> None:
    row = {
        "clip_id": "clip-1",
        "model_id": "model-a",
        "sample_stratum": "balanced_background",
        "manual_review_status": "pending",
        **{field: "no" for field in MANUAL_REVIEW_FIELDS},
    }
    with pytest.raises(ManifestError, match="1 pending"):
        summarize_completed_manual_review([row])

    row["manual_review_status"] = "complete"
    row["hallucination"] = "yes"
    summary = summarize_completed_manual_review([row])
    hallucination = next(
        item
        for item in summary
        if item["sample_stratum"] == "all" and item["category"] == "hallucination"
    )
    assert hallucination["yes"] == "1"
    assert hallucination["error_or_feature_rate"] == "1.000000"


def test_automatic_edit_summary_aggregates_counts() -> None:
    rows = [
        {
            "model_id": "model-a",
            "sample_stratum": "balanced_background",
            "word_substitutions": "2",
            "word_deletions": "1",
            "word_insertions": "3",
            "reference_words": "10",
        },
        {
            "model_id": "model-a",
            "sample_stratum": "whisper_high_insertion",
            "word_substitutions": "1",
            "word_deletions": "0",
            "word_insertions": "50",
            "reference_words": "10",
        },
    ]
    summary = summarize_manual_review_edits(rows)
    combined = next(row for row in summary if row["sample_stratum"] == "all")
    assert combined["substitutions"] == "3"
    assert combined["insertions"] == "53"
    assert combined["reference_words"] == "20"
    assert combined["wer"] == "2.850000"
