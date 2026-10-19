import csv
import json
from pathlib import Path

import pytest

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.manual_review_web import (
    LABEL_TO_WORKSHEET_FIELD,
    ManualReviewStore,
    finalize_label_review,
    parse_labels,
)

FIELDS = [
    "example_id",
    "clip_id",
    "reference",
    "model_code",
    "model_name",
    "prediction",
    "suggested_labels",
    "confidence",
    "evidence",
    "needs_audio_review",
    "human_status",
    "human_labels",
    "human_notes",
]


def _write_fixture(tmp_path: Path) -> ManualReviewStore:
    review = tmp_path / "review.csv"
    progress = tmp_path / "progress.json"
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for example_id in range(1, 51):
            for code in ("W", "S", "M", "O", "X"):
                writer.writerow(
                    {
                        "example_id": example_id,
                        "clip_id": f"clip-{example_id}",
                        "reference": "reference",
                        "model_code": code,
                        "model_name": code,
                        "prediction": "prediction",
                        "suggested_labels": "KING, VAR",
                        "confidence": "high",
                        "evidence": "evidence",
                        "needs_audio_review": "no",
                        "human_status": "pending",
                        "human_labels": "",
                        "human_notes": "",
                    }
                )
    progress.write_text(
        json.dumps(
            {
                "total_target_examples": 250,
                "batch_size": 10,
                "completed_batches": [1, 2, 3, 4, 5],
                "human_validated_batches": [],
                "next_example_index": 1,
                "last_update_timestamp": "initial",
            }
        ),
        encoding="utf-8",
    )
    return ManualReviewStore(review, progress)


def test_parse_labels_rejects_unknown_label() -> None:
    with pytest.raises(ManifestError, match="Unknown manual-review labels"):
        parse_labels("KING, OTHER")


def test_accept_preserves_suggestion_and_records_human_labels(tmp_path: Path) -> None:
    store = _write_fixture(tmp_path)
    store.apply([{"example_id": 1, "model_code": "W", "action": "accept"}])
    rows, progress = store.load()
    row = next(row for row in rows if row["example_id"] == "1" and row["model_code"] == "W")
    assert row["suggested_labels"] == "KING, VAR"
    assert row["human_labels"] == "KING, VAR"
    assert row["human_status"] == "validated"
    assert progress["next_example_index"] == 1


def test_correct_allows_an_explicit_empty_label_set(tmp_path: Path) -> None:
    store = _write_fixture(tmp_path)
    store.apply(
        [
            {
                "example_id": 1,
                "model_code": "W",
                "action": "correct",
                "human_labels": [],
                "human_notes": "No substantive error.",
            }
        ]
    )
    rows, _ = store.load()
    row = next(row for row in rows if row["example_id"] == "1" and row["model_code"] == "W")
    assert row["human_labels"] == ""
    assert row["human_notes"] == "No substantive error."
    assert row["human_status"] == "validated"


def test_batch_is_validated_only_after_all_fifty_rows(tmp_path: Path) -> None:
    store = _write_fixture(tmp_path)
    decisions = [
        {"example_id": example_id, "model_code": code, "action": "accept"}
        for example_id in range(1, 11)
        for code in ("W", "S", "M", "O", "X")
    ]
    progress = store.apply(decisions)
    assert progress["human_validated_batches"] == [1]
    assert progress["next_example_index"] == 11


def test_finalize_label_review_preserves_suggestions_and_uncertainty() -> None:
    review_rows = []
    worksheet_rows = []
    model_ids = {
        "W": "whisper_large_v3",
        "S": "sunflower_asr_51",
        "M": "mms_1b_all",
        "O": "omnilingual_ctc_1b",
        "X": "xlsr_swahili",
    }
    for example_id in range(1, 51):
        for code, model_id in model_ids.items():
            labels = "AMT, UNCERTAIN" if example_id == 1 and code == "W" else "KING"
            review_rows.append(
                {
                    "clip_id": f"clip-{example_id}",
                    "model_code": code,
                    "human_status": "validated",
                    "human_labels": labels,
                    "human_notes": "",
                }
            )
            worksheet_rows.append(
                {
                    "clip_id": f"clip-{example_id}",
                    "model_id": model_id,
                    "sample_stratum": "background",
                    "manual_review_status": "pending",
                    **{field: "" for field in LABEL_TO_WORKSHEET_FIELD.values()},
                    "manual_notes": "",
                }
            )

    finalized, summary = finalize_label_review(review_rows, worksheet_rows)

    first = finalized[0]
    assert first["manual_review_status"] == "complete"
    assert first["monetary_amount_error"] == "yes"
    assert first["kingwana_specific_form_error"] == "no"
    assert "UNCERTAIN" in first["manual_notes"]
    uncertain = next(
        row
        for row in summary
        if row["model_id"] == "whisper_large_v3"
        and row["sample_stratum"] == "all"
        and row["label"] == "UNCERTAIN"
    )
    assert uncertain["count"] == "1"
    assert uncertain["reviewed_comparisons"] == "50"
