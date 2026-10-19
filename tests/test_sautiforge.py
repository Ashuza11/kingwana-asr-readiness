from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.sautiforge import (
    apply_transcript_corrections,
    build_manifest_rows,
    current_revisions,
    select_balanced_smoke_rows,
    select_preflight_row,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def create_export(tmp_path: Path) -> Path:
    root = tmp_path / "private-export"
    audio = root / "audio" / "fictional.opus"
    audio.parent.mkdir(parents=True)
    audio.touch()
    write_jsonl(
        root / "metadata/participants.jsonl",
        [{"id": "participant-id", "speaker_code": "SPK_TEST"}],
    )
    write_jsonl(
        root / "metadata/scenarios.jsonl",
        [{"id": "scenario-id", "expected_intent": "record_sale"}],
    )
    write_jsonl(
        root / "metadata/recordings.jsonl",
        [
            {
                "id": "recording-id",
                "display_id": "CLIP_TEST",
                "participant_id": "participant-id",
                "scenario_id": "scenario-id",
                "audio_path": "audio/fictional.opus",
                "duration_ms": 1250,
                "code_switching_status": "ambiguous",
                "noise_level": "low",
            }
        ],
    )
    write_jsonl(
        root / "metadata/transcriptions.jsonl",
        [
            {
                "id": "old-transcription",
                "recording_id": "recording-id",
                "supersedes_id": None,
                "verbatim_text": "Old fictional text",
                "language_tags_json": '["Swahili"]',
            },
            {
                "id": "current-transcription",
                "recording_id": "recording-id",
                "supersedes_id": "old-transcription",
                "verbatim_text": "Bei ni Mia!",
                "language_tags_json": '["Swahili", "French"]',
            },
        ],
    )
    return root


def test_build_rows_resolves_revision_and_normalizes_verbatim(tmp_path: Path) -> None:
    root = create_export(tmp_path)
    manifest = tmp_path / "manifests" / "evaluation.csv"
    rows = build_manifest_rows(root, manifest)
    assert len(rows) == 1
    assert rows[0]["clip_id"] == "CLIP_TEST"
    assert rows[0]["speaker_id"] == "SPK_TEST"
    assert rows[0]["reference_verbatim"] == "Bei ni Mia!"
    assert rows[0]["reference_eval"] == "bei ni mia"
    assert rows[0]["languages"] == "swc+fr"
    assert rows[0]["languages_raw"] == "Swahili+French"
    assert rows[0]["code_switched"] == "ambiguous"
    assert rows[0]["code_switch_status"] == "ambiguous"
    assert rows[0]["noise_level"] == "low"
    assert rows[0]["noise_level_raw"] == "low"
    assert rows[0]["quality_rating"] == "unknown"
    assert (manifest.parent / rows[0]["audio_path"]).resolve().is_file()


def test_transcript_correction_preserves_source_expectation_and_renormalizes(
    tmp_path: Path,
) -> None:
    corrections = tmp_path / "corrections.csv"
    corrections.write_text(
        "clip_id,original_reference_verbatim,corrected_reference_verbatim,notes\n"
        "CLIP_TEST,Bei ni Mia!,Bei ni 100!,researcher verified audio\n",
        encoding="utf-8",
    )
    rows = [
        {
            "clip_id": "CLIP_TEST",
            "reference_verbatim": "Bei ni Mia!",
            "reference_eval": "bei ni mia",
        }
    ]

    corrected = apply_transcript_corrections(rows, corrections)

    assert rows[0]["reference_verbatim"] == "Bei ni Mia!"
    assert corrected[0]["reference_verbatim"] == "Bei ni 100!"
    assert corrected[0]["reference_eval"] == "bei ni 100"


def test_transcript_correction_rejects_changed_source(tmp_path: Path) -> None:
    corrections = tmp_path / "corrections.csv"
    corrections.write_text(
        "clip_id,original_reference_verbatim,corrected_reference_verbatim,notes\n"
        "CLIP_TEST,different text,corrected text,researcher verified audio\n",
        encoding="utf-8",
    )
    rows = [
        {
            "clip_id": "CLIP_TEST",
            "reference_verbatim": "source text",
            "reference_eval": "source text",
        }
    ]

    with pytest.raises(ManifestError, match="source text has changed"):
        apply_transcript_corrections(rows, corrections)


def test_current_revision_rejects_unknown_parent() -> None:
    with pytest.raises(ManifestError, match="unknown supersedes_id"):
        current_revisions([{"id": "revision", "supersedes_id": "missing"}], label="fixture")


def test_balanced_smoke_selection_is_deterministic() -> None:
    rows = [
        {
            "clip_id": f"C{speaker}{scenario}{repeat}",
            "speaker_id": f"S{speaker}",
            "scenario": f"scenario_{scenario}",
        }
        for speaker in range(4)
        for scenario in range(2)
        for repeat in range(3)
    ]
    first = select_balanced_smoke_rows(rows, size=8, seed=42)
    second = select_balanced_smoke_rows(rows, size=8, seed=42)
    assert [row["clip_id"] for row in first] == [row["clip_id"] for row in second]
    assert Counter(row["speaker_id"] for row in first) == {
        "S0": 2,
        "S1": 2,
        "S2": 2,
        "S3": 2,
    }
    assert Counter(row["scenario"] for row in first) == {
        "scenario_0": 4,
        "scenario_1": 4,
    }


def test_preflight_prefers_clear_high_quality_representative_clip() -> None:
    rows = [
        {
            "clip_id": "lower-quality",
            "duration_s": "7.0",
            "quality_rating": "3",
            "noise_level": "low",
            "code_switch_status": "present",
            "languages": "swc+fr",
        },
        {
            "clip_id": "representative",
            "duration_s": "7.1",
            "quality_rating": "5",
            "noise_level": "low",
            "code_switch_status": "present",
            "languages": "swc+fr",
        },
        {
            "clip_id": "noisy",
            "duration_s": "7.0",
            "quality_rating": "5",
            "noise_level": "high",
            "code_switch_status": "present",
            "languages": "swc+fr",
        },
    ]
    assert select_preflight_row(rows)["clip_id"] == "representative"
    assert select_balanced_smoke_rows(rows, size=1)[0]["clip_id"] == "representative"
