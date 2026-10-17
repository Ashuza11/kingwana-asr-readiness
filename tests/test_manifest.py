from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kingwana_asr.data.manifest import REQUIRED_COLUMNS, ManifestError, read_manifest
from kingwana_asr.data.validation import validate_manifest


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def row(audio_path: Path, **overrides: str) -> dict[str, str]:
    payload = {
        "clip_id": "clip_001",
        "speaker_id": "SPK_001",
        "audio_path": str(audio_path),
        "duration_s": "1.25",
        "reference_verbatim": "Niko hapa.",
        "reference_eval": "niko hapa",
        "scenario": "airtime_sale",
        "languages": "kingwana",
        "code_switched": "false",
        "noise_level": "low",
    }
    payload.update(overrides)
    return payload


def test_valid_manifest_reports_duration_and_groups(tmp_path: Path) -> None:
    first_audio = tmp_path / "first.fixture"
    second_audio = tmp_path / "second.fixture"
    first_audio.touch()
    second_audio.touch()
    manifest = tmp_path / "evaluation.csv"
    write_manifest(
        manifest,
        [
            row(first_audio),
            row(
                second_audio,
                clip_id="clip_002",
                speaker_id="SPK_002",
                duration_s="2.75",
                scenario="mobile_money",
            ),
        ],
    )
    summary = validate_manifest(manifest)
    assert summary.clip_count == 2
    assert summary.total_duration_s == pytest.approx(4.0)
    assert summary.counts_by_speaker == {"SPK_001": 1, "SPK_002": 1}
    assert summary.counts_by_scenario == {"airtime_sale": 1, "mobile_money": 1}


def test_duplicate_clip_ids_fail(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.audio"
    audio.touch()
    manifest = tmp_path / "evaluation.csv"
    write_manifest(manifest, [row(audio), row(audio)])
    with pytest.raises(ManifestError, match="duplicate clip_id"):
        validate_manifest(manifest)


def test_missing_audio_path_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "evaluation.csv"
    write_manifest(manifest, [row(tmp_path / "absent.audio")])
    with pytest.raises(ManifestError, match="audio file does not exist"):
        validate_manifest(manifest)


def test_empty_gold_transcript_fails_loudly(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.audio"
    audio.touch()
    manifest = tmp_path / "evaluation.csv"
    write_manifest(manifest, [row(audio, reference_eval="")])
    with pytest.raises(ManifestError, match="reference_eval.*empty"):
        read_manifest(manifest)


def test_missing_required_column_fails_loudly(tmp_path: Path) -> None:
    manifest = tmp_path / "evaluation.csv"
    manifest.write_text("clip_id,speaker_id\nclip_001,SPK_001\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="missing required columns"):
        read_manifest(manifest)


def test_invalid_boolean_fails_loudly(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.audio"
    audio.touch()
    manifest = tmp_path / "evaluation.csv"
    write_manifest(manifest, [row(audio, code_switched="sometimes")])
    with pytest.raises(ManifestError, match="code_switched must be true/present"):
        read_manifest(manifest)


@pytest.mark.parametrize("value", ["ambiguous", "unknown"])
def test_unavailable_code_switch_label_is_preserved(tmp_path: Path, value: str) -> None:
    audio = tmp_path / "fixture.audio"
    audio.touch()
    manifest = tmp_path / "evaluation.csv"
    write_manifest(manifest, [row(audio, code_switched=value)])
    assert read_manifest(manifest)[0].code_switched is None


@pytest.mark.parametrize(("value", "expected"), [("present", True), ("none", False)])
def test_source_code_switch_labels_are_supported(
    tmp_path: Path, value: str, expected: bool
) -> None:
    audio = tmp_path / "fixture.audio"
    audio.touch()
    manifest = tmp_path / "evaluation.csv"
    write_manifest(manifest, [row(audio, code_switched=value)])
    assert read_manifest(manifest)[0].code_switched is expected


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_duration_fails_loudly(tmp_path: Path, duration: str) -> None:
    audio = tmp_path / "fixture.audio"
    audio.touch()
    manifest = tmp_path / "evaluation.csv"
    write_manifest(manifest, [row(audio, duration_s=duration)])
    with pytest.raises(ManifestError, match="duration_s must be finite and greater than zero"):
        read_manifest(manifest)
