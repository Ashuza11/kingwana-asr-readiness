from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from kingwana_asr.data.manifest import ManifestError, read_manifest
from kingwana_asr.data.tico19 import prepare_tico19_test, validate_tico19_shards


def _fixture_shards(tmp_path: Path) -> tuple[Path, dict[str, tuple[int, str]]]:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    expected: dict[str, tuple[int, str]] = {}
    for index, content in enumerate((b"first", b"second", b"third")):
        name = f"test-{index:05d}-of-00003.parquet"
        (shard_dir / name).write_bytes(content)
        expected[name] = (len(content), hashlib.sha256(content).hexdigest())
    return shard_dir, expected


def test_prepare_tico19_builds_separate_normalized_manifest(tmp_path: Path) -> None:
    shard_dir, expected = _fixture_shards(tmp_path)
    source_rows = [
        {
            "audio": {"bytes": b"fictional-mp3-a", "path": "session_125_1_swc.mp3"},
            "session": "125",
            "sentence_id": "CMU_1:1",
            "content_id": "1",
            "swc": "Ni tangu siku ngapi?",
            "duration": 5.04,
        },
        {
            "audio": {"bytes": b"fictional-mp3-b", "path": "session_126_1_swc.mp3"},
            "session": "126",
            "sentence_id": "CMU_1:1",
            "content_id": "1",
            "swc": "Dawa iko tayari!",
            "duration": 4.5,
        },
    ]
    manifest = tmp_path / "manifests" / "tico19.csv"
    provenance = tmp_path / "manifests" / "tico19.provenance.json"

    summary = prepare_tico19_test(
        shard_dir,
        tmp_path / "audio",
        manifest,
        provenance,
        expected_shards=expected,
        expected_rows=2,
        expected_speakers=2,
        row_reader=lambda _shards: iter(source_rows),
    )

    assert summary.clip_count == 2
    assert summary.speaker_count == 2
    assert summary.total_duration_s == pytest.approx(9.54)
    rows = read_manifest(manifest)
    assert [row.clip_id for row in rows] == ["TICO19_TEST_125_1", "TICO19_TEST_126_1"]
    assert [row.reference_eval for row in rows] == [
        "ni tangu siku ngapi",
        "dawa iko tayari",
    ]
    assert all(row.scenario == "tico19_medical_read_speech" for row in rows)
    assert all(row.languages == "swc" for row in rows)
    assert all(row.code_switched is None for row in rows)
    assert {path.name for path in summary.audio_dir.iterdir()} == {
        "TICO19_TEST_125_1.mp3",
        "TICO19_TEST_126_1.mp3",
    }
    with manifest.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    assert raw[0]["notes"] == "TICO-19 sentence_id=CMU_1:1"
    assert provenance.is_file()


def test_validate_tico19_shards_rejects_checksum_mismatch(tmp_path: Path) -> None:
    shard_dir, expected = _fixture_shards(tmp_path)
    name = next(iter(expected))
    size, _digest = expected[name]
    expected[name] = (size, "0" * 64)

    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        validate_tico19_shards(shard_dir, expected)


def test_prepare_tico19_rejects_unexpected_row_count(tmp_path: Path) -> None:
    shard_dir, expected = _fixture_shards(tmp_path)
    source_rows = [
        {
            "audio": {"bytes": b"fictional-mp3", "path": "session_125_1_swc.mp3"},
            "session": "125",
            "sentence_id": "CMU_1:1",
            "content_id": "1",
            "swc": "Ni tangu siku ngapi?",
            "duration": 5.04,
        }
    ]

    with pytest.raises(ManifestError, match="row count mismatch"):
        prepare_tico19_test(
            shard_dir,
            tmp_path / "audio",
            tmp_path / "manifest.csv",
            tmp_path / "provenance.json",
            expected_shards=expected,
            expected_rows=2,
            expected_speakers=1,
            row_reader=lambda _shards: iter(source_rows),
        )
