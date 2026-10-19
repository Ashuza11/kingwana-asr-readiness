from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path

import pytest

from kingwana_asr.data.common_voice import prepare_common_voice_test
from kingwana_asr.data.manifest import ManifestError, read_manifest


def _add_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _create_archive(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    archive_path = tmp_path / "common-voice-fixture.tar.gz"
    fields = ["client_id", "path", "sentence", "variant"]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, dialect="excel-tab")
    writer.writeheader()
    writer.writerows(rows)
    root = "cv-corpus-fixture/sw"
    with tarfile.open(archive_path, "w:gz") as archive:
        _add_bytes(archive, f"{root}/test.tsv", text.getvalue().encode())
        for row in rows:
            _add_bytes(archive, f"{root}/clips/{row['path']}", row["path"].encode())
    return archive_path


def test_prepare_common_voice_retains_only_kingwana_test_rows(tmp_path: Path) -> None:
    archive = _create_archive(
        tmp_path,
        [
            {
                "client_id": "public-client-a",
                "path": "kingwana-1.mp3",
                "sentence": "Bei ni francs elfu tano.",
                "variant": "sw-kingwana",
            },
            {
                "client_id": "public-client-b",
                "path": "standard-sw.mp3",
                "sentence": "Habari.",
                "variant": "sw-standard",
            },
            {
                "client_id": "public-client-a",
                "path": "kingwana-2.mp3",
                "sentence": "Réseaux iko tayari!",
                "variant": "sw-kingwana",
            },
        ],
    )
    audio_dir = tmp_path / "private" / "audio"
    manifest = tmp_path / "manifests" / "common_voice.csv"
    provenance = tmp_path / "manifests" / "common_voice.provenance.json"

    summary = prepare_common_voice_test(
        archive,
        audio_dir,
        manifest,
        provenance,
        expected_sha256=None,
        duration_probe=lambda _path: 1.25,
    )

    assert summary.clip_count == 2
    assert summary.speaker_count == 1
    assert summary.total_duration_s == pytest.approx(2.5)
    assert sorted(path.name for path in audio_dir.iterdir()) == [
        "kingwana-1.mp3",
        "kingwana-2.mp3",
    ]
    parsed = read_manifest(manifest)
    assert [row.reference_verbatim for row in parsed] == [
        "Bei ni francs elfu tano.",
        "Réseaux iko tayari!",
    ]
    assert [row.reference_eval for row in parsed] == [
        "bei ni francs elfu tano",
        "réseaux iko tayari",
    ]
    assert all(row.languages == "swc" for row in parsed)
    assert all(row.scenario == "common_voice_read_speech" for row in parsed)
    assert all(row.code_switched is None for row in parsed)
    assert provenance.is_file()


def test_prepare_common_voice_rejects_archive_without_target_variant(tmp_path: Path) -> None:
    archive = _create_archive(
        tmp_path,
        [
            {
                "client_id": "public-client",
                "path": "standard-sw.mp3",
                "sentence": "Habari.",
                "variant": "sw-standard",
            }
        ],
    )

    with pytest.raises(ManifestError, match="contains no rows"):
        prepare_common_voice_test(
            archive,
            tmp_path / "audio",
            tmp_path / "manifest.csv",
            tmp_path / "provenance.json",
            expected_sha256=None,
            duration_probe=lambda _path: 1.0,
        )


def test_prepare_common_voice_rejects_unsafe_clip_path(tmp_path: Path) -> None:
    archive = _create_archive(
        tmp_path,
        [
            {
                "client_id": "public-client",
                "path": "../outside.mp3",
                "sentence": "Bei ni tano.",
                "variant": "sw-kingwana",
            }
        ],
    )

    with pytest.raises(ManifestError, match="unsafe clip path"):
        prepare_common_voice_test(
            archive,
            tmp_path / "audio",
            tmp_path / "manifest.csv",
            tmp_path / "provenance.json",
            expected_sha256=None,
            duration_probe=lambda _path: 1.0,
        )


def test_prepare_common_voice_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archive = _create_archive(
        tmp_path,
        [
            {
                "client_id": "public-client",
                "path": "kingwana.mp3",
                "sentence": "Bei ni tano.",
                "variant": "sw-kingwana",
            }
        ],
    )

    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        prepare_common_voice_test(
            archive,
            tmp_path / "audio",
            tmp_path / "manifest.csv",
            tmp_path / "provenance.json",
            expected_sha256="0" * 64,
            duration_probe=lambda _path: 1.0,
        )
