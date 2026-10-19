"""Prepare an exact Common Voice Kingwana test subset from an official archive."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.sautiforge import write_manifest_rows
from kingwana_asr.data.validation import validate_manifest
from kingwana_asr.evaluation.normalization import normalize_text

COMMON_VOICE_RELEASE = "Common Voice Scripted Speech 27.0 - Swahili"
COMMON_VOICE_DATASET_ID = "cmu5sgles002zo1071nphgtby"
COMMON_VOICE_ARCHIVE_FILENAME = "common-voice-scripted-speech-27-0-swahil-a1aabcc5.tar.gz"
COMMON_VOICE_ARCHIVE_SHA256 = "45811dfd9c6146513e6ef350c5d60b7bc97ce55411e553e204ac082b31bdc2ba"
COMMON_VOICE_LOCALE = "sw"
COMMON_VOICE_VARIANT = "sw-kingwana"
COMMON_VOICE_SPLIT = "test"


@dataclass(frozen=True, slots=True)
class CommonVoicePreparationSummary:
    """Summary of the retained test subset."""

    clip_count: int
    speaker_count: int
    total_duration_s: float
    manifest_path: Path
    audio_dir: Path
    archive_sha256: str | None


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file checksum without loading the archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration(path: Path, *, ffprobe: str = "ffprobe") -> float:
    """Read decoded audio duration with FFprobe."""
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ManifestError("FFprobe is required to measure Common Voice clip durations") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown FFprobe error"
        raise ManifestError(f"Could not read audio duration for {path}: {detail}") from exc
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise ManifestError(f"FFprobe returned an invalid duration for {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ManifestError(f"Audio duration must be positive and finite for {path}")
    return duration


def _normalized_member_name(name: str) -> str:
    return PurePosixPath(name).as_posix().removeprefix("./")


def _read_target_test_rows(archive_path: Path) -> tuple[PurePosixPath, list[dict[str, str]]]:
    test_member = None
    rows: list[dict[str, str]] = []
    try:
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                member_path = PurePosixPath(_normalized_member_name(member.name))
                if not member.isfile() or member_path.name != "test.tsv":
                    continue
                if len(member_path.parts) < 2 or member_path.parent.name != COMMON_VOICE_LOCALE:
                    continue
                if test_member is not None:
                    raise ManifestError("Archive contains more than one Swahili test.tsv")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ManifestError("Could not read Swahili test.tsv from the archive")
                test_member = member_path
                # A streaming tar member is not seekable, while TextIOWrapper probes
                # seekability during construction. test.tsv is metadata rather than audio,
                # so buffering this one member keeps the archive scan itself streaming.
                with io.StringIO(extracted.read().decode("utf-8-sig"), newline="") as handle:
                    reader = csv.DictReader(handle, dialect="excel-tab", strict=True)
                    required = {"client_id", "path", "sentence", "variant"}
                    if reader.fieldnames is None:
                        raise ManifestError("Common Voice test.tsv has no header")
                    missing = required - set(reader.fieldnames)
                    if missing:
                        raise ManifestError(
                            "Common Voice test.tsv is missing columns: "
                            + ", ".join(sorted(missing))
                        )
                    for row_number, row in enumerate(reader, start=2):
                        if None in row or any(value is None for value in row.values()):
                            raise ManifestError(
                                f"Common Voice test.tsv row {row_number} is malformed"
                            )
                        if row["variant"].strip() == COMMON_VOICE_VARIANT:
                            rows.append({key: value.strip() for key, value in row.items()})
                break
    except (tarfile.TarError, csv.Error) as exc:
        raise ManifestError(f"Could not parse Common Voice archive: {exc}") from exc

    if test_member is None:
        raise ManifestError("Archive does not contain a Swahili test.tsv")
    if not rows:
        raise ManifestError(
            f"Common Voice test.tsv contains no rows for variant {COMMON_VOICE_VARIANT!r}"
        )
    return test_member.parent, rows


def _validate_test_rows(rows: list[dict[str, str]]) -> None:
    paths: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        for field in ("client_id", "path", "sentence"):
            if not row[field]:
                raise ManifestError(
                    f"Selected Common Voice row {row_number} has an empty {field!r}"
                )
        path = PurePosixPath(row["path"])
        if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
            raise ManifestError(
                f"Selected Common Voice row {row_number} has an unsafe clip path: {row['path']!r}"
            )
        if row["path"] in paths:
            raise ManifestError(f"Duplicate Common Voice test clip path: {row['path']}")
        paths.add(row["path"])


def _extract_selected_clips(
    archive_path: Path,
    locale_root: PurePosixPath,
    rows: list[dict[str, str]],
    output_audio_dir: Path,
    *,
    force: bool,
) -> None:
    expected = {(locale_root / "clips" / row["path"]).as_posix(): row["path"] for row in rows}
    found: set[str] = set()
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                member_name = _normalized_member_name(member.name)
                clip_name = expected.get(member_name)
                if clip_name is None:
                    continue
                if clip_name in found:
                    raise ManifestError(f"Archive contains duplicate clip member: {clip_name}")
                if not member.isfile() or member.size <= 0:
                    raise ManifestError(f"Archive clip is not a non-empty file: {clip_name}")
                destination = output_audio_dir / clip_name
                if destination.exists() and not force:
                    if not destination.is_file() or destination.stat().st_size != member.size:
                        raise ManifestError(
                            f"Existing extracted clip does not match archive size: {destination}"
                        )
                    found.add(clip_name)
                    if len(found) == len(expected):
                        break
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ManifestError(f"Could not read archive clip: {clip_name}")
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                try:
                    with source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    if temporary.stat().st_size != member.size:
                        raise ManifestError(f"Incomplete extraction for clip: {clip_name}")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                found.add(clip_name)
                if len(found) == len(expected):
                    break
    except tarfile.TarError as exc:
        raise ManifestError(f"Could not extract Common Voice archive: {exc}") from exc

    missing = sorted(set(row["path"] for row in rows) - found)
    if missing:
        preview = ", ".join(missing[:5])
        raise ManifestError(
            f"Archive is missing {len(missing)} selected Common Voice clips; first: {preview}"
        )


def _speaker_id(client_id: str) -> str:
    value = f"{COMMON_VOICE_DATASET_ID}\0{client_id}".encode()
    return "CVSPK_" + hashlib.sha256(value).hexdigest()[:12].upper()


def _build_manifest_rows(
    selected_rows: list[dict[str, str]],
    output_audio_dir: Path,
    manifest_path: Path,
    duration_probe: Callable[[Path], float],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    clip_ids: set[str] = set()
    speaker_sources: dict[str, str] = {}
    for selected in selected_rows:
        clip_path = output_audio_dir / selected["path"]
        stem = Path(selected["path"]).stem
        clip_id = f"CV27_SW_KINGWANA_TEST_{stem}"
        if clip_id in clip_ids:
            raise ManifestError(f"Derived duplicate Common Voice clip_id: {clip_id}")
        clip_ids.add(clip_id)

        speaker_id = _speaker_id(selected["client_id"])
        existing_source = speaker_sources.get(speaker_id)
        if existing_source is not None and existing_source != selected["client_id"]:
            raise ManifestError("A collision occurred while pseudonymizing Common Voice speakers")
        speaker_sources[speaker_id] = selected["client_id"]

        reference_verbatim = selected["sentence"]
        reference_eval = normalize_text(reference_verbatim)
        if not reference_eval:
            raise ManifestError(f"Normalized reference is empty for Common Voice clip {clip_id}")
        duration = duration_probe(clip_path)
        relative_audio = os.path.relpath(clip_path, start=manifest_path.parent.resolve())
        rows.append(
            {
                "clip_id": clip_id,
                "speaker_id": speaker_id,
                "audio_path": Path(relative_audio).as_posix(),
                "duration_s": f"{duration:.6f}",
                "reference_verbatim": reference_verbatim,
                "reference_eval": reference_eval,
                "scenario": "common_voice_read_speech",
                "languages": "swc",
                "code_switched": "unknown",
                "noise_level": "unknown",
            }
        )
    return rows


def _write_provenance(path: Path, payload: dict[str, object], *, force: bool) -> None:
    if path.exists() and not force:
        raise ManifestError(f"Refusing to overwrite existing provenance without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_common_voice_test(
    archive_path: Path,
    output_audio_dir: Path,
    manifest_path: Path,
    provenance_path: Path,
    *,
    expected_sha256: str | None = COMMON_VOICE_ARCHIVE_SHA256,
    force: bool = False,
    duration_probe: Callable[[Path], float] = probe_duration,
) -> CommonVoicePreparationSummary:
    """Retain only Kingwana rows and audio from the official Swahili test split."""
    archive_path = archive_path.expanduser().resolve()
    output_audio_dir = output_audio_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    provenance_path = provenance_path.expanduser().resolve()
    if not archive_path.is_file():
        raise ManifestError(f"Common Voice archive does not exist: {archive_path}")
    if archive_path == output_audio_dir or archive_path == manifest_path:
        raise ManifestError("Common Voice source and output paths must be separate")
    if manifest_path.exists() and not force:
        raise ManifestError(
            f"Refusing to overwrite existing manifest without --force: {manifest_path}"
        )
    if provenance_path.exists() and not force:
        raise ManifestError(
            f"Refusing to overwrite existing provenance without --force: {provenance_path}"
        )

    actual_sha256 = sha256_file(archive_path) if expected_sha256 is not None else None
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise ManifestError(
            "Common Voice archive SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    locale_root, selected_rows = _read_target_test_rows(archive_path)
    _validate_test_rows(selected_rows)
    _extract_selected_clips(
        archive_path,
        locale_root,
        selected_rows,
        output_audio_dir,
        force=force,
    )
    manifest_rows = _build_manifest_rows(
        selected_rows,
        output_audio_dir,
        manifest_path,
        duration_probe,
    )
    write_manifest_rows(manifest_path, manifest_rows, force=force)
    summary = validate_manifest(manifest_path)
    provenance = {
        "source": "Mozilla Data Collective",
        "dataset_name": COMMON_VOICE_RELEASE,
        "dataset_id": COMMON_VOICE_DATASET_ID,
        "archive_filename": archive_path.name,
        "archive_sha256": actual_sha256,
        "expected_archive_sha256": expected_sha256,
        "split": COMMON_VOICE_SPLIT,
        "locale": COMMON_VOICE_LOCALE,
        "variant": COMMON_VOICE_VARIANT,
        "selection_rule": "test.tsv rows where variant equals sw-kingwana exactly",
        "normalization": "project deterministic normalize_text applied unchanged",
        "clip_count": summary.clip_count,
        "speaker_count": len(summary.counts_by_speaker),
        "total_duration_s": summary.total_duration_s,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_provenance(provenance_path, provenance, force=force)
    return CommonVoicePreparationSummary(
        clip_count=summary.clip_count,
        speaker_count=len(summary.counts_by_speaker),
        total_duration_s=summary.total_duration_s,
        manifest_path=manifest_path,
        audio_dir=output_audio_dir,
        archive_sha256=actual_sha256,
    )
