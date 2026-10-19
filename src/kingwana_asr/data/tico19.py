"""Prepare the pinned TICO-19 Congolese Swahili test split for ASR evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.sautiforge import write_manifest_rows
from kingwana_asr.data.validation import validate_manifest
from kingwana_asr.evaluation.normalization import normalize_text

TICO19_DATASET_ID = "CLEAR-Global/TICO-19-swc-speech"
TICO19_REVISION = "7393279cb0c4633cebe0f58b2796185852aab408"
TICO19_CONFIG = "read"
TICO19_SPLIT = "test"
TICO19_LICENSE = "CC-BY-NC-4.0"
TICO19_EXPECTED_ROWS = 2_124
TICO19_EXPECTED_SPEAKERS = 6
TICO19_TEST_SHARDS = {
    "test-00000-of-00003.parquet": (
        41_345_191,
        "12f2fb2840c2f44d1d3851996c975f88d1dc9e6e7e8d9102fe8b28ceefc7ac8d",
    ),
    "test-00001-of-00003.parquet": (
        58_140_655,
        "33c35cb827185d3197befef1604d03571507021486d7948e95a53036192cdcb5",
    ),
    "test-00002-of-00003.parquet": (
        52_410_220,
        "8ca46db4c3a30bb1c2cc7feccab24bc7b3092bd11f661069a38430217839e5b2",
    ),
}
_PARQUET_COLUMNS = ("audio", "session", "sentence_id", "content_id", "swc", "duration")


@dataclass(frozen=True, slots=True)
class Tico19PreparationSummary:
    """Summary of the extracted external evaluation split."""

    clip_count: int
    speaker_count: int
    total_duration_s: float
    manifest_path: Path
    audio_dir: Path


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tico19_shards(
    shard_dir: Path,
    expected_shards: Mapping[str, tuple[int, str]] = TICO19_TEST_SHARDS,
) -> list[Path]:
    """Require the exact pinned test shards and verify byte size and SHA-256."""
    paths: list[Path] = []
    for filename, (expected_size, expected_sha256) in expected_shards.items():
        path = shard_dir / filename
        if not path.is_file():
            raise ManifestError(f"Required TICO-19 test shard does not exist: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ManifestError(
                f"TICO-19 shard size mismatch for {filename}: "
                f"expected {expected_size}, got {actual_size}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ManifestError(
                f"TICO-19 shard SHA-256 mismatch for {filename}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        paths.append(path)
    return paths


def iter_tico19_rows(shards: list[Path]) -> Iterable[dict[str, Any]]:
    """Yield selected fields in shard order without loading embedded audio all at once."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ManifestError(
            "PyArrow is required for TICO-19 preparation; install the project's "
            "omnilingual extra or pyarrow==20.0.0"
        ) from exc

    for shard in shards:
        parquet_file = parquet.ParquetFile(shard)
        missing = sorted(set(_PARQUET_COLUMNS) - set(parquet_file.schema_arrow.names))
        if missing:
            raise ManifestError(
                f"TICO-19 shard {shard.name} is missing columns: {', '.join(missing)}"
            )
        for batch in parquet_file.iter_batches(batch_size=32, columns=list(_PARQUET_COLUMNS)):
            yield from batch.to_pylist()


def _required_text(row: Mapping[str, Any], field: str, row_number: int) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ManifestError(f"TICO-19 row {row_number} has an empty {field!r}")
    return value


def _safe_original_audio_name(audio: Mapping[str, Any], row_number: int) -> str:
    original = str(audio.get("path") or "").strip()
    path = PurePosixPath(original)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ManifestError(f"TICO-19 row {row_number} has an unsafe audio path: {original!r}")
    return path.name


def _persist_audio(destination: Path, content: bytes, *, force: bool) -> None:
    if destination.exists() and not force:
        if (
            destination.is_file()
            and _sha256_file(destination) == hashlib.sha256(content).hexdigest()
        ):
            return
        raise ManifestError(f"Existing TICO-19 audio differs from source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_tico19_manifest_rows(
    source_rows: Iterable[Mapping[str, Any]],
    output_audio_dir: Path,
    manifest_path: Path,
    *,
    force: bool,
) -> list[dict[str, str]]:
    """Extract embedded MP3 audio and build source-preserving manifest rows."""
    rows: list[dict[str, str]] = []
    clip_ids: set[str] = set()
    original_paths: set[str] = set()
    for row_number, source in enumerate(source_rows, start=1):
        session = _required_text(source, "session", row_number)
        content_id = _required_text(source, "content_id", row_number)
        sentence_id = _required_text(source, "sentence_id", row_number)
        reference_verbatim = _required_text(source, "swc", row_number)
        reference_eval = normalize_text(reference_verbatim)
        if not reference_eval:
            raise ManifestError(f"TICO-19 row {row_number} has an empty normalized reference")

        audio = source.get("audio")
        if not isinstance(audio, Mapping):
            raise ManifestError(f"TICO-19 row {row_number} has malformed embedded audio")
        original_name = _safe_original_audio_name(audio, row_number)
        if original_name in original_paths:
            raise ManifestError(f"TICO-19 contains duplicate original audio path: {original_name}")
        original_paths.add(original_name)
        content = audio.get("bytes")
        if not isinstance(content, bytes) or not content:
            raise ManifestError(f"TICO-19 row {row_number} has empty embedded audio bytes")

        clip_id = f"TICO19_TEST_{session}_{content_id}"
        if clip_id in clip_ids:
            raise ManifestError(f"TICO-19 contains duplicate derived clip_id: {clip_id}")
        clip_ids.add(clip_id)
        destination = output_audio_dir / f"{clip_id}.mp3"
        _persist_audio(destination, content, force=force)

        try:
            duration = float(source.get("duration"))
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"TICO-19 row {row_number} has invalid duration") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ManifestError(f"TICO-19 row {row_number} duration must be positive and finite")
        relative_audio = os.path.relpath(destination, start=manifest_path.parent.resolve())
        rows.append(
            {
                "clip_id": clip_id,
                "speaker_id": f"TICO19_SPK_{session}",
                "audio_path": Path(relative_audio).as_posix(),
                "duration_s": f"{duration:.6f}",
                "reference_verbatim": reference_verbatim,
                "reference_eval": reference_eval,
                "scenario": "tico19_medical_read_speech",
                "languages": "swc",
                "code_switched": "unknown",
                "noise_level": "unknown",
                "notes": f"TICO-19 sentence_id={sentence_id}",
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


def prepare_tico19_test(
    shard_dir: Path,
    output_audio_dir: Path,
    manifest_path: Path,
    provenance_path: Path,
    *,
    force: bool = False,
    expected_shards: Mapping[str, tuple[int, str]] = TICO19_TEST_SHARDS,
    expected_rows: int = TICO19_EXPECTED_ROWS,
    expected_speakers: int = TICO19_EXPECTED_SPEAKERS,
    row_reader: Callable[[list[Path]], Iterable[dict[str, Any]]] = iter_tico19_rows,
) -> Tico19PreparationSummary:
    """Verify, extract, normalize, and validate the pinned TICO-19 test split."""
    shard_dir = shard_dir.expanduser().resolve()
    output_audio_dir = output_audio_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    provenance_path = provenance_path.expanduser().resolve()
    if manifest_path.exists() and not force:
        raise ManifestError(
            f"Refusing to overwrite existing manifest without --force: {manifest_path}"
        )
    if provenance_path.exists() and not force:
        raise ManifestError(
            f"Refusing to overwrite existing provenance without --force: {provenance_path}"
        )

    shards = validate_tico19_shards(shard_dir, expected_shards)
    rows = build_tico19_manifest_rows(
        row_reader(shards), output_audio_dir, manifest_path, force=force
    )
    if len(rows) != expected_rows:
        raise ManifestError(
            f"TICO-19 test row count mismatch: expected {expected_rows}, got {len(rows)}"
        )
    speaker_count = len({row["speaker_id"] for row in rows})
    if speaker_count != expected_speakers:
        raise ManifestError(
            "TICO-19 test speaker count mismatch: "
            f"expected {expected_speakers}, got {speaker_count}"
        )

    write_manifest_rows(manifest_path, rows, force=force)
    validated = validate_manifest(manifest_path)
    provenance = {
        "dataset_id": TICO19_DATASET_ID,
        "dataset_revision": TICO19_REVISION,
        "config": TICO19_CONFIG,
        "split": TICO19_SPLIT,
        "license": TICO19_LICENSE,
        "reference_field": "swc",
        "audio_source": "MP3 bytes embedded in the pinned Parquet test shards",
        "normalization": "project deterministic normalize_text applied unchanged",
        "scenario": "tico19_medical_read_speech",
        "clip_count": validated.clip_count,
        "speaker_count": len(validated.counts_by_speaker),
        "total_duration_s": validated.total_duration_s,
        "shards": [
            {"filename": name, "size_bytes": size, "sha256": digest}
            for name, (size, digest) in expected_shards.items()
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_provenance(provenance_path, provenance, force=force)
    return Tico19PreparationSummary(
        clip_count=validated.clip_count,
        speaker_count=len(validated.counts_by_speaker),
        total_duration_s=validated.total_duration_s,
        manifest_path=manifest_path,
        audio_dir=output_audio_dir,
    )
