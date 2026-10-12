"""Evaluation-manifest validation and summary reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from kingwana_asr.data.manifest import ManifestError, ManifestRow, read_manifest


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    manifest: Path
    clip_count: int
    total_duration_s: float
    counts_by_speaker: dict[str, int]
    counts_by_scenario: dict[str, int]


def validate_rows(rows: list[ManifestRow], manifest: Path) -> ManifestSummary:
    """Validate cross-row constraints and that every referenced audio file exists."""
    errors: list[str] = []
    seen: dict[str, int] = {}
    for row in rows:
        previous = seen.get(row.clip_id)
        if previous is not None:
            errors.append(
                f"Row {row.source_row}: duplicate clip_id {row.clip_id!r} "
                f"(first seen on row {previous})"
            )
        else:
            seen[row.clip_id] = row.source_row
        if not row.audio_path.is_file():
            errors.append(f"Row {row.source_row}: audio file does not exist: {row.audio_path}")

    if errors:
        raise ManifestError("Manifest validation failed:\n- " + "\n- ".join(errors))

    return ManifestSummary(
        manifest=manifest.resolve(),
        clip_count=len(rows),
        total_duration_s=sum(row.duration_s for row in rows),
        counts_by_speaker=dict(sorted(Counter(row.speaker_id for row in rows).items())),
        counts_by_scenario=dict(sorted(Counter(row.scenario for row in rows).items())),
    )


def validate_manifest(path: Path) -> ManifestSummary:
    return validate_rows(read_manifest(path), path)
