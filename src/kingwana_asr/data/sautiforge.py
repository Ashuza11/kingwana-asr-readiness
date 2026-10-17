"""Strict import of a private SautiForge annotated export into evaluation manifests."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from kingwana_asr.data.manifest import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, ManifestError
from kingwana_asr.evaluation.normalization import normalize_text

_METADATA_FILES = {
    "participants": "metadata/participants.jsonl",
    "recordings": "metadata/recordings.jsonl",
    "scenarios": "metadata/scenarios.jsonl",
    "transcriptions": "metadata/transcriptions.jsonl",
}
_CODE_SWITCH_VALUES = {
    "present": "true",
    "none": "false",
    "ambiguous": "ambiguous",
    "unknown": "unknown",
}
_LANGUAGE_VALUES = {
    "Swahili": "swc",
    "Kiswahili": "swc",
    "Swahil": "swc",
    "French": "fr",
    "Frenchie": "fr",
    "English": "en",
}
_LANGUAGE_ORDER = {"swc": 0, "fr": 1, "en": 2}
_NOISE_VALUES = {
    "Low": "low",
    "low": "low",
    "quiet": "low",
    "medium": "medium",
    "moderate": "medium",
    "high": "high",
    "very_high": "high",
    "ambiguous": "unknown",
    "unknown": "unknown",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ManifestError(f"Required export file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"Malformed JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"Expected an object in {path} line {line_number}")
            records.append(value)
    return records


def _index_unique(records: Iterable[dict[str, Any]], field: str, label: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for record in records:
        value = str(record.get(field) or "").strip()
        if not value:
            raise ManifestError(f"{label} contains an empty {field}")
        if value in indexed:
            raise ManifestError(f"{label} contains duplicate {field}: {value}")
        indexed[value] = record
    return indexed


def current_revisions(records: Iterable[dict[str, Any]], *, label: str) -> list[dict[str, Any]]:
    """Resolve revision chains by retaining records not superseded by another record."""
    materialized = list(records)
    by_id = _index_unique(materialized, "id", label)
    superseded: set[str] = set()
    for record in materialized:
        previous = record.get("supersedes_id")
        if previous is None:
            continue
        previous_id = str(previous).strip()
        if previous_id not in by_id:
            raise ManifestError(f"{label} revision refers to unknown supersedes_id: {previous_id}")
        if previous_id in superseded:
            raise ManifestError(f"{label} revision is superseded more than once: {previous_id}")
        superseded.add(previous_id)
    return [record for record in materialized if str(record["id"]) not in superseded]


def _parse_string_list(value: Any, *, field: str, clip_id: str) -> list[str]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Clip {clip_id}: {field} is not valid JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ManifestError(f"Clip {clip_id}: {field} must be a non-empty list")
    cleaned = [str(item).strip() for item in parsed]
    if any(not item for item in cleaned):
        raise ManifestError(f"Clip {clip_id}: {field} contains an empty label")
    return cleaned


def build_manifest_rows(export_root: Path, manifest_path: Path) -> list[dict[str, str]]:
    """Build rows without writing, preserving private source labels and verbatim text."""
    export_root = export_root.expanduser().resolve()
    data = {key: _read_jsonl(export_root / relative) for key, relative in _METADATA_FILES.items()}
    participants = _index_unique(data["participants"], "id", "participants")
    scenarios = _index_unique(data["scenarios"], "id", "scenarios")
    recordings = _index_unique(data["recordings"], "id", "recordings")
    current = current_revisions(data["transcriptions"], label="transcriptions")
    transcriptions = _index_unique(current, "recording_id", "current transcriptions")
    if set(transcriptions) != set(recordings):
        missing = len(set(recordings) - set(transcriptions))
        extra = len(set(transcriptions) - set(recordings))
        raise ManifestError(
            "Current transcription coverage does not match recordings: "
            f"missing={missing}, extra={extra}"
        )

    speaker_codes = [
        str(record.get("speaker_code") or "").strip() for record in participants.values()
    ]
    if any(not value for value in speaker_codes) or len(set(speaker_codes)) != len(speaker_codes):
        raise ManifestError("speaker_code values must be non-empty and unique")
    display_ids = [str(record.get("display_id") or "").strip() for record in recordings.values()]
    if any(not value for value in display_ids) or len(set(display_ids)) != len(display_ids):
        raise ManifestError("recording display_id values must be non-empty and unique")

    rows: list[dict[str, str]] = []
    for recording in recordings.values():
        clip_id = str(recording["display_id"]).strip()
        participant_id = str(recording.get("participant_id") or "")
        scenario_id = str(recording.get("scenario_id") or "")
        if participant_id not in participants:
            raise ManifestError(f"Clip {clip_id}: unknown participant_id")
        if scenario_id not in scenarios:
            raise ManifestError(f"Clip {clip_id}: unknown scenario_id")

        source_audio = (export_root / str(recording.get("audio_path") or "")).resolve()
        if not source_audio.is_file() or not source_audio.is_relative_to(export_root):
            raise ManifestError(f"Clip {clip_id}: audio path is missing or leaves the export root")
        try:
            duration_s = float(recording["duration_ms"]) / 1000
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Clip {clip_id}: invalid duration_ms") from exc
        if duration_s <= 0:
            raise ManifestError(f"Clip {clip_id}: duration must be positive")

        transcription = transcriptions[str(recording["id"])]
        verbatim = str(transcription.get("verbatim_text") or "").strip()
        reference_eval = normalize_text(verbatim)
        if not verbatim or not reference_eval:
            raise ManifestError(f"Clip {clip_id}: verbatim or normalized evaluation text is empty")

        languages = _parse_string_list(
            transcription.get("language_tags_json"),
            field="language_tags_json",
            clip_id=clip_id,
        )
        unsupported_languages = sorted(set(languages) - set(_LANGUAGE_VALUES))
        if unsupported_languages:
            raise ManifestError(
                f"Clip {clip_id}: unsupported language labels: {', '.join(unsupported_languages)}"
            )
        canonical_languages = sorted(
            {_LANGUAGE_VALUES[value] for value in languages}, key=_LANGUAGE_ORDER.__getitem__
        )
        raw_code_switch = str(recording.get("code_switching_status") or "").strip().lower()
        if raw_code_switch not in _CODE_SWITCH_VALUES:
            raise ManifestError(f"Clip {clip_id}: unsupported code_switching_status")
        noise_level = str(recording.get("noise_level") or "").strip()
        if noise_level not in _NOISE_VALUES:
            raise ManifestError(f"Clip {clip_id}: unsupported noise_level")
        raw_quality = recording.get("quality_rating")
        if raw_quality is None:
            quality_rating = "unknown"
        else:
            try:
                quality_value = int(raw_quality)
            except (TypeError, ValueError) as exc:
                raise ManifestError(f"Clip {clip_id}: invalid quality_rating") from exc
            if quality_value not in {1, 2, 3, 4, 5}:
                raise ManifestError(f"Clip {clip_id}: quality_rating must be 1 through 5")
            quality_rating = str(quality_value)
        scenario = str(scenarios[scenario_id].get("expected_intent") or "").strip()
        if not scenario:
            raise ManifestError(f"Clip {clip_id}: scenario expected_intent is empty")

        relative_audio = os.path.relpath(source_audio, start=manifest_path.parent.resolve())
        rows.append(
            {
                "clip_id": clip_id,
                "speaker_id": str(participants[participant_id]["speaker_code"]).strip(),
                "audio_path": Path(relative_audio).as_posix(),
                "duration_s": f"{duration_s:.3f}",
                "reference_verbatim": verbatim,
                "reference_eval": reference_eval,
                "scenario": scenario,
                "languages": "+".join(canonical_languages),
                "languages_raw": "+".join(languages),
                "code_switched": _CODE_SWITCH_VALUES[raw_code_switch],
                "code_switch_status": raw_code_switch,
                "noise_level": _NOISE_VALUES[noise_level],
                "noise_level_raw": noise_level,
                "quality_rating": quality_rating,
            }
        )
    return sorted(rows, key=lambda row: row["clip_id"])


def select_balanced_smoke_rows(
    rows: list[dict[str, str]], *, size: int = 20, seed: int = 42
) -> list[dict[str, str]]:
    """Select equal speaker/scenario coverage, retrying deterministic shuffled orders."""
    if size == 1:
        return [select_preflight_row(rows)]
    speakers = sorted({row["speaker_id"] for row in rows})
    scenarios = sorted({row["scenario"] for row in rows})
    if not speakers or not scenarios or size % len(speakers) or size % len(scenarios):
        raise ManifestError("Smoke size must divide evenly across speakers and scenarios")
    speaker_quota = size // len(speakers)
    scenario_quota = size // len(scenarios)
    ordered = sorted(rows, key=lambda row: row["clip_id"])
    for attempt in range(10_000):
        candidates = list(ordered)
        random.Random(seed + attempt).shuffle(candidates)
        speaker_counts: Counter[str] = Counter()
        scenario_counts: Counter[str] = Counter()
        selected: list[dict[str, str]] = []
        for row in candidates:
            if speaker_counts[row["speaker_id"]] >= speaker_quota:
                continue
            if scenario_counts[row["scenario"]] >= scenario_quota:
                continue
            selected.append(row)
            speaker_counts[row["speaker_id"]] += 1
            scenario_counts[row["scenario"]] += 1
            if len(selected) == size:
                return sorted(selected, key=lambda row: row["clip_id"])
    raise ManifestError("Could not construct a balanced smoke subset from the supplied rows")


def select_preflight_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """Choose one clear, representative clip for model-integration preflight checks."""
    if not rows:
        raise ManifestError("Cannot select a preflight clip from an empty manifest")
    noise_rank = {"low": 0, "medium": 1, "high": 2, "unknown": 3}
    code_switch_rank = {"present": 0, "ambiguous": 1, "unknown": 2, "none": 3}

    def score(row: dict[str, str]) -> tuple[object, ...]:
        quality = row.get("quality_rating", "unknown")
        quality_rank = -int(quality) if quality.isdigit() else 0
        return (
            quality_rank,
            noise_rank.get(row.get("noise_level", "unknown"), 4),
            code_switch_rank.get(row.get("code_switch_status", "unknown"), 4),
            row.get("languages") != "swc+fr",
            abs(float(row["duration_s"]) - 7.0),
            row["clip_id"],
        )

    return min(rows, key=score)


def write_manifest_rows(path: Path, rows: list[Mapping[str, str]], *, force: bool = False) -> None:
    """Atomically write private manifest rows, refusing overwrite by default."""
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise ManifestError(f"Refusing to overwrite existing manifest without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        optional = [field for field in OPTIONAL_COLUMNS if any(field in row for row in rows)]
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(*REQUIRED_COLUMNS, *optional),
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
