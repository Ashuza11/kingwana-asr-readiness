"""Strict prediction loading and reproducible evaluation-table export."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kingwana_asr.data.manifest import read_manifest
from kingwana_asr.evaluation.aggregate import aggregate_predictions
from kingwana_asr.models.base import PredictionRecord
from kingwana_asr.models.registry import load_model_registry


class EvaluationError(RuntimeError):
    """Raised when predictions cannot be aligned and scored safely."""


SUMMARY_FIELDS = (
    "model_id",
    "clip_count",
    "wer",
    "cer",
    "speaker_macro_wer",
    "speaker_macro_cer",
    "word_substitutions",
    "word_deletions",
    "word_insertions",
    "reference_words",
    "character_substitutions",
    "character_deletions",
    "character_insertions",
    "reference_characters",
)

GROUP_FIELDS = (
    "model_id",
    "group",
    "clip_count",
    "wer",
    "cer",
    "word_substitutions",
    "word_deletions",
    "word_insertions",
    "reference_words",
    "character_substitutions",
    "character_deletions",
    "character_insertions",
    "reference_characters",
)


def _read_manifest_annotations(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise EvaluationError(f"Manifest has no header: {path}")
        try:
            rows = list(reader)
        except csv.Error as exc:
            raise EvaluationError(f"Malformed manifest CSV: {path}") from exc
    return {str(row["clip_id"]).strip(): row for row in rows}


def _load_predictions(path: Path, expected_model: str) -> list[PredictionRecord]:
    if not path.is_file():
        raise EvaluationError(f"Missing predictions for {expected_model}: {path}")
    records: list[PredictionRecord] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise EvaluationError(f"Prediction file has no header: {path}")
        try:
            for row_number, row in enumerate(reader, start=2):
                try:
                    record = PredictionRecord.model_validate(row)
                except ValidationError as exc:
                    raise EvaluationError(
                        f"Invalid prediction at row {row_number}: {path}"
                    ) from exc
                if record.model_id != expected_model:
                    raise EvaluationError(
                        f"Row {row_number} in {path} has model_id={record.model_id!r}; "
                        f"expected {expected_model!r}"
                    )
                key = (record.model_id, record.clip_id)
                if key in seen:
                    raise EvaluationError(
                        f"Duplicate model_id + clip_id at row {row_number}: {key}"
                    )
                seen.add(key)
                records.append(record)
        except csv.Error as exc:
            raise EvaluationError(f"Malformed prediction CSV: {path}") from exc
    if not records:
        raise EvaluationError(f"Prediction file contains no records: {path}")
    return records


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _group_rows(model_id: str, groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model_id": model_id,
            "group": group,
            "clip_count": metrics["utterance_count"],
            **{field: metrics[field] for field in GROUP_FIELDS[3:]},
        }
        for group, metrics in groups.items()
    ]


def evaluate_outputs(
    *,
    manifest: Path,
    output_root: Path = Path("outputs"),
    reports_root: Path = Path("reports/tables"),
    models_config: Path = Path("configs/models.yaml"),
    model_keys: Sequence[str] | None = None,
    minimum_scenario_clips: int = 1,
) -> dict[str, Path]:
    """Validate complete model outputs, aggregate them, and atomically write tables."""
    manifest_rows = read_manifest(manifest)
    annotations = _read_manifest_annotations(manifest)
    manifest_by_id = {row.clip_id: row for row in manifest_rows}
    expected_ids = set(manifest_by_id)

    registry = load_model_registry(models_config)
    selected = (
        list(model_keys)
        if model_keys
        else [key for key, config in registry.items() if config.get("enabled", False)]
    )
    if not selected:
        raise EvaluationError("No models were selected for evaluation")
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise EvaluationError(f"Unknown model keys: {', '.join(unknown)}")

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {
        "per_speaker.csv": [],
        "per_scenario.csv": [],
        "per_code_switch.csv": [],
        "per_code_switch_status.csv": [],
        "per_language_mix.csv": [],
        "per_noise_level.csv": [],
        "per_quality_rating.csv": [],
    }
    group_sources = {
        "per_speaker.csv": "per_speaker",
        "per_scenario.csv": "per_scenario",
        "per_code_switch.csv": "by_code_switch",
        "per_code_switch_status.csv": "by_code_switch_status",
        "per_language_mix.csv": "by_language_mix",
        "per_noise_level.csv": "by_noise_level",
        "per_quality_rating.csv": "by_quality_rating",
    }

    for model_id in selected:
        predictions = _load_predictions(output_root / model_id / "predictions.csv", model_id)
        prediction_ids = {record.clip_id for record in predictions}
        missing = sorted(expected_ids - prediction_ids)
        extra = sorted(prediction_ids - expected_ids)
        if missing or extra:
            raise EvaluationError(
                f"{model_id} predictions do not match the manifest "
                f"(missing={len(missing)}, extra={len(extra)})"
            )

        aligned: list[dict[str, Any]] = []
        for prediction in predictions:
            manifest_row = manifest_by_id[prediction.clip_id]
            if prediction.speaker_id != manifest_row.speaker_id:
                raise EvaluationError(f"Speaker mismatch for {model_id}/{prediction.clip_id}")
            if prediction.reference != manifest_row.reference_eval:
                raise EvaluationError(f"Reference mismatch for {model_id}/{prediction.clip_id}")
            raw = annotations[prediction.clip_id]
            aligned.append(
                {
                    **prediction.model_dump(mode="json"),
                    "scenario": manifest_row.scenario,
                    "languages": manifest_row.languages,
                    "code_switched": manifest_row.code_switched,
                    "code_switch_status": raw.get("code_switch_status", "")
                    or raw.get("code_switched", ""),
                    "noise_level": manifest_row.noise_level,
                    "quality_rating": raw.get("quality_rating", "") or "unknown",
                }
            )

        report = aggregate_predictions(aligned, minimum_scenario_clips=minimum_scenario_clips)
        global_metrics = report["global"]
        summary_rows.append(
            {
                "model_id": model_id,
                "clip_count": global_metrics["utterance_count"],
                "wer": global_metrics["wer"],
                "cer": global_metrics["cer"],
                "speaker_macro_wer": report["speaker_macro_wer"],
                "speaker_macro_cer": report["speaker_macro_cer"],
                **{field: global_metrics[field] for field in SUMMARY_FIELDS[6:]},
            }
        )
        for filename, source in group_sources.items():
            grouped[filename].extend(_group_rows(model_id, report[source]))

    paths = {"model_summary.csv": reports_root / "model_summary.csv"}
    _atomic_csv(paths["model_summary.csv"], SUMMARY_FIELDS, summary_rows)
    for filename, rows in grouped.items():
        paths[filename] = reports_root / filename
        _atomic_csv(paths[filename], GROUP_FIELDS, rows)
    return paths
