"""Crash-tolerant prediction and run-metadata persistence."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kingwana_asr.models.base import PredictionRecord

PREDICTION_FIELDS = tuple(PredictionRecord.model_fields)


class PredictionStoreError(RuntimeError):
    """Raised when persisted predictions are malformed or inconsistent."""


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PredictionStore:
    """Append attempts to JSONL and atomically export one latest row per clip/model."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.jsonl_path = output_dir / "predictions.jsonl"
        self.csv_path = output_dir / "predictions.csv"
        self.error_log_path = output_dir / "errors.log"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_path.touch(exist_ok=True)
        self._latest: dict[tuple[str, str], PredictionRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = PredictionRecord.model_validate_json(line)
                except (ValidationError, ValueError) as exc:
                    raise PredictionStoreError(
                        f"Malformed prediction JSONL at line {line_number}: {self.jsonl_path}"
                    ) from exc
                self._latest[(record.model_id, record.clip_id)] = record

    def is_complete(self, model_id: str, clip_id: str) -> bool:
        record = self._latest.get((model_id, clip_id))
        return record is not None and record.status == "success"

    def append(self, record: PredictionRecord) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._latest[(record.model_id, record.clip_id)] = record
        if record.status == "error":
            with self.error_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{record.timestamp.isoformat()}\t{record.clip_id}\t{record.error}\n")
        self.export_csv()

    def export_csv(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".predictions.", suffix=".csv", dir=self.output_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
                writer.writeheader()
                for key in sorted(self._latest):
                    writer.writerow(self._latest[key].model_dump(mode="json"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.csv_path)
        finally:
            temporary.unlink(missing_ok=True)


def write_run_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    serialized = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text_write(output_dir / "run_metadata.json", serialized)
