"""CSV manifest schema and strict parser."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = (
    "clip_id",
    "speaker_id",
    "audio_path",
    "duration_s",
    "reference_verbatim",
    "reference_eval",
    "scenario",
    "languages",
    "code_switched",
    "noise_level",
)

OPTIONAL_COLUMNS = (
    "languages_raw",
    "code_switch_status",
    "noise_level_raw",
    "quality_rating",
    "amount_present",
    "amount_gold",
    "currency_gold",
    "customer_present",
    "product_present",
    "notes",
)

TRUE_VALUES = {"true", "1", "yes"}
FALSE_VALUES = {"false", "0", "no"}
UNAVAILABLE_VALUES = {"ambiguous", "unknown"}


class ManifestError(ValueError):
    """Raised when any manifest row violates the evaluation schema."""


def parse_bool(value: str, *, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ManifestError(
        f"Row {row_number}: {field} must be one of true/false, 1/0, yes/no; got {value!r}"
    )


def parse_optional_bool(value: str, *, field: str, row_number: int) -> bool | None:
    """Parse a boolean while preserving explicitly unavailable annotations."""
    normalized = value.strip().lower()
    if normalized in UNAVAILABLE_VALUES:
        return None
    if normalized == "present":
        return True
    if normalized == "none":
        return False
    try:
        return parse_bool(value, field=field, row_number=row_number)
    except ManifestError as exc:
        raise ManifestError(
            f"Row {row_number}: {field} must be true/present, false/none, ambiguous, "
            f"or unknown; got {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ManifestRow:
    clip_id: str
    speaker_id: str
    audio_path: Path
    duration_s: float
    reference_verbatim: str
    reference_eval: str
    scenario: str
    languages: str
    code_switched: bool | None
    noise_level: str
    source_row: int


def _required_text(row: dict[str, str], field: str, row_number: int) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ManifestError(f"Row {row_number}: required field {field!r} is empty")
    return value.strip()


def read_manifest(path: Path) -> list[ManifestRow]:
    """Read every CSV row strictly; malformed rows are never skipped."""
    if not path.is_file():
        raise ManifestError(f"Manifest does not exist: {path}")

    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise ManifestError(f"Manifest has no header: {path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ManifestError(f"Manifest is missing required columns: {', '.join(missing)}")
        unknown = [
            column
            for column in reader.fieldnames
            if column not in REQUIRED_COLUMNS and column not in OPTIONAL_COLUMNS
        ]
        if unknown:
            raise ManifestError(f"Manifest has unknown columns: {', '.join(unknown)}")

        try:
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ManifestError(f"Row {row_number}: too many CSV fields")
                if any(value is None for value in row.values()):
                    raise ManifestError(f"Row {row_number}: too few CSV fields")
                duration_raw = _required_text(row, "duration_s", row_number)
                try:
                    duration = float(duration_raw)
                except ValueError as exc:
                    raise ManifestError(
                        f"Row {row_number}: duration_s is not numeric: {duration_raw!r}"
                    ) from exc
                if not math.isfinite(duration) or duration <= 0:
                    raise ManifestError(
                        f"Row {row_number}: duration_s must be finite and greater than zero"
                    )

                audio_raw = _required_text(row, "audio_path", row_number)
                audio_path = Path(audio_raw).expanduser()
                if not audio_path.is_absolute():
                    audio_path = (path.parent / audio_path).resolve()

                rows.append(
                    ManifestRow(
                        clip_id=_required_text(row, "clip_id", row_number),
                        speaker_id=_required_text(row, "speaker_id", row_number),
                        audio_path=audio_path,
                        duration_s=duration,
                        reference_verbatim=_required_text(row, "reference_verbatim", row_number),
                        reference_eval=_required_text(row, "reference_eval", row_number),
                        scenario=_required_text(row, "scenario", row_number),
                        languages=_required_text(row, "languages", row_number),
                        code_switched=parse_optional_bool(
                            _required_text(row, "code_switched", row_number),
                            field="code_switched",
                            row_number=row_number,
                        ),
                        noise_level=_required_text(row, "noise_level", row_number),
                        source_row=row_number,
                    )
                )
        except csv.Error as exc:
            raise ManifestError(f"Malformed CSV near row {reader.line_num}: {exc}") from exc

    if not rows:
        raise ManifestError("Manifest contains no data rows")
    return rows
