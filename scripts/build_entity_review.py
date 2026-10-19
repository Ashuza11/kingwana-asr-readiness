"""Build a private worksheet for validating business-critical entity gold labels."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.sautiforge import (
    _read_jsonl,
    apply_transcript_corrections,
    current_revisions,
)
from kingwana_asr.evaluation.entity_validation import (
    ENTITY_FIELDS,
    EntityReviewError,
    apply_currency_policy,
    apply_numeric_provenance_policy,
    apply_payment_method_policy,
    apply_product_mapping,
    apply_review_decisions,
    apply_transaction_intent_policy,
    audit_entity_payload,
    finalize_entity_review,
)

app = typer.Typer(add_completion=False)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"clip_id", "speaker_id", "scenario", "reference_verbatim"}
        if reader.fieldnames is None:
            raise ManifestError(f"Manifest has no header: {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ManifestError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def _payload(value: Any, *, clip_id: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Clip {clip_id}: annotation payload is malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise ManifestError(f"Clip {clip_id}: annotation payload must be an object")
    return parsed


def build_review_rows(
    manifest: Path,
    export_root: Path,
    *,
    manifest_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Join current annotations to manifest rows and add non-destructive audit columns."""

    if manifest_rows is None:
        manifest_rows = _read_manifest(manifest)
    recordings = _read_jsonl(export_root / "metadata/recordings.jsonl")
    current_annotations = current_revisions(
        _read_jsonl(export_root / "metadata/annotations.jsonl"), label="annotations"
    )
    recording_by_clip = {str(row.get("display_id") or "").strip(): row for row in recordings}
    annotation_by_recording = {
        str(row.get("recording_id") or "").strip(): row for row in current_annotations
    }
    manifest_ids = {row["clip_id"] for row in manifest_rows}
    if manifest_ids != set(recording_by_clip):
        raise ManifestError("Manifest and recording clip sets do not match")
    if set(annotation_by_recording) != {str(row["id"]) for row in recordings}:
        raise ManifestError("Current annotation coverage does not match recordings")

    output: list[dict[str, str]] = []
    for row in manifest_rows:
        clip_id = row["clip_id"]
        recording = recording_by_clip[clip_id]
        annotation = annotation_by_recording[str(recording["id"])]
        payload = _payload(annotation.get("payload_json"), clip_id=clip_id)
        audit = audit_entity_payload(
            reference_verbatim=row["reference_verbatim"],
            expected_scenario=row["scenario"],
            payload=payload,
        )
        review_row = {
            "clip_id": clip_id,
            "speaker_id": row["speaker_id"],
            "scenario": row["scenario"],
            "reference_verbatim": row["reference_verbatim"],
            "reference_numbers": " | ".join(audit.reference_numbers),
            "audit_flags": " | ".join(audit.flags),
            "gold_review_status": "pending",
            "review_notes": "",
        }
        for field in ENTITY_FIELDS:
            raw = payload.get(field)
            value = "" if raw is None else str(raw)
            review_row[f"source_{field}"] = value
            review_row[f"audit_{field}"] = audit.field_statuses[field]
            review_row[f"reviewed_{field}"] = value
            review_row[f"review_status_{field}"] = "pending"
            if field in {"quantity", "amount", "amountPaid", "outstandingDebt"}:
                review_row[f"value_origin_{field}"] = "pending_review"
        output.append(review_row)
    return sorted(output, key=lambda row: row["clip_id"])


def _write_atomic(path: Path, rows: list[dict[str, str]], *, force: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise ManifestError(f"Refusing to overwrite existing review file without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@app.command()
def main(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    export_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    decisions: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Researcher decisions CSV"),
    ] = None,
    entity_config: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Researcher-approved entity canonicalization YAML",
        ),
    ] = None,
    transcript_corrections: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Researcher-confirmed transcript corrections CSV",
        ),
    ] = None,
    force: Annotated[bool, typer.Option(help="Replace an existing derived worksheet")] = False,
) -> None:
    """Create a private entity validation worksheet without changing source annotations."""

    try:
        manifest_rows = _read_manifest(manifest)
        if transcript_corrections is not None:
            manifest_rows = apply_transcript_corrections(manifest_rows, transcript_corrections)
        rows = build_review_rows(manifest, export_root, manifest_rows=manifest_rows)
        if entity_config is not None:
            rows = apply_transaction_intent_policy(rows)
            rows = apply_product_mapping(rows, entity_config)
            rows = apply_currency_policy(rows, entity_config)
            rows = apply_numeric_provenance_policy(rows, entity_config)
            rows = apply_payment_method_policy(rows, entity_config)
        if decisions is not None:
            rows = apply_review_decisions(rows, decisions)
        rows = finalize_entity_review(rows)
        _write_atomic(output, rows, force=force)
    except (ManifestError, EntityReviewError) as exc:
        typer.echo(f"Entity review build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    flag_counts: Counter[str] = Counter(
        flag for row in rows for flag in row["audit_flags"].split(" | ") if flag
    )
    typer.echo(f"Entity review worksheet: {len(rows)} clips")
    for flag, count in sorted(flag_counts.items()):
        typer.echo(f"  {flag}: {count}")


if __name__ == "__main__":
    app()
