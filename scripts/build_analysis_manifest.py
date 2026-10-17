"""Enrich a standardized private manifest with approved annotation strata."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.sautiforge import build_manifest_rows, write_manifest_rows
from kingwana_asr.data.validation import validate_manifest

app = typer.Typer(add_completion=False)


def build_analysis_rows(standardized_manifest: Path, export_root: Path) -> list[dict[str, str]]:
    """Join source annotations without changing inference inputs or references."""
    with standardized_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise ManifestError(f"Manifest has no header: {standardized_manifest}")
        standardized = list(reader)

    source_rows = build_manifest_rows(export_root, standardized_manifest)
    source_by_id = {row["clip_id"]: row for row in source_rows}
    standardized_ids = {row["clip_id"] for row in standardized}
    if standardized_ids != set(source_by_id):
        raise ManifestError("Standardized and source-export clip sets do not match")

    immutable_fields = (
        "speaker_id",
        "reference_verbatim",
        "reference_eval",
        "scenario",
        "code_switched",
    )
    enriched: list[dict[str, str]] = []
    for row in standardized:
        clip_id = row["clip_id"]
        source = source_by_id[clip_id]
        for field in immutable_fields:
            if row[field] != source[field]:
                raise ManifestError(f"Clip {clip_id}: source mismatch for {field}")
        if row["languages"] != source["languages_raw"]:
            raise ManifestError(f"Clip {clip_id}: raw language labels do not match the source")
        if row["noise_level"] != source["noise_level_raw"]:
            raise ManifestError(f"Clip {clip_id}: raw noise label does not match the source")

        enriched.append(
            {
                **row,
                "languages": source["languages"],
                "languages_raw": source["languages_raw"],
                "code_switch_status": source["code_switch_status"],
                "noise_level": source["noise_level"],
                "noise_level_raw": source["noise_level_raw"],
                "quality_rating": source["quality_rating"],
            }
        )
    return enriched


@app.command()
def main(
    standardized_manifest: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ],
    export_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output_manifest: Annotated[Path, typer.Option(dir_okay=False)],
    force: Annotated[bool, typer.Option(help="Replace an existing derived manifest")] = False,
) -> None:
    """Create a private analysis manifest with canonical and raw annotation labels."""
    try:
        rows = build_analysis_rows(standardized_manifest, export_root)
        write_manifest_rows(output_manifest, rows, force=force)
        summary = validate_manifest(output_manifest)
    except ManifestError as exc:
        typer.echo(f"Analysis-manifest build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Analysis manifest: {summary.clip_count} clips, {summary.total_duration_s:.3f} seconds"
    )


if __name__ == "__main__":
    app()
