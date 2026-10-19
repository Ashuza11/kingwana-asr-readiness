"""Build a private, balanced 50-clip manual ASR review worksheet."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Annotated

import typer
import yaml

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.manual_review import (
    build_manual_review_rows,
    select_error_stratified_sample,
    summarize_manual_review_edits,
    validate_manual_sample,
)

app = typer.Typer(add_completion=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise ManifestError(f"CSV has no header: {path}")
        return list(reader)


def _write_atomic(path: Path, rows: list[dict[str, str]]) -> None:
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


def _model_order(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    try:
        return [key for key, value in configuration["models"].items() if value["enabled"]]
    except (KeyError, TypeError) as exc:
        raise ManifestError("Models configuration is malformed") from exc


@app.command()
def main(
    entity_review: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    code_switch_review: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    outputs_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    models_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    selection_output: Annotated[Path, typer.Option(dir_okay=False)],
    worksheet_output: Annotated[Path, typer.Option(dir_okay=False)],
    automatic_summary_output: Annotated[Path, typer.Option(dir_okay=False)],
    size: Annotated[int, typer.Option(min=1)] = 50,
    seed: Annotated[int, typer.Option()] = 42,
    insertion_threshold: Annotated[int, typer.Option(min=1)] = 50,
) -> None:
    """Create the fixed manual-review sample and its five-model worksheet."""

    try:
        entity_rows = _read_csv(entity_review)
        code_switch_by_id = {row["clip_id"]: row for row in _read_csv(code_switch_review)}
        if {row["clip_id"] for row in entity_rows} != set(code_switch_by_id):
            raise ManifestError("Entity and code-switch review clip IDs do not align")
        candidates = [
            {
                **row,
                "audio_path": code_switch_by_id[row["clip_id"]]["audio_path"],
                "duration_s": code_switch_by_id[row["clip_id"]]["duration_s"],
                "noise_level": code_switch_by_id[row["clip_id"]]["noise_level"],
                "quality_rating": code_switch_by_id[row["clip_id"]]["quality_rating"],
                "code_switch_reviewed": code_switch_by_id[row["clip_id"]]["code_switch_reviewed"],
            }
            for row in entity_rows
        ]
        model_order = _model_order(models_config)
        predictions_by_model = {
            model_id: _read_csv(outputs_root / model_id / "predictions.csv")
            for model_id in model_order
        }
        selected = select_error_stratified_sample(
            candidates,
            predictions_by_model["whisper_large_v3"],
            size=size,
            seed=seed,
            insertion_threshold=insertion_threshold,
        )
        validate_manual_sample(selected, expected_size=size)
        worksheet = build_manual_review_rows(
            selected,
            predictions_by_model,
            model_order=model_order,
            seed=seed,
        )
        selection_rows = [
            {
                "selection_seed": str(seed),
                "clip_id": row["clip_id"],
                "speaker_id": row["speaker_id"],
                "scenario": row["scenario"],
                "audio_path": row["audio_path"],
                "duration_s": row["duration_s"],
                "noise_level": row["noise_level"],
                "quality_rating": row["quality_rating"],
                "code_switch_reviewed": row["code_switch_reviewed"],
                "sample_stratum": row["sample_stratum"],
                "selection_whisper_insertions": row["selection_whisper_insertions"],
                "reference_verbatim": row["reference_verbatim"],
            }
            for row in selected
        ]
        _write_atomic(selection_output, selection_rows)
        _write_atomic(worksheet_output, worksheet)
        _write_atomic(automatic_summary_output, summarize_manual_review_edits(worksheet))
    except (ManifestError, FileNotFoundError) as exc:
        typer.echo(f"Manual-review build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Manual-review clips: {len(selected)} -> {selection_output}")
    typer.echo(f"Model comparisons: {len(worksheet)} -> {worksheet_output}")
    typer.echo(f"Automatic edit summary -> {automatic_summary_output}")


if __name__ == "__main__":
    app()
