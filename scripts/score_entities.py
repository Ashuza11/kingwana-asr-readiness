"""Score strict business-entity preservation for completed model outputs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.entity_scoring import load_scoring_aliases, score_predictions

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
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@app.command()
def main(
    review: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    outputs_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    entity_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Create a table of strict entity-preservation scores for every model output."""

    try:
        review_rows = _read_csv(review)
        aliases = load_scoring_aliases(entity_config)
        prediction_files = sorted(outputs_root.glob("*/predictions.csv"))
        if not prediction_files:
            raise ManifestError(f"No model predictions found under: {outputs_root}")
        results = []
        for prediction_file in prediction_files:
            results.extend(score_predictions(review_rows, _read_csv(prediction_file), aliases))
        rows = [
            {
                "model_id": result.model_id,
                "category": result.category,
                "correct_mentions": str(result.correct_mentions),
                "gold_mentions": str(result.gold_mentions),
                "mention_accuracy": f"{result.mention_accuracy:.6f}",
                "exact_clips": str(result.exact_clips),
                "eligible_clips": str(result.eligible_clips),
                "clip_accuracy": f"{result.clip_accuracy:.6f}",
                "scoring_policy": "strict_explicit_entity_preservation_v1",
            }
            for result in results
        ]
        _write_atomic(output, rows)
    except ManifestError as exc:
        typer.echo(f"Entity scoring failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Entity accuracy table: {len(rows)} model/category rows -> {output}")


if __name__ == "__main__":
    app()
