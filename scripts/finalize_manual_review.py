"""Finalize human-validated web labels without modifying source worksheets."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.manual_review_web import finalize_label_review

app = typer.Typer(add_completion=False)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
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


@app.command()
def main(
    review: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    worksheet: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output_worksheet: Annotated[Path, typer.Option(dir_okay=False)],
    output_summary: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Validate all decisions and write final compatibility and label summaries."""

    try:
        finalized, summary = finalize_label_review(_read(review), _read(worksheet))
        _write_atomic(output_worksheet, finalized)
        _write_atomic(output_summary, summary)
    except ManifestError as exc:
        typer.echo(f"Manual-review finalization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Manual review finalized: {len(finalized)} comparisons; {len(summary)} summary rows"
    )


if __name__ == "__main__":
    app()
