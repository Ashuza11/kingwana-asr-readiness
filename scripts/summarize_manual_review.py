"""Validate and summarize a completed private manual-error worksheet."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.manual_review import summarize_completed_manual_review

app = typer.Typer(add_completion=False)


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
    worksheet: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Fail on incomplete judgments; otherwise create model/category counts."""

    try:
        with worksheet.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None:
                raise ManifestError(f"Worksheet has no header: {worksheet}")
            rows = list(reader)
        summary = summarize_completed_manual_review(rows)
        _write_atomic(output, summary)
    except ManifestError as exc:
        typer.echo(f"Manual-review summary failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Manual-review summary: {len(summary)} rows -> {output}")


if __name__ == "__main__":
    app()
