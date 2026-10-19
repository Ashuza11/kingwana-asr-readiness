"""Build a derived analysis manifest with researcher-reviewed code-switch labels."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.code_switching import apply_reviewed_code_switch_labels

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
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    review: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Join complete reviewed labels without changing references or audio paths."""

    try:
        rows = apply_reviewed_code_switch_labels(_read(manifest), _read(review))
        _write_atomic(output, rows)
    except ManifestError as exc:
        typer.echo(f"Reviewed code-switch manifest failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    counts = Counter(row["code_switch_reviewed"] for row in rows)
    typer.echo(
        f"Reviewed code-switch manifest: {len(rows)} clips; "
        + ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    )


if __name__ == "__main__":
    app()
