"""Export descriptive differences between two completed corpus summaries."""

import csv
from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.evaluation.cross_corpus import compare_model_summaries

app = typer.Typer(add_completion=False)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


@app.command()
def main(
    business_summary: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    comparator_summary: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Write business-minus-comparator WER/CER differences in percentage points."""

    rows = compare_model_summaries(_read(business_summary), _read(comparator_summary))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    typer.echo(f"Wrote {len(rows)} model comparisons to {output}")


if __name__ == "__main__":
    app()
