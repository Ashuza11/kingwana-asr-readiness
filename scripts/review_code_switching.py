"""Create a private, derived worksheet for code-switch annotation review."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.evaluation.code_switching import review_code_switching

app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    force: Annotated[bool, typer.Option(help="Replace an existing derived review file")] = False,
) -> None:
    """Write proposals while retaining original labels and transcripts."""

    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise typer.BadParameter(f"Refusing to overwrite existing review file: {output}")

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise typer.BadParameter(f"Manifest has no header: {manifest}")
        required = {"clip_id", "reference_verbatim", "code_switch_status"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise typer.BadParameter(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    reviewed: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        decision = review_code_switching(row["reference_verbatim"])
        counts[decision.status] += 1
        reviewed.append(
            {
                **row,
                "code_switch_original": row["code_switch_status"],
                "code_switch_reviewed": decision.status,
                "code_switch_evidence": " | ".join(decision.evidence),
                "code_switch_review_reason": decision.rationale,
                "code_switch_review_status": (
                    "researcher_confirmation_required"
                    if decision.status == "needs_review"
                    else "rule_assigned"
                ),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fieldnames = list(reviewed[0]) if reviewed else []
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(reviewed)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    typer.echo(
        f"Reviewed {len(reviewed)} rows: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )


if __name__ == "__main__":
    app()
