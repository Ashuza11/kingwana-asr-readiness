"""Build private evaluation and deterministic preflight manifests from a SautiForge export."""

from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.sautiforge import (
    build_manifest_rows,
    select_balanced_smoke_rows,
    write_manifest_rows,
)
from kingwana_asr.data.validation import validate_manifest

app = typer.Typer(add_completion=False)


@app.command()
def main(
    export_root: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    evaluation_manifest: Annotated[Path, typer.Option(dir_okay=False)],
    smoke_manifest: Annotated[Path, typer.Option(dir_okay=False)],
    smoke_size: Annotated[int, typer.Option(min=1)] = 1,
    seed: Annotated[int, typer.Option()] = 42,
    force: Annotated[bool, typer.Option(help="Replace existing generated manifests")] = False,
) -> None:
    """Resolve current revisions and create private, validated manifests."""
    try:
        rows = build_manifest_rows(export_root, evaluation_manifest)
        smoke_rows = select_balanced_smoke_rows(rows, size=smoke_size, seed=seed)
        write_manifest_rows(evaluation_manifest, rows, force=force)
        write_manifest_rows(smoke_manifest, smoke_rows, force=force)
        evaluation = validate_manifest(evaluation_manifest)
        smoke = validate_manifest(smoke_manifest)
    except ManifestError as exc:
        typer.echo(f"Manifest build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Evaluation manifest: {evaluation.clip_count} clips, "
        f"{evaluation.total_duration_s:.3f} seconds"
    )
    typer.echo(f"Smoke manifest: {smoke.clip_count} clips, {smoke.total_duration_s:.3f} seconds")
    typer.echo(f"Smoke speakers: {len(smoke.counts_by_speaker)}")
    typer.echo(f"Smoke scenarios: {len(smoke.counts_by_scenario)}")


if __name__ == "__main__":
    app()
