"""Prepare the pinned TICO-19 Congolese Swahili test split."""

from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.tico19 import prepare_tico19_test

app = typer.Typer(add_completion=False)


@app.command()
def main(
    shard_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output_audio_dir: Annotated[Path, typer.Option(file_okay=False)],
    manifest: Annotated[Path, typer.Option(dir_okay=False)],
    provenance: Annotated[Path, typer.Option(dir_okay=False)],
    force: Annotated[
        bool,
        typer.Option(help="Replace extracted audio, manifest, and provenance"),
    ] = False,
) -> None:
    """Extract only the pinned, official TICO-19 test rows and embedded audio."""
    try:
        summary = prepare_tico19_test(
            shard_dir,
            output_audio_dir,
            manifest,
            provenance,
            force=force,
        )
    except ManifestError as exc:
        typer.echo(f"TICO-19 preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Manifest: {summary.manifest_path}")
    typer.echo(f"Clips: {summary.clip_count}")
    typer.echo(f"Pseudonymous speakers: {summary.speaker_count}")
    typer.echo(f"Total duration: {summary.total_duration_s:.3f} seconds")
    typer.echo(f"Audio directory: {summary.audio_dir}")


if __name__ == "__main__":
    app()
