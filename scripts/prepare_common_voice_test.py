"""Retain only the Common Voice sw-kingwana test subset from an official archive."""

from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.common_voice import (
    COMMON_VOICE_ARCHIVE_SHA256,
    prepare_common_voice_test,
)
from kingwana_asr.data.manifest import ManifestError

app = typer.Typer(add_completion=False)


@app.command()
def main(
    archive: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output_audio_dir: Annotated[Path, typer.Option(file_okay=False)],
    manifest: Annotated[Path, typer.Option(dir_okay=False)],
    provenance: Annotated[Path, typer.Option(dir_okay=False)],
    verify_checksum: Annotated[
        bool,
        typer.Option(
            "--verify-checksum/--skip-checksum",
            help="Verify the official CV 27.0 Swahili archive SHA-256",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(help="Replace existing selected clips, manifest, and provenance"),
    ] = False,
) -> None:
    """Extract exact sw-kingwana rows from test.tsv; ignore all other splits/variants."""
    try:
        summary = prepare_common_voice_test(
            archive,
            output_audio_dir,
            manifest,
            provenance,
            expected_sha256=COMMON_VOICE_ARCHIVE_SHA256 if verify_checksum else None,
            force=force,
        )
    except ManifestError as exc:
        typer.echo(f"Common Voice preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Manifest: {summary.manifest_path}")
    typer.echo(f"Retained clips: {summary.clip_count}")
    typer.echo(f"Pseudonymous speakers: {summary.speaker_count}")
    typer.echo(f"Total duration: {summary.total_duration_s:.3f} seconds")
    typer.echo(f"Audio directory: {summary.audio_dir}")


if __name__ == "__main__":
    app()
