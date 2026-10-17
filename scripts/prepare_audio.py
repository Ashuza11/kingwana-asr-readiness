"""Prepare manifest audio as source-preserving 16 kHz mono PCM WAV files."""

import csv
import os
from pathlib import Path
from typing import Annotated

import typer

from kingwana_asr.data.audio import AudioPreparationError, prepare_audio
from kingwana_asr.data.manifest import REQUIRED_COLUMNS, ManifestError, read_manifest

app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option(file_okay=False)],
    output_manifest: Annotated[
        Path | None,
        typer.Option(help="Write a copy whose audio paths point to prepared WAV files"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Prepare every manifest clip, naming derivatives by unique clip ID."""
    try:
        rows = read_manifest(manifest)
        prepared_paths: dict[str, Path] = {}
        for row in rows:
            destination = output_dir / f"{row.clip_id}.wav"
            result = prepare_audio(row.audio_path, destination, dry_run=dry_run)
            prepared_paths[row.clip_id] = result.destination
            prefix = "Would create" if result.dry_run else "Created"
            typer.echo(f"{prefix}: {result.destination}")
        if output_manifest is not None:
            _write_prepared_manifest(
                manifest,
                output_manifest,
                prepared_paths,
                dry_run=dry_run,
            )
    except (ManifestError, AudioPreparationError) as exc:
        typer.echo(f"Audio preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _write_prepared_manifest(
    source: Path,
    destination: Path,
    prepared_paths: dict[str, Path],
    *,
    dry_run: bool,
) -> None:
    destination = destination.expanduser().resolve()
    if destination == source.expanduser().resolve():
        raise ManifestError("Prepared manifest must not overwrite its source manifest")
    if destination.exists():
        raise ManifestError(f"Prepared manifest already exists: {destination}")
    if dry_run:
        typer.echo(f"Would create manifest: {destination}")
        return

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise ManifestError(f"Manifest has no header: {source}")
        missing = [field for field in REQUIRED_COLUMNS if field not in reader.fieldnames]
        if missing:
            raise ManifestError(f"Manifest is missing required columns: {', '.join(missing)}")
        records = list(reader)

    for record in records:
        clip_id = record["clip_id"]
        if clip_id not in prepared_paths:
            raise ManifestError(f"No prepared audio path for clip: {clip_id}")
        relative = os.path.relpath(prepared_paths[clip_id], start=destination.parent)
        record["audio_path"] = Path(relative).as_posix()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    typer.echo(f"Created manifest: {destination}")


if __name__ == "__main__":
    app()
