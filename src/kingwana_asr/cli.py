"""Command-line interface for Phase 1 validation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.validation import validate_manifest

app = typer.Typer(
    name="kingwana-asr",
    help="Reproducible ASR evaluation for private Kingwana speech.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Run reproducible Kingwana ASR evaluation workflows."""


@app.command()
def validate(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a manifest and report duration, speaker, and scenario counts."""
    try:
        summary = validate_manifest(manifest)
    except ManifestError as exc:
        console.print(f"[bold red]Validation failed[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]Valid manifest:[/bold green] {summary.manifest}")
    console.print(f"Clips: {summary.clip_count}")
    console.print(f"Total duration: {summary.total_duration_s:.3f} seconds")
    for title, counts in (
        ("Clips by speaker", summary.counts_by_speaker),
        ("Clips by scenario", summary.counts_by_scenario),
    ):
        table = Table(title=title)
        table.add_column("Label")
        table.add_column("Count", justify="right")
        for label, count in counts.items():
            table.add_row(label, str(count))
        console.print(table)


if __name__ == "__main__":
    app()
