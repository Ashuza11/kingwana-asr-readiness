"""Command-line interface for Phase 1 validation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.data.validation import validate_manifest
from kingwana_asr.evaluation.runner import EvaluationError, evaluate_outputs
from kingwana_asr.inference import run_transcription

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


@app.command()
def transcribe(
    model: Annotated[str, typer.Option(help="Model key from configs/models.yaml")],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    models_config: Annotated[Path, typer.Option()] = Path("configs/models.yaml"),
    output_root: Annotated[Path, typer.Option(file_okay=False)] = Path("outputs"),
    device: Annotated[str, typer.Option(help="auto, cuda, mps, or cpu")] = "auto",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    clip_id: Annotated[list[str] | None, typer.Option(help="Repeat to select clip IDs")] = None,
    force: Annotated[bool, typer.Option(help="Repeat successful predictions")] = False,
) -> None:
    """Run resumable transcription for one implemented model adapter."""
    try:
        summary = run_transcription(
            model_key=model,
            manifest=manifest,
            models_config=models_config,
            output_root=output_root,
            device=device,
            limit=limit,
            clip_ids=clip_id,
            force=force,
        )
    except Exception as exc:
        console.print(f"[bold red]Transcription failed[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Selected: {summary.selected}; completed: {summary.completed}; "
        f"skipped: {summary.skipped}; failed: {summary.failed}"
    )
    console.print(f"Outputs: {summary.output_dir}")


@app.command()
def evaluate(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output_root: Annotated[Path, typer.Option(file_okay=False)] = Path("outputs"),
    reports_root: Annotated[Path, typer.Option(file_okay=False)] = Path("reports/tables"),
    models_config: Annotated[Path, typer.Option()] = Path("configs/models.yaml"),
    model: Annotated[
        list[str] | None,
        typer.Option(help="Repeat to evaluate selected models; default is all enabled models"),
    ] = None,
) -> None:
    """Validate completed predictions and export reproducible metric tables."""
    try:
        paths = evaluate_outputs(
            manifest=manifest,
            output_root=output_root,
            reports_root=reports_root,
            models_config=models_config,
            model_keys=model,
        )
    except (EvaluationError, ManifestError, ValueError) as exc:
        console.print(f"[bold red]Evaluation failed[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold green]Evaluation complete:[/bold green] {len(paths)} tables")
    for path in paths.values():
        console.print(path)


if __name__ == "__main__":
    app()
