"""Model-independent, resumable per-clip transcription orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from kingwana_asr.data.validation import validate_manifest
from kingwana_asr.models.base import ASRModel, PredictionRecord
from kingwana_asr.models.registry import create_model, load_model_registry
from kingwana_asr.persistence import PredictionStore, write_run_metadata
from kingwana_asr.utils.reproducibility import environment_metadata


@dataclass(frozen=True, slots=True)
class TranscriptionSummary:
    selected: int
    completed: int
    skipped: int
    failed: int
    output_dir: Path


def _select_rows(
    rows: Sequence[Any], clip_ids: Sequence[str] | None, limit: int | None
) -> list[Any]:
    selected = list(rows)
    if clip_ids:
        requested = set(clip_ids)
        available = {row.clip_id for row in rows}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Unknown clip IDs: {', '.join(missing)}")
        selected = [row for row in rows if row.clip_id in requested]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        selected = selected[:limit]
    return selected


def run_transcription(
    *,
    model_key: str,
    manifest: Path,
    models_config: Path = Path("configs/models.yaml"),
    output_root: Path = Path("outputs"),
    device: str = "auto",
    limit: int | None = None,
    clip_ids: Sequence[str] | None = None,
    force: bool = False,
    model_factory: Callable[[str, dict[str, dict[str, Any]], str], ASRModel] | None = None,
) -> TranscriptionSummary:
    """Transcribe selected rows, persist each attempt, and continue after clip failures."""
    manifest_summary = validate_manifest(manifest)
    from kingwana_asr.data.manifest import read_manifest

    rows = _select_rows(read_manifest(manifest), clip_ids, limit)
    registry = load_model_registry(models_config)
    factory = model_factory or (
        lambda key, values, requested: create_model(key, values, device=requested)
    )
    model = factory(model_key, registry, device)
    output_dir = output_root / model_key
    store = PredictionStore(output_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        **environment_metadata(),
        "started_at": started_at,
        "status": "loading",
        "model_key": model_key,
        "model_repository_id": registry[model_key]["hf_id"],
        "model_config": registry[model_key],
        "manifest_path": str(manifest_summary.manifest),
        "manifest_version": manifest.stem,
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "manifest_clip_count": manifest_summary.clip_count,
        "selected_clip_count": len(rows),
        "device_requested": device,
        "force": force,
    }
    write_run_metadata(output_dir, metadata)

    completed = skipped = failed = 0
    try:
        model.load()
        device_spec = getattr(model, "device_spec", None)
        metadata.update(
            status="running",
            model_revision=getattr(model, "model_revision", None),
            language_config=getattr(model, "language_config", ""),
            device=asdict(device_spec) if device_spec is not None else None,
            cuda_version=getattr(
                getattr(getattr(model, "_torch", None), "version", None), "cuda", None
            ),
        )
        write_run_metadata(output_dir, metadata)

        for row in rows:
            if not force and store.is_complete(model_key, row.clip_id):
                skipped += 1
                continue
            started = perf_counter()
            try:
                prediction = model.transcribe(row.audio_path)
                if not prediction.strip():
                    raise RuntimeError("Model returned an empty transcription")
                status = "success"
                error = ""
                completed += 1
            except Exception as exc:  # per-clip isolation is intentional
                prediction = ""
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                failed += 1
            elapsed = perf_counter() - started
            record = PredictionRecord(
                clip_id=row.clip_id,
                speaker_id=row.speaker_id,
                model_id=model_key,
                reference=row.reference_eval,
                prediction=prediction,
                runtime_seconds=elapsed,
                status=status,
                error=error,
                language_config=getattr(model, "language_config", ""),
                device=getattr(device_spec, "device", "") if device_spec else "",
                dtype=getattr(device_spec, "dtype", "") if device_spec else "",
                model_revision=getattr(model, "model_revision", None) or "",
                timestamp=datetime.now(timezone.utc),
            )
            store.append(record)
    except Exception:
        metadata.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat())
        write_run_metadata(output_dir, metadata)
        raise
    finally:
        model.unload()

    metadata.update(
        status="complete_with_errors" if failed else "complete",
        finished_at=datetime.now(timezone.utc).isoformat(),
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
    write_run_metadata(output_dir, metadata)
    return TranscriptionSummary(len(rows), completed, skipped, failed, output_dir)
