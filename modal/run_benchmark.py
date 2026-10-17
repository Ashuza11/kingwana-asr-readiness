"""Modal GPU backend for the shared Kingwana ASR transcription pipeline."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path

import modal

APP_NAME = "kingwana-asr-readiness"
DATA_VOLUME_NAME = os.environ.get("KINGWANA_MODAL_DATA_VOLUME", "kingwana-asr-private-data")
OUTPUT_VOLUME_NAME = os.environ.get("KINGWANA_MODAL_OUTPUT_VOLUME", "kingwana-asr-outputs")
CACHE_VOLUME_NAME = os.environ.get("KINGWANA_MODAL_CACHE_VOLUME", "kingwana-asr-hf-cache")
HF_SECRET_NAME = "Speech2sql_project"
REMOTE_DATA_ROOT = Path("/data")
REMOTE_OUTPUT_ROOT = Path("/outputs")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFE_KEY = re.compile(r"[A-Za-z0-9_.-]+")

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "accelerate>=1.2,<2",
        "huggingface-hub>=0.27,<1",
        "jiwer>=3.1,<4",
        "librosa>=0.10,<1",
        "numpy>=1.26,<3",
        "omnilingual-asr==0.2.0",
        "pyarrow==20.0.0",
        "pydantic>=2.9,<3",
        "PyYAML>=6.0,<7",
        "rich>=13.9,<15",
        "soundfile>=0.12,<1",
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "transformers>=4.47,<5",
        "typer>=0.12,<1",
    )
    .add_local_dir(PROJECT_ROOT / "src" / "kingwana_asr", "/root/kingwana_asr")
    .add_local_dir(PROJECT_ROOT / "configs", "/root/configs")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    timeout=7_200,
    volumes={
        str(REMOTE_DATA_ROOT): data_volume,
        str(REMOTE_OUTPUT_ROOT): output_volume,
        "/cache": cache_volume,
    },
    env={
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_CACHE": "/cache/huggingface/hub",
    },
    secrets=[hf_secret],
)
def transcribe_remote(
    model_key: str,
    manifest_path: str,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Run the normal resumable transcription pipeline on one Modal GPU."""
    from kingwana_asr.inference import run_transcription

    try:
        summary = run_transcription(
            model_key=model_key,
            manifest=Path(manifest_path),
            models_config=Path("/root/configs/models.yaml"),
            output_root=REMOTE_OUTPUT_ROOT,
            device="cuda",
            limit=limit,
            force=force,
        )
        return {
            "selected": summary.selected,
            "completed": summary.completed,
            "skipped": summary.skipped,
            "failed": summary.failed,
            "output_dir": str(summary.output_dir),
        }
    finally:
        output_volume.commit()
        cache_volume.commit()


def _safe_key(value: str, *, field: str) -> str:
    if SAFE_KEY.fullmatch(value) is None:
        raise ValueError(f"{field} may contain only letters, numbers, dot, underscore, and dash")
    return value


def _stage_private_manifest(local_manifest: Path, dataset_key: str, *, force: bool) -> str:
    """Upload only manifest-referenced audio plus a path-rewritten private manifest."""
    local_manifest = local_manifest.expanduser().resolve()
    if not local_manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {local_manifest}")
    dataset_key = _safe_key(dataset_key, field="dataset_key")
    remote_base = f"/{dataset_key}"
    remote_manifest = f"{remote_base}/{local_manifest.name}"

    with local_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None or "clip_id" not in reader.fieldnames:
            raise ValueError("Manifest must contain a clip_id column")
        if "audio_path" not in reader.fieldnames:
            raise ValueError("Manifest must contain an audio_path column")
        fieldnames = reader.fieldnames
        records = list(reader)
    if not records:
        raise ValueError("Manifest contains no records")

    uploads: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for row in records:
        clip_id = _safe_key(str(row["clip_id"]), field="clip_id")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id in manifest: {clip_id}")
        seen.add(clip_id)
        local_audio = Path(str(row["audio_path"])).expanduser()
        if not local_audio.is_absolute():
            local_audio = (local_manifest.parent / local_audio).resolve()
        if not local_audio.is_file():
            raise FileNotFoundError(f"Audio file does not exist for clip {clip_id}")
        suffix = local_audio.suffix.lower()
        remote_audio = f"{remote_base}/audio/{clip_id}{suffix}"
        uploads.append((local_audio, remote_audio))
        row["audio_path"] = f"/data{remote_audio}"

    with tempfile.TemporaryDirectory(prefix="kingwana-modal-") as temporary_directory:
        staged_manifest = Path(temporary_directory) / "manifest.csv"
        with staged_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(records)
        with data_volume.batch_upload(force=force) as batch:
            for local_audio, remote_audio in uploads:
                batch.put_file(local_audio, remote_audio)
            batch.put_file(staged_manifest, remote_manifest)
    return f"/data{remote_manifest}"


@app.local_entrypoint()
def main(
    model: str = "whisper_large_v3",
    manifest: str = "data/manifests/smoke_v0.1_1clip.csv",
    dataset_key: str = "preflight-v0.1",
    upload: bool = False,
    force_upload: bool = False,
    limit: int | None = None,
    force: bool = False,
) -> None:
    """Optionally upload an explicit manifest subset, then run it on Modal."""
    _safe_key(model, field="model")
    _safe_key(dataset_key, field="dataset_key")
    remote_manifest = f"/data/{dataset_key}/{Path(manifest).name}"
    if upload:
        remote_manifest = _stage_private_manifest(Path(manifest), dataset_key, force=force_upload)
        print(f"Uploaded only the {dataset_key!r} manifest and its referenced audio files.")
    result = transcribe_remote.remote(model, remote_manifest, limit, force)
    print(
        "Modal run complete: "
        f"selected={result['selected']} completed={result['completed']} "
        f"skipped={result['skipped']} failed={result['failed']}"
    )
    print(f"Persistent output directory: {result['output_dir']}")
