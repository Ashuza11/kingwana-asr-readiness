from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from kingwana_asr.inference import run_transcription
from kingwana_asr.models.base import ASRModel
from kingwana_asr.utils.device import DeviceSpec


class FakeModel(ASRModel):
    model_id = "whisper_large_v3"
    display_name = "Synthetic model"
    language_config = "language=swahili;task=transcribe"
    model_revision = "synthetic-revision"
    device_spec = DeviceSpec(
        requested="cpu",
        device="cpu",
        dtype="float32",
        accelerator_name=None,
        total_memory_bytes=None,
    )

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[str] = []

    def load(self) -> None:
        return None

    def transcribe(self, audio_path: Path) -> str:
        self.calls.append(audio_path.stem)
        if audio_path.stem in self.failures:
            raise RuntimeError("synthetic clip failure")
        return f"prediction for {audio_path.stem}"

    def unload(self) -> None:
        return None


def write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "evaluation_v0.1.csv"
    fields = [
        "clip_id",
        "speaker_id",
        "audio_path",
        "duration_s",
        "reference_verbatim",
        "reference_eval",
        "scenario",
        "languages",
        "code_switched",
        "noise_level",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in (1, 2):
            audio = tmp_path / f"audio_{index}.fixture"
            audio.touch()
            writer.writerow(
                {
                    "clip_id": f"clip_{index}",
                    "speaker_id": f"SPK_{index}",
                    "audio_path": audio,
                    "duration_s": "1.0",
                    "reference_verbatim": f"Reference {index}.",
                    "reference_eval": f"reference {index}",
                    "scenario": "synthetic",
                    "languages": "kingwana",
                    "code_switched": "false",
                    "noise_level": "low",
                }
            )
    return manifest


def factory_for(model: FakeModel):
    return lambda key, registry, device: model


def test_inference_continues_after_failure_and_resumes(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    output_root = tmp_path / "outputs"
    first_model = FakeModel({"audio_2"})
    first = run_transcription(
        model_key="whisper_large_v3",
        manifest=manifest,
        output_root=output_root,
        model_factory=factory_for(first_model),
    )
    assert first.completed == 1
    assert first.failed == 1
    assert first_model.calls == ["audio_1", "audio_2"]

    second_model = FakeModel()
    second = run_transcription(
        model_key="whisper_large_v3",
        manifest=manifest,
        output_root=output_root,
        model_factory=factory_for(second_model),
    )
    assert second.completed == 1
    assert second.skipped == 1
    assert second.failed == 0
    assert second_model.calls == ["audio_2"]

    output_dir = output_root / "whisper_large_v3"
    with (output_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"success"}
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["model_revision"] == "synthetic-revision"
    assert metadata["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert (output_dir / "errors.log").is_file()


def test_limit_and_clip_selection_are_enforced(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    model = FakeModel()
    summary = run_transcription(
        model_key="whisper_large_v3",
        manifest=manifest,
        output_root=tmp_path / "outputs",
        clip_ids=["clip_2"],
        limit=1,
        model_factory=factory_for(model),
    )
    assert summary.selected == 1
    assert model.calls == ["audio_2"]
