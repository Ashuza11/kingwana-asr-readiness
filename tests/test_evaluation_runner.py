import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from kingwana_asr.evaluation.runner import EvaluationError, evaluate_outputs
from kingwana_asr.models.base import PredictionRecord
from kingwana_asr.persistence import PREDICTION_FIELDS


def write_fixture(
    tmp_path: Path, *, include_second_prediction: bool = True
) -> tuple[Path, Path, Path]:
    audio_one = tmp_path / "one.wav"
    audio_two = tmp_path / "two.wav"
    audio_one.touch()
    audio_two.touch()
    manifest = tmp_path / "evaluation_test.csv"
    manifest_fields = [
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
        "code_switch_status",
        "quality_rating",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "clip_id": "c1",
                    "speaker_id": "S1",
                    "audio_path": audio_one,
                    "duration_s": "1.0",
                    "reference_verbatim": "Mimi nina deni.",
                    "reference_eval": "mimi nina deni",
                    "scenario": "debt",
                    "languages": "swc",
                    "code_switched": "none",
                    "noise_level": "low",
                    "code_switch_status": "none",
                    "quality_rating": "5",
                },
                {
                    "clip_id": "c2",
                    "speaker_id": "S2",
                    "audio_path": audio_two,
                    "duration_s": "2.0",
                    "reference_verbatim": "Chargeur iko hapa.",
                    "reference_eval": "chargeur iko hapa",
                    "scenario": "inventory",
                    "languages": "swc+fr",
                    "code_switched": "present",
                    "noise_level": "medium",
                    "code_switch_status": "present",
                    "quality_rating": "3",
                },
            ]
        )

    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "test_model": {
                        "enabled": True,
                        "hf_id": "example/test",
                        "family": "wav2vec2",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    output_dir = output_root / "test_model"
    output_dir.mkdir(parents=True)
    predictions = [
        PredictionRecord(
            clip_id="c1",
            speaker_id="S1",
            model_id="test_model",
            reference="mimi nina deni",
            prediction="mimi deni",
            runtime_seconds=0.1,
            status="success",
            timestamp=datetime.now(timezone.utc),
        )
    ]
    if include_second_prediction:
        predictions.append(
            PredictionRecord(
                clip_id="c2",
                speaker_id="S2",
                model_id="test_model",
                reference="chargeur iko hapa",
                prediction="chargeur iko hapa",
                runtime_seconds=0.1,
                status="success",
                timestamp=datetime.now(timezone.utc),
            )
        )
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(record.model_dump(mode="json") for record in predictions)
    return manifest, config, output_root


def test_evaluation_exports_summary_and_required_group_tables(tmp_path: Path) -> None:
    manifest, config, output_root = write_fixture(tmp_path)
    reports_root = tmp_path / "reports"

    paths = evaluate_outputs(
        manifest=manifest,
        output_root=output_root,
        reports_root=reports_root,
        models_config=config,
    )

    assert set(paths) >= {"model_summary.csv", "per_speaker.csv", "per_scenario.csv"}
    with paths["model_summary.csv"].open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 1
    assert summary[0]["model_id"] == "test_model"
    assert summary[0]["clip_count"] == "2"
    assert float(summary[0]["wer"]) == pytest.approx(1 / 6)
    with paths["per_speaker.csv"].open(encoding="utf-8", newline="") as handle:
        speakers = list(csv.DictReader(handle))
    assert {row["group"] for row in speakers} == {"S1", "S2"}


def test_evaluation_rejects_incomplete_model_output(tmp_path: Path) -> None:
    manifest, config, output_root = write_fixture(tmp_path, include_second_prediction=False)

    with pytest.raises(EvaluationError, match="missing=1"):
        evaluate_outputs(
            manifest=manifest,
            output_root=output_root,
            reports_root=tmp_path / "reports",
            models_config=config,
        )
