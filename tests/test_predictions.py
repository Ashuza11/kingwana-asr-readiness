from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from kingwana_asr.models.base import PredictionRecord


def base_record() -> dict[str, object]:
    return {
        "clip_id": "clip_001",
        "speaker_id": "SPK_001",
        "model_id": "whisper_large_v3",
        "reference": "niko hapa",
        "prediction": "niko hapa",
        "runtime_seconds": 0.5,
        "status": "success",
        "error": "",
        "timestamp": datetime.now(timezone.utc),
    }


def test_prediction_schema_accepts_success() -> None:
    prediction = PredictionRecord.model_validate(base_record())
    assert prediction.status == "success"


def test_prediction_schema_rejects_unexplained_failure() -> None:
    payload = base_record()
    payload.update(status="error", error="")
    with pytest.raises(ValidationError, match="failed predictions must record an error"):
        PredictionRecord.model_validate(payload)


def test_prediction_schema_forbids_unknown_fields() -> None:
    payload = base_record()
    payload["participant_name"] = "must not be persisted"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictionRecord.model_validate(payload)
