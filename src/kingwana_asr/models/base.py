"""Common model and prediction interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionRecord(BaseModel):
    """Standard persisted output from every future ASR adapter."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    clip_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    prediction: str
    runtime_seconds: float = Field(ge=0)
    status: Literal["success", "error"]
    error: str = ""
    language_config: str = ""
    device: str = ""
    dtype: str = ""
    model_revision: str = ""
    timestamp: datetime

    @model_validator(mode="after")
    def status_matches_error(self) -> PredictionRecord:
        if self.status == "success" and self.error:
            raise ValueError("successful predictions must not contain an error")
        if self.status == "error" and not self.error:
            raise ValueError("failed predictions must record an error")
        return self


class ASRModel(ABC):
    model_id: str
    display_name: str

    @abstractmethod
    def load(self) -> None:
        """Load model resources."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Transcribe one audio file."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
