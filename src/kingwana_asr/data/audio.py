"""Lossless source-preserving audio preparation with FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

EVALUATION_SAMPLE_RATE_HZ = 16_000
EVALUATION_CHANNELS = 1
EVALUATION_ENCODING = "PCM signed 16-bit little-endian"


class AudioPreparationError(RuntimeError):
    """Raised when an evaluation derivative cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class AudioPreparationResult:
    source: Path
    destination: Path
    command: tuple[str, ...]
    dry_run: bool


def prepare_audio(
    source: Path,
    destination: Path,
    *,
    dry_run: bool = False,
    ffmpeg_binary: str = "ffmpeg",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> AudioPreparationResult:
    """Create a 16 kHz mono PCM WAV derivative without overwriting any file."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise AudioPreparationError(f"Source audio does not exist: {source}")
    if source == destination:
        raise AudioPreparationError("Source and destination must be different files")
    if destination.exists():
        raise AudioPreparationError(f"Destination already exists: {destination}")
    if shutil.which(ffmpeg_binary) is None:
        raise AudioPreparationError(f"FFmpeg executable was not found: {ffmpeg_binary}")

    command = (
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(EVALUATION_CHANNELS),
        "-ar",
        str(EVALUATION_SAMPLE_RATE_HZ),
        "-c:a",
        "pcm_s16le",
        str(destination),
    )
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            runner(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "unknown FFmpeg error").strip()
            raise AudioPreparationError(f"FFmpeg failed for {source}: {detail}") from exc

    return AudioPreparationResult(source, destination, command, dry_run)
