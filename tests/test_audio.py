from pathlib import Path
from unittest.mock import Mock

import pytest

from kingwana_asr.data.audio import AudioPreparationError, prepare_audio


def test_prepare_audio_dry_run_preserves_source_and_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.input"
    source.write_bytes(b"synthetic-not-real-audio")
    destination = tmp_path / "processed" / "clip.wav"
    runner = Mock()

    result = prepare_audio(
        source,
        destination,
        dry_run=True,
        ffmpeg_binary="true",
        runner=runner,
    )

    assert source.read_bytes() == b"synthetic-not-real-audio"
    assert not destination.exists()
    assert result.command[-1] == str(destination.resolve())
    assert "pcm_s16le" in result.command
    assert "16000" in result.command
    runner.assert_not_called()


def test_prepare_audio_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.input"
    destination = tmp_path / "clip.wav"
    source.touch()
    destination.touch()
    with pytest.raises(AudioPreparationError, match="Destination already exists"):
        prepare_audio(source, destination, dry_run=True, ffmpeg_binary="true")


def test_prepare_audio_refuses_source_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.touch()
    with pytest.raises(AudioPreparationError, match="must be different"):
        prepare_audio(source, source, dry_run=True, ffmpeg_binary="true")
