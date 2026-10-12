"""Explicit wrappers around jiwer WER and CER calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jiwer import process_characters, process_words

from kingwana_asr.evaluation.normalization import normalize_text


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    wer: float
    cer: float
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    reference_words: int
    character_substitutions: int
    character_deletions: int
    character_insertions: int
    reference_characters: int
    utterance_count: int


def calculate_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalize: bool = True,
    cer_includes_whitespace: bool = True,
) -> ErrorMetrics:
    """Calculate corpus-micro WER/CER after the declared normalization pipeline."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one reference/hypothesis pair is required")

    normalized_references = [normalize_text(text) if normalize else text for text in references]
    normalized_hypotheses = [normalize_text(text) if normalize else text for text in hypotheses]
    if any(not text for text in normalized_references):
        raise ValueError("references must not be empty after normalization")

    word_output = process_words(normalized_references, normalized_hypotheses)
    character_references = normalized_references
    character_hypotheses = normalized_hypotheses
    if not cer_includes_whitespace:
        character_references = [text.replace(" ", "") for text in character_references]
        character_hypotheses = [text.replace(" ", "") for text in character_hypotheses]
    character_output = process_characters(character_references, character_hypotheses)

    return ErrorMetrics(
        wer=word_output.wer,
        cer=character_output.cer,
        word_substitutions=word_output.substitutions,
        word_deletions=word_output.deletions,
        word_insertions=word_output.insertions,
        reference_words=(word_output.hits + word_output.substitutions + word_output.deletions),
        character_substitutions=character_output.substitutions,
        character_deletions=character_output.deletions,
        character_insertions=character_output.insertions,
        reference_characters=(
            character_output.hits + character_output.substitutions + character_output.deletions
        ),
        utterance_count=len(references),
    )
