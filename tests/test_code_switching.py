import pytest

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.code_switching import (
    apply_reviewed_code_switch_labels,
    review_code_switching,
)


def test_isolated_calendar_word_is_not_code_switching() -> None:
    result = review_code_switching("Atalipa samedi.")
    assert result.status == "none"


def test_calendar_word_does_not_hide_a_french_phrase() -> None:
    result = review_code_switching("Deni depuis le 30 septembre.")
    assert result.status == "present"
    assert "depuis le" in result.evidence


def test_vendor_nouns_in_kingwana_frame_are_not_switches() -> None:
    result = review_code_switching("Unités ngapi zili lala mu stock jana ku réseaux zote?")
    assert result.status == "none"


def test_bare_french_command_is_code_switching() -> None:
    result = review_code_switching("Changer prix d'achat Orange ku 22.5.")
    assert result.status == "present"
    assert "changer" in result.evidence


def test_ku_prefixed_french_verb_is_integrated() -> None:
    result = review_code_switching("Nataka ku verifier stock.")
    assert result.status == "none"


def test_reste_is_code_switching() -> None:
    result = review_code_switching("Atalipa reste mardi.")
    assert result.status == "present"
    assert "reste" in result.evidence


def test_reviewed_labels_preserve_original_analysis_values() -> None:
    manifest = [
        {
            "clip_id": "clip-1",
            "reference_verbatim": "text",
            "audio_path": "audio.wav",
            "code_switched": "ambiguous",
            "code_switch_status": "ambiguous",
        }
    ]
    review = [
        {
            "clip_id": "clip-1",
            "code_switch_reviewed": "none",
            "code_switch_evidence": "",
            "code_switch_review_reason": "integrated vocabulary",
            "code_switch_review_status": "rule_assigned",
        }
    ]

    rows = apply_reviewed_code_switch_labels(manifest, review)

    assert rows[0]["reference_verbatim"] == "text"
    assert rows[0]["audio_path"] == "audio.wav"
    assert rows[0]["code_switched_original"] == "ambiguous"
    assert rows[0]["code_switch_status_original"] == "ambiguous"
    assert rows[0]["code_switched"] == "false"
    assert rows[0]["code_switch_status"] == "none"


def test_reviewed_labels_require_exact_clip_alignment() -> None:
    with pytest.raises(ManifestError, match="clip sets do not match"):
        apply_reviewed_code_switch_labels(
            [{"clip_id": "clip-1"}],
            [{"clip_id": "clip-2", "code_switch_reviewed": "present"}],
        )
