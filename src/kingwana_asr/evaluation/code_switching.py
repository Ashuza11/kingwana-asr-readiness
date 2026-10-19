"""Transparent rule-assisted review of corpus code-switch annotations.

The rules in this module encode decisions made with the corpus researcher. They
are annotation aids, not automatic language identification. Original source
labels must always be retained beside reviewed labels.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kingwana_asr.data.manifest import ManifestError


@dataclass(frozen=True)
class CodeSwitchReview:
    """A proposed recording-level label and the evidence supporting it."""

    status: str
    evidence: tuple[str, ...]
    rationale: str


_CLEAR_FRENCH_PHRASES = (
    re.compile(r"\bprix\s+d\s+achat\b"),
    re.compile(r"\bprix\s+de\s+vente\b"),
    re.compile(r"\bdepuis\s+le\b"),
    re.compile(r"\bune\s+semaine\b"),
    re.compile(r"\bdans\s+une\b"),
    re.compile(r"\bil\s+faut\b"),
    re.compile(r"\ble\s+\d+\b"),
)

_FRENCH_SWITCH_WORDS = {
    "ajouter",
    "changer",
    "devoir",
    "enregistre",
    "enregistrer",
    "reste",
    "verifier",
    "vérifier",
}


def _review_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    return re.sub(r"[^\w\s'-]", " ", normalized, flags=re.UNICODE)


def review_code_switching(text: str) -> CodeSwitchReview:
    """Propose a label under the approved borrowing/code-switch distinction.

    Isolated vendor nouns, brands, calendar terms, numbers, and currencies are
    neutral evidence. The researcher-confirmed French switch words are evidence
    of switching; a French verb immediately preceded by the Kingwana infinitival
    prefix ``ku`` is treated as integrated. Multiword French grammatical
    constructions are evidence of switching.
    """

    normalized = _review_text(text)
    tokens = normalized.split()
    evidence: set[str] = set()

    for pattern in _CLEAR_FRENCH_PHRASES:
        if match := pattern.search(normalized):
            evidence.add(match.group())

    for index, token in enumerate(tokens):
        if token in _FRENCH_SWITCH_WORDS and not (index > 0 and tokens[index - 1] == "ku"):
            evidence.add(token)

    if evidence:
        return CodeSwitchReview(
            status="present",
            evidence=tuple(sorted(evidence)),
            rationale="clear French lexical switch or multiword French grammatical construction",
        )
    return CodeSwitchReview(
        status="none",
        evidence=(),
        rationale=(
            "Kingwana frame with no clear switch; isolated vendor vocabulary, brands, "
            "calendar terms, numbers, and currencies are treated as integrated lexical items"
        ),
    )


def apply_reviewed_code_switch_labels(
    manifest_rows: Sequence[Mapping[str, str]],
    review_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Create analysis rows using reviewed labels while retaining originals."""

    manifest_by_id = {row["clip_id"]: row for row in manifest_rows}
    review_by_id = {row["clip_id"]: row for row in review_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ManifestError("Analysis manifest contains duplicate clip IDs")
    if len(review_by_id) != len(review_rows):
        raise ManifestError("Code-switch review contains duplicate clip IDs")
    if set(manifest_by_id) != set(review_by_id):
        raise ManifestError("Analysis manifest and code-switch review clip sets do not match")

    output: list[dict[str, str]] = []
    for source in manifest_rows:
        review = review_by_id[source["clip_id"]]
        status = review.get("code_switch_reviewed", "").strip().lower()
        if status not in {"present", "none"}:
            raise ManifestError(
                f"Clip {source['clip_id']}: invalid reviewed code-switch label {status!r}"
            )
        output.append(
            {
                **source,
                "code_switched_original": source.get("code_switched", ""),
                "code_switch_status_original": source.get("code_switch_status", ""),
                "code_switched": "true" if status == "present" else "false",
                "code_switch_status": status,
                "code_switch_reviewed": status,
                "code_switch_evidence": review.get("code_switch_evidence", ""),
                "code_switch_review_reason": review.get("code_switch_review_reason", ""),
                "code_switch_review_status": review.get("code_switch_review_status", ""),
            }
        )
    return output
