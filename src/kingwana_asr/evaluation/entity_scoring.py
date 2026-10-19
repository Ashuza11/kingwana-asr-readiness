"""Strict, reproducible preservation scoring for validated business entities."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.entity_validation import extract_numeric_values
from kingwana_asr.evaluation.normalization import normalize_text

FINAL_REVIEW_STATUSES = {"confirmed", "corrected", "excluded"}
MONETARY_FIELDS = ("amount", "amountPaid", "outstandingDebt")


@dataclass(frozen=True)
class EntityScore:
    """Mention-level preservation and all-mentions-correct clip accuracy."""

    model_id: str
    category: str
    correct_mentions: int
    gold_mentions: int
    exact_clips: int
    eligible_clips: int

    @property
    def mention_accuracy(self) -> float:
        return self.correct_mentions / self.gold_mentions

    @property
    def clip_accuracy(self) -> float:
        return self.exact_clips / self.eligible_clips


_DIGIT_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}(?:[ ,.\u202f]\d{3})+|\d+(?:[.,]\d+)?)(?!\d)")

_FR_SMALL = {
    "zéro": 0,
    "zero": 0,
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    "vingt": 20,
    "trente": 30,
    "quarante": 40,
    "cinquante": 50,
    "soixante": 60,
}
_EN_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SW_SMALL = {
    "sifuri": 0,
    "moja": 1,
    "moya": 1,
    "mbili": 2,
    "tatu": 3,
    "nne": 4,
    "ine": 4,
    "tano": 5,
    "sita": 6,
    "saba": 7,
    "nane": 8,
    "tisa": 9,
    "kumi": 10,
    "ishirini": 20,
    "thelathini": 30,
    "arobaini": 40,
    "hamsini": 50,
    "sitini": 60,
    "sabini": 70,
    "themanini": 80,
    "tisini": 90,
}


def _canonical_digit(candidate: str) -> str:
    compact = candidate.replace(" ", "").replace("\u202f", "")
    groups = re.split(r"[,.]", compact)
    if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]):
        return "".join(groups)
    return compact.replace(",", ".")


def _parse_western_number(tokens: Sequence[str], *, language: str) -> int | None:
    small = _FR_SMALL if language == "fr" else _EN_SMALL
    hundred = "cent" if language == "fr" else "hundred"
    thousand = "mille" if language == "fr" else "thousand"
    millions = {"million", "millions"}
    connectors = {"et"} if language == "fr" else {"and"}
    total = 0
    current = 0
    consumed = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in connectors or (language == "fr" and token == "cents"):
            index += 1
            continue
        if language == "fr" and token == "quatre" and index + 1 < len(tokens):
            if tokens[index + 1] in {"vingt", "vingts"}:
                current += 80
                consumed = True
                index += 2
                continue
        if language == "fr" and token == "dix" and index + 1 < len(tokens):
            teen = small.get(tokens[index + 1])
            if teen is not None and 7 <= teen <= 9:
                current += 10 + teen
                consumed = True
                index += 2
                continue
        if token in small:
            current += small[token]
        elif token in {hundred, "cents"}:
            current = max(current, 1) * 100
        elif token == thousand:
            total += max(current, 1) * 1_000
            current = 0
        elif token in millions:
            total += max(current, 1) * 1_000_000
            current = 0
        else:
            return None
        consumed = True
        index += 1
    return total + current if consumed else None


def _parse_swahili_number(tokens: Sequence[str]) -> int | None:
    total = 0
    current = 0
    consumed = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "na":
            index += 1
            continue
        if token in {"elfu", "mia"}:
            scale = 1_000 if token == "elfu" else 100
            multiplier = 1
            if index + 1 < len(tokens) and tokens[index + 1] in _SW_SMALL:
                multiplier = _SW_SMALL[tokens[index + 1]]
                index += 1
            total += scale * multiplier
        elif token in {"milioni", "milion"}:
            multiplier = 1
            if index + 1 < len(tokens) and tokens[index + 1] in _SW_SMALL:
                multiplier = _SW_SMALL[tokens[index + 1]]
                index += 1
            total += 1_000_000 * multiplier
        elif token in _SW_SMALL:
            current += _SW_SMALL[token]
        else:
            return None
        consumed = True
        index += 1
    return total + current if consumed else None


def _word_number_values(text: str) -> set[str]:
    tokens = normalize_text(text.replace("-", " ")).split()
    language_vocabulary = {
        "fr": set(_FR_SMALL) | {"et", "cent", "cents", "mille", "million", "millions", "vingts"},
        "en": set(_EN_SMALL) | {"and", "hundred", "thousand", "million", "millions"},
        "sw": set(_SW_SMALL) | {"na", "mia", "elfu", "milioni", "milion"},
    }
    values: set[str] = set()
    for language, vocabulary in language_vocabulary.items():
        index = 0
        while index < len(tokens):
            if tokens[index] not in vocabulary:
                index += 1
                continue
            end = index
            while end < len(tokens) and tokens[end] in vocabulary:
                end += 1
            phrase = tokens[index:end]
            value = (
                _parse_swahili_number(phrase)
                if language == "sw"
                else _parse_western_number(phrase, language=language)
            )
            if value is not None:
                values.add(str(value))
            index = end
    return values


def extract_scoring_numbers(text: str) -> set[str]:
    """Extract grouped digits and well-formed French, English, or Swahili numbers."""

    values = {_canonical_digit(match.group()) for match in _DIGIT_PATTERN.finditer(text)}
    values.update(_word_number_values(text))
    return values


def _contains_alias(text: str, alias: str) -> bool:
    normalized = normalize_text(text)
    normalized_alias = normalize_text(alias)
    if not normalized_alias:
        return alias in text
    return re.search(rf"(?<!\w){re.escape(normalized_alias)}(?!\w)", normalized) is not None


def canonical_entities(text: str, aliases: Mapping[str, Sequence[str]]) -> set[str]:
    """Return canonical labels whose configured aliases occur exactly in text."""

    return {
        canonical
        for canonical, forms in aliases.items()
        if any(_contains_alias(text, form) for form in forms)
    }


def load_scoring_aliases(path: Path) -> dict[str, dict[str, list[str]]]:
    with path.open(encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    try:
        scoring = configuration["entity_scoring"]
    except (KeyError, TypeError) as exc:
        raise ManifestError("Entity configuration has no entity_scoring section") from exc
    required = {"currency", "product_service", "operator", "payment_method"}
    if required - set(scoring):
        raise ManifestError("Entity scoring aliases are incomplete")
    return scoring


def gold_entities(
    row: Mapping[str, str], aliases: Mapping[str, Mapping[str, Sequence[str]]]
) -> dict[str, set[str]]:
    """Build explicitly spoken gold entity sets for one reviewed clip."""

    if row["gold_review_status"] != "complete":
        raise ManifestError(f"Clip {row['clip_id']}: entity gold review is incomplete")
    reference = row["reference_verbatim"]
    reference_numbers = extract_scoring_numbers(reference)
    monetary: set[str] = set()
    for field in MONETARY_FIELDS:
        if row[f"review_status_{field}"] not in FINAL_REVIEW_STATUSES - {"excluded"}:
            continue
        for value in extract_numeric_values(row[f"reviewed_{field}"]):
            if value in reference_numbers:
                monetary.add(value)

    currency = set()
    if row["review_status_currency"] in {"confirmed", "corrected"}:
        value = row["reviewed_currency"]
        if value:
            currency.add(value)

    reference_products = canonical_entities(reference, aliases["product_service"])
    products = set()
    if row["review_status_productOrService"] in {"confirmed", "corrected"}:
        products = {
            value
            for value in row["reviewed_productOrService"].split("|")
            if value and value in reference_products
        }

    payment = set()
    if row["review_status_paymentMethod"] == "confirmed":
        value = row["reviewed_paymentMethod"]
        if value:
            payment.add(value)

    return {
        "monetary_amount": monetary,
        "currency": currency,
        "product_service": products,
        "operator": canonical_entities(reference, aliases["operator"]),
        "payment_method": payment,
    }


def predicted_entities(
    prediction: str, aliases: Mapping[str, Mapping[str, Sequence[str]]]
) -> dict[str, set[str]]:
    return {
        "monetary_amount": extract_scoring_numbers(prediction),
        "currency": canonical_entities(prediction, aliases["currency"]),
        "product_service": canonical_entities(prediction, aliases["product_service"]),
        "operator": canonical_entities(prediction, aliases["operator"]),
        "payment_method": canonical_entities(prediction, aliases["payment_method"]),
    }


def score_predictions(
    review_rows: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, str]],
    aliases: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[EntityScore]:
    """Score one model after strict ID and success-status validation."""

    review_by_id = {row["clip_id"]: row for row in review_rows}
    prediction_by_id = {row["clip_id"]: row for row in prediction_rows}
    if len(review_by_id) != len(review_rows) or len(prediction_by_id) != len(prediction_rows):
        raise ManifestError("Entity scoring inputs contain duplicate clip IDs")
    if set(review_by_id) != set(prediction_by_id):
        raise ManifestError("Entity review and prediction clip IDs do not align")
    model_ids = {row["model_id"] for row in prediction_rows}
    if len(model_ids) != 1:
        raise ManifestError("Prediction file must contain exactly one model_id")
    if any(row["status"] != "success" for row in prediction_rows):
        raise ManifestError("Entity scoring requires successful predictions for every clip")

    model_id = next(iter(model_ids))
    counts = {
        category: {"correct": 0, "gold": 0, "exact": 0, "clips": 0}
        for category in (
            "monetary_amount",
            "currency",
            "product_service",
            "operator",
            "payment_method",
        )
    }
    for clip_id, review in review_by_id.items():
        gold = gold_entities(review, aliases)
        predicted = predicted_entities(prediction_by_id[clip_id]["prediction"], aliases)
        for category, gold_values in gold.items():
            if not gold_values:
                continue
            correct = len(gold_values & predicted[category])
            counts[category]["correct"] += correct
            counts[category]["gold"] += len(gold_values)
            counts[category]["exact"] += correct == len(gold_values)
            counts[category]["clips"] += 1

    return [
        EntityScore(
            model_id=model_id,
            category=category,
            correct_mentions=value["correct"],
            gold_mentions=value["gold"],
            exact_clips=value["exact"],
            eligible_clips=value["clips"],
        )
        for category, value in counts.items()
        if value["gold"]
    ]
