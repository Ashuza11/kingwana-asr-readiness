"""Auditable checks for business-entity gold annotations.

These checks identify records requiring human review. They do not silently
correct source annotations and they do not score ASR predictions.
"""

from __future__ import annotations

import csv
import itertools
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

NON_DEFINITIVE_VALUES = {"ambiguous", "unknown", "unavailable"}
NUMERIC_FIELDS = ("quantity", "amount", "amountPaid", "outstandingDebt")
ENTITY_FIELDS = (
    "transactionIntent",
    "productOrService",
    "quantity",
    "amount",
    "currency",
    "amountPaid",
    "outstandingDebt",
    "paymentMethod",
)
DECISION_STATUSES = {"confirmed", "corrected", "excluded"}
NUMERIC_VALUE_ORIGINS = {
    "explicit",
    "calculated",
    "inferred",
    "corrected",
    "excluded",
    "pending_review",
}


class EntityReviewError(ValueError):
    """Raised when a researcher decision file is malformed or inconsistent."""


@dataclass(frozen=True)
class EntityAudit:
    """Machine-checkable evidence to assist human gold-label validation."""

    reference_numbers: tuple[str, ...]
    field_statuses: dict[str, str]
    flags: tuple[str, ...]


def apply_review_decisions(
    rows: list[dict[str, str]], decisions_path: Path
) -> list[dict[str, str]]:
    """Apply explicit researcher decisions to reviewed fields, never source fields."""

    with decisions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"clip_id", "field", "reviewed_value", "review_status", "notes"}
        if reader.fieldnames is None:
            raise EntityReviewError(f"Decisions file has no header: {decisions_path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise EntityReviewError(
                f"Decisions file is missing columns: {', '.join(sorted(missing))}"
            )
        decisions = list(reader)

    by_id = {row["clip_id"]: dict(row) for row in rows}
    seen: set[tuple[str, str]] = set()
    for decision in decisions:
        clip_id = decision["clip_id"].strip()
        field = decision["field"].strip()
        status = decision["review_status"].strip()
        key = (clip_id, field)
        if clip_id not in by_id:
            raise EntityReviewError(f"Decision refers to unknown clip_id: {clip_id}")
        if field not in ENTITY_FIELDS:
            raise EntityReviewError(f"Decision refers to unsupported field: {field}")
        if status not in DECISION_STATUSES:
            raise EntityReviewError(f"Decision has unsupported review_status: {status}")
        if key in seen:
            raise EntityReviewError(f"Duplicate decision for {clip_id} field {field}")
        seen.add(key)
        row = by_id[clip_id]
        row[f"reviewed_{field}"] = decision["reviewed_value"].strip()
        row[f"review_status_{field}"] = status
        requested_origin = (decision.get("value_origin") or "").strip()
        if field in NUMERIC_FIELDS:
            if requested_origin and requested_origin not in NUMERIC_VALUE_ORIGINS:
                raise EntityReviewError(
                    f"Decision has unsupported numeric value_origin: {requested_origin}"
                )
            row[f"value_origin_{field}"] = requested_origin or (
                "excluded" if status == "excluded" else "corrected"
            )
        elif requested_origin:
            raise EntityReviewError(
                f"Decision supplies value_origin for non-numeric field: {field}"
            )
        note = decision["notes"].strip()
        if note:
            existing = row["review_notes"].strip()
            row["review_notes"] = "; ".join(value for value in (existing, note) if value)
    return sorted(by_id.values(), key=lambda row: row["clip_id"])


def apply_product_mapping(
    rows: list[dict[str, str]], configuration_path: Path
) -> list[dict[str, str]]:
    """Apply researcher-approved product mappings to reviewed worksheet columns."""

    with configuration_path.open(encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    try:
        mapping = configuration["product_or_service"]["source_to_canonical"]
    except (KeyError, TypeError) as exc:
        raise EntityReviewError("Entity configuration has no product source mapping") from exc
    if not isinstance(mapping, dict):
        raise EntityReviewError("Product source mapping must be an object")

    output: list[dict[str, str]] = []
    observed: set[str] = set()
    for source_row in rows:
        row = dict(source_row)
        source = row["source_productOrService"].strip()
        if not source:
            row["reviewed_productOrService"] = ""
            row["review_status_productOrService"] = "excluded"
            output.append(row)
            continue
        observed.add(source)
        if source not in mapping:
            raise EntityReviewError(f"No approved product mapping for source value: {source}")
        canonical = mapping[source]
        if canonical is None:
            row["reviewed_productOrService"] = ""
            row["review_status_productOrService"] = "excluded"
        else:
            if not isinstance(canonical, list) or not canonical:
                raise EntityReviewError(f"Invalid product mapping for source value: {source}")
            values = [str(value).strip() for value in canonical]
            if any(not value for value in values) or len(values) != len(set(values)):
                raise EntityReviewError(f"Invalid canonical product set for source value: {source}")
            reviewed = "|".join(values)
            row["reviewed_productOrService"] = reviewed
            row["review_status_productOrService"] = (
                "confirmed" if reviewed == source else "corrected"
            )
        output.append(row)

    unused = set(mapping) - observed
    if unused:
        raise EntityReviewError(f"Configured product mappings were not observed: {sorted(unused)}")
    return output


def detect_currency_values(reference: str) -> tuple[str, ...]:
    """Return currencies explicitly expressed in a verbatim transcript."""

    lowered = unicodedata.normalize("NFC", reference).lower()
    values: list[str] = []
    if re.search(r"\b(?:cdf|fc)\b|francs?\s+congolais", lowered):
        values.append("CDF")
    if "$" in lowered or re.search(r"\b(?:usd|dollars?)\b", lowered):
        values.append("USD")
    return tuple(values)


def apply_currency_policy(
    rows: list[dict[str, str]], configuration_path: Path
) -> list[dict[str, str]]:
    """Apply the approved explicit-currency-only policy to review columns."""

    with configuration_path.open(encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    try:
        policy = configuration["currency"]["policy"]
    except (KeyError, TypeError) as exc:
        raise EntityReviewError("Entity configuration has no currency policy") from exc
    if policy != "explicitly_spoken_only":
        raise EntityReviewError(f"Unsupported currency policy: {policy}")

    output: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        source = row["source_currency"].strip()
        if not source:
            row["reviewed_currency"] = ""
            row["review_status_currency"] = "excluded"
            output.append(row)
            continue
        detected = detect_currency_values(row["reference_verbatim"])
        if len(detected) > 1:
            raise EntityReviewError(
                f"Clip {row['clip_id']} contains multiple explicit currencies and needs review"
            )
        if not detected:
            row["reviewed_currency"] = ""
            row["review_status_currency"] = "excluded"
        else:
            reviewed = detected[0]
            row["reviewed_currency"] = reviewed
            row["review_status_currency"] = "confirmed" if reviewed == source else "corrected"
        output.append(row)
    return output


def apply_transaction_intent_policy(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Confirm source intents that agree with the elicitation scenario."""

    output: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        if row["audit_transactionIntent"] == "match":
            row["reviewed_transactionIntent"] = row["source_transactionIntent"]
            row["review_status_transactionIntent"] = "confirmed"
        output.append(row)
    return output


def finalize_entity_review(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Mark rows complete only when every entity field has a final decision."""

    output: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        statuses = [row[f"review_status_{field}"] for field in ENTITY_FIELDS]
        row["gold_review_status"] = (
            "complete" if all(status in DECISION_STATUSES for status in statuses) else "pending"
        )
        output.append(row)
    return output


def apply_payment_method_policy(
    rows: list[dict[str, str]], configuration_path: Path
) -> list[dict[str, str]]:
    """Confirm only payment methods explicitly expressed in the transcript."""

    with configuration_path.open(encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    try:
        policy = configuration["payment_method"]
        patterns = policy["explicit_patterns"]
        excluded_values = set(policy["exclude_source_values"])
    except (KeyError, TypeError) as exc:
        raise EntityReviewError("Entity configuration has no payment-method policy") from exc
    if policy.get("policy") != "explicitly_spoken_only":
        raise EntityReviewError("Payment-method policy must be explicitly_spoken_only")
    if not isinstance(patterns, dict) or not all(
        isinstance(pattern, str) for pattern in patterns.values()
    ):
        raise EntityReviewError("Payment-method patterns must map labels to regex strings")

    output: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        source = row["source_paymentMethod"].strip()
        if not source or source in excluded_values:
            row["reviewed_paymentMethod"] = ""
            row["review_status_paymentMethod"] = "excluded"
        elif source in patterns and re.search(
            patterns[source], row["reference_verbatim"], re.IGNORECASE
        ):
            row["reviewed_paymentMethod"] = source
            row["review_status_paymentMethod"] = "confirmed"
        output.append(row)
    return output


def _spoken_number_values(reference: str, aliases: dict[str, list[str]]) -> set[str]:
    """Return configured number values expressed as standalone words."""

    normalized = _normalized_text(reference)
    values: set[str] = set()
    for value, forms in aliases.items():
        for form in forms:
            normalized_form = _normalized_text(form)
            if re.search(rf"(?<!\w){re.escape(normalized_form)}(?!\w)", normalized):
                values.add(_canonical_number(str(value)))
                break
    return values


def classify_numeric_value_origin(
    *,
    field: str,
    value: Any,
    reference: str,
    number_word_aliases: dict[str, list[str]],
) -> str:
    """Classify numeric gold provenance without changing the gold value.

    Values not supported by deterministic evidence remain pending for researcher
    review. A calculated label means that a single gold value is the sum of at
    least two numbers expressed in the transcript; it does not claim that an
    arbitrary downstream system emitted that calculation.
    """

    if value is None or not str(value).strip():
        return "excluded"
    if _normalized_text(value) in NON_DEFINITIVE_VALUES:
        return "excluded"

    # Quantity annotation commas or semicolons delimit multiple values (for
    # example, ``750,250``); transcript commas may instead group thousands.
    expected = (
        tuple(
            _canonical_number(item.strip())
            for item in re.split(r"[,;]", str(value))
            if item.strip()
        )
        if field == "quantity"
        else extract_numeric_values(str(value))
    )
    if not expected:
        return "pending_review"

    reference_numbers = extract_numeric_values(reference)
    spoken_values = set(reference_numbers) | _spoken_number_values(reference, number_word_aliases)
    if all(item in spoken_values for item in expected):
        return "explicit"

    # Zero is a structured outcome unless the speaker actually says zero.
    if expected == ("0",):
        return "inferred"

    if len(expected) == 1:
        try:
            target = Decimal(expected[0])
            candidates = [Decimal(item) for item in reference_numbers]
        except InvalidOperation:
            return "pending_review"
        for size in range(2, len(candidates) + 1):
            if any(sum(items) == target for items in itertools.combinations(candidates, size)):
                return "calculated"

    return "pending_review"


def apply_numeric_provenance_policy(
    rows: list[dict[str, str]], configuration_path: Path
) -> list[dict[str, str]]:
    """Apply the approved numeric provenance policy to worksheet columns."""

    with configuration_path.open(encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    try:
        policy = configuration["numeric_fields"]
        aliases = policy["number_word_aliases"]
    except (KeyError, TypeError) as exc:
        raise EntityReviewError("Entity configuration has no numeric-fields policy") from exc
    if policy.get("asr_entity_scoring") != "explicit_only":
        raise EntityReviewError("Numeric ASR entity scoring must use explicit values only")
    if not isinstance(aliases, dict) or not all(
        isinstance(forms, list) for forms in aliases.values()
    ):
        raise EntityReviewError("Numeric number-word aliases must map values to lists")

    output: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        for field in NUMERIC_FIELDS:
            origin = classify_numeric_value_origin(
                field=field,
                value=row[f"source_{field}"],
                reference=row["reference_verbatim"],
                number_word_aliases=aliases,
            )
            if field == "quantity" and origin == "pending_review":
                product_count = len(
                    [value for value in row["reviewed_productOrService"].split("|") if value]
                )
                try:
                    quantity = int(row["source_quantity"])
                except ValueError:
                    quantity = -1
                if product_count and quantity == product_count:
                    origin = "inferred"
            row[f"value_origin_{field}"] = origin
            if origin in {"explicit", "calculated", "inferred"}:
                row[f"review_status_{field}"] = "confirmed"
            elif origin == "excluded":
                row[f"reviewed_{field}"] = ""
                row[f"review_status_{field}"] = "excluded"
        output.append(row)
    return output


def _canonical_number(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if number == number.to_integral():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def extract_numeric_values(text: str) -> tuple[str, ...]:
    """Extract canonical numeric values while distinguishing lists from separators."""

    values: list[str] = []
    for candidate in re.findall(r"\d+(?:[.,]\d+)*", text):
        if "," in candidate:
            groups = candidate.split(",")
            thousands = len(groups[0]) <= 3 and all(len(group) == 3 for group in groups[1:])
            candidates = ["".join(groups)] if thousands else groups
        else:
            candidates = [candidate]
        for item in candidates:
            if item:
                values.append(_canonical_number(item))
    return tuple(values)


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split())


def _numeric_status(value: Any, reference_numbers: tuple[str, ...]) -> str:
    if value is None or not str(value).strip():
        return "missing"
    if _normalized_text(value) in NON_DEFINITIVE_VALUES:
        return "non_definitive"
    expected = extract_numeric_values(str(value))
    if not expected:
        return "invalid"
    present = sum(item in reference_numbers for item in expected)
    if present == len(expected):
        return "match"
    if present:
        return "partial_match"
    return "not_found"


def _currency_status(value: Any, reference: str) -> str:
    if value is None or not str(value).strip():
        return "missing"
    canonical = _normalized_text(value)
    if canonical in NON_DEFINITIVE_VALUES:
        return "non_definitive"
    supported = {"cdf": "CDF", "usd": "USD"}
    if canonical not in supported:
        return "unsupported_value"
    return "match" if supported[canonical] in detect_currency_values(reference) else "not_found"


def _intent_status(value: Any, expected_scenario: str) -> str:
    if value is None or not str(value).strip():
        return "missing"
    return "match" if str(value).strip() == expected_scenario else "mismatch"


def _product_status(value: Any) -> str:
    if value is None or not str(value).strip():
        return "missing"
    if _normalized_text(value) in NON_DEFINITIVE_VALUES:
        return "non_definitive"
    return "requires_alias_review"


def _payment_status(value: Any, reference: str) -> str:
    if value is None or not str(value).strip():
        return "missing"
    canonical = _normalized_text(value)
    if canonical in NON_DEFINITIVE_VALUES:
        return "non_definitive"
    if canonical == "cash":
        return "match" if re.search(r"\bcash\b", reference, re.IGNORECASE) else "not_found"
    return "requires_alias_review"


def audit_entity_payload(
    *, reference_verbatim: str, expected_scenario: str, payload: dict[str, Any]
) -> EntityAudit:
    """Audit one annotation payload against its transcript and scenario."""

    reference_numbers = extract_numeric_values(reference_verbatim)
    statuses = {
        "transactionIntent": _intent_status(payload.get("transactionIntent"), expected_scenario),
        "productOrService": _product_status(payload.get("productOrService")),
        "currency": _currency_status(payload.get("currency"), reference_verbatim),
        "paymentMethod": _payment_status(payload.get("paymentMethod"), reference_verbatim),
    }
    statuses.update(
        {field: _numeric_status(payload.get(field), reference_numbers) for field in NUMERIC_FIELDS}
    )

    benign = {"match", "missing"}
    flags = tuple(
        f"{field}:{status}"
        for field, status in statuses.items()
        if status not in benign or (field == "transactionIntent" and status == "missing")
    )
    return EntityAudit(
        reference_numbers=reference_numbers,
        field_statuses=statuses,
        flags=flags,
    )
