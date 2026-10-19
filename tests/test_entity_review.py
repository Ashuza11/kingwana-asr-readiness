import csv
from pathlib import Path

import pytest

from kingwana_asr.evaluation.entity_validation import (
    ENTITY_FIELDS,
    EntityReviewError,
    apply_currency_policy,
    apply_numeric_provenance_policy,
    apply_payment_method_policy,
    apply_product_mapping,
    apply_review_decisions,
    apply_transaction_intent_policy,
    finalize_entity_review,
)


def _write_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "clip_id",
                "field",
                "reviewed_value",
                "review_status",
                "notes",
                "value_origin",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_apply_decision_changes_only_review_columns(tmp_path: Path) -> None:
    rows = [
        {
            "clip_id": "clip-1",
            "source_transactionIntent": "wrong",
            "reviewed_transactionIntent": "wrong",
            "review_status_transactionIntent": "pending",
            "review_notes": "",
        }
    ]
    decisions = tmp_path / "decisions.csv"
    _write_decisions(
        decisions,
        [
            {
                "clip_id": "clip-1",
                "field": "transactionIntent",
                "reviewed_value": "record_sale",
                "review_status": "corrected",
                "notes": "researcher confirmed",
            }
        ],
    )

    reviewed = apply_review_decisions(rows, decisions)

    assert reviewed[0]["source_transactionIntent"] == "wrong"
    assert reviewed[0]["reviewed_transactionIntent"] == "record_sale"
    assert reviewed[0]["review_status_transactionIntent"] == "corrected"
    assert reviewed[0]["review_notes"] == "researcher confirmed"


def test_duplicate_decision_fails_loudly(tmp_path: Path) -> None:
    rows = [
        {
            "clip_id": "clip-1",
            "reviewed_transactionIntent": "wrong",
            "review_status_transactionIntent": "pending",
            "review_notes": "",
        }
    ]
    decisions = tmp_path / "decisions.csv"
    decision = {
        "clip_id": "clip-1",
        "field": "transactionIntent",
        "reviewed_value": "record_sale",
        "review_status": "corrected",
        "notes": "",
    }
    _write_decisions(decisions, [decision, decision])

    with pytest.raises(EntityReviewError, match="Duplicate decision"):
        apply_review_decisions(rows, decisions)


def test_numeric_decision_can_record_calculated_origin(tmp_path: Path) -> None:
    rows = [
        {
            "clip_id": "clip-1",
            "reviewed_amount": "6500",
            "review_status_amount": "pending",
            "value_origin_amount": "pending_review",
            "review_notes": "",
        }
    ]
    decisions = tmp_path / "decisions.csv"
    _write_decisions(
        decisions,
        [
            {
                "clip_id": "clip-1",
                "field": "amount",
                "reviewed_value": "7500",
                "review_status": "corrected",
                "notes": "confirmed sum",
                "value_origin": "calculated",
            }
        ],
    )

    reviewed = apply_review_decisions(rows, decisions)[0]

    assert reviewed["reviewed_amount"] == "7500"
    assert reviewed["review_status_amount"] == "corrected"
    assert reviewed["value_origin_amount"] == "calculated"


def test_product_mapping_preserves_source_and_excludes_not_applicable(tmp_path: Path) -> None:
    configuration = tmp_path / "entities.yaml"
    configuration.write_text(
        "product_or_service:\n"
        "  source_to_canonical:\n"
        "    'Unités téléphonique': [airtime]\n"
        "    unavailable: null\n",
        encoding="utf-8",
    )
    rows = [
        {
            "source_productOrService": "Unités téléphonique",
            "reviewed_productOrService": "Unités téléphonique",
            "review_status_productOrService": "pending",
        },
        {
            "source_productOrService": "unavailable",
            "reviewed_productOrService": "unavailable",
            "review_status_productOrService": "pending",
        },
    ]

    reviewed = apply_product_mapping(rows, configuration)

    assert reviewed[0]["source_productOrService"] == "Unités téléphonique"
    assert reviewed[0]["reviewed_productOrService"] == "airtime"
    assert reviewed[0]["review_status_productOrService"] == "corrected"
    assert reviewed[1]["reviewed_productOrService"] == ""
    assert reviewed[1]["review_status_productOrService"] == "excluded"


def test_matching_intent_is_confirmed_and_complete_rows_are_finalized() -> None:
    row = {
        "source_transactionIntent": "record_sale",
        "reviewed_transactionIntent": "record_sale",
        "audit_transactionIntent": "match",
    }
    for field in ENTITY_FIELDS:
        row.setdefault(f"review_status_{field}", "excluded")

    reviewed = apply_transaction_intent_policy([row])
    finalized = finalize_entity_review(reviewed)

    assert finalized[0]["review_status_transactionIntent"] == "confirmed"
    assert finalized[0]["gold_review_status"] == "complete"


def test_currency_policy_corrects_explicit_currency_and_excludes_unspoken(
    tmp_path: Path,
) -> None:
    configuration = tmp_path / "entities.yaml"
    configuration.write_text(
        "currency:\n  policy: explicitly_spoken_only\n",
        encoding="utf-8",
    )
    rows = [
        {
            "clip_id": "usd-clip",
            "reference_verbatim": "Alibeba airpod ya 10$.",
            "source_currency": "CDF",
            "reviewed_currency": "CDF",
            "review_status_currency": "pending",
        },
        {
            "clip_id": "implicit-clip",
            "reference_verbatim": "Alilipa elfu tano.",
            "source_currency": "CDF",
            "reviewed_currency": "CDF",
            "review_status_currency": "pending",
        },
    ]

    reviewed = apply_currency_policy(rows, configuration)

    assert reviewed[0]["reviewed_currency"] == "USD"
    assert reviewed[0]["review_status_currency"] == "corrected"
    assert reviewed[1]["reviewed_currency"] == ""
    assert reviewed[1]["review_status_currency"] == "excluded"


def test_numeric_policy_preserves_values_and_marks_origins(tmp_path: Path) -> None:
    configuration = tmp_path / "entities.yaml"
    configuration.write_text(
        "numeric_fields:\n"
        "  asr_entity_scoring: explicit_only\n"
        "  number_word_aliases:\n"
        "    '2': [mbili]\n",
        encoding="utf-8",
    )
    rows = [
        {
            "reference_verbatim": "USB mbili za 3000 FC na chargeur ya 5000 FC",
            "reviewed_productOrService": "usb",
            "source_quantity": "2",
            "source_amount": "8000",
            "source_amountPaid": "0",
            "source_outstandingDebt": "unknown",
            "reviewed_quantity": "2",
            "reviewed_amount": "8000",
            "reviewed_amountPaid": "0",
            "reviewed_outstandingDebt": "unknown",
            "review_status_quantity": "pending",
            "review_status_amount": "pending",
            "review_status_amountPaid": "pending",
            "review_status_outstandingDebt": "pending",
        }
    ]

    reviewed = apply_numeric_provenance_policy(rows, configuration)[0]

    assert reviewed["value_origin_quantity"] == "explicit"
    assert reviewed["value_origin_amount"] == "calculated"
    assert reviewed["value_origin_amountPaid"] == "inferred"
    assert reviewed["value_origin_outstandingDebt"] == "excluded"
    assert reviewed["source_outstandingDebt"] == "unknown"
    assert reviewed["reviewed_outstandingDebt"] == ""


def test_numeric_policy_marks_product_count_quantity_as_inferred(tmp_path: Path) -> None:
    configuration = tmp_path / "entities.yaml"
    configuration.write_text(
        "numeric_fields:\n  asr_entity_scoring: explicit_only\n  number_word_aliases: {}\n",
        encoding="utf-8",
    )
    rows = [
        {
            "reference_verbatim": "Aliuza screen na pochette ku 15000 FC",
            "reviewed_productOrService": "phone_screen|phone_case",
            "source_quantity": "2",
            "source_amount": "15000",
            "source_amountPaid": "",
            "source_outstandingDebt": "",
            "reviewed_quantity": "2",
            "reviewed_amount": "15000",
            "reviewed_amountPaid": "",
            "reviewed_outstandingDebt": "",
            "review_status_quantity": "pending",
            "review_status_amount": "pending",
            "review_status_amountPaid": "pending",
            "review_status_outstandingDebt": "pending",
        }
    ]

    reviewed = apply_numeric_provenance_policy(rows, configuration)[0]

    assert reviewed["value_origin_quantity"] == "inferred"
    assert reviewed["review_status_quantity"] == "confirmed"


def test_payment_policy_requires_explicit_method_wording(tmp_path: Path) -> None:
    configuration = tmp_path / "entities.yaml"
    configuration.write_text(
        "payment_method:\n"
        "  policy: explicitly_spoken_only\n"
        "  exclude_source_values: [unknown, unavailable, ambiguous]\n"
        "  explicit_patterns:\n"
        "    cash: '\\bcash\\b'\n"
        "    mobile_money: '\\b(?:airtel|orange)\\s+money\\b|\\bvirtuel(?:le)?\\b'\n",
        encoding="utf-8",
    )
    rows = [
        {
            "source_paymentMethod": "cash",
            "reviewed_paymentMethod": "cash",
            "review_status_paymentMethod": "pending",
            "reference_verbatim": "Ali lipa cash.",
        },
        {
            "source_paymentMethod": "mobile_money",
            "reviewed_paymentMethod": "mobile_money",
            "review_status_paymentMethod": "pending",
            "reference_verbatim": "Ali tuma virtuelle ku Orange money.",
        },
        {
            "source_paymentMethod": "mobile_money",
            "reviewed_paymentMethod": "mobile_money",
            "review_status_paymentMethod": "pending",
            "reference_verbatim": "Amani anatuma 17000fc.",
        },
        {
            "source_paymentMethod": "unknown",
            "reviewed_paymentMethod": "unknown",
            "review_status_paymentMethod": "pending",
            "reference_verbatim": "Alilipa yote.",
        },
    ]

    reviewed = apply_payment_method_policy(rows, configuration)

    assert reviewed[0]["review_status_paymentMethod"] == "confirmed"
    assert reviewed[1]["review_status_paymentMethod"] == "confirmed"
    assert reviewed[2]["review_status_paymentMethod"] == "pending"
    assert reviewed[3]["review_status_paymentMethod"] == "excluded"
    assert reviewed[3]["reviewed_paymentMethod"] == ""
