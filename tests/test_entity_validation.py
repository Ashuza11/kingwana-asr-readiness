from kingwana_asr.evaluation.entity_validation import (
    audit_entity_payload,
    classify_numeric_value_origin,
    extract_numeric_values,
)


def test_extract_numeric_values_handles_thousands_decimals_and_lists() -> None:
    assert extract_numeric_values("25,000 FC na 22.5; kiasi 5,2") == (
        "25000",
        "22.5",
        "5",
        "2",
    )


def test_audit_matches_canonical_numbers_and_currency_aliases() -> None:
    result = audit_entity_payload(
        reference_verbatim="Alilipa 5,000 FC na akabaki na 2$.",
        expected_scenario="record_debt_repayment",
        payload={
            "transactionIntent": "record_debt_repayment",
            "amountPaid": "5000",
            "outstandingDebt": "2",
            "currency": "CDF",
        },
    )
    assert result.field_statuses["amountPaid"] == "match"
    assert result.field_statuses["outstandingDebt"] == "match"
    assert result.field_statuses["currency"] == "match"


def test_audit_flags_intent_and_numeric_mismatches() -> None:
    result = audit_entity_payload(
        reference_verbatim="Aliuza chargeur ku 5000 FC.",
        expected_scenario="record_sale",
        payload={"transactionIntent": "record_customer_debt", "amount": "50000"},
    )
    assert result.field_statuses["transactionIntent"] == "mismatch"
    assert result.field_statuses["amount"] == "not_found"
    assert "transactionIntent:mismatch" in result.flags
    assert "amount:not_found" in result.flags


def test_missing_required_intent_is_flagged() -> None:
    result = audit_entity_payload(
        reference_verbatim="Onyesha deni zote.",
        expected_scenario="query_business_information",
        payload={},
    )
    assert result.field_statuses["transactionIntent"] == "missing"
    assert "transactionIntent:missing" in result.flags


def test_non_definitive_values_are_not_treated_as_gold() -> None:
    result = audit_entity_payload(
        reference_verbatim="Alilipa deni.",
        expected_scenario="record_debt_repayment",
        payload={"paymentMethod": "unknown", "currency": "ambiguous"},
    )
    assert result.field_statuses["paymentMethod"] == "non_definitive"
    assert result.field_statuses["currency"] == "non_definitive"


def test_product_requires_alias_review_instead_of_literal_assumption() -> None:
    result = audit_entity_payload(
        reference_verbatim="Aliuza chargeur moya.",
        expected_scenario="record_sale",
        payload={"productOrService": "phone_charger"},
    )
    assert result.field_statuses["productOrService"] == "requires_alias_review"


def test_numeric_origin_distinguishes_explicit_calculated_and_inferred() -> None:
    aliases = {"1": ["moya", "moja"], "2": ["mbili"]}

    assert (
        classify_numeric_value_origin(
            field="quantity",
            value="2",
            reference="USB mbili",
            number_word_aliases=aliases,
        )
        == "explicit"
    )
    assert (
        classify_numeric_value_origin(
            field="amount",
            value="6500",
            reference="sim card ku 1500 FC na batterie ku 5000 FC",
            number_word_aliases=aliases,
        )
        == "calculated"
    )
    assert (
        classify_numeric_value_origin(
            field="amountPaid",
            value="0",
            reference="atalipa mardi",
            number_word_aliases=aliases,
        )
        == "inferred"
    )
    assert (
        classify_numeric_value_origin(
            field="amount",
            value="67000",
            reference="ku 67500 FC",
            number_word_aliases=aliases,
        )
        == "pending_review"
    )


def test_quantity_commas_are_lists_not_thousands_separators() -> None:
    assert (
        classify_numeric_value_origin(
            field="quantity",
            value="750,250",
            reference="750 unités na 250 unités",
            number_word_aliases={},
        )
        == "explicit"
    )
    assert (
        classify_numeric_value_origin(
            field="quantity",
            value="7000;5000",
            reference="7000 unités na 5000 unités",
            number_word_aliases={},
        )
        == "explicit"
    )
