from kingwana_asr.evaluation.entity_scoring import (
    canonical_entities,
    extract_scoring_numbers,
    score_predictions,
)


def test_scoring_numbers_accept_grouped_digits_and_well_formed_words() -> None:
    values = extract_scoring_numbers(
        "67 000fc, 25,000FC, vingt-cinq mille, elfu tano, and twenty three dollars"
    )
    assert {"67000", "25000", "5000", "23"} <= values


def test_canonical_entities_use_exact_aliases_without_fuzzy_matching() -> None:
    aliases = {"CDF": ["FC", "franc", "francs"], "USD": ["USD", "dollar"]}
    assert canonical_entities("bei ni francs", aliases) == {"CDF"}
    assert canonical_entities("bei ni frang", aliases) == set()


def test_score_predictions_uses_only_explicit_monetary_gold() -> None:
    aliases = {
        "currency": {"CDF": ["FC", "franc", "francs"]},
        "product_service": {"phone_charger": ["chargeur"]},
        "operator": {"airtel": ["airtel"]},
        "payment_method": {"cash": ["cash"]},
    }
    review = [
        {
            "clip_id": "clip-1",
            "gold_review_status": "complete",
            "reference_verbatim": "Chargeur ya 5000 FC na alilipa cash ku Airtel.",
            "review_status_amount": "confirmed",
            "reviewed_amount": "5000",
            "review_status_amountPaid": "confirmed",
            "reviewed_amountPaid": "5000",
            "review_status_outstandingDebt": "confirmed",
            "reviewed_outstandingDebt": "0",
            "review_status_currency": "confirmed",
            "reviewed_currency": "CDF",
            "review_status_productOrService": "confirmed",
            "reviewed_productOrService": "phone_charger",
            "review_status_paymentMethod": "confirmed",
            "reviewed_paymentMethod": "cash",
        }
    ]
    predictions = [
        {
            "clip_id": "clip-1",
            "model_id": "test-model",
            "prediction": "Chargeur 5,000 francs cash Airtel",
            "status": "success",
        }
    ]

    scores = {score.category: score for score in score_predictions(review, predictions, aliases)}

    assert scores["monetary_amount"].gold_mentions == 1
    assert scores["monetary_amount"].correct_mentions == 1
    assert all(score.mention_accuracy == 1 for score in scores.values())
