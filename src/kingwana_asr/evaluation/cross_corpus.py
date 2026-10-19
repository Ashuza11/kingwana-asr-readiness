"""Cross-corpus metric comparisons without causal domain claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def compare_model_summaries(
    business_rows: Sequence[Mapping[str, str]],
    comparator_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Return percentage-point differences after exact model-ID validation."""

    business = {row["model_id"]: row for row in business_rows}
    comparator = {row["model_id"]: row for row in comparator_rows}
    if len(business) != len(business_rows) or len(comparator) != len(comparator_rows):
        raise ValueError("Model summaries contain duplicate model IDs")
    if set(business) != set(comparator):
        raise ValueError("Model summaries do not contain the same model IDs")

    results: list[dict[str, str]] = []
    for model_id in business:
        business_wer = float(business[model_id]["wer"]) * 100
        comparator_wer = float(comparator[model_id]["wer"]) * 100
        business_cer = float(business[model_id]["cer"]) * 100
        comparator_cer = float(comparator[model_id]["cer"]) * 100
        results.append(
            {
                "model_id": model_id,
                "business_wer_percent": f"{business_wer:.6f}",
                "comparator_wer_percent": f"{comparator_wer:.6f}",
                "wer_cross_corpus_difference_pp": f"{business_wer - comparator_wer:.6f}",
                "business_cer_percent": f"{business_cer:.6f}",
                "comparator_cer_percent": f"{comparator_cer:.6f}",
                "cer_cross_corpus_difference_pp": f"{business_cer - comparator_cer:.6f}",
            }
        )
    return results
