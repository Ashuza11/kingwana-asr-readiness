"""Corpus, speaker, scenario, and code-switch metric aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from statistics import fmean
from typing import Any

from kingwana_asr.evaluation.metrics import calculate_metrics


def _code_switch_label(value: Any) -> str:
    if isinstance(value, bool):
        code_switched = value
    elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        code_switched = value.strip().lower() == "true"
    else:
        raise ValueError(f"Invalid code_switched value: {value!r}")
    return "code_switched" if code_switched else "not_code_switched"


def _score(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = calculate_metrics(
        [str(record["reference"]) for record in records],
        [str(record["prediction"]) for record in records],
    )
    return asdict(metrics)


def aggregate_predictions(
    records: Iterable[Mapping[str, Any]], *, minimum_scenario_clips: int = 1
) -> dict[str, Any]:
    """Aggregate only successful records; never silently accept failed records."""
    materialized = list(records)
    failures = [record for record in materialized if record.get("status") != "success"]
    if failures:
        failed_ids = ", ".join(str(record.get("clip_id", "<unknown>")) for record in failures)
        raise ValueError(f"Cannot score failed or incomplete predictions: {failed_ids}")
    if not materialized:
        raise ValueError("No prediction records were supplied")

    by_speaker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_code_switch: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in materialized:
        by_speaker[str(record["speaker_id"])].append(record)
        by_scenario[str(record["scenario"])].append(record)
        label = _code_switch_label(record["code_switched"])
        by_code_switch[label].append(record)

    speaker_metrics = {key: _score(value) for key, value in sorted(by_speaker.items())}
    scenario_metrics = {
        key: _score(value)
        for key, value in sorted(by_scenario.items())
        if len(value) >= minimum_scenario_clips
    }
    code_switch_metrics = {key: _score(value) for key, value in sorted(by_code_switch.items())}
    return {
        "global": _score(materialized),
        "per_speaker": speaker_metrics,
        "speaker_macro_wer": fmean(item["wer"] for item in speaker_metrics.values()),
        "speaker_macro_cer": fmean(item["cer"] for item in speaker_metrics.values()),
        "per_scenario": scenario_metrics,
        "by_code_switch": code_switch_metrics,
    }
