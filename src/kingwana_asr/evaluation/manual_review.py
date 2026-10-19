"""Deterministic sampling and worksheet creation for manual ASR error review."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from kingwana_asr.data.manifest import ManifestError
from kingwana_asr.evaluation.metrics import calculate_metrics
from kingwana_asr.evaluation.normalization import normalize_text

MANUAL_REVIEW_FIELDS = (
    "monetary_amount_error",
    "currency_error",
    "person_name_error",
    "product_operator_error",
    "debt_payment_terminology_error",
    "kingwana_specific_form_error",
    "french_code_switch_error",
    "english_code_switch_error",
    "acceptable_variant",
    "semantic_equivalent",
    "hallucination",
)
MANUAL_DECISIONS = {"yes", "no", "not_applicable", "uncertain"}


def select_speaker_scenario_sample(
    rows: Sequence[Mapping[str, str]], *, size: int = 50, seed: int = 42
) -> list[dict[str, str]]:
    """Select equal counts from every observed speaker-by-scenario cell."""

    if not rows:
        raise ManifestError("Cannot select a manual-review sample from no rows")
    speakers = sorted({row["speaker_id"] for row in rows})
    scenarios = sorted({row["scenario"] for row in rows})
    cell_count = len(speakers) * len(scenarios)
    if size <= 0 or size % cell_count:
        raise ManifestError(
            f"Manual-review size must be a positive multiple of {cell_count} speaker-scenario cells"
        )
    per_cell = size // cell_count
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["speaker_id"], row["scenario"])].append(row)

    selected: list[dict[str, str]] = []
    randomizer = random.Random(seed)
    for speaker in speakers:
        for scenario in scenarios:
            candidates = sorted(
                grouped.get((speaker, scenario), []), key=lambda row: row["clip_id"]
            )
            if len(candidates) < per_cell:
                raise ManifestError(
                    f"Speaker-scenario cell {speaker}/{scenario} has "
                    f"{len(candidates)} clips; {per_cell} required"
                )
            selected.extend(dict(row) for row in randomizer.sample(candidates, per_cell))
    return sorted(selected, key=lambda row: row["clip_id"])


def select_error_stratified_sample(
    rows: Sequence[Mapping[str, str]],
    target_predictions: Sequence[Mapping[str, str]],
    *,
    size: int = 50,
    seed: int = 42,
    insertion_threshold: int = 50,
) -> list[dict[str, str]]:
    """Include all high-insertion outliers, then fill exact speaker/scenario quotas."""

    speakers = sorted({row["speaker_id"] for row in rows})
    scenarios = sorted({row["scenario"] for row in rows})
    if size % len(speakers) or size % len(scenarios):
        raise ManifestError("Error-stratified sample size must balance speakers and scenarios")
    speaker_target = size // len(speakers)
    scenario_target = size // len(scenarios)
    by_id = {row["clip_id"]: dict(row) for row in rows}
    prediction_by_id = {row["clip_id"]: row for row in target_predictions}
    if set(by_id) != set(prediction_by_id):
        raise ManifestError("Target-model predictions do not align with review candidates")

    insertion_counts: dict[str, int] = {}
    high_ids: set[str] = set()
    for clip_id, row in by_id.items():
        metrics = calculate_metrics(
            [row["reference_verbatim"]], [prediction_by_id[clip_id]["prediction"]]
        )
        insertion_counts[clip_id] = metrics.word_insertions
        if metrics.word_insertions >= insertion_threshold:
            high_ids.add(clip_id)
    if len(high_ids) > size:
        raise ManifestError("High-insertion stratum is larger than the manual-review sample")

    high_rows = [by_id[clip_id] for clip_id in sorted(high_ids)]
    speaker_remaining = {
        speaker: speaker_target - sum(row["speaker_id"] == speaker for row in high_rows)
        for speaker in speakers
    }
    scenario_remaining = {
        scenario: scenario_target - sum(row["scenario"] == scenario for row in high_rows)
        for scenario in scenarios
    }
    if min((*speaker_remaining.values(), *scenario_remaining.values())) < 0:
        raise ManifestError("High-insertion stratum exceeds a balancing quota")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for clip_id, row in by_id.items():
        if clip_id not in high_ids:
            grouped[(row["speaker_id"], row["scenario"])].append(row)
    randomizer = random.Random(seed)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: row["clip_id"])
        randomizer.shuffle(candidates)

    allocation: dict[tuple[str, str], int] = {}

    def allocate_speaker(index: int) -> bool:
        if index == len(speakers):
            return all(value == 0 for value in scenario_remaining.values())
        speaker = speakers[index]
        required = speaker_remaining[speaker]
        options: list[tuple[int, ...]] = []

        def build_options(scenario_index: int, left: int, values: list[int]) -> None:
            if scenario_index == len(scenarios):
                if left == 0:
                    options.append(tuple(values))
                return
            scenario = scenarios[scenario_index]
            maximum = min(
                left,
                scenario_remaining[scenario],
                len(grouped[(speaker, scenario)]),
            )
            for count in range(maximum + 1):
                build_options(scenario_index + 1, left - count, [*values, count])

        build_options(0, required, [])
        randomizer.shuffle(options)
        for option in options:
            for scenario, count in zip(scenarios, option, strict=True):
                scenario_remaining[scenario] -= count
                allocation[(speaker, scenario)] = count
            if allocate_speaker(index + 1):
                return True
            for scenario, count in zip(scenarios, option, strict=True):
                scenario_remaining[scenario] += count
                allocation.pop((speaker, scenario), None)
        return False

    if not allocate_speaker(0):
        raise ManifestError("Could not fill balanced background sampling quotas")

    selected: list[dict[str, str]] = []
    for row in high_rows:
        selected.append(
            {
                **row,
                "sample_stratum": "whisper_high_insertion",
                "selection_whisper_insertions": str(insertion_counts[row["clip_id"]]),
            }
        )
    for speaker in speakers:
        for scenario in scenarios:
            count = allocation.get((speaker, scenario), 0)
            for row in grouped[(speaker, scenario)][:count]:
                selected.append(
                    {
                        **row,
                        "sample_stratum": "balanced_background",
                        "selection_whisper_insertions": str(insertion_counts[row["clip_id"]]),
                    }
                )
    selected.sort(key=lambda row: row["clip_id"])
    validate_manual_sample(selected, expected_size=size)
    return selected


def build_manual_review_rows(
    selected_rows: Sequence[Mapping[str, str]],
    predictions_by_model: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    model_order: Sequence[str],
    seed: int,
) -> list[dict[str, str]]:
    """Create one pending manual-review row per selected clip and model."""

    selected_ids = {row["clip_id"] for row in selected_rows}
    if len(selected_ids) != len(selected_rows):
        raise ManifestError("Selected manual-review rows contain duplicate clip IDs")
    if set(predictions_by_model) != set(model_order):
        raise ManifestError("Prediction model set does not match configured model order")

    indexed_predictions: dict[str, dict[str, Mapping[str, str]]] = {}
    for model_id, predictions in predictions_by_model.items():
        by_id = {row["clip_id"]: row for row in predictions}
        if len(by_id) != len(predictions):
            raise ManifestError(f"Model {model_id} predictions contain duplicate clip IDs")
        missing = selected_ids - set(by_id)
        if missing:
            raise ManifestError(
                f"Model {model_id} is missing selected clips: {', '.join(sorted(missing))}"
            )
        if any(by_id[clip_id]["status"] != "success" for clip_id in selected_ids):
            raise ManifestError(f"Model {model_id} has failed selected predictions")
        indexed_predictions[model_id] = by_id

    output: list[dict[str, str]] = []
    for selected in selected_rows:
        reference = selected["reference_verbatim"]
        for model_id in model_order:
            prediction = indexed_predictions[model_id][selected["clip_id"]]["prediction"]
            metrics = calculate_metrics([reference], [prediction])
            row = {
                "selection_seed": str(seed),
                "clip_id": selected["clip_id"],
                "speaker_id": selected["speaker_id"],
                "scenario": selected["scenario"],
                "audio_path": selected.get("audio_path", ""),
                "duration_s": selected.get("duration_s", ""),
                "noise_level": selected.get("noise_level", ""),
                "quality_rating": selected.get("quality_rating", ""),
                "code_switch_reviewed": selected.get("code_switch_reviewed", ""),
                "sample_stratum": selected.get("sample_stratum", ""),
                "selection_whisper_insertions": selected.get("selection_whisper_insertions", ""),
                "model_id": model_id,
                "reference_verbatim": reference,
                "reference_eval": normalize_text(reference),
                "prediction_verbatim": prediction,
                "prediction_eval": normalize_text(prediction),
                "wer": f"{metrics.wer:.6f}",
                "cer": f"{metrics.cer:.6f}",
                "word_substitutions": str(metrics.word_substitutions),
                "word_deletions": str(metrics.word_deletions),
                "word_insertions": str(metrics.word_insertions),
                "reference_words": str(metrics.reference_words),
                "manual_review_status": "pending",
                **{field: "" for field in MANUAL_REVIEW_FIELDS},
                "manual_notes": "",
            }
            output.append(row)
    return output


def validate_manual_sample(rows: Sequence[Mapping[str, str]], *, expected_size: int) -> None:
    """Fail if a selected clip sample is not balanced as declared."""

    if len(rows) != expected_size:
        raise ManifestError(f"Expected {expected_size} selected clips; found {len(rows)}")
    speakers = {row["speaker_id"] for row in rows}
    scenarios = {row["scenario"] for row in rows}
    expected_speaker_count = expected_size // len(speakers)
    expected_scenario_count = expected_size // len(scenarios)
    if set(Counter(row["speaker_id"] for row in rows).values()) != {expected_speaker_count}:
        raise ManifestError("Manual-review sample is not balanced by speaker")
    if set(Counter(row["scenario"] for row in rows).values()) != {expected_scenario_count}:
        raise ManifestError("Manual-review sample is not balanced by scenario")


def summarize_completed_manual_review(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Validate completed judgments and summarize each model and sample stratum."""

    if not rows:
        raise ManifestError("Manual-review worksheet is empty")
    keys = [(row["clip_id"], row["model_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ManifestError("Manual-review worksheet contains duplicate clip/model rows")
    pending = [row for row in rows if row["manual_review_status"] != "complete"]
    if pending:
        raise ManifestError(f"Manual review has {len(pending)} pending comparison rows")
    for row in rows:
        for field in MANUAL_REVIEW_FIELDS:
            value = row[field].strip().lower()
            if value not in MANUAL_DECISIONS:
                raise ManifestError(
                    f"Clip {row['clip_id']} model {row['model_id']}: "
                    f"invalid {field} decision {row[field]!r}"
                )

    models = sorted({row["model_id"] for row in rows})
    strata = sorted({row["sample_stratum"] for row in rows})
    output: list[dict[str, str]] = []
    for model_id in models:
        for stratum in ["all", *strata]:
            selected = [
                row
                for row in rows
                if row["model_id"] == model_id
                and (stratum == "all" or row["sample_stratum"] == stratum)
            ]
            for category in MANUAL_REVIEW_FIELDS:
                counts = Counter(row[category].strip().lower() for row in selected)
                applicable = counts["yes"] + counts["no"]
                output.append(
                    {
                        "model_id": model_id,
                        "sample_stratum": stratum,
                        "category": category,
                        "yes": str(counts["yes"]),
                        "no": str(counts["no"]),
                        "not_applicable": str(counts["not_applicable"]),
                        "uncertain": str(counts["uncertain"]),
                        "applicable_decisions": str(applicable),
                        "error_or_feature_rate": (
                            f"{counts['yes'] / applicable:.6f}" if applicable else ""
                        ),
                        "reviewed_comparisons": str(len(selected)),
                    }
                )
    return output


def summarize_manual_review_edits(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Aggregate automatic word edits by model and declared sampling stratum."""

    if not rows:
        raise ManifestError("Manual-review worksheet is empty")
    output: list[dict[str, str]] = []
    models = sorted({row["model_id"] for row in rows})
    strata = sorted({row["sample_stratum"] for row in rows})
    for model_id in models:
        for stratum in ["all", *strata]:
            selected = [
                row
                for row in rows
                if row["model_id"] == model_id
                and (stratum == "all" or row["sample_stratum"] == stratum)
            ]
            substitutions = sum(int(row["word_substitutions"]) for row in selected)
            deletions = sum(int(row["word_deletions"]) for row in selected)
            insertions = sum(int(row["word_insertions"]) for row in selected)
            reference_words = sum(int(row["reference_words"]) for row in selected)
            output.append(
                {
                    "model_id": model_id,
                    "sample_stratum": stratum,
                    "clips": str(len(selected)),
                    "substitutions": str(substitutions),
                    "deletions": str(deletions),
                    "insertions": str(insertions),
                    "reference_words": str(reference_words),
                    "wer": f"{(substitutions + deletions + insertions) / reference_words:.6f}",
                }
            )
    return output
