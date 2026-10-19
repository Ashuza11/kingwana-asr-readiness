"""Local persistence for the human-in-the-loop manual review interface."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from kingwana_asr.data.manifest import ManifestError

REVIEW_LABELS = (
    "AMT",
    "CUR",
    "NAME",
    "PROD",
    "DEBT",
    "KING",
    "FR",
    "EN",
    "VAR",
    "SEM",
    "HALL",
    "UNCERTAIN",
)
MODEL_CODES = ("W", "S", "M", "O", "X")
MODEL_IDS = {
    "W": "whisper_large_v3",
    "S": "sunflower_asr_51",
    "M": "mms_1b_all",
    "O": "omnilingual_ctc_1b",
    "X": "xlsr_swahili",
}
LABEL_TO_WORKSHEET_FIELD = {
    "AMT": "monetary_amount_error",
    "CUR": "currency_error",
    "NAME": "person_name_error",
    "PROD": "product_operator_error",
    "DEBT": "debt_payment_terminology_error",
    "KING": "kingwana_specific_form_error",
    "FR": "french_code_switch_error",
    "EN": "english_code_switch_error",
    "VAR": "acceptable_variant",
    "SEM": "semantic_equivalent",
    "HALL": "hallucination",
}


def parse_labels(value: str | Sequence[str]) -> list[str]:
    """Return unique labels in the project's canonical display order."""

    raw = value.split(",") if isinstance(value, str) else value
    provided = {item.strip().upper() for item in raw if item.strip()}
    unknown = provided - set(REVIEW_LABELS)
    if unknown:
        raise ManifestError(f"Unknown manual-review labels: {', '.join(sorted(unknown))}")
    return [label for label in REVIEW_LABELS if label in provided]


def finalize_label_review(
    review_rows: Sequence[Mapping[str, str]],
    worksheet_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Merge validated labels into a new worksheet and produce label counts.

    The web review is a complete multi-label judgment for every comparison. A
    missing label therefore becomes ``no`` in the compatibility worksheet.
    ``UNCERTAIN`` remains an independent label in the authoritative review and
    summary; it is also preserved in the worksheet notes.
    """

    if len(review_rows) != 250 or len(worksheet_rows) != 250:
        raise ManifestError("Final manual review requires exactly 250 rows in both inputs")
    review_keys = {(row["clip_id"], row["model_code"]) for row in review_rows}
    if len(review_keys) != 250:
        raise ManifestError("Label review contains duplicate clip/model rows")
    pending = [row for row in review_rows if row["human_status"] != "validated"]
    if pending:
        raise ManifestError(f"Label review has {len(pending)} pending rows")

    indexed = {(row["clip_id"], row["model_code"]): row for row in review_rows}
    finalized: list[dict[str, str]] = []
    summary_source: list[tuple[str, str, set[str]]] = []
    for source in worksheet_rows:
        model_id = source["model_id"]
        matching_codes = [code for code, value in MODEL_IDS.items() if value == model_id]
        if len(matching_codes) != 1:
            raise ManifestError(f"Unsupported worksheet model ID: {model_id}")
        key = (source["clip_id"], matching_codes[0])
        if key not in indexed:
            raise ManifestError(f"Missing validated label review for {key[0]}/{key[1]}")
        review = indexed[key]
        labels = set(parse_labels(review["human_labels"]))
        row = dict(source)
        row["manual_review_status"] = "complete"
        for label, field in LABEL_TO_WORKSHEET_FIELD.items():
            row[field] = "yes" if label in labels else "no"
        notes = review["human_notes"].strip()
        if "UNCERTAIN" in labels:
            uncertainty = "Human label UNCERTAIN retained in authoritative label review."
            notes = f"{notes} {uncertainty}".strip()
        row["manual_notes"] = notes
        finalized.append(row)
        summary_source.append((model_id, source["sample_stratum"], labels))

    if len({(row["clip_id"], row["model_id"]) for row in finalized}) != 250:
        raise ManifestError("Finalized worksheet contains duplicate clip/model rows")

    summary: list[dict[str, str]] = []
    strata = sorted({stratum for _, stratum, _ in summary_source})
    for model_id in MODEL_IDS.values():
        for stratum in ("all", *strata):
            selected = [
                labels
                for candidate_model, candidate_stratum, labels in summary_source
                if candidate_model == model_id
                and (stratum == "all" or candidate_stratum == stratum)
            ]
            counts = Counter(label for labels in selected for label in labels)
            for label in REVIEW_LABELS:
                summary.append(
                    {
                        "model_id": model_id,
                        "sample_stratum": stratum,
                        "label": label,
                        "count": str(counts[label]),
                        "reviewed_comparisons": str(len(selected)),
                        "comparison_rate": f"{counts[label] / len(selected):.6f}",
                    }
                )
    return finalized, summary


class ManualReviewStore:
    """Read and atomically update the local review CSV and progress JSON."""

    def __init__(self, review_path: Path, progress_path: Path) -> None:
        self.review_path = review_path
        self.progress_path = progress_path
        self._lock = threading.Lock()

    def load(self) -> tuple[list[dict[str, str]], dict[str, object]]:
        with self._lock:
            return self._load_rows(), self._load_progress()

    def apply(self, decisions: Sequence[Mapping[str, object]]) -> dict[str, object]:
        """Apply explicit accept/correct/reset decisions and return updated progress."""

        if not decisions:
            raise ManifestError("No review decisions were supplied")
        with self._lock:
            rows = self._load_rows()
            progress = self._load_progress()
            indexed = {(row["example_id"], row["model_code"]): row for row in rows}
            seen: set[tuple[str, str]] = set()
            for decision in decisions:
                example_id = str(decision.get("example_id", "")).strip()
                model_code = str(decision.get("model_code", "")).strip().upper()
                key = (example_id, model_code)
                if key in seen:
                    raise ManifestError(f"Duplicate decision for example {example_id}/{model_code}")
                seen.add(key)
                if key not in indexed:
                    raise ManifestError(f"Unknown review row: example {example_id}/{model_code}")
                action = str(decision.get("action", "")).strip().lower()
                row = indexed[key]
                if action == "accept":
                    if not row["suggested_labels"]:
                        raise ManifestError(f"Example {example_id}/{model_code} has no suggestion")
                    row["human_labels"] = ", ".join(parse_labels(row["suggested_labels"]))
                    row["human_notes"] = str(decision.get("human_notes", "")).strip()
                    row["human_status"] = "validated"
                elif action == "correct":
                    labels = decision.get("human_labels", [])
                    if not isinstance(labels, (str, list, tuple)):
                        raise ManifestError("human_labels must be a string or list")
                    row["human_labels"] = ", ".join(parse_labels(labels))
                    row["human_notes"] = str(decision.get("human_notes", "")).strip()
                    row["human_status"] = "validated"
                elif action == "reset":
                    row["human_labels"] = ""
                    row["human_notes"] = ""
                    row["human_status"] = "pending"
                else:
                    raise ManifestError(f"Unsupported review action: {action or '<empty>'}")

            self._write_rows(rows)
            self._update_progress(progress, rows)
            self._write_json(self.progress_path, progress)
            return progress

    def _load_rows(self) -> list[dict[str, str]]:
        try:
            with self.review_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except FileNotFoundError as exc:
            raise ManifestError(f"Manual-review CSV does not exist: {self.review_path}") from exc
        if len(rows) != 250:
            raise ManifestError(f"Expected 250 manual-review rows; found {len(rows)}")
        keys = {(row["example_id"], row["model_code"]) for row in rows}
        if len(keys) != len(rows):
            raise ManifestError("Manual-review CSV contains duplicate example/model rows")
        return rows

    def _load_progress(self) -> dict[str, object]:
        try:
            with self.progress_path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError as exc:
            raise ManifestError(f"Progress JSON does not exist: {self.progress_path}") from exc
        if not isinstance(value, dict):
            raise ManifestError("Manual-review progress must be a JSON object")
        return value

    def _write_rows(self, rows: Sequence[Mapping[str, str]]) -> None:
        fieldnames = list(rows[0])
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{self.review_path.name}.",
                suffix=".tmp",
                dir=self.review_path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.review_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @staticmethod
    def _update_progress(progress: dict[str, object], rows: Sequence[Mapping[str, str]]) -> None:
        validated_batches: list[int] = []
        for batch in range(1, 6):
            lower = (batch - 1) * 10 + 1
            upper = batch * 10
            batch_rows = [row for row in rows if lower <= int(row["example_id"]) <= upper]
            if len(batch_rows) == 50 and all(
                row["human_status"] == "validated" for row in batch_rows
            ):
                validated_batches.append(batch)
        pending_examples = sorted(
            {int(row["example_id"]) for row in rows if row["human_status"] != "validated"}
        )
        progress["human_validated_batches"] = validated_batches
        progress["next_example_index"] = pending_examples[0] if pending_examples else 51
        progress["last_update_timestamp"] = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
