#!/usr/bin/env python3
"""Derive duration-capped variants of raw Paza predictions (post-processing only).

Paza's model card describes truncating each hypothesis to
``max_words = audio_duration * words_per_second`` to suppress Whisper-style
trailing hallucinations. This script applies only that length cap, as a
transparent post-processing step on the SAVED raw predictions:

* the raw ``outputs/<source_key>/predictions.csv`` is never modified;
* each cap value gets its own output key, ``<source_key>_wps<N>``, with
  ``model_id`` rewritten so ``kingwana_asr.cli evaluate`` accepts it;
* a ``run_metadata.json`` records exactly how each variant was derived;
* a small models config is generated so ``evaluate --models-config`` can
  resolve the derived keys without touching ``configs/models.yaml``.

The card's optional repetition-detection pass is NOT implemented (its
parameters are not published).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SOURCE_KEY = "paza_whisper_large_v3_turbo"
DEFAULT_WPS = (2.0, 3.0, 4.0, 6.0)


def wps_label(wps: float) -> str:
    if wps <= 0:
        raise ValueError("words-per-second must be positive")
    return str(int(wps)) if float(wps).is_integer() else str(wps).replace(".", "p")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_durations(manifest: Path) -> dict[str, float]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = set(reader.fieldnames or [])
        if not {"clip_id", "duration_s"} <= fields:
            raise SystemExit(f"Manifest needs clip_id and duration_s columns: {manifest}")
        durations: dict[str, float] = {}
        for row in reader:
            clip_id = str(row["clip_id"]).strip()
            duration = float(row["duration_s"])
            if duration <= 0:
                raise SystemExit(f"Non-positive duration for {clip_id}")
            durations[clip_id] = duration
    if not durations:
        raise SystemExit(f"Manifest has no rows: {manifest}")
    return durations


def truncate(text: str, duration_s: float, wps: float) -> tuple[str, int]:
    """Cap at floor(duration * wps) whitespace-delimited words (minimum 1)."""
    words = text.split()
    cap = max(1, math.floor(duration_s * wps))
    if len(words) <= cap:
        return text, 0
    return " ".join(words[:cap]), len(words) - cap


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_variant(
    *,
    raw_rows: list[dict[str, str]],
    fieldnames: list[str],
    durations: dict[str, float],
    source_key: str,
    variant_key: str,
    wps: float,
    output_root: Path,
    source_csv: Path,
    manifest: Path,
    force: bool,
) -> dict[str, Any]:
    out_dir = output_root / variant_key
    out_csv = out_dir / "predictions.csv"
    if out_csv.exists() and not force:
        raise SystemExit(f"{out_csv} exists; pass --force to overwrite derived variants")

    rows: list[dict[str, str]] = []
    truncated_clips = 0
    words_before = words_after = 0
    for row in raw_rows:
        new = dict(row)
        new["model_id"] = variant_key
        if row.get("status") == "success":
            clip_id = row["clip_id"]
            if clip_id not in durations:
                raise SystemExit(f"Clip {clip_id} is missing from the manifest")
            text = row["prediction"]
            capped, removed = truncate(text, durations[clip_id], wps)
            new["prediction"] = capped
            words_before += len(text.split())
            words_after += len(capped.split())
            truncated_clips += 1 if removed else 0
        rows.append(new)

    out_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".predictions.", dir=out_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out_csv)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "variant_key": variant_key,
        "words_per_second": wps,
        "clips": len(rows),
        "clips_truncated": truncated_clips,
        "words_before": words_before,
        "words_after": words_after,
    }
    metadata = {
        "derived": True,
        "derivation": "duration_word_cap",
        "rule": "max_words = max(1, floor(audio_duration_s * words_per_second))",
        "source_model_key": source_key,
        "source_predictions_sha256": sha256_of(source_csv),
        "duration_manifest": str(manifest),
        "duration_manifest_sha256": sha256_of(manifest),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repetition_detection_applied": False,
        "note": (
            "Post-processing of saved raw predictions using a pre-declared grid of caps. "
            "Choosing the best-scoring cap on the evaluation data is oracle tuning and "
            "must be reported as an upper bound, not as a deployable result."
        ),
        **summary,
    }
    atomic_write(
        out_dir / "run_metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest with clip_id and duration_s (e.g. evaluation_v0.1_standardized.csv)")
    parser.add_argument("--source-key", default=DEFAULT_SOURCE_KEY)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--wps", type=float, nargs="+", default=list(DEFAULT_WPS),
                        help="Words-per-second caps to derive (default: 2 3 4 6)")
    parser.add_argument("--models-config-out", type=Path,
                        default=Path("configs/models_paza_variants.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_csv = args.output_root / args.source_key / "predictions.csv"
    if not source_csv.is_file():
        raise SystemExit(f"Raw predictions not found: {source_csv}")
    durations = read_durations(args.manifest)

    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fieldnames = list(reader.fieldnames or [])
        raw_rows = list(reader)
    if not raw_rows:
        raise SystemExit(f"No rows in {source_csv}")
    wrong = {r["model_id"] for r in raw_rows} - {args.source_key}
    if wrong:
        raise SystemExit(f"Unexpected model_id values in raw predictions: {sorted(wrong)}")

    summaries = []
    entries: dict[str, Any] = {}
    for wps in args.wps:
        key = f"{args.source_key}_wps{wps_label(wps)}"
        summaries.append(
            write_variant(
                raw_rows=raw_rows, fieldnames=fieldnames, durations=durations,
                source_key=args.source_key, variant_key=key, wps=wps,
                output_root=args.output_root, source_csv=source_csv,
                manifest=args.manifest, force=args.force,
            )
        )
        entries[key] = {
            "enabled": False,
            "display_name": f"Paza Whisper Large v3 Turbo (cap {wps:g} words/s)",
            "hf_id": "microsoft/paza-whisper-large-v3-turbo",
            "family": "derived",
            "derived_from": args.source_key,
            "postprocess": {"type": "duration_word_cap", "words_per_second": wps},
        }

    config_text = yaml.safe_dump(
        {"schema_version": 1, "models": entries}, sort_keys=False, allow_unicode=True
    )
    atomic_write(args.models_config_out, config_text)

    print(f"Raw predictions untouched: {source_csv}")
    print(f"Wrote {args.models_config_out}\n")
    print(f"{'variant':<42}{'wps':>5}{'truncated':>12}{'words before':>14}{'after':>8}")
    for s in summaries:
        pct = 100 * s["clips_truncated"] / s["clips"]
        print(f"{s['variant_key']:<42}{s['words_per_second']:>5g}"
              f"{s['clips_truncated']:>6} ({pct:4.1f}%){s['words_before']:>14}{s['words_after']:>8}")


if __name__ == "__main__":
    sys.exit(main())