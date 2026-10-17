# Kingwana ASR Readiness

**Working title:** *From Spoken Transactions to Structured Records: Evaluating ASR Readiness
for Kingwana Small-Business Speech*

This repository provides a reproducible evaluation pipeline for measuring how existing
open-source ASR systems handle natural Kingwana (Congolese Swahili) in small-business and
telecom transactions. Initial outcomes are corpus, speaker, scenario, and code-switch WER/CER,
followed by qualitative error analysis. Business-critical entity evaluation will be added only
after suitable gold labels are validated.

> This repository evaluates existing ASR models on Kingwana small-business speech. It does not
> yet train a new ASR model or implement Speech-to-SQL.

## Benchmark models

The planned primary benchmark contains exactly these systems:

| Key | Model | Role |
| --- | --- | --- |
| `whisper_large_v3` | `openai/whisper-large-v3` | General multilingual baseline |
| `sunflower_asr_51` | `Sunbird/SunflowerASR-51-african-languages` | African-language Whisper adaptation |
| `mms_1b_all` | `facebook/mms-1b-all` | Massively multilingual baseline |
| `omnilingual_ctc_1b` | `facebook/omniASR-CTC-1B` | Modern massively multilingual baseline |
| `xlsr_swahili` | `thinkKenya/wav2vec2-large-xls-r-300m-sw` | Swahili-specific baseline |

`configs/models.yaml` records intended language configuration. All five primary models have
executable adapters. Each adapter is checked against its official implementation and the
effective configuration is persisted in run metadata before use. MMS uses its explicit `swh`
adapter. OmniASR CTC does not accept language conditioning, so its metadata distinguishes the
Kingwana evaluation language (`swc`) from the closest supported label (`swh_Latn`) without
passing an ignored language argument.

## Privacy and data governance

The recordings contain human voices and are private research data. Do not commit audio,
participant names, phone numbers, consent forms, identifiers, private metadata, credentials,
model weights, caches, or unapproved dataset exports. The repository makes no assumption that the
corpus can be released publicly. See [`data/README.md`](data/README.md) and `.gitignore`.

Only pseudonymous speaker IDs belong in manifests. Review `git status` and staged diffs for
sensitive content before every commit. Raw audio must remain unchanged; later preparation will
write derivatives to a separate ignored location.

## Setup

Python 3.11 is preferred; Python 3.10 through 3.13 is supported. The current WSL environment has
Python 3.12 and `uv`, so the reproducible local setup is:

```bash
uv sync --extra dev
uv run pytest
```

`uv.lock` is committed after dependency resolution. A standard virtual environment also works:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

ASR dependencies are optional and are not needed for Phase 1 tests. Later phases will use
`uv sync --extra dev --extra audio --extra asr`; Omnilingual's official package has its own
`omnilingual` extra.

## Manifest

The evaluation unit is a versioned UTF-8 CSV such as
`data/manifests/evaluation_v0.1.csv`. Required columns are:

```text
clip_id,speaker_id,audio_path,duration_s,reference_verbatim,reference_eval,scenario,languages,code_switched,noise_level
```

Paths may be absolute or relative to the manifest. Validation rejects missing files, duplicate
clip IDs, empty gold text, unknown or malformed fields, invalid code-switch labels, and
non-positive duration; it reports total duration and counts by speaker/scenario. Code-switch
labels may be true, false, ambiguous, or unknown; the unavailable labels are excluded from that
breakdown. The committed example contains fictional rows only and intentionally references
nonexistent files. Real manifests are Git-ignored because they contain private transcripts.

The verified private SautiForge pilot export contains 500 clips, 10 pseudonymous speakers, five
balanced scenarios, and 3,645.319 seconds of declared audio. Build the full and deterministic
evaluation and deterministic one-clip preflight manifests without changing the export:

```bash
uv run python scripts/build_sautiforge_manifests.py \
  --export-root data/private/sautiforge-kingwana-pilot-2026-10-16 \
  --evaluation-manifest data/manifests/evaluation_v0.1.csv \
  --smoke-manifest data/manifests/smoke_v0.1_1clip_source.csv \
  --smoke-size 1 --seed 42
```

```bash
uv run python -m kingwana_asr.cli validate \
  --manifest data/manifests/evaluation_v0.1.csv
```

Never silently replace a frozen benchmark. Add a new manifest version when data changes.

## Normalization and metrics

The scoring transform is deterministic: Unicode NFC, lowercase, Unicode punctuation replaced by
spaces, then whitespace collapsed. It does not translate Kingwana, rewrite it as Standard Swahili,
or alter vocabulary based on predictions. `reference_verbatim` is preserved; `reference_eval` is
the scoring source. CER currently includes normalized spaces, as recorded in
`configs/evaluation.yaml`.

Corpus language labels use **Kingwana (Congolese Swahili / Bukavu Swahili)** with ISO 639-3
`swc`; French and English use `fr` and `en`. Raw export spellings remain in provenance columns.
This corpus label is distinct from the Standard Swahili (`swh`) proxy required by some benchmark
model APIs. Noise labels are canonicalized to `low`, `medium`, `high`, or `unknown`, with the raw
value retained. Every audio-quality rating remains eligible and is retained for later stratified
reporting.

The evaluation layer wraps `jiwer` and provides corpus-micro WER/CER, per-speaker WER/CER,
speaker-macro WER/CER, per-scenario results, annotated code-switch status, canonical language-mix,
noise-level, and audio-quality strata. A binary code-switch comparison includes only definitive
`present`/`none` labels; ambiguous and unknown labels remain visible in the categorical report.
Failed predictions are rejected from scoring rather than silently omitted. Entity metrics are
intentionally deferred.

## One-clip preflight and model runs

The local preflight manifest selects one clear, representative clip. Each model's first run uses
this same standardized clip to verify
download/load, language configuration, device, non-empty output, safe persistence, resumability,
and clip/model alignment before any full run.

The intended later commands are:

```bash
python -m kingwana_asr.cli transcribe --model whisper_large_v3 \
  --manifest data/manifests/smoke_v0.1_1clip.csv
python -m kingwana_asr.cli evaluate \
  --manifest data/manifests/evaluation_v0.1_analysis.csv
```

The `transcribe` command supports `--device`, `--limit`, repeatable `--clip-id`, and `--force`.
Attempts are appended safely to JSONL, while `predictions.csv` and `run_metadata.json` are written
atomically under `outputs/<model_key>/`. Successful model/clip pairs are skipped on resume and
per-clip errors are logged without ending the run. No checkpoint is downloaded during setup or
tests.

Full inference uses the source-preserving 16 kHz WAV manifest
`evaluation_v0.1_standardized.csv`. The derived private `evaluation_v0.1_analysis.csv` retains
those exact clip IDs, references, paths, and decoded durations while adding the approved canonical
language/noise labels, their raw source labels, code-switch status, and quality rating. This keeps
inference inputs distinct from analysis annotations. Both manifests and all prediction files are
Git-ignored private research artifacts.

Evaluation validates exact manifest coverage, model IDs, speakers, references, prediction status,
and duplicate keys before atomically exporting tables under `reports/tables/`. Missing or failed
clips stop scoring instead of producing partial results.

Prepare source-preserving 16 kHz mono PCM WAV derivatives with FFmpeg:

```bash
uv run python scripts/prepare_audio.py \
  --manifest data/manifests/smoke_v0.1_1clip_source.csv \
  --output-dir data/processed/smoke_v0.1_1clip \
  --output-manifest data/manifests/smoke_v0.1_1clip.csv \
  --dry-run
```

Remove `--dry-run` only after reviewing the deterministic destinations. Existing files and source
paths are never overwritten.

## Compute backends

The core package remains a normal Linux/WSL Python project. Kaggle and Colab will install this same
package rather than fork benchmark logic. Modal is a thin GPU backend around the shared inference
pipeline; see [`modal/README.md`](modal/README.md) for the data-transfer warning. Nothing creates
secrets or uploads participant recordings automatically.

## Reproducibility policy

Each later inference run will record the model ID and revision, package/PyTorch/CUDA versions,
device, dtype, decoding settings, explicit language configuration, seed, timestamp, and frozen
manifest version. Architecture-specific decoding differences must be declared. Per-clip failures
will be persisted and inference will continue; successful `model_id + clip_id` pairs will resume
unless `--force` is explicitly used.

## Current status

The foundation and all five primary model adapters are complete. Source-preserving audio
preparation, device selection, per-clip failure isolation, incremental prediction storage,
resume/force behavior, and preflight selection controls are implemented. Whisper Large v3,
SunflowerASR-51, MMS-1B-All, OmniASR CTC 1B, and XLS-R Swahili have all passed the same one-clip
Modal engineering preflight and the 500-clip evaluation on an NVIDIA L4. Each model has 500
successful, manifest-aligned, non-empty persisted predictions and an empty error log. Private model
outputs are downloaded under `outputs/<model_key>/`; the eight local evaluation tables cover global,
speaker, scenario, code-switch, language-mix, noise, and quality analyses. Entity scoring remains a
future extension.
