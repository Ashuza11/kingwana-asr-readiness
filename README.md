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

`configs/models.yaml` records intended language configuration. Phase 1 does not contain inference
adapters or download model weights. Each adapter will be checked against its official
implementation and the effective configuration persisted in run metadata before it is enabled.

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
clip IDs, empty gold text, unknown or malformed fields, invalid booleans, and non-positive
duration; it reports total duration and counts by speaker/scenario. The committed example contains
fictional rows only and intentionally references nonexistent files.

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

The evaluation layer wraps `jiwer` and provides corpus-micro WER/CER, per-speaker WER/CER,
speaker-macro WER/CER, per-scenario results, and code-switch group results. Failed predictions are
rejected from scoring rather than silently omitted. Entity metrics are intentionally deferred.

## Planned smoke test and model runs

Inference is not available in Phase 1. Phase 2 will add Whisper Large v3 first and exercise only a
small, balanced selection (about 20 clips across speakers/scenarios) after the researcher supplies
a valid private manifest. The smoke test must verify download/load, language configuration,
device, non-empty output, safe persistence, resumability, and clip/model alignment before a full
run.

The intended later commands are:

```bash
python -m kingwana_asr.cli transcribe --model whisper_large_v3 \
  --manifest data/manifests/evaluation_v0.1.csv --limit 20
python -m kingwana_asr.cli evaluate \
  --manifest data/manifests/evaluation_v0.1.csv
```

These are the target interface and are not exposed until implemented. No multi-gigabyte checkpoint
is downloaded during foundation setup.

## Compute backends

The core package remains a normal Linux/WSL Python project. Kaggle and Colab will install this same
package rather than fork benchmark logic. Modal support is deferred until the local architecture
is stable; see [`modal/README.md`](modal/README.md) for the data-transfer warning. Nothing
configures secrets or uploads participant recordings automatically.

## Reproducibility policy

Each later inference run will record the model ID and revision, package/PyTorch/CUDA versions,
device, dtype, decoding settings, explicit language configuration, seed, timestamp, and frozen
manifest version. Architecture-specific decoding differences must be declared. Per-clip failures
will be persisted and inference will continue; successful `model_id + clip_id` pairs will resume
unless `--force` is explicitly used.

## Current status

Phase 1 foundation is implemented: packaging, privacy controls, model/evaluation configuration,
strict manifest parsing and validation, normalization, WER/CER, aggregation, prediction schema,
CLI validation, and synthetic unit tests. Audio conversion, device management, model adapters,
resumable inference, report export, and cloud execution remain intentionally deferred.
