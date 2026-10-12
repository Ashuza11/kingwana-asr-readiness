# Evaluation manifest

The frozen benchmark unit is a UTF-8 CSV file. Create the real private manifest as
`evaluation_v0.1.csv`; do not replace it after a comparison begins. New data requires a new
version such as `evaluation_v0.2.csv`.

Required columns:

| Column | Meaning |
| --- | --- |
| `clip_id` | Unique, stable clip identifier |
| `speaker_id` | Pseudonymous speaker identifier only |
| `audio_path` | Absolute path or path relative to the manifest directory |
| `duration_s` | Positive duration in seconds |
| `reference_verbatim` | Preserved human transcription |
| `reference_eval` | Gold text selected for deterministic scoring |
| `scenario` | Controlled scenario/domain label |
| `languages` | Declared languages, e.g. `kingwana` or `kingwana+french` |
| `code_switched` | Strict boolean: `true/false`, `1/0`, or `yes/no` |
| `noise_level` | Controlled annotation such as `low`, `medium`, `high` |

Optional future columns are `amount_present`, `amount_gold`, `currency_gold`,
`customer_present`, `product_present`, and `notes`. Entity metrics must not be enabled until those
labels have been reviewed and validated.

[`evaluation_v0.1.example.csv`](evaluation_v0.1.example.csv) contains fictional schema examples.
Its audio paths intentionally do not exist, so it is documentation—not an executable benchmark.
Do not copy fictional text into the research manifest.
