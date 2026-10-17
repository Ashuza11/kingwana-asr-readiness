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
| `languages` | Canonical ISO 639-3 tags, e.g. `swc`, `swc+fr`, or `swc+fr+en` |
| `code_switched` | Boolean, or `ambiguous`/`unknown` when the annotation is unavailable |
| `noise_level` | Controlled annotation such as `low`, `medium`, `high` |

Optional provenance/stratification columns are `languages_raw`, `code_switch_status`,
`noise_level_raw`, and `quality_rating`. Future entity columns are `amount_present`,
`amount_gold`, `currency_gold`, `customer_present`, `product_present`, and `notes`. Entity metrics
must not be enabled until those labels have been reviewed and validated.

[`evaluation_v0.1.example.csv`](evaluation_v0.1.example.csv) contains fictional schema examples.
Its audio paths intentionally do not exist, so it is documentation—not an executable benchmark.
Do not copy fictional text into the research manifest.

Real manifests are generated locally and ignored by Git. For the SautiForge export, current
annotation revisions are resolved explicitly, `reference_verbatim` is preserved, and
`reference_eval` is derived from the repository's deterministic normalization function. Source
language/noise labels are retained in provenance columns while canonical analysis labels follow
`configs/evaluation.yaml`. The primary language is named **Kingwana (Congolese Swahili / Bukavu
Swahili)** and coded `swc`; this does not rewrite its lexical forms as Standard Swahili (`swh`).
`ambiguous` and `unknown` code-switch rows are retained as annotated and excluded from binary
code-switch aggregation rather than being treated as negative examples. All audio-quality ratings
remain in the benchmark and should be available for stratified reporting.
