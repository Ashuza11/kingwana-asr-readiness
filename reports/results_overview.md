# Results Overview — All Models, Both Corpora

Last updated: 2026-11-10

This page consolidates headline WER/CER for every model evaluated in this project — the five
primary benchmark models plus the non-primary Paza extension — on both corpora. It is a summary
index; for full methodology, per-speaker/per-scenario breakdowns, entity preservation, qualitative
error analysis, and reproducibility details, follow the links in each section.

## Kingwana evaluation corpus (500 business-speech clips)

| Model | WER (%) | CER (%) | Status |
| --- | ---: | ---: | --- |
| OmniASR CTC 1B | **68.63** | **32.50** | Primary |
| SunflowerASR-51 | 70.51 | 35.96 | Primary |
| MMS-1B-All | 74.16 | 34.66 | Primary |
| XLS-R Swahili | 80.21 | 45.64 | Primary |
| Paza Whisper-Large-v3-Turbo (best: wps2 duration cap, oracle-tuned) | 103.83 | 80.54 | Extension |
| Whisper Large v3 | 117.70 | 72.34 | Primary |
| Paza Whisper-Large-v3-Turbo (raw, as-shipped) | 685.80 | 673.57 | Extension |

## TICO-19 external comparison corpus (2,124 medical read-speech clips)

| Model | WER (%) | CER (%) | Status |
| --- | ---: | ---: | --- |
| OmniASR CTC 1B | **21.25** | **6.87** | Primary |
| MMS-1B-All | 22.17 | 6.96 | Primary |
| SunflowerASR-51 | 28.79 | 15.66 | Primary |
| XLS-R Swahili | 34.50 | 13.07 | Primary |
| Whisper Large v3 | 40.53 | 17.18 | Primary |
| Paza Whisper-Large-v3-Turbo (best: wps2 duration cap, oracle-tuned) | 64.51 | 41.44 | Extension |
| Paza Whisper-Large-v3-Turbo (raw, as-shipped) | 270.71 | 221.57 | Extension |

## Reading these numbers together

- Among the five primary models, OmniASR CTC 1B is best on both corpora, and every primary model
  scores substantially lower WER/CER on TICO-19 than on the Kingwana business corpus — a
  descriptive cross-corpus difference, not a controlled domain-degradation estimate (see the
  training-overlap caveat linked below).
- Paza is a non-primary extension, evaluated separately because it was fine-tuned only on six
  Kenyan languages and never on Congolese Swahili. Even at its best oracle-tuned configuration, it
  does not beat any primary model on TICO-19, and only edges out the weakest primary model
  (Whisper Large v3) on the Kingwana corpus.
- WER above 100% is mathematically possible when insertions exceed the reference word count; it
  is reported as-is and should not, by itself, be equated with hallucination without audio-level
  confirmation (established for Whisper's insertion-heavy failures; established with audio/clip
  evidence for Paza's repetition-loop failures).

## Full detail

| Topic | Document |
| --- | --- |
| Five primary models: full Kingwana results (scenario, speaker, code-switch, noise, entity preservation, qualitative error analysis), TICO-19 comparison table, and reproducibility details | [`reports/submission_results_summary.md`](submission_results_summary.md) |
| Paza extension: full methodology, decoding/precision ablations, duration-cap mitigation analysis, and dialect-mismatch interpretation | [`reports/paza_extension_results.md`](paza_extension_results.md) |
| TICO-19 training-data overlap audit (applies to all models evaluated on TICO-19) | [`reports/tico19_training_overlap_audit.md`](tico19_training_overlap_audit.md) |
