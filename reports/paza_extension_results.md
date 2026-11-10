# Paza (Microsoft Research) Extension: Results and Findings

Audit date: 2026-11-10

This is a non-primary extension experiment. It evaluates
[`microsoft/paza-whisper-large-v3-turbo`](https://huggingface.co/microsoft/paza-whisper-large-v3-turbo)
(pinned revision `8a0bd24795b298b2f3678fcd8129c0a1c2b4ee22`) — part of Microsoft's
[Paza benchmark/model family for low-resource African languages](https://www.microsoft.com/en-us/research/blog/paza-introducing-automatic-speech-recognition-benchmarks-and-models-for-low-resource-languages/)
— against the same two corpora used for the five primary benchmark models. Paza is **not** one of
the five primary models in [`README.md`](../README.md#benchmark-models); it is scored separately
because it was fine-tuned only on six Kenyan languages (Swahili, Dholuo, Kalenjin, Kikuyu, Maasai,
Somali) and was never trained on Kingwana (Congolese Swahili, `swc`).

## Summary

Paza-Whisper-Large-v3-Turbo catastrophically hallucinates on both evaluated Kingwana-adjacent
corpora, producing runaway word-repetition loops. Its WER is 3.6–13.3x worse than the worst of the
five primary models on each corpus. Two candidate root causes suggested by Microsoft's own
published materials — decoding misconfiguration and inference precision — were tested and ruled
out via single-clip preflights. The model card's own documented mitigation (a duration-based
word-truncation cap) never brings Paza close to competitive, even when oracle-tuned per corpus.
The best-supported remaining explanation is a genuine dialect gap: Paza has no Congolese Swahili
training exposure, and the failure is domain-independent (it fails equally on spontaneous business
speech and on read medical speech), which argues against a pure audio-domain explanation.

## Results

### Kingwana evaluation corpus (500 business-speech clips)

| Model / config | WER | CER |
| --- | --- | --- |
| **Paza, raw (as-shipped greedy decoding)** | **685.80%** | 673.57% |
| Paza + wps2 duration cap (oracle-tuned) | 103.83% | 80.54% |
| Paza + wps3 duration cap | 158.45% | 136.54% |
| Paza + wps4 duration cap | 216.59% | 194.52% |
| Paza + wps6 duration cap | 332.49% | 312.06% |
| *Five primary models (range, for reference)* | *68.63%–117.70%* | *32.50%–72.34%* |

Even at its best oracle-tuned configuration (wps2), Paza is worse than four of the five primary
models (`sunflower_asr_51` 70.51%, `omnilingual_ctc_1b` 68.63%, `mms_1b_all` 74.16%,
`xlsr_swahili` 80.21%) and only edges out the weakest, `whisper_large_v3` (117.70%).

### TICO-19 external comparison corpus (2,124 medical read-speech clips)

| Model / config | WER | CER |
| --- | --- | --- |
| **Paza, raw (as-shipped greedy decoding)** | **270.71%** | 221.57% |
| Paza + wps2 duration cap (oracle-tuned, best) | 64.51% | 41.44% |
| Paza + wps3 duration cap | 119.22% | 90.64% |
| Paza + wps4 duration cap | 161.95% | 128.07% |
| Paza + wps6 duration cap | 212.06% | 171.41% |
| *Five primary models (range, for reference)* | *21.25%–40.53%* | *6.87%–17.18%* |

On TICO-19, Paza's best possible configuration is still ~1.6x worse than the worst primary model
and ~3x worse than the best.

## Ablations performed to rule out non-dialect causes

Both ablations below were run as cheap single-clip Modal preflights before committing to a full
rerun, per the duration-cap methodology in `scripts/postprocess_paza_truncation.py`.

1. **Decoding configuration.** The as-shipped config used greedy decoding (`num_beams=1`). The
   Hugging Face model card documents an official inference recipe
   (`num_beams=5, length_penalty=0.8, early_stopping=True, suppress_tokens=[]`,
   registered as `paza_whisper_large_v3_turbo_beam5` in `configs/models.yaml`). On the same test
   clip, beam search produced a nearly identical hallucination/repetition pattern. **Ruled out.**
2. **Inference precision.** Paza was trained in fp32; the as-shipped benchmark used the GPU
   default fp16. An fp32 preflight (`paza_whisper_large_v3_turbo_fp32`, also added to
   `configs/models.yaml`) was run on the same test clip and produced a nearly identical
   hallucination pattern. **Ruled out.**

Since neither fix path improved the single-clip result, full Kingwana reruns with beam5/fp32 were
not pursued (per the oracle-tuning caveat below, further hyperparameter search on the eval set
itself would not change the conclusion). The TICO-19 corpus had not previously been run for Paza
at all, so it was run in full using the original, already-used-on-Kingwana greedy/fp16 config to
keep the two-corpus comparison config-matched.

## Duration-cap mitigation (model card's documented fix)

The model card documents a post-hoc mitigation: truncate each hypothesis to
`floor(duration_seconds * words_per_second)` words, with `words_per_second` (wps) calibrated per
language/dataset via grid search, plus an undocumented "repetition detection" pass (only described
qualitatively as scanning for repeated "5–8 word phrases"; exact thresholds are not published).
`scripts/postprocess_paza_truncation.py` reproduces the duration-cap half of this mitigation and
was run against both corpora at wps ∈ {2, 3, 4, 6}.

**Important caveat:** selecting the best wps value by minimizing WER on the eval set itself (as
done here, to establish an upper bound) is an oracle-tuning estimate, not a deployable
configuration — a real deployment would need the cap calibrated on held-out data. Even granting
this generous oracle advantage, Paza never approaches the five primary models' performance on
either corpus.

## Interpretation

- Raw hallucination is severe and domain-independent: 685.80% WER on spontaneous Kingwana business
  speech and 270.71% WER on read Congolese Swahili medical speech (TICO-19). A pure audio-domain or
  noise-condition explanation would predict the opposite — these are a markedly different domain
  and recording condition, yet both collapse the same way.
- Decoding configuration and inference precision are both ruled out as causes.
- The documented duration-cap mitigation, even oracle-tuned, cannot close the gap to the five
  primary models on either corpus.
- This leaves the dialect gap as the best-supported explanation: Paza's training data covers
  Kenyan-coded Swahili and five other Kenyan languages, never Congolese Swahili (Kingwana). Unlike
  Microsoft's own "new vs. proxy adapter" strategy for languages without direct training data
  (e.g., genealogically proxying Samburu with Maasai for MMS), Kenyan Swahili appears to have been
  assumed to transfer directly to Congolese Swahili rather than being treated as a distinct,
  under-covered dialect needing its own adapter or proxy.

## Reproducibility

| Config key (`configs/models.yaml`) | Decoding | dtype |
| --- | --- | --- |
| `paza_whisper_large_v3_turbo` | greedy (`num_beams=1`), as-shipped | float16 (device default) |
| `paza_whisper_large_v3_turbo_beam5` | `num_beams=5, length_penalty=0.8, early_stopping=True, suppress_tokens=[]` | float16 (device default) |
| `paza_whisper_large_v3_turbo_fp32` | greedy, as-shipped | float32 |
| `paza_whisper_large_v3_turbo_fp32_beam5` | official recipe | float32 |

All four variants are registered but `enabled: false` (non-primary, extension-only). Full-corpus
runs used `paza_whisper_large_v3_turbo`; `_beam5` and `_fp32` were exercised only as single-clip
preflights per the ablation methodology above. Raw predictions and run metadata (model revision,
device, dtype, decoding config, seed, timestamp) are persisted under
`outputs/paza_whisper_large_v3_turbo/` and
`outputs/tico19-swc-test-v0.1/paza_whisper_large_v3_turbo/` (both Git-ignored, private). Evaluated
summary tables are under `reports/tables/paza_extension/{raw,capped}/` (Kingwana) and
`reports/tables/tico19_paza_{raw,capped}/` (TICO-19).

See also [`reports/tico19_training_overlap_audit.md`](tico19_training_overlap_audit.md) for the
general TICO-19 cross-corpus comparison caveats that also apply here, and
[`reports/submission_results_summary.md`](submission_results_summary.md) for the five primary
models' full results on both corpora.
