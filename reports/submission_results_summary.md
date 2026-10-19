# Submission-ready evaluation results

## Evaluation coverage

All five systems were evaluated zero-shot on the same 500-utterance Kingwana business corpus. The standardized evaluation audio totals 3,631.420 seconds (approximately 60.5 minutes) and contains ten pseudonymous speakers, with 50 recordings per speaker. The corpus is balanced across five scenarios, with 100 recordings each for business-information queries, customer-debt recording, debt-repayment recording, sales recording, and inventory updates. Each prediction file contains 500 unique clip identifiers, all aligned with the evaluation manifest; no inference failure or empty prediction is present in the final outputs.

Evaluation used deterministic normalization applied identically to references and hypotheses: NFC Unicode normalization, Unicode-aware lowercasing, replacement of Unicode punctuation with spaces, and whitespace collapsing. Numbers were not converted between written and digit forms, filled pauses were not removed, and Kingwana, French, and borrowed lexical forms were not translated or standardized. Corpus-micro WER and CER were calculated with `jiwer`; CER includes whitespace.

## Overall ASR performance

| Model | WER (%) | CER (%) | Speaker-macro WER (%) | Speaker-macro CER (%) |
|---|---:|---:|---:|---:|
| OmniASR CTC 1B | **68.63** | **32.50** | **68.64** | **32.51** |
| SunflowerASR-51 | 70.51 | 35.96 | 70.51 | 35.97 |
| MMS-1B-All | 74.16 | 34.66 | 74.16 | 34.67 |
| XLS-R Swahili | 80.21 | 45.64 | 80.21 | 45.65 |
| Whisper Large v3 | 117.70 | 72.34 | 117.51 | 72.34 |

OmniASR produced the lowest overall WER and CER. SunflowerASR-51 had the second-lowest WER, while MMS-1B-All had the second-lowest CER. The near equality of corpus-micro and speaker-macro scores is consistent with the balanced design of 50 clips per speaker. None of the evaluated systems achieved a WER below 68%, indicating that substantial recognition difficulty remains for natural Kingwana transactional speech.

## Word-level edit counts

| Model | Substitutions | Deletions | Insertions | Reference words | WER (%) |
|---|---:|---:|---:|---:|---:|
| OmniASR CTC 1B | 2,811 | 1,231 | 96 | 6,029 | 68.63 |
| SunflowerASR-51 | 3,286 | 629 | 336 | 6,029 | 70.51 |
| MMS-1B-All | 3,478 | 695 | 298 | 6,029 | 74.16 |
| XLS-R Swahili | 3,549 | 1,186 | 101 | 6,029 | 80.21 |
| Whisper Large v3 | 3,670 | 613 | 2,813 | 6,029 | 117.70 |

Substitutions were the largest edit category for four systems. OmniASR and XLS-R also showed comparatively high deletion counts. Whisper differed sharply: its 2,813 insertions, in addition to 3,670 substitutions and 613 deletions, raised its WER above 100%. A WER above 100% is mathematically possible because insertions increase the edit count beyond the number of reference words; it should not, by itself, be equated with hallucination.

## Scenario-level performance

| Scenario (100 clips each) | OmniASR WER (%) | Sunflower WER (%) | MMS WER (%) | XLS-R WER (%) | Whisper WER (%) |
|---|---:|---:|---:|---:|---:|
| Business-information query | 68.24 | **67.22** | 73.43 | 77.50 | 94.07 |
| Customer-debt recording | **66.72** | 72.09 | 73.53 | 79.66 | 150.04 |
| Debt-repayment recording | **64.07** | 67.05 | 65.90 | 75.74 | 90.47 |
| Sales recording | **73.40** | 76.36 | 81.71 | 83.95 | 121.81 |
| Inventory update | 70.85 | **69.61** | 76.55 | 84.15 | 132.62 |

Performance varied by scenario. OmniASR achieved the lowest WER for customer-debt, debt-repayment, and sales recordings, while SunflowerASR-51 was best for business-information queries and inventory updates. Whisper's largest error rates occurred for customer-debt recording and inventory updates, driven in part by insertion-heavy outputs. These descriptive differences were not tested for statistical significance.

## Speaker-level performance

| Model | Lowest speaker WER (%) | Highest speaker WER (%) | Speaker-macro WER (%) |
|---|---:|---:|---:|
| OmniASR CTC 1B | 63.74 | 75.33 | 68.64 |
| SunflowerASR-51 | 61.92 | 78.61 | 70.51 |
| MMS-1B-All | 67.05 | 80.13 | 74.16 |
| XLS-R Swahili | 74.79 | 85.24 | 80.21 |
| Whisper Large v3 | 69.93 | 195.72 | 117.51 |

The results show speaker-dependent variation for every model. Variation was particularly large for Whisper, whose speaker-level WER ranged from 69.93% to 195.72%. Because speaker identity may be confounded with speaking style, recording conditions, and utterance content, these ranges should be treated descriptively rather than as evidence of demographic bias.

## Code-switching analysis

Following researcher review using the study's distinction between integrated borrowing and genuine language switching, 85 recordings were labelled code-switched and 415 were labelled not code-switched.

| Model | Code-switched WER (%) | Not code-switched WER (%) | Code-switched CER (%) | Not code-switched CER (%) |
|---|---:|---:|---:|---:|
| OmniASR CTC 1B | 67.51 | 68.87 | 29.58 | 33.11 |
| SunflowerASR-51 | 65.16 | 71.65 | 30.94 | 37.02 |
| MMS-1B-All | 71.09 | 74.81 | 31.88 | 35.24 |
| XLS-R Swahili | 81.45 | 79.95 | 46.00 | 45.56 |
| Whisper Large v3 | 207.53 | 98.49 | 163.20 | 53.17 |

For OmniASR, Sunflower, and MMS, WER was descriptively lower in the code-switched subset; XLS-R showed a small increase. Whisper showed a much higher WER in the code-switched subset, but this subset contained insertion-heavy failures. These results do not estimate a causal effect of code-switching: the groups are unequal in size and may differ in speaker, scenario, utterance complexity, and lexical content.

## Noise and recording quality

| Model | Low-noise WER (%) (n=183) | Medium-noise WER (%) (n=222) | High-noise WER (%) (n=76) |
|---|---:|---:|---:|
| OmniASR CTC 1B | 69.58 | 67.92 | 69.32 |
| SunflowerASR-51 | 71.95 | 69.59 | 71.40 |
| MMS-1B-All | 76.41 | 73.32 | 72.49 |
| XLS-R Swahili | 80.83 | 79.97 | 81.33 |
| Whisper Large v3 | 127.37 | 103.18 | 147.82 |

The remaining 19 recordings had unknown noise labels. Across the observed noise and quality-rating groups, error rates did not follow a simple monotonic pattern. This should not be interpreted as evidence that noise has no effect: the labels were subjective and noise condition may be confounded with speaker, scenario, clip complexity, microphone placement, and other recording characteristics. Quality ratings were retained for all recordings; the distribution was one clip rated 1, 123 rated 2, 116 rated 3, 168 rated 4, 90 rated 5, and two with missing ratings.

## Business-entity preservation

Entity preservation was scored only for researcher-validated and explicitly spoken gold values. The strict mention-level score required an independently extracted canonical entity or exact numeric value to be present in the normalized ASR output. The score therefore measures literal preservation, not semantic recovery by a downstream information-extraction model.

| Model | Monetary amounts (n=328) | Currencies (n=150) | Products/services (n=342) | Operators (n=351) | Payment methods (n=37) |
|---|---:|---:|---:|---:|---:|
| OmniASR CTC 1B | 5/328 (1.52%) | **39/150 (26.00%)** | **102/342 (29.82%)** | **158/351 (45.01%)** | 2/37 (5.41%) |
| SunflowerASR-51 | 124/328 (37.80%) | 8/150 (5.33%) | 50/342 (14.62%) | 148/351 (42.17%) | **5/37 (13.51%)** |
| MMS-1B-All | 74/328 (22.56%) | 37/150 (24.67%) | 84/342 (24.56%) | 129/351 (36.75%) | 3/37 (8.11%) |
| XLS-R Swahili | 0/328 (0.00%) | 0/150 (0.00%) | 6/342 (1.75%) | 14/351 (3.99%) | 0/37 (0.00%) |
| Whisper Large v3 | **182/328 (55.49%)** | 22/150 (14.67%) | 56/342 (16.37%) | 74/351 (21.08%) | 4/37 (10.81%) |

The ranking depended strongly on the business field. Whisper preserved the largest proportion of exact monetary mentions despite having the highest overall WER. OmniASR preserved the most currencies, products/services, and operators, while Sunflower preserved the most payment methods. This illustrates why generic WER alone is insufficient for transactional use: a system can have a relatively poor transcript score while retaining a critical amount, or a better transcript score while losing it. The very low numerical preservation scores for OmniASR and XLS-R should be interpreted under the strict surface-preservation rule and do not establish that a downstream semantic normalizer could never recover the intended values.

Customer/person-name and debt/payment terminology accuracy were not automatically scored over all 500 clips. These categories were assessed in the human-reviewed sample described below and should not be presented as corpus-wide entity accuracy.

## Manual qualitative error analysis

Fifty utterances were compared against all five model outputs, producing 250 human-validated model--utterance judgments. The complete sample was balanced at five clips per speaker and ten clips per scenario. It contained 34 deterministically selected background clips and 16 clips deliberately selected to inspect extreme Whisper insertion behaviour. Multiple labels could be assigned to one judgment, so counts do not sum to 50.

| Model | AMT | CUR | NAME | PROD | DEBT | KING | FR | EN | VAR | SEM | HALL | UNCERTAIN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniASR CTC 1B | 37 | 11 | 14 | 15 | 10 | 49 | 6 | 0 | 39 | 9 | 0 | 1 |
| SunflowerASR-51 | 20 | 11 | 13 | 24 | 11 | 46 | 7 | 0 | 41 | 20 | 0 | 9 |
| MMS-1B-All | 23 | 4 | 15 | 18 | 10 | 48 | 4 | 0 | 42 | 22 | 0 | 3 |
| XLS-R Swahili | 26 | 15 | 21 | 35 | 17 | 49 | 9 | 0 | 34 | 18 | 0 | 10 |
| Whisper Large v3 | 21 | 14 | 22 | 30 | 16 | 48 | 11 | 0 | 32 | 7 | 16 | 0 |

AMT denotes monetary-amount errors; CUR currency errors; NAME person-name errors; PROD product or operator errors; DEBT debt/payment terminology errors; KING errors affecting locally characteristic Kingwana forms; FR and EN genuine French and English switch errors; VAR acceptable variants; SEM semantic equivalents; HALL hallucination candidates; and UNCERTAIN unresolved judgments.

Errors involving Kingwana-specific forms occurred in 46--49 of the 50 reviewed outputs for every model, while acceptable variants were also frequent (32--42 outputs). This combination indicates both substantial recognition difficulty and the importance of not treating every orthographic difference as a substantive error. Product/operator errors were most frequent for XLS-R and Whisper in this sample. No English code-switch error was identified.

All 16 hallucination candidates occurred in the deliberately selected high-insertion Whisper stratum; no hallucination candidate was annotated in the 34 background Whisper clips or in the other four systems. Consequently, 16/50 must not be reported as an estimate of corpus-wide hallucination prevalence. The result establishes a specific insertion-heavy Whisper failure mode that warrants audio-confirmed investigation.

## Reproducibility details

| Model | Repository | Evaluated revision | Language/decoding configuration | Precision |
|---|---|---|---|---|
| Whisper Large v3 | `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | `language=swahili; task=transcribe`; one-beam decoding | FP16 |
| SunflowerASR-51 | `Sunbird/SunflowerASR-51-african-languages` | `4583c9dd251c422a4325fb0a0a509a3828097be2` | `language=swa; language_token_id=50318; task=transcribe`; one-beam decoding | FP16 |
| MMS-1B-All | `facebook/mms-1b-all` | `3d33597edbdaaba14a8e858e2c8caa76e3cec0cd` | Swahili adapter/tokenizer `swh`; greedy CTC | FP16 |
| OmniASR CTC 1B | `facebook/omniASR-CTC-1B` | `8c22e3ffdaa4aab6431b128b84b991a7d9c2515c` | evaluation language `swc`; closest supported label `swh_Latn`; no explicit language conditioning; greedy CTC | BF16 |
| XLS-R Swahili | `thinkKenya/wav2vec2-large-xls-r-300m-sw` | `6b8dbc0596fba56750b0b2c1a5db29b62d51af0e` | checkpoint language `sw`, evaluation language `swc`; greedy CTC | FP16 |

Inference used Python 3.11.12, PyTorch 2.8.0, torchaudio 2.8.0, Transformers 4.57.6, and CUDA 12.8 on NVIDIA L4 GPUs. Audio was standardized to mono, 16-kHz PCM WAV while preserving original recordings. Each run recorded the model revision, language configuration, device, precision, package versions, manifest version and hash, and timestamp.

## External-corpus comparison status

The Common Voice comparison was not completed and is not required to report the primary Kingwana business-corpus findings. The available Common Voice Swahili release is distributed as a large monolithic archive, making extraction of only the `sw-kingwana` test subset impractical under the submission deadline. It should therefore be described as future work rather than shown as an incomplete results table.

As a feasible external comparator, all five systems were also evaluated on the TICO-19 Congolese Swahili (`swc`) read-speech test corpus. The comparator contains 2,124 clips from six speakers/sessions and 39,388.240 seconds (10.94 hours) of medical-domain speech. All five prediction files contained exactly 2,124 unique aligned clip identifiers, with no failed or empty predictions. The same deterministic transcript normalization and metric implementation used for the business corpus were retained.

| Model | TICO-19 WER (%) | Business WER (%) | WER difference (pp) | TICO-19 CER (%) | Business CER (%) | CER difference (pp) |
|---|---:|---:|---:|---:|---:|---:|
| OmniASR CTC 1B | **21.25** | 68.63 | +47.39 | **6.87** | 32.50 | +25.63 |
| MMS-1B-All | 22.17 | 74.16 | +51.99 | 6.96 | 34.66 | +27.70 |
| SunflowerASR-51 | 28.79 | 70.51 | +41.72 | 15.66 | 35.96 | +20.30 |
| XLS-R Swahili | 34.50 | 80.21 | +45.71 | 13.07 | 45.64 | +32.57 |
| Whisper Large v3 | 40.53 | 117.70 | +77.17 | 17.18 | 72.34 | +55.16 |

Differences are calculated as business-corpus error rate minus TICO-19 error rate in percentage points. Every model obtained substantially lower WER and CER on TICO-19. These values are descriptive cross-corpus differences, not estimates of domain degradation: TICO-19 and the business corpus differ in domain, speaking style, elicitation procedure, speakers, recording conditions, and utterance duration. In addition, exact training overlap is unverifiable for Whisper and Sunflower; disclosed sources do not document TICO-19 overlap for MMS, OmniASR, or XLS-R, but absence cannot be guaranteed without clip-level training manifests.

For recordings exceeding architecture-specific input limits, the implementation used deterministic long-audio handling while preserving the normal model decoder: OmniASR divided audio longer than 30 seconds into consecutive 30-second chunks and concatenated their greedy CTC transcripts, whereas Whisper enabled its official timestamp-based long-form generation mode above 30 seconds. These settings were recorded in run metadata.

## Concise conclusion

The results do not support deployment-ready Kingwana transactional ASR. OmniASR achieved the best general transcription accuracy, but its 68.63% WER remains high. Field-level results reveal a different ranking: Whisper retained the most exact monetary mentions, whereas OmniASR was strongest for several categorical business entities. The models also showed substantial errors on Kingwana-specific forms, names, products, and payment/debt terminology. These findings support application-specific evaluation and human verification of critical transactional fields rather than reliance on aggregate WER alone.
