# TICO-19 Training-Overlap Audit

Audit date: 2026-10-18

External evaluation dataset: [`CLEAR-Global/TICO-19-swc-speech`](https://huggingface.co/datasets/CLEAR-Global/TICO-19-swc-speech), config `read`, split `test`.

Pinned dataset revision: `7393279cb0c4633cebe0f58b2796185852aab408`.

The TICO-19 test recordings were collected in 2021 and served as development/evaluation data for
the IWSLT 2021 low-resource speech-translation task. The task prohibited using these recordings
for training within that shared-task setup. This restriction does not prove that later ASR model
developers never included the publicly available corpus.

## Classification criteria

- **No documented overlap:** the disclosed training sources do not name TICO-19 or Gamayun.
  This is evidence against known overlap, not cryptographic proof of absence.
- **Unverifiable:** the available documentation does not enumerate training sources sufficiently
  to decide whether TICO-19 was included.

## Findings

| Model | Exact evaluated revision | Finding | Evidence and interpretation |
|---|---|---|---|
| Whisper Large v3 | `06f233fe06e710322aca913c1bc4249a0d71fce1` | **Unverifiable; possible overlap** | The [model card](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md) reports one million hours of weakly labelled audio plus four million hours of pseudo-labelled audio, but does not publish a source-level manifest. Because TICO-19 was public before Large v3 was trained, exact overlap cannot be excluded. |
| SunflowerASR-51 | `4583c9dd251c422a4325fb0a0a509a3828097be2` | **Unverifiable; possible overlap** | Public model metadata names `huwenjie333/speech-cer-filtered-0.4-full` and `huwenjie333/speech-cer-filtered-0.4-top200-cer-gap` as training datasets and Whisper Large v3 plus `huwenjie333/whisper-v3-ft-af51-0903` as bases. The named source repositories are not publicly inspectable, and the base Whisper data is undisclosed. No TICO-19 claim was found, but absence cannot be verified. |
| MMS-1B-All | `3d33597edbdaaba14a8e858e2c8caa76e3cec0cd` | **No documented overlap; low apparent risk** | The [MMS paper](https://arxiv.org/abs/2305.13516) documents multi-domain supervised training on MMS-lab, FLEURS, Common Voice, VoxPopuli and MLS. TICO-19 and Gamayun are not named. The evaluated Swahili adapter is `swh`, not a separately documented `swc` adapter. |
| OmniASR CTC 1B | `8c22e3ffdaa4aab6431b128b84b991a7d9c2515c` | **No documented overlap; low-to-moderate apparent risk** | The [Omnilingual ASR paper](https://arxiv.org/abs/2511.09690) lists public supervised sources including ALFFA, Common Voice 22, FLEURS, MMS-lab, MLS and other corpora, but does not name TICO-19 or Gamayun. Its language table reports `swh_Latn` with 461.8 training hours and does not list `swc_Latn`. The very broad mixed training corpus prevents a stronger absence claim without a clip-level manifest. |
| XLS-R Swahili | `6b8dbc0596fba56750b0b2c1a5db29b62d51af0e` | **No documented fine-tuning overlap; low apparent risk** | The [checkpoint card](https://huggingface.co/thinkKenya/wav2vec2-large-xls-r-300m-sw/blob/6b8dbc0596fba56750b0b2c1a5db29b62d51af0e/README.md) declares fine-tuning on Common Voice 11.0 Swahili. It does not name TICO-19 or Gamayun. The underlying XLS-R pretraining data is broader, so this finding is limited to disclosed sources rather than guaranteed clip-level exclusion. |

## Reporting consequence

TICO-19 results may be reported for all five models, but the corpus must not be described as a
guaranteed unseen benchmark. Results for Whisper and Sunflower require an explicit
training-overlap-unknown caveat. The comparison should be called an **external cross-corpus
comparison**, not domain degradation: TICO-19 is translated medical read speech, while the study
corpus is natural scenario-guided small-business speech.
