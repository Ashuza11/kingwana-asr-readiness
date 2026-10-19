# Manual review protocol

The fixed manual sample contains 50 clips and remains exactly balanced at five
clips per speaker and ten clips per scenario (seed 42). It includes all 16 clips
where Whisper produced at least 50 word insertions, plus 34 deterministically
sampled background clips that fill the speaker and scenario quotas. This
purposeful high-insertion stratum supports hallucination analysis; proportions
from the 50 clips must not be treated as corpus-prevalence estimates. Every
selected clip is reviewed against all five model predictions, producing 250
comparison rows.

Use `yes`, `no`, `not_applicable`, or `uncertain` in each manual category. Set
`manual_review_status` to `complete` only after every category has been assessed.

Categories:

- `monetary_amount_error`: a spoken amount is missing or changed.
- `currency_error`: a spoken currency is missing or changed.
- `person_name_error`: an explicitly spoken person name is missing or changed.
- `product_operator_error`: a product, service, or telecom operator is missing or changed.
- `debt_payment_terminology_error`: debt, repayment, balance, or payment meaning is changed.
- `kingwana_specific_form_error`: a Kingwana lexical or grammatical form is mistranscribed.
- `french_code_switch_error`: explicitly switched French material is mistranscribed.
- `english_code_switch_error`: explicitly switched English material is mistranscribed.
- `acceptable_variant`: the output differs orthographically but remains acceptable.
- `semantic_equivalent`: different wording preserves the intended meaning.
- `hallucination`: unsupported content is added by the model.

Do not revise the gold transcript based on a model prediction. If the gold itself
appears questionable, note it separately and return it to the transcript-review
process. Automatic WER, CER, substitution, deletion, and insertion counts use the
same deterministic normalization as the main benchmark.

After all 250 rows are complete, validate and summarize them with:

```bash
python scripts/summarize_manual_review.py \
  --worksheet data/private/reviews/manual_review_50x5_v0.1.csv \
  --output reports/tables/manual_error_summary.csv
```

The summary is reported separately for the balanced background and targeted
Whisper high-insertion strata. Do not use the combined hallucination percentage
as an estimate of hallucination prevalence in the full 500-clip corpus.
