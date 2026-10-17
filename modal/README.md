# Modal GPU backend

[`run_benchmark.py`](run_benchmark.py) is a thin GPU wrapper around the same
`kingwana_asr.inference.run_transcription` function used locally. It does not contain a second ASR
implementation. All five primary model adapters use this same backend.

## Privacy boundary

Using `--upload` transfers the selected private manifest, its transcripts, pseudonymous IDs, and
only the audio files referenced by that manifest from this PC to Modal-managed infrastructure.
Without `--upload`, the script transfers no research data and expects the named dataset to exist
already. Do not upload the full evaluation corpus merely to test the pipeline. Modal credentials
and secrets are never created by this repository.

The default names can be overridden with `KINGWANA_MODAL_DATA_VOLUME`,
`KINGWANA_MODAL_OUTPUT_VOLUME`, and `KINGWANA_MODAL_CACHE_VOLUME`. Model functions attach the
Modal secret named `Speech2sql_project`. It must contain an `HF_TOKEN` key; the repository never
stores or prints its value.

## One-time setup

Install/authenticate the Modal client yourself, then create three persistent volumes:

```bash
modal volume create kingwana-asr-private-data
modal volume create kingwana-asr-outputs
modal volume create kingwana-asr-hf-cache
```

The local environment currently has Modal client 1.5.1 and an active profile. No Hugging Face
secret is needed for public benchmark checkpoints. SunflowerASR-51 is gated: first obtain access
to its Hugging Face repository, then create the `Speech2sql_project` Modal secret containing a
read-only `HF_TOKEN`. Run it without writing the token into this repository:

```bash
modal run modal/run_benchmark.py --model sunflower_asr_51 \
  --dataset-key preflight-v0.1
```

## One-clip preflight

Review `data/manifests/smoke_v0.1_1clip.csv` before explicitly approving its transfer.
This command uploads exactly one standardized WAV file and the private manifest, builds the
GPU image, downloads the public model into the persistent cache, and runs the shared resumable
pipeline on an L4 GPU:

```bash
modal run modal/run_benchmark.py --upload --model whisper_large_v3 \
  --manifest data/manifests/smoke_v0.1_1clip.csv \
  --dataset-key preflight-v0.1
```

Rerun without `--upload`; completed `model_id + clip_id` pairs will be skipped:

```bash
modal run modal/run_benchmark.py --model whisper_large_v3 \
  --dataset-key preflight-v0.1
```

Outputs persist in `kingwana-asr-outputs`; the Hugging Face cache persists separately. Retrieve
the model directory after a run without printing private transcripts in the terminal:

```bash
modal volume get kingwana-asr-outputs whisper_large_v3 outputs/whisper_large_v3
```

For long runs, `--detach` keeps the remote function alive if the local WSL terminal disconnects:

```bash
modal run --detach modal/run_benchmark.py --model whisper_large_v3 \
  --manifest data/manifests/evaluation_v0.1_standardized.csv \
  --dataset-key evaluation-v0.1
```

An inactive app with zero live containers is normal after an ephemeral run completes. Completion
must be verified from `run_metadata.json` and full manifest/prediction alignment, not from dashboard
activity alone.

The data volume continues to contain participant audio after the job. Retention and deletion are
research-governance decisions; this repository never deletes remote data automatically.
