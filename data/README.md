# Private evaluation data

Human voice recordings and participant metadata are private research data. Put original files
under `data/private/` only when local storage is appropriate, or reference an approved external
path from the manifest. Prepared derivatives belong in `data/processed/`. Both directories are
ignored by Git except for their `.gitkeep` files.

Never add participant names, phone numbers, consent records, credentials, or unapproved exports
to this repository. A manifest committed to Git must contain pseudonymous speaker IDs and no
identifying notes.

Audio preparation will be implemented before inference. It will preserve originals and produce
16 kHz, mono, PCM WAV derivatives without overwriting source recordings.
