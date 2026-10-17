# Private evaluation data

Human voice recordings and participant metadata are private research data. Put original files
under `data/private/` only when local storage is appropriate, or reference an approved external
path from the manifest. Prepared derivatives belong in `data/processed/`. Both directories are
ignored by Git except for their `.gitkeep` files.

Never add participant names, phone numbers, consent records, credentials, or unapproved exports
to this repository. A manifest committed to Git must contain pseudonymous speaker IDs and no
identifying notes.

Audio preparation preserves originals and produces 16 kHz, mono, PCM WAV derivatives without
overwriting source recordings. Generated manifests containing real transcripts are also ignored
by Git until an explicit data-release decision is made.

The locally verified 2026-10-16 SautiForge pilot export contains 500 recordings from 10
pseudonymous speakers across five scenarios. Its outer archive SHA-256 is
`81b3ea95d78acd1091e277e35d2a64c28b276016dcaed4968a65ebdd182f9d66`. The archive and its
extracted files remain under `data/private/`; the checksum documents local provenance and does
not authorize publication. The export's supplied normalized annotations are preserved unchanged.
For benchmark scoring, including the 17 rows without a supplied normalized value,
`reference_eval` is deterministically derived from the corresponding `reference_verbatim`.
