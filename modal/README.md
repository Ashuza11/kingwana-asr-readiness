# Modal backend (planned Phase 6)

Modal support is intentionally not implemented in Phase 1. It will be a thin execution layer over
the installed `kingwana_asr` package, not a separate benchmark.

Using Modal would transfer any explicitly mounted evaluation audio and manifest data from the
local PC to Modal-managed infrastructure. No participant audio will be uploaded, mounted, or
configured automatically. The researcher must make and document that data-governance decision,
configure credentials themselves, and select persistent output/model-cache storage.
