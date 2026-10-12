.PHONY: install test lint validate

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .

validate:
	uv run python -m kingwana_asr.cli validate --manifest data/manifests/evaluation_v0.1.csv
