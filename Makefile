.PHONY: setup ingest test lint typecheck check eval-retrieval

setup:
	uv sync --frozen

ingest:
	uv run python -m emporio.ingestion

test:
	uv run pytest --cov --cov-report=term

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy --strict src

check: lint typecheck test

eval-retrieval:
	uv run python -m emporio.evals_retrieval
