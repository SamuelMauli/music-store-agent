.PHONY: setup ingest chat api test lint typecheck check eval-retrieval

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

chat:
	uv run python -m emporio.interfaces.cli $(ARGS)

api:
	uv run uvicorn --factory emporio.interfaces.api:create_app --host 127.0.0.1 --port 8000
