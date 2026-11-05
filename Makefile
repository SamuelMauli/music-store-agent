.PHONY: setup ingest chat api test lint typecheck check eval-retrieval eval eval-replay eval-live examples

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

# offline (CI): LLM fake, todos os casos; falha abaixo do piso de qualidade
eval:
	uv run python -m emporio.evals --mode fake

# offline: repete o cassete gravado pelo eval-live (casos com live: true)
eval-replay:
	uv run python -m emporio.evals --mode replay

# provedor real (EMPORIO_LLM_*): subconjunto live, grava cassete e relatorio em evals/reports/
# throttle p/ plano gratuito: EMPORIO_EVAL_SLEEP=62 make eval-live
eval-live:
	uv run python -m emporio.evals --mode live

chat:
	uv run python -m emporio.interfaces.cli $(ARGS)

api:
	uv run uvicorn --factory emporio.interfaces.api:create_app --host 127.0.0.1 --port 8000

# transcricoes reais (EMPORIO_LLM_*) em examples/; plano gratuito: EMPORIO_EVAL_SLEEP=62 make examples
examples:
	uv run python -m emporio.evals.examples
