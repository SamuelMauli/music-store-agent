.PHONY: train-guard llm-check docker-build compose-up compose-down setup setup-min ingest chat api test lint typecheck check eval-retrieval eval eval-replay eval-live examples

# ambiente de desenvolvimento = ambiente do CI (todos os extras): make check passa sem ajuste
setup:
	uv sync --frozen --all-extras

# só a dependência base (runtime enxuto); provedores e estado compartilhado ficam de fora
setup-min:
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

# imagem local (EXTRAS=all inclui provedores, postgres e redis; EXTRAS=postgres para enxuta)
docker-build:
	docker build --build-arg EXTRAS=$(or $(EXTRAS),all) -t emporio-api:local .

# stack completa (api+postgres+redis+otel); requer deploy/.env (cp deploy/.env.example deploy/.env)
# com Jaeger: make compose-up PROFILE=observability
compose-up:
	docker compose -f deploy/docker-compose.yml --env-file deploy/.env $(if $(PROFILE),--profile $(PROFILE)) up -d --build

compose-down:
	docker compose -f deploy/docker-compose.yml --env-file deploy/.env $(if $(PROFILE),--profile $(PROFILE)) down

# diagnostico do LLM: credencial + chamada real (tool calling e saida estruturada) por papel
llm-check:
	uv run python -m emporio.agent.llm.doctor

# guarda de injecao: regenera evals/guard/{train,test}.jsonl, treina (fastembed + logistica),
# grava src/emporio/guardrails/injection_model.json e evals/guard/metrics.json
train-guard:
	uv run python -m emporio.evals.guard_train
