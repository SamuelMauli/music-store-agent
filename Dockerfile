# syntax=docker/dockerfile:1
# Imagem da API (SSE em /v1/chat). Estágio 1 resolve dependências com uv; o estágio final
# leva só o venv e os dados, roda sem root e já traz o banco gerado por `make ingest`.

# Bases fixadas por digest do índice multi-arch (obtido via HEAD no registry, sem baixar a imagem).
# Para atualizar: troque o digest após revisar a nova versão da tag.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Camada de dependências: só muda quando uv.lock ou pyproject.toml mudam.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Camada do projeto, instalada como pacote (sem editable): o código-fonte não vai ao runtime.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd --system --gid 10001 emporio \
 && useradd --system --uid 10001 --gid emporio --home-dir /app --shell /usr/sbin/nologin emporio
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY data/raw ./data/raw
COPY data/contracts ./data/contracts

# Bronze -> silver -> gold + relatório de qualidade. O banco é artefato reprodutível,
# então nasce no build e a imagem sobe pronta, sem passo manual.
RUN python -m emporio.ingestion \
 && chown -R emporio:emporio /app/data

USER emporio

# Memória de sessão e chamados (SQLite) ficam em /app/data: monte um volume para persistir.
VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["uvicorn", "--factory", "emporio.interfaces.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
