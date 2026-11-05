"""Configuração por ambiente (12-factor) via pydantic-settings."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo

from pydantic import BaseModel, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_STORE_TZ = "America/Campo_Grande"
POLICY_PDF_NAME = "politicas_da_loja.pdf"


class ModelPrice(BaseModel):
    """Preço de um modelo em US$ por milhão de tokens."""

    input: float
    output: float


class Settings(BaseSettings):
    """Configurações do sistema; todas sobrescrevíveis por ``EMPORIO_*``.

    Attributes:
        db_path: Arquivo DuckDB com as camadas bronze/silver/gold.
        raw_dir: Diretório com os CSVs originais (imutáveis).
        contracts_dir: Diretório com os data contracts ODCS.
        quality_report_path: Destino do relatório de qualidade gerado.
        timezone: Fuso IANA da loja.
        metro_cities: Cidades da região metropolitana de Campo Grande (§5.1); sem lista
            oficial no manual, o default é só a capital (``EMPORIO_METRO_CITIES`` em JSON).
        policy_retriever: ``long_context`` (manual inteiro) ou ``hybrid`` (BM25 + denso + RRF).
        reference_datetime: "Agora" simulado; os dados vão de out/2025 a mar/2026.
        checkpoint_path: SQLite com a memória por sessão (checkpointer do LangGraph).
        tickets_path: SQLite dos chamados (somente anexação).
        rate_limit_per_minute: Mensagens por sessão por minuto (CLI e API).
        rate_limit_per_ip_per_minute: Mensagens por IP de cliente por minuto (API).
        trusted_proxy_hops: Proxies confiáveis na frente da API; com N>0 o IP do cliente é o
            N-ésimo da direita em ``X-Forwarded-For`` (0 = ignora o cabeçalho).
        session_secret: Segredo HMAC dos ids de sessão da API (``EMPORIO_SESSION_SECRET``);
            ausente, gera um aleatório por processo (sessões não sobrevivem a restart).
        verification_ttl_seconds: Validade da verificação de cliente na sessão.
        verify_max_attempts_per_target: Falhas por pedido e por contato antes do bloqueio.
        verify_max_attempts_per_ip: Falhas de verificação por IP antes do bloqueio.
        verify_lockout_seconds: Duração do bloqueio por alvo/IP (TTL das falhas).
        use_system_clock: Usa o relógio real em vez de ``reference_datetime``.
        llm_provider: ``openai_compatible`` (Groq, Ollama, OpenAI...) ou ``fake`` (offline).
        llm_base_url: Endpoint OpenAI-compatível.
        llm_api_key: Chave do provedor; nunca logada.
        llm_model_small: Modelo do roteador, das guardas e do resumo.
        llm_model_medium: Modelo do nó de resposta.
        llm_model_fallback: Modelo alternativo quando o principal falha (opcional).
        llm_timeout_seconds: Limite por chamada ao provedor.
        llm_max_tokens: Teto de tokens de saída por chamada.
        llm_price_*: Preço em US$ por milhão de tokens, usado só para estimar custo do turno.
        llm_price_table: Preço por nome de modelo (``EMPORIO_LLM_PRICE_TABLE`` em JSON, ex.:
            ``{"openai/gpt-oss-20b": {"input": 0.075, "output": 0.3}}``); vence o preço do tier.
        eval_sleep: Segundos entre turnos no ``make eval-live`` (limite de tokens por minuto).
        eval_judge_model: Modelo do LLM-juiz (família diferente da do gerador).
        otel_exporter: ``none`` (no-op), ``console``, ``jsonl`` (arquivo) ou ``otlp``.
        otel_jsonl_path: Destino do exportador ``jsonl`` (ignorado no git).
        otel_endpoint: Endpoint OTLP/HTTP dos traces (exige o pacote do exportador OTLP).
        env: ``dev`` ou ``prod``; em ``prod`` a aplicação recusa subir com configuração
            insegura ou que não escala (ver ``emporio.production``).
        state_backend: ``sqlite`` (processo único) ou ``postgres`` (memória de sessão e
            chamados compartilhados entre réplicas).
        database_url: DSN do Postgres (``postgresql://...``) quando ``state_backend=postgres``.
        redis_url: URL do Redis; presente, limites de taxa e tentativas de verificação ficam
            no Redis (valem entre réplicas). Ausente, ficam em memória do processo.
        api_keys: Chaves aceitas na API (``Authorization: Bearer`` ou ``X-API-Key``); mais de
            uma permite rotação sem downtime. Vazio em ``dev`` = API aberta.
        log_format: ``text`` (humano) ou ``json`` (uma linha por evento, com ``request_id``).
        log_level: Nível do logger raiz.
        metrics_enabled: Expõe ``GET /metrics`` no formato Prometheus.
        cors_origins: Origens liberadas para CORS (vazio = CORS desligado).
        max_message_chars: Tamanho máximo de uma mensagem do cliente.
    """

    model_config = SettingsConfigDict(env_prefix="EMPORIO_", env_file=".env", extra="ignore")

    db_path: Path = Path("data/emporio.duckdb")
    raw_dir: Path = Path("data/raw")
    contracts_dir: Path = Path("data/contracts")
    quality_report_path: Path = Path("data/quality_report.json")
    timezone: str = _STORE_TZ
    metro_cities: list[str] = ["Campo Grande"]
    policy_retriever: Literal["long_context", "hybrid"] = "long_context"
    reference_datetime: datetime = datetime(2026, 3, 25, 10, 0, tzinfo=ZoneInfo(_STORE_TZ))
    checkpoint_path: Path = Path("data/checkpoints.sqlite")
    tickets_path: Path = Path("data/tickets.sqlite")
    use_system_clock: bool = False
    rate_limit_per_minute: int = 20
    rate_limit_per_ip_per_minute: int = 60
    trusted_proxy_hops: int = 0
    session_secret: SecretStr | None = None
    verification_ttl_seconds: int = 900
    verify_max_attempts_per_target: int = 5
    verify_max_attempts_per_ip: int = 10
    verify_lockout_seconds: int = 900
    llm_provider: Literal["openai_compatible", "fake"] = "openai_compatible"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: SecretStr | None = None
    llm_model_small: str = "openai/gpt-oss-20b"
    llm_model_medium: str = "openai/gpt-oss-120b"
    llm_model_fallback: str | None = None
    llm_timeout_seconds: float = 20.0
    llm_max_tokens: int = 700
    llm_price_small_input: float = 0.075
    llm_price_small_output: float = 0.30
    llm_price_medium_input: float = 0.15
    llm_price_medium_output: float = 0.60
    llm_price_table: dict[str, ModelPrice] = {}
    eval_sleep: float = 0.0
    eval_judge_model: str | None = None
    otel_exporter: Literal["none", "console", "jsonl", "otlp"] = "none"
    otel_jsonl_path: Path = Path("data/traces.jsonl")
    otel_endpoint: str = "http://localhost:4318/v1/traces"
    # --- Produção: perfil, estado compartilhado e operação --------------------------------
    env: Literal["dev", "prod"] = "dev"
    state_backend: Literal["sqlite", "postgres"] = "sqlite"
    database_url: SecretStr | None = None
    redis_url: SecretStr | None = None
    api_keys: list[SecretStr] = []
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"
    metrics_enabled: bool = True
    cors_origins: list[str] = []
    max_message_chars: int = 2000

    def price_for(self, model: str, tier: Literal["small", "medium"]) -> tuple[float, float]:
        """Preço (entrada, saída) por milhão de tokens: tabela por modelo, senão o do tier."""
        if (entry := self.llm_price_table.get(model)) is not None:
            return entry.input, entry.output
        if tier == "small":
            return self.llm_price_small_input, self.llm_price_small_output
        return self.llm_price_medium_input, self.llm_price_medium_output

    @property
    def policy_pdf(self) -> Path:
        """PDF do manual de políticas, dentro de ``raw_dir``."""
        return self.raw_dir / POLICY_PDF_NAME

    @model_validator(mode="after")
    def _localize_reference(self) -> Self:
        if self.reference_datetime.tzinfo is None:
            self.reference_datetime = self.reference_datetime.replace(
                tzinfo=ZoneInfo(self.timezone)
            )
        return self
