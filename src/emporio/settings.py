"""Configuração por ambiente (12-factor) via pydantic-settings."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_STORE_TZ = "America/Campo_Grande"
POLICY_PDF_NAME = "politicas_da_loja.pdf"


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
    use_system_clock: bool = False
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
