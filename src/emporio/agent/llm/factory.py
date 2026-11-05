"""Fábrica do provedor de LLM a partir da configuração (``EMPORIO_LLM_*``)."""

from __future__ import annotations

from collections.abc import Iterable

from emporio.agent.llm import providers
from emporio.agent.llm.chat import LangChainLLM
from emporio.agent.llm.fake import FakeLLM
from emporio.agent.llm.resilient import ResilientLLM
from emporio.agent.llm.spec import fallback_specs, role_specs
from emporio.agent.llm.types import LLMProvider
from emporio.settings import Settings


def validate(settings: Settings) -> None:
    """Valida no boot todos os papéis (small, medium, reservas).

    Args:
        settings: Configuração.

    Raises:
        providers.LLMConfigError: Provedor desconhecido, extra ausente ou credencial faltando.
    """
    for spec in role_specs(settings).values():
        providers.validate(spec, settings)


def build(settings: Settings, catalog_names: Iterable[str] = ()) -> LLMProvider:
    """Monta o provedor configurado.

    Args:
        settings: Configuração; ``llm_provider=fake`` usa o fake offline. Os demais montam a
            cadeia ``principal → EMPORIO_LLM_FALLBACKS`` (specs ``provedor:modelo``).
        catalog_names: Nomes de produtos, usados só pelo fake para reconhecer marcas.

    Returns:
        ``FakeLLM`` ou ``ResilientLLM`` sobre a cadeia de adaptadores.

    Raises:
        providers.LLMConfigError: (um ``ValueError``) se algum papel estiver mal configurado.
    """
    if settings.llm_provider == "fake":
        return FakeLLM(catalog_names)
    validate(settings)
    reserves = [LangChainLLM(settings, model_override=s) for s in fallback_specs(settings)]
    return ResilientLLM(LangChainLLM(settings), fallbacks=reserves)
