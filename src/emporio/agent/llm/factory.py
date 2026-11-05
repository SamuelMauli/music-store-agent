"""Fábrica do provedor de LLM a partir da configuração (``EMPORIO_LLM_*``)."""

from __future__ import annotations

from collections.abc import Iterable

from emporio.agent.llm.fake import FakeLLM
from emporio.agent.llm.openai_compat import OpenAICompatibleLLM
from emporio.agent.llm.resilient import ResilientLLM
from emporio.agent.llm.types import LLMProvider
from emporio.settings import Settings


def build(settings: Settings, catalog_names: Iterable[str] = ()) -> LLMProvider:
    """Monta o provedor configurado.

    Args:
        settings: Configuração; ``llm_provider`` escolhe entre OpenAI-compatível e fake.
        catalog_names: Nomes de produtos, usados só pelo fake para reconhecer marcas.

    Returns:
        ``FakeLLM`` ou ``ResilientLLM`` sobre o adaptador OpenAI-compatível.

    Raises:
        ValueError: Se o provedor real for pedido sem chave para endpoint que a exige.
    """
    if settings.llm_provider == "fake":
        return FakeLLM(catalog_names)
    local = settings.llm_base_url.startswith(("http://localhost", "http://127.0.0.1"))
    if settings.llm_api_key is None and not local:
        raise ValueError("EMPORIO_LLM_API_KEY é obrigatório para este endpoint.")
    primary = OpenAICompatibleLLM(settings)
    fallback = (
        OpenAICompatibleLLM(settings, model_override=settings.llm_model_fallback)
        if settings.llm_model_fallback
        else None
    )
    return ResilientLLM(primary, fallback)
