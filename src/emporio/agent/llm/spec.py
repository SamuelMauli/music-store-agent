"""Especificação de modelo ``provedor:modelo`` e papéis configurados (small, medium, reservas)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

from emporio.settings import LLMProviderName, Settings

PROVIDERS: frozenset[str] = frozenset(get_args(LLMProviderName))


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Modelo concreto de um provedor.

    Attributes:
        provider: Nome registrado do provedor (``anthropic``, ``openai``...).
        model: Nome do modelo como o provedor o conhece (pode conter ``/`` e ``:``).
    """

    provider: str
    model: str

    @property
    def label(self) -> str:
        """Forma canônica ``provedor:modelo``."""
        return f"{self.provider}:{self.model}"


def parse_spec(raw: str, default_provider: str) -> ModelSpec:
    """Interpreta ``provedor:modelo``; sem prefixo conhecido, usa o provedor padrão.

    Só um provedor registrado conta como prefixo: ``qwen2.5:7b`` e ``openai/gpt-oss-120b``
    continuam sendo nomes de modelo do provedor padrão (retrocompatível com o ``.env``).

    Args:
        raw: Texto da configuração.
        default_provider: Provedor das specs sem prefixo (``EMPORIO_LLM_PROVIDER``).

    Returns:
        A spec resolvida.

    Raises:
        ValueError: Se a spec ficar sem nome de modelo.
    """
    text = raw.strip()
    prefix, sep, rest = text.partition(":")
    if sep and prefix in PROVIDERS:
        provider, model = prefix, rest.strip()
    else:
        provider, model = default_provider, text
    if not model:
        raise ValueError(f"spec de modelo sem nome de modelo: {raw!r}")
    return ModelSpec(provider, model)


def role_specs(settings: Settings) -> dict[str, ModelSpec]:
    """Specs por papel, na ordem de uso: ``small``, ``medium`` e ``fallback[i]``.

    Args:
        settings: Configuração (``EMPORIO_LLM_MODEL_*`` e ``EMPORIO_LLM_FALLBACKS``).

    Returns:
        Mapa papel → spec.
    """
    default = settings.llm_provider
    roles = {
        "small": parse_spec(settings.llm_model_small, default),
        "medium": parse_spec(settings.llm_model_medium, default),
    }
    for index, spec in enumerate(fallback_specs(settings)):
        roles[f"fallback[{index}]"] = spec
    return roles


def fallback_specs(settings: Settings) -> list[ModelSpec]:
    """Cadeia de reservas: ``llm_fallbacks`` e, ao fim, o alias legado ``llm_model_fallback``.

    Args:
        settings: Configuração.

    Returns:
        Specs sem repetição, na ordem configurada.
    """
    raws = [*settings.llm_fallbacks]
    if settings.llm_model_fallback:
        raws.append(settings.llm_model_fallback)
    specs: list[ModelSpec] = []
    for raw in raws:
        spec = parse_spec(raw, settings.llm_provider)
        if spec not in specs:
            specs.append(spec)
    return specs
