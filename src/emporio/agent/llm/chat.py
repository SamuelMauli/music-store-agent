"""Adaptador único de :class:`LLMProvider` sobre qualquer ``BaseChatModel`` do LangChain."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from emporio.agent.llm import providers
from emporio.agent.llm.errors import classify
from emporio.agent.llm.messages import to_langchain
from emporio.agent.llm.spec import ModelSpec, parse_spec
from emporio.agent.llm.types import LLMError, LLMResponse, Message, Tier, ToolCall, Usage
from emporio.settings import Settings

logger = logging.getLogger("emporio.llm")

ChatFactory = Callable[[str, float, int], BaseChatModel]
_PER_MILLION = 1_000_000


class LangChainLLM:
    """Implementa :class:`LLMProvider` para qualquer provedor registrado.

    Cada tier tem sua spec (``provedor:modelo``), então o roteador pode rodar num provedor
    barato e a resposta noutro. Erros do SDK viram ``LLMTransientError``/``LLMError``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        model_override: str | ModelSpec | None = None,
        chat_factory: ChatFactory | None = None,
    ) -> None:
        """Cria o adaptador.

        Args:
            settings: Configuração (specs, credenciais, preços, limites).
            model_override: Spec usada nos dois tiers (um elo da cadeia de fallback).
            chat_factory: Fábrica ``(modelo, temperatura, max_tokens)``; injetável em testes.
        """
        self._settings = settings
        default = settings.llm_provider
        if isinstance(model_override, str):
            model_override = parse_spec(model_override, default)
        self._specs: dict[Tier, ModelSpec] = {
            "small": model_override or parse_spec(settings.llm_model_small, default),
            "medium": model_override or parse_spec(settings.llm_model_medium, default),
        }
        self._factory = chat_factory

    @property
    def name(self) -> str:
        """Rótulo para logs (``provedor:modelo`` do tier medium)."""
        return self._specs["medium"].label

    def spec(self, tier: Tier) -> ModelSpec:
        """Spec usada no tier."""
        return self._specs[tier]

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Chama o modelo do tier, com tools se informadas."""
        chat = self._chat(tier, temperature)
        runnable: Any = chat.bind_tools(tools) if tools else chat
        prompt = self._convert(messages, tier)
        message = self._invoke(lambda: runnable.invoke(prompt))
        calls = tuple(
            ToolCall(str(c.get("id") or f"call_{i}"), c["name"], dict(c.get("args") or {}))
            for i, c in enumerate(getattr(message, "tool_calls", None) or [])
        )
        return LLMResponse(text_of(message).strip(), calls, self._usage(message, tier))

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Saída estruturada com o ``method`` adequado ao provedor, validada por ``schema``.

        Provedor sem ``with_structured_output`` cai para tool calling forçado.
        """
        chat = self._chat(tier, temperature)
        prompt = self._convert(messages, tier)
        try:
            runnable: Any = chat.with_structured_output(
                schema, method=self._method(tier), include_raw=True
            )
        except NotImplementedError:
            return self._structured_via_tools(chat, prompt, schema, tier)
        result = self._invoke(lambda: runnable.invoke(prompt))
        parsed = result.get("parsed")
        if not isinstance(parsed, schema):
            logger.warning("saída estruturada inválida: %s", type(result.get("parsing_error")))
            raise LLMError("O modelo não devolveu a saída estruturada esperada.")
        return parsed, self._usage(result.get("raw"), tier)

    def _structured_via_tools[T: BaseModel](
        self, chat: BaseChatModel, prompt: list[BaseMessage], schema: type[T], tier: Tier
    ) -> tuple[T, Usage]:
        runnable: Any = chat.bind_tools([schema], tool_choice="any")
        message = self._invoke(lambda: runnable.invoke(prompt))
        calls = getattr(message, "tool_calls", None) or []
        try:
            parsed = schema.model_validate(calls[0]["args"])
        except (IndexError, KeyError, ValidationError) as exc:
            logger.warning("saída estruturada inválida (tools): %s", type(exc).__name__)
            raise LLMError("O modelo não devolveu a saída estruturada esperada.") from exc
        return parsed, self._usage(message, tier)

    def _method(self, tier: Tier) -> providers.StructuredMethod:
        if self._factory is not None:
            return "function_calling"
        if self._settings.llm_structured_method != "auto":
            return self._settings.llm_structured_method
        return providers.info_for(self._specs[tier].provider).structured_method

    def _convert(self, messages: list[Message], tier: Tier) -> list[BaseMessage]:
        provider = self._specs[tier].provider
        hoist = providers.REGISTRY.get(provider, providers.REGISTRY["openai"]).hoist_system
        return to_langchain(messages, hoist_system=hoist)

    def _chat(self, tier: Tier, temperature: float) -> BaseChatModel:
        spec, max_tokens = self._specs[tier], self._settings.llm_max_tokens
        if self._factory is not None:
            return self._factory(spec.model, temperature, max_tokens)
        return providers.build_chat(
            spec, self._settings, temperature=temperature, max_tokens=max_tokens
        )

    @staticmethod
    def _invoke(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except Exception as exc:
            raise classify(exc) from exc

    def _usage(self, message: Any, tier: Tier) -> Usage:
        meta = cast("dict[str, int]", getattr(message, "usage_metadata", None) or {})
        tokens_in, tokens_out = meta.get("input_tokens", 0), meta.get("output_tokens", 0)
        spec = self._specs[tier]
        key = spec.label if spec.label in self._settings.llm_price_table else spec.model
        price_in, price_out = self._settings.price_for(key, tier)
        cost = (tokens_in * price_in + tokens_out * price_out) / _PER_MILLION
        return Usage(tokens_in, tokens_out, cost, 1)


def text_of(message: Any) -> str:
    """Texto da resposta, seja ``content`` string (OpenAI) ou lista de blocos (Claude, Gemini).

    Args:
        message: ``AIMessage`` (ou ``None``).

    Returns:
        Só os blocos de texto concatenados; ``tool_use``/``thinking`` ficam de fora.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block if isinstance(block, str) else str(block.get("text") or "")
        for block in content
        if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
    ]
    return "".join(parts)
