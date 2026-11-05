"""Adaptador único para qualquer endpoint OpenAI-compatível (Groq, Ollama, OpenAI...)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, cast

import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from emporio.agent.llm.types import (
    LLMError,
    LLMResponse,
    LLMTransientError,
    Message,
    Tier,
    ToolCall,
    Usage,
)
from emporio.settings import Settings

logger = logging.getLogger("emporio.llm")

ChatFactory = Callable[[str, float, int], BaseChatModel]
_PER_MILLION = 1_000_000

_TRANSIENT = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def to_langchain(messages: list[Message]) -> list[BaseMessage]:
    """Converte mensagens no formato OpenAI para o LangChain.

    Args:
        messages: Lista com ``role`` system, user, assistant ou tool.

    Returns:
        Mensagens equivalentes, preservando ``tool_calls`` e ``tool_call_id``.
    """
    converted: list[BaseMessage] = []
    for message in messages:
        content = str(message.get("content") or "")
        match message["role"]:
            case "system":
                converted.append(SystemMessage(content))
            case "assistant":
                calls = [
                    {
                        "id": c["id"],
                        "name": c["function"]["name"],
                        "args": json.loads(c["function"]["arguments"]),
                        "type": "tool_call",
                    }
                    for c in message.get("tool_calls", [])
                ]
                converted.append(AIMessage(content, tool_calls=calls))
            case "tool":
                converted.append(ToolMessage(content, tool_call_id=message["tool_call_id"]))
            case _:
                converted.append(HumanMessage(content))
    return converted


class OpenAICompatibleLLM:
    """Implementa :class:`LLMProvider` sobre ``ChatOpenAI`` com ``base_url`` configurável."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_override: str | None = None,
        chat_factory: ChatFactory | None = None,
    ) -> None:
        """Cria o adaptador.

        Args:
            settings: Configuração (endpoint, chave, modelos, preços, limites).
            model_override: Usa este modelo nos dois tiers (caminho de fallback).
            chat_factory: Fábrica do chat model; injetável para testes.
        """
        self._settings = settings
        self._override = model_override
        self._factory = chat_factory or self._default_factory

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
        message = self._invoke(lambda: runnable.invoke(to_langchain(messages)))
        calls = tuple(
            ToolCall(str(c.get("id") or f"call_{i}"), c["name"], dict(c.get("args") or {}))
            for i, c in enumerate(getattr(message, "tool_calls", None) or [])
        )
        text = message.content if isinstance(message.content, str) else ""
        return LLMResponse(text.strip(), calls, self._usage(message, tier))

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Saída estruturada por function calling, validada por ``schema``."""
        chat = self._chat(tier, temperature)
        runnable: Any = chat.with_structured_output(
            schema, method="function_calling", include_raw=True
        )
        result = self._invoke(lambda: runnable.invoke(to_langchain(messages)))
        parsed = result.get("parsed")
        if not isinstance(parsed, schema):
            logger.warning("saída estruturada inválida: %s", result.get("parsing_error"))
            raise LLMError("O modelo não devolveu a saída estruturada esperada.")
        return parsed, self._usage(result.get("raw"), tier)

    def _chat(self, tier: Tier, temperature: float) -> BaseChatModel:
        settings = self._settings
        model = self._override or (
            settings.llm_model_small if tier == "small" else settings.llm_model_medium
        )
        return self._factory(model, temperature, settings.llm_max_tokens)

    def _default_factory(self, model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        settings = self._settings
        key = settings.llm_api_key or SecretStr("ollama")  # Ollama ignora a chave
        return ChatOpenAI(
            model=model,
            base_url=settings.llm_base_url,
            api_key=key,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retentativa é do ResilientLLM, com circuit breaker
        )

    @staticmethod
    def _invoke(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except _TRANSIENT as exc:
            raise LLMTransientError(type(exc).__name__) from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"provedor recusou a chamada (HTTP {exc.status_code})") from exc

    def _usage(self, message: Any, tier: Tier) -> Usage:
        meta = cast("dict[str, int]", getattr(message, "usage_metadata", None) or {})
        tokens_in, tokens_out = meta.get("input_tokens", 0), meta.get("output_tokens", 0)
        s = self._settings
        price_in, price_out = (
            (s.llm_price_small_input, s.llm_price_small_output)
            if tier == "small"
            else (s.llm_price_medium_input, s.llm_price_medium_output)
        )
        cost = (tokens_in * price_in + tokens_out * price_out) / _PER_MILLION
        return Usage(tokens_in, tokens_out, cost, 1)
