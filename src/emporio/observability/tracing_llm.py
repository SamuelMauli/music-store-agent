"""Decorator de LLMProvider que emite um span GenAI por chamada ao modelo."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from emporio.agent.llm.spec import provider_label
from emporio.agent.llm.types import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    Tier,
    Usage,
)
from emporio.observability.telemetry import AttributeValue, Telemetry
from emporio.settings import Settings


class TracingLLM:
    """Registra modelo, tokens e custo (nunca o conteúdo das mensagens, que pode ter PII)."""

    def __init__(self, inner: LLMProvider, telemetry: Telemetry, settings: Settings) -> None:
        """Cria o decorator.

        Args:
            inner: Provedor decorado.
            telemetry: Fachada de tracing.
            settings: Para resolver o modelo por tier e o nome do provedor.
        """
        self._inner = inner
        self._telemetry = telemetry
        self._settings = settings

    def _model(self, tier: Tier) -> str:
        if self._settings.llm_provider == "fake":
            return "fake"
        s = self._settings
        return s.llm_model_small if tier == "small" else s.llm_model_medium

    def _provider(self, tier: Tier) -> str:
        return provider_label(self._settings, tier)

    def _span(self, tier: Tier, temperature: float) -> Any:
        model = self._model(tier)
        attributes: dict[str, AttributeValue] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": self._provider(tier),
            "gen_ai.request.model": model,
            "gen_ai.request.temperature": temperature,
            "emporio.tier": tier,
        }
        return self._telemetry.span(f"chat {model}", **attributes)

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Chat com span GenAI."""
        with self._span(tier, temperature) as span:
            span.set_attribute("emporio.tools_offered", len(tools or []))
            try:
                response = self._inner.complete(
                    messages, tier=tier, temperature=temperature, tools=tools
                )
            except LLMError as exc:
                span.set_attribute("error.type", type(exc).__name__)
                raise
            self._usage(span, response.usage)
            span.set_attribute("emporio.tool_calls", ",".join(c.name for c in response.tool_calls))
            return response

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Saída estruturada com span GenAI."""
        with self._span(tier, temperature) as span:
            span.set_attribute("emporio.schema", schema.__name__)
            try:
                parsed, usage = self._inner.structured(
                    messages, schema, tier=tier, temperature=temperature
                )
            except LLMError as exc:
                span.set_attribute("error.type", type(exc).__name__)
                raise
            self._usage(span, usage)
            return parsed, usage

    @staticmethod
    def _usage(span: Any, usage: Usage) -> None:
        span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        span.set_attribute("emporio.cost_usd", usage.cost_usd)
