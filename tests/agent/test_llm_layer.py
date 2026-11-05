from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel
from tenacity import wait_none

from emporio.agent.intents import Intent, RouteDecision
from emporio.agent.llm import factory
from emporio.agent.llm.fake import FakeLLM
from emporio.agent.llm.openai_compat import OpenAICompatibleLLM, to_langchain
from emporio.agent.llm.resilient import CircuitBreaker, ResilientLLM
from emporio.agent.llm.types import (
    LLMError,
    LLMResponse,
    LLMTransientError,
    LLMUnavailableError,
    Message,
    Tier,
    ToolCall,
    Usage,
    assistant_message,
    tool_message,
)
from emporio.settings import Settings

MSGS: list[Message] = [{"role": "user", "content": "oi"}]


class Scripted:
    """Provedor que executa uma sequência de resultados ou exceções."""

    def __init__(self, *steps: LLMResponse | Exception) -> None:
        self.steps = list(steps)
        self.calls = 0

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        raise NotImplementedError


OK = LLMResponse("ok")


def test_retry_recovers_from_transient_error() -> None:
    primary = Scripted(LLMTransientError("429"), OK)
    llm = ResilientLLM(primary, wait=wait_none())
    assert llm.complete(MSGS).text == "ok"
    assert primary.calls == 2


def test_permanent_error_is_not_retried_and_uses_fallback() -> None:
    primary, fallback = Scripted(LLMError("401")), Scripted(OK)
    llm = ResilientLLM(primary, fallback, wait=wait_none())
    assert llm.complete(MSGS).text == "ok"
    assert (primary.calls, fallback.calls) == (1, 1)


def test_unavailable_when_everything_fails() -> None:
    primary = Scripted(*[LLMTransientError("a")] * 5)
    llm = ResilientLLM(primary, wait=wait_none())
    with pytest.raises(LLMUnavailableError):
        llm.complete(MSGS)


def test_breaker_opens_after_threshold_and_half_opens() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, reset_seconds=30, monotonic=lambda: now[0])
    primary = Scripted(LLMError("x"), LLMError("x"), OK)
    fallback = Scripted(OK, OK, OK)
    llm = ResilientLLM(primary, fallback, breaker=breaker, wait=wait_none())
    llm.complete(MSGS)
    llm.complete(MSGS)
    assert breaker.is_open
    llm.complete(MSGS)  # circuito aberto: nem tenta o principal
    assert primary.calls == 2
    now[0] = 31.0
    assert not breaker.is_open
    assert llm.complete(MSGS).text == "ok"  # prova half-open fecha o circuito
    assert primary.calls == 3
    breaker_state = breaker.is_open
    assert breaker_state is False


def test_structured_goes_through_the_guard() -> None:
    llm = ResilientLLM(FakeLLM(), wait=wait_none())
    messages: list[Message] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Quais são as promoções ativas?"},
    ]
    decision, usage = llm.structured(messages, RouteDecision)
    assert decision.intents == [Intent.ACTIVE_PROMOTIONS]
    assert usage.calls == 1


def test_to_langchain_converts_all_roles() -> None:
    response = LLMResponse("", (ToolCall("c1", "get_product", {"name_or_id": "GD20"}),))
    converted = to_langchain(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            assistant_message(response),
            tool_message(response.tool_calls[0], "{}"),
        ]
    )
    assert [type(m) for m in converted] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]
    assert converted[2].tool_calls[0]["args"] == {"name_or_id": "GD20"}  # type: ignore[attr-defined]
    assert converted[3].tool_call_id == "c1"  # type: ignore[attr-defined]


def test_assistant_message_serializes_arguments_as_json() -> None:
    message = assistant_message(LLMResponse("", (ToolCall("c", "t", {"a": 1}),)))
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


class StubChat(GenericFakeChatModel):
    """Chat model do LangChain com resposta fixa e metadados de uso."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _chat_factory(message: AIMessage) -> Callable[[str, float, int], BaseChatModel]:
    def build(model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        return StubChat(messages=iter([message]))

    return build


def test_adapter_maps_text_tool_calls_and_cost() -> None:
    message = AIMessage(
        "olá",
        tool_calls=[{"id": "t1", "name": "get_product", "args": {"name_or_id": "95"}}],
        usage_metadata={
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "total_tokens": 2_000_000,
        },
    )
    llm = OpenAICompatibleLLM(Settings(), chat_factory=_chat_factory(message))
    response = llm.complete(MSGS, tier="medium", tools=[{"type": "function", "function": {}}])
    assert response.text == "olá"
    assert response.tool_calls == (ToolCall("t1", "get_product", {"name_or_id": "95"}),)
    assert response.usage.cost_usd == pytest.approx(0.15 + 0.60)


def test_adapter_uses_the_small_model_for_the_small_tier() -> None:
    seen: list[str] = []

    def build(model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        seen.append(model)
        return StubChat(messages=iter([AIMessage("x")]))

    settings = Settings(llm_model_small="pequeno", llm_model_medium="medio")
    llm = OpenAICompatibleLLM(settings, chat_factory=build)
    llm.complete(MSGS, tier="small")
    assert seen == ["pequeno"]
    OpenAICompatibleLLM(settings, model_override="reserva", chat_factory=build).complete(MSGS)
    assert seen == ["pequeno", "reserva"]


class RaisingChat(BaseChatModel):
    error: Exception

    @property
    def _llm_type(self) -> str:
        return "raising"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://x")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_request())


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (openai.APITimeoutError(_request()), LLMTransientError),
        (
            openai.RateLimitError(
                "limit", response=httpx.Response(429, request=_request()), body=None
            ),
            LLMTransientError,
        ),
        (
            openai.AuthenticationError(
                "no", response=httpx.Response(401, request=_request()), body=None
            ),
            LLMError,
        ),
    ],
)
def test_adapter_translates_provider_errors(error: Exception, expected: type[LLMError]) -> None:
    def build(model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        return RaisingChat(error=error)

    with pytest.raises(expected) as caught:
        OpenAICompatibleLLM(Settings(), chat_factory=build).complete(MSGS)
    assert not isinstance(caught.value, LLMUnavailableError)
    if expected is LLMError:
        assert type(caught.value) is LLMError


def test_adapter_structured_invalid_output_is_llm_error() -> None:
    class Plain(GenericFakeChatModel):
        def with_structured_output(self, *a: Any, **k: Any) -> Any:
            class Runner:
                def invoke(self, _messages: Any) -> dict[str, Any]:
                    return {"parsed": None, "parsing_error": ValueError("ruim"), "raw": None}

            return Runner()

    def build(model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        return Plain(messages=iter([]))

    with pytest.raises(LLMError):
        OpenAICompatibleLLM(Settings(), chat_factory=build).structured(MSGS, RouteDecision)


def test_factory_builds_fake_without_key() -> None:
    llm = factory.build(Settings(llm_provider="fake"), ["Takamine GD20"])
    assert isinstance(llm, FakeLLM)


def test_factory_requires_key_for_remote_endpoint() -> None:
    with pytest.raises(ValueError, match="EMPORIO_LLM_API_KEY"):
        factory.build(Settings(llm_provider="openai_compatible", llm_api_key=None))


def test_factory_allows_local_endpoint_without_key() -> None:
    settings = Settings(llm_base_url="http://localhost:11434/v1", llm_api_key=None)
    assert isinstance(factory.build(settings), ResilientLLM)


def test_factory_wraps_real_provider_with_fallback() -> None:
    settings = Settings(llm_api_key="k", llm_model_fallback="reserva")  # type: ignore[arg-type]
    assert isinstance(factory.build(settings), ResilientLLM)


def test_settings_read_the_documented_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMPORIO_LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMPORIO_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMPORIO_LLM_API_KEY", "segredo")
    monkeypatch.setenv("EMPORIO_LLM_MODEL_SMALL", "p")
    monkeypatch.setenv("EMPORIO_LLM_MODEL_MEDIUM", "m")
    settings = Settings()
    assert (settings.llm_provider, settings.llm_model_small, settings.llm_model_medium) == (
        "fake",
        "p",
        "m",
    )
    assert settings.llm_api_key is not None
    assert "segredo" not in repr(settings)
