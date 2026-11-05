"""Cadeia de fallback entre provedores, saída estruturada portável, fábrica e diagnóstico."""

from __future__ import annotations

import io
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from tenacity import wait_none

from emporio.agent.llm import doctor, factory, providers
from emporio.agent.llm.chat import LangChainLLM
from emporio.agent.llm.fake import FakeLLM
from emporio.agent.llm.resilient import CircuitBreaker, ResilientLLM
from emporio.agent.llm.spec import ModelSpec
from emporio.agent.llm.types import (
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMTransientError,
    LLMUnavailableError,
    Message,
    Tier,
    ToolCall,
    Usage,
)
from emporio.settings import Settings

MSGS: list[Message] = [{"role": "user", "content": "oi"}]
OK = LLMResponse("ok")


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class Probe(BaseModel):
    tema: str
    confianca: float


class Scripted:
    """Provedor que executa uma sequência de resultados ou exceções."""

    def __init__(self, name: str, *steps: Any) -> None:
        self.name = name
        self.steps = list(steps)
        self.calls = 0

    def _next(self) -> Any:
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        result: LLMResponse = self._next()
        return result

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        result: tuple[T, Usage] = self._next()
        return result


# --- cadeia de fallback -------------------------------------------------------------------


def test_transient_is_retried_then_next_provider() -> None:
    a = Scripted("anthropic:c", *[LLMTransientError("529")] * 3)
    b = Scripted("openai:g", OK)
    llm = ResilientLLM(a, fallbacks=[b], attempts=3, wait=wait_none())
    assert llm.complete(MSGS).text == "ok"
    assert (a.calls, b.calls) == (3, 1)


def test_permanent_goes_straight_to_next_and_chain_walks_to_the_end() -> None:
    a = Scripted("a", LLMError("401"))
    b = Scripted("b", LLMError("400"))
    c = Scripted("c", OK)
    llm = ResilientLLM(a, fallbacks=[b, c], wait=wait_none())
    assert llm.complete(MSGS).text == "ok"
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)
    assert llm.chain == ["a", "b", "c"]


def test_each_provider_has_its_own_breaker() -> None:
    now = [0.0]

    def make() -> CircuitBreaker:
        return CircuitBreaker(failure_threshold=1, reset_seconds=30, monotonic=lambda: now[0])

    a = Scripted("a", LLMError("x"), OK)
    b = Scripted("b", LLMError("x"))
    c = Scripted("c", OK, OK)
    llm = ResilientLLM(a, fallbacks=[b, c], breaker=make(), breaker_factory=make, wait=wait_none())
    llm.complete(MSGS)  # a e b falham e abrem; c atende
    assert llm.complete(MSGS).text == "ok"  # a e b pulados
    assert (a.calls, b.calls, c.calls) == (1, 1, 2)
    now[0] = 31.0
    assert llm.complete(MSGS).text == "ok"  # a volta em half-open e fecha
    assert a.calls == 2


def test_chain_exhausted_is_unavailable() -> None:
    llm = ResilientLLM(
        Scripted("a", LLMError("x")), fallbacks=[Scripted("b", LLMError("y"))], wait=wait_none()
    )
    with pytest.raises(LLMUnavailableError):
        llm.complete(MSGS)


def test_structured_also_walks_the_chain() -> None:
    a = Scripted("a", LLMError("schema"))
    b = Scripted("b", (Probe(tema="t", confianca=1.0), Usage(calls=1)))
    probe, _ = ResilientLLM(a, fallbacks=[b], wait=wait_none()).structured(MSGS, Probe)
    assert probe.tema == "t"


def test_unnamed_providers_get_positional_names() -> None:
    llm = ResilientLLM(FakeLLM(), FakeLLM())
    assert llm.chain == ["FakeLLM[0]", "FakeLLM[1]"]


# --- adaptador: saída estruturada portável e custo ----------------------------------------


class SpyChat(GenericFakeChatModel):
    """Chat que registra o ``method`` pedido e responde via tool calling."""

    methods: list[str] = []  # noqa: RUF012 - estado de teste
    reply: AIMessage | None = None
    supports_structured: bool = True

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        if not self.supports_structured:
            raise NotImplementedError
        self.methods.append(kwargs["method"])
        reply = self.reply

        class Runner:
            def invoke(self, _messages: Any) -> dict[str, Any]:
                return {"parsed": schema(tema="x", confianca=0.5), "raw": reply}

        return Runner()


def _spy_factory(chat: SpyChat) -> Any:
    def build(spec: ModelSpec, settings: Settings, **_: Any) -> BaseChatModel:
        return chat

    return build


@pytest.mark.parametrize(
    ("provider", "method"),
    [
        ("google", "json_schema"),
        ("anthropic", "function_calling"),
        ("openai", "function_calling"),
        ("bedrock", "function_calling"),
        ("groq", "function_calling"),
    ],
)
def test_structured_uses_the_method_of_each_provider(
    monkeypatch: pytest.MonkeyPatch, provider: str, method: str
) -> None:
    chat = SpyChat(messages=iter([]), methods=[])
    monkeypatch.setattr(providers, "build_chat", _spy_factory(chat))
    llm = LangChainLLM(_settings(), model_override=f"{provider}:m")
    parsed, usage = llm.structured(MSGS, Probe)
    assert parsed.tema == "x"
    assert chat.methods == [method]
    assert usage.calls == 1


def test_structured_method_override_beats_the_provider_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # gpt-oss-20b no Groq às vezes responde em texto com tool forçada (HTTP 400 "did not
    # call a tool"); EMPORIO_LLM_STRUCTURED_METHOD=json_schema resolve só por configuração.
    chat = SpyChat(messages=iter([]), methods=[])
    monkeypatch.setattr(providers, "build_chat", _spy_factory(chat))
    settings = _settings(llm_structured_method="json_schema")
    LangChainLLM(settings, model_override="groq:openai/gpt-oss-20b").structured(MSGS, Probe)
    assert chat.methods == ["json_schema"]


def test_structured_falls_back_to_forced_tool_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = AIMessage(
        "",
        tool_calls=[{"id": "1", "name": "Probe", "args": {"tema": "devolução", "confianca": 0.9}}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    chat = SpyChat(messages=iter([reply]), supports_structured=False)
    monkeypatch.setattr(providers, "build_chat", _spy_factory(chat))
    parsed, usage = LangChainLLM(_settings(), model_override="ollama:m").structured(MSGS, Probe)
    assert parsed == Probe(tema="devolução", confianca=0.9)
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)


@pytest.mark.parametrize(
    "reply",
    [
        AIMessage("sem tool"),
        AIMessage("", tool_calls=[{"id": "1", "name": "Probe", "args": {"tema": 3}}]),
    ],
)
def test_forced_tool_calling_invalid_output_is_llm_error(
    monkeypatch: pytest.MonkeyPatch, reply: AIMessage
) -> None:
    chat = SpyChat(messages=iter([reply]), supports_structured=False)
    monkeypatch.setattr(providers, "build_chat", _spy_factory(chat))
    with pytest.raises(LLMError):
        LangChainLLM(_settings(), model_override="ollama:m").structured(MSGS, Probe)


def test_mixed_providers_per_tier_and_price_by_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    message = AIMessage(
        [{"type": "text", "text": "resposta"}],
        usage_metadata={
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "total_tokens": 2_000_000,
        },
    )

    def build(spec: ModelSpec, settings: Settings, **_: Any) -> BaseChatModel:
        seen.append(spec.label)
        return SpyChat(messages=iter([message]))

    monkeypatch.setattr(providers, "build_chat", build)
    settings = _settings(
        llm_model_small="google:gemini-flash",
        llm_model_medium="anthropic:claude",
        llm_price_table={"anthropic:claude": {"input": 3.0, "output": 15.0}},
    )
    llm = LangChainLLM(settings)
    response = llm.complete(MSGS)
    assert response.text == "resposta"  # content em blocos (Claude/Gemini)
    assert response.usage.cost_usd == pytest.approx(18.0)
    assert llm.name == "anthropic:claude"
    assert llm.spec("small") == ModelSpec("google", "gemini-flash")
    assert seen == ["anthropic:claude"]


# --- fábrica ------------------------------------------------------------------------------


def test_factory_builds_the_configured_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in ("OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(env, "k")
    settings = _settings(
        llm_api_key="groq",
        llm_fallbacks=["openai:gpt-5-mini", "google:gemini-2.5-flash"],
        llm_model_fallback="ollama:qwen2.5:7b",
    )
    llm = factory.build(settings)
    assert isinstance(llm, ResilientLLM)
    assert llm.chain == [
        "openai_compatible:openai/gpt-oss-120b",
        "openai:gpt-5-mini",
        "google:gemini-2.5-flash",
        "ollama:qwen2.5:7b",
    ]


def test_factory_fails_at_boot_when_a_fallback_has_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = _settings(llm_api_key="k", llm_fallbacks=["anthropic:claude-sonnet-5-5"])
    with pytest.raises(providers.LLMConfigError, match="ANTHROPIC_API_KEY"):
        factory.build(settings)


def test_factory_bedrock_needs_no_key() -> None:
    settings = _settings(
        llm_provider="bedrock",
        llm_model_small="meta.llama3-8b-instruct-v1:0",
        llm_model_medium="anthropic.claude-3-haiku-20240307-v1:0",
    )
    assert isinstance(factory.build(settings), ResilientLLM)


# --- doctor -------------------------------------------------------------------------------


class _DoctorLLM:
    def __init__(self, tool: bool = True, structured: bool = True) -> None:
        self.tool, self.ok = tool, structured

    def complete(self, messages: list[Message], **_: Any) -> LLMResponse:
        calls = (ToolCall("1", "ping", {"eco": "ok"}),) if self.tool else ()
        return LLMResponse("", calls, Usage(5, 2, 0.0, 1))

    def structured(self, messages: list[Message], schema: type[Any], **_: Any) -> Any:
        if not self.ok:
            raise LLMTransientError("ResourceExhausted (HTTP 429)")
        return schema(tema="devolução", confianca=0.9), Usage(4, 3, 0.0, 1)


def _builder(llm: Any) -> Any:
    def build(settings: Settings, spec: ModelSpec) -> LLMProvider:
        result: LLMProvider = llm
        return result

    return build


def test_doctor_reports_each_role_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "nao-mostrar")
    settings = _settings(
        llm_provider="google",
        llm_model_small="gemini-flash",
        llm_model_medium="google:gemini-pro",
        llm_fallbacks=["ollama:qwen"],
    )
    out = io.StringIO()
    assert doctor.run(settings, out, _builder(_DoctorLLM())) == 0
    report = out.getvalue()
    assert "[small] google:gemini-flash" in report
    assert "[fallback[0]] ollama:qwen" in report
    assert "presente (GOOGLE_API_KEY)" in report
    assert "não exigida" in report
    assert "ping({'eco': 'ok'})" in report
    assert "nao-mostrar" not in report
    assert report.rstrip().endswith("resultado: OK")


@pytest.mark.parametrize(
    ("llm", "expected"),
    [
        (_DoctorLLM(tool=False), "sem tool call"),
        (_DoctorLLM(structured=False), "LLMTransientError"),
    ],
)
def test_doctor_fails_when_a_call_fails(llm: Any, expected: str) -> None:
    out = io.StringIO()
    settings = _settings(llm_provider="ollama", llm_model_small="a", llm_model_medium="a")
    assert doctor.run(settings, out, _builder(llm)) == 1
    assert expected in out.getvalue()


def test_doctor_reports_missing_credential_without_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = io.StringIO()
    settings = _settings(llm_provider="anthropic", llm_model_small="c", llm_model_medium="c")
    assert doctor.run(settings, out, _builder(None)) == 1
    assert "ausente" in out.getvalue()
    assert "LLMConfigError" in out.getvalue()


def test_doctor_classifies_unexpected_errors() -> None:
    def broken() -> str:
        raise ConnectionError("rede")

    result = doctor._timed("x", broken)
    assert not result.ok
    assert result.detail.startswith("LLMTransientError")


def test_doctor_skips_fake_and_main_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    assert doctor.run(_settings(llm_provider="fake"), out) == 0
    monkeypatch.setenv("EMPORIO_LLM_PROVIDER", "fake")
    with pytest.raises(SystemExit) as caught:
        doctor.main()
    assert caught.value.code == 0
