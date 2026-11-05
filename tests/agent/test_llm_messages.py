"""Conversão de mensagens portável e contrato de tool calling por provedor (sem rede)."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws.chat_models.bedrock_converse import _messages_to_bedrock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai.chat_models import _parse_chat_history
from langchain_openai import ChatOpenAI

from emporio.agent.llm import providers
from emporio.agent.llm.chat import LangChainLLM, text_of
from emporio.agent.llm.messages import EMPTY_TOOL_RESULT, to_langchain
from emporio.agent.llm.spec import ModelSpec
from emporio.agent.llm.types import (
    LLMResponse,
    LLMTransientError,
    Message,
    ToolCall,
    assistant_message,
    tool_message,
)
from emporio.settings import Settings

CALL = ToolCall("call_1", "get_product", {"name_or_id": "GD20"})
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Busca um produto.",
            "parameters": {
                "type": "object",
                "properties": {"name_or_id": {"type": "string"}},
                "required": ["name_or_id"],
            },
        },
    }
]


def _conversation() -> list[Message]:
    """Turno real do agente: system, pedido, tool call com content vazio, resultado, system."""
    assistant = assistant_message(LLMResponse("", (CALL,)))
    return [
        {"role": "system", "content": "Você é o atendente."},
        {"role": "user", "content": "Tem o violão GD20?"},
        assistant,
        tool_message(CALL, '{"preco": 1200}'),
        {"role": "system", "content": "Responda em uma frase."},
    ]


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


# --- to_langchain -------------------------------------------------------------------------


def test_in_place_system_is_kept_for_openai_family() -> None:
    converted = to_langchain(_conversation())
    assert [type(m) for m in converted] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        SystemMessage,
    ]


def test_hoist_system_merges_every_system_message_on_top_in_order() -> None:
    converted = to_langchain(_conversation(), hoist_system=True)
    assert [type(m) for m in converted] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]
    assert converted[0].content == "Você é o atendente.\n\nResponda em uma frase."


def test_hoist_without_system_adds_nothing() -> None:
    converted = to_langchain([{"role": "user", "content": "oi"}], hoist_system=True)
    assert [type(m) for m in converted] == [HumanMessage]


def test_assistant_with_none_content_and_tool_calls() -> None:
    message = assistant_message(LLMResponse("", (CALL,)))
    message["content"] = None
    converted = to_langchain([message])
    ai = converted[0]
    assert isinstance(ai, AIMessage)
    assert ai.content == ""
    assert ai.tool_calls[0]["args"] == {"name_or_id": "GD20"}
    assert ai.tool_calls[0]["id"] == "call_1"


def test_tool_call_arguments_accept_dict_empty_and_non_object() -> None:
    message: Message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "a", "function": {"name": "t", "arguments": {"x": 1}}},
            {"id": "b", "function": {"name": "t", "arguments": ""}},
            {"id": "c", "function": {"name": "t", "arguments": "[1]"}},
        ],
    }
    ai = to_langchain([message])[0]
    assert isinstance(ai, AIMessage)
    assert [c["args"] for c in ai.tool_calls] == [{"x": 1}, {}, {"value": [1]}]


def test_tool_message_keeps_id_and_name_and_fills_empty_content() -> None:
    tool = to_langchain([tool_message(CALL, "")])[0]
    assert isinstance(tool, ToolMessage)
    assert (tool.tool_call_id, tool.name, tool.content) == (
        "call_1",
        "get_product",
        EMPTY_TOOL_RESULT,
    )


def test_text_of_handles_string_blocks_and_missing() -> None:
    blocks = AIMessage(
        [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "Olá, "},
            "tudo bem?",
            {"type": "tool_use", "id": "t", "name": "x", "input": {}},
        ]
    )
    assert text_of(blocks) == "Olá, tudo bem?"
    assert text_of(AIMessage("simples")) == "simples"
    assert text_of(None) == ""
    assert text_of(type("M", (), {"content": 3})()) == ""


# --- payload real de cada SDK (offline) ---------------------------------------------------


def test_anthropic_payload_has_top_level_system_and_paired_tool_result() -> None:
    chat = ChatAnthropic(model="claude-sonnet-5-5", api_key="k")  # type: ignore[call-arg]
    payload = chat._get_request_payload(to_langchain(_conversation(), hoist_system=True))
    assert "Responda em uma frase." in json.dumps(payload["system"], ensure_ascii=False)
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["user", "assistant", "user"]
    tool_use = payload["messages"][1]["content"]
    assert any(b["type"] == "tool_use" and b["id"] == "call_1" for b in tool_use)
    assert not any(b.get("type") == "text" and not b.get("text") for b in tool_use)
    result = payload["messages"][2]["content"][0]
    assert (result["type"], result["tool_use_id"]) == ("tool_result", "call_1")


def test_gemini_history_keeps_late_system_and_function_response() -> None:
    lc = to_langchain(_conversation(), hoist_system=True)
    system, contents = _parse_chat_history(lc, model="gemini-2.5-flash")
    assert system is not None
    assert "Responda em uma frase." in "".join(p.text for p in system.parts)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "get_product"
    assert contents[2].parts[0].function_response.name == "get_product"


def test_gemini_would_drop_a_non_leading_system_without_hoist() -> None:
    """Documenta o porquê do hoist: sem ele o Gemini descarta a instrução em silêncio."""
    late_only = [{"role": "user", "content": "oi"}, {"role": "system", "content": "regra"}]
    system, _ = _parse_chat_history(to_langchain(late_only), model="gemini-2.5-flash")
    assert system is None
    hoisted, _ = _parse_chat_history(to_langchain(late_only, hoist_system=True))
    assert hoisted is not None


def test_bedrock_converse_messages_and_system() -> None:
    lc = to_langchain(_conversation(), hoist_system=True)
    messages, system = _messages_to_bedrock(lc)
    assert "Responda em uma frase." in json.dumps(system, ensure_ascii=False)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][-1]["toolUse"]["toolUseId"] == "call_1"
    assert messages[2]["content"][0]["toolResult"]["toolUseId"] == "call_1"


# --- contrato fim a fim com transporte HTTP mockado ---------------------------------------

ANTHROPIC_TOOL_USE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [
        {"type": "text", "text": "Vou consultar o catálogo."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_product",
            "input": {"name_or_id": "GD20"},
        },
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 120, "output_tokens": 30},
}

OPENAI_TOOL_CALL = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-5-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "get_product", "arguments": '{"name_or_id":"GD20"}'},
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 80, "completion_tokens": 12, "total_tokens": 92},
}


class Recorder:
    """Transporte httpx/httpx2 que grava o pedido e devolve uma resposta fixa."""

    def __init__(self, body: dict[str, Any], status: int = 200, http: Any = httpx) -> None:
        self.body, self.status, self.http = body, status, http
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: Any) -> Any:
        self.requests.append(json.loads(request.content))
        return self.http.Response(self.status, json=self.body)

    def client(self) -> Any:
        return self.http.Client(transport=self.http.MockTransport(self))


def test_anthropic_contract_tool_use_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-teste")
    recorder = Recorder(ANTHROPIC_TOOL_USE, http=httpx2)  # o SDK da Anthropic usa httpx2
    settings = _settings(llm_provider="anthropic", llm_model_medium="claude-sonnet-5-5")

    def factory(model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        chat = providers.build_chat(
            ModelSpec("anthropic", model), settings, temperature=temperature, max_tokens=max_tokens
        )
        client = anthropic.Anthropic(api_key="k-teste", http_client=recorder.client())
        chat.__dict__["_client"] = client
        return chat

    llm = LangChainLLM(settings, chat_factory=factory)
    response = llm.complete(_conversation(), tools=TOOLS)
    assert response.text == "Vou consultar o catálogo."
    assert response.tool_calls == (ToolCall("toolu_01", "get_product", {"name_or_id": "GD20"}),)
    assert (response.usage.input_tokens, response.usage.output_tokens) == (120, 30)
    sent = recorder.requests[0]
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["tools"][0]["name"] == "get_product"
    assert "Responda em uma frase." in json.dumps(sent["system"], ensure_ascii=False)
    assert sent["messages"][2]["content"][0]["tool_use_id"] == "call_1"


def test_anthropic_contract_overloaded_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "x"}}
    recorder = Recorder(body, status=529, http=httpx2)
    settings = _settings(llm_provider="anthropic", llm_api_key="k", llm_model_medium="claude")

    def factory(model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        chat = providers.build_chat(
            ModelSpec("anthropic", model), settings, temperature=temperature, max_tokens=max_tokens
        )
        chat.__dict__["_client"] = anthropic.Anthropic(
            api_key="k", max_retries=0, http_client=recorder.client()
        )
        return chat

    with pytest.raises(LLMTransientError):
        LangChainLLM(settings, chat_factory=factory).complete(_conversation())


def test_openai_contract_tool_call_end_to_end() -> None:
    recorder = Recorder(OPENAI_TOOL_CALL)
    settings = _settings(llm_provider="openai", llm_model_medium="gpt-5-mini")

    def factory(model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        return ChatOpenAI(
            model=model,
            api_key="k-teste",  # type: ignore[arg-type]
            max_retries=0,
            http_client=recorder.client(),
        )

    response = LangChainLLM(settings, chat_factory=factory).complete(_conversation(), tools=TOOLS)
    assert response.text == ""
    assert response.tool_calls == (ToolCall("call_9", "get_product", {"name_or_id": "GD20"}),)
    assert (response.usage.input_tokens, response.usage.output_tokens) == (80, 12)
    sent = recorder.requests[0]
    assert [m["role"] for m in sent["messages"]][-1] == "system"  # OpenAI aceita no lugar
    assert sent["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"preco": 1200}',
    }
