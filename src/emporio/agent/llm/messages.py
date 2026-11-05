"""Conversão das mensagens internas (formato OpenAI) para o LangChain, portável entre provedores.

Diferenças que a conversão absorve:

* Anthropic, Gemini e Bedrock só aceitam instrução de sistema no topo; o Gemini chega a
  descartar em silêncio um ``system`` que não seja o primeiro. Com ``hoist_system`` todas as
  mensagens de sistema viram uma só, no início, na ordem original.
* Assistente com ``tool_calls`` e ``content`` vazio/``None`` é válido em todos.
* Resultado de tool vazio é rejeitado por alguns provedores: vira um marcador explícito.
* ``ToolMessage`` leva ``name`` (o Gemini casa a resposta da função por nome).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from emporio.agent.llm.types import Message

EMPTY_TOOL_RESULT = "(sem conteúdo)"


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    parsed = json.loads(raw or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _assistant(message: Message, content: str) -> AIMessage:
    calls = [
        {
            "id": c["id"],
            "name": c["function"]["name"],
            "args": _arguments(c["function"].get("arguments")),
            "type": "tool_call",
        }
        for c in message.get("tool_calls") or []
    ]
    return AIMessage(content, tool_calls=calls)


def _tool(message: Message, content: str) -> ToolMessage:
    return ToolMessage(
        content or EMPTY_TOOL_RESULT,
        tool_call_id=message["tool_call_id"],
        name=message.get("name"),
    )


def to_langchain(messages: list[Message], *, hoist_system: bool = False) -> list[BaseMessage]:
    """Converte mensagens no formato OpenAI para o LangChain.

    Args:
        messages: Lista com ``role`` system, user, assistant ou tool.
        hoist_system: Junta todas as mensagens de sistema numa só, no início (provedores que
            não aceitam ``system`` no meio da conversa).

    Returns:
        Mensagens equivalentes, preservando ``tool_calls`` e ``tool_call_id``.
    """
    converted: list[BaseMessage] = []
    system_parts: list[str] = []
    for message in messages:
        content = str(message.get("content") or "")
        match message["role"]:
            case "system" if hoist_system:
                if content:
                    system_parts.append(content)
            case "system":
                converted.append(SystemMessage(content))
            case "assistant":
                converted.append(_assistant(message, content))
            case "tool":
                converted.append(_tool(message, content))
            case _:
                converted.append(HumanMessage(content))
    if system_parts:
        converted.insert(0, SystemMessage("\n\n".join(system_parts)))
    return converted
