"""Tipos próprios da camada de LLM: o grafo nunca depende de classes do LangChain."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel

Tier = Literal["small", "medium"]
Message = dict[str, Any]  # formato OpenAI: role, content, tool_calls, tool_call_id


class LLMError(Exception):
    """Falha permanente do provedor (credencial, pedido inválido, saída fora do schema)."""


class LLMTransientError(LLMError):
    """Falha passageira (timeout, limite de taxa, 5xx): vale tentar de novo."""


class LLMUnavailableError(LLMError):
    """Provedor indisponível após retentativas e fallback (circuito aberto)."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Pedido de execução de uma tool feito pelo modelo."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    """Consumo de uma ou mais chamadas.

    Attributes:
        input_tokens: Tokens de entrada.
        output_tokens: Tokens de saída.
        cost_usd: Custo estimado em dólares.
        calls: Quantidade de chamadas ao provedor.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Soma dois consumos."""
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
            self.calls + other.calls,
        )

    def to_dict(self) -> dict[str, float | int]:
        """Forma JSON-friendly para o estado do grafo."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage:
        """Reconstrói a partir de :meth:`to_dict`."""
        data = data or {}
        return cls(
            int(data.get("input_tokens", 0)),
            int(data.get("output_tokens", 0)),
            float(data.get("cost_usd", 0.0)),
            int(data.get("calls", 0)),
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Resposta de uma chamada de chat."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)


def assistant_message(response: LLMResponse) -> Message:
    """Mensagem do assistente (formato OpenAI) com as tool calls pedidas."""
    return {
        "role": "assistant",
        "content": response.text,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.args, default=str)},
            }
            for c in response.tool_calls
        ],
    }


def tool_message(call: ToolCall, content: str) -> Message:
    """Mensagem com o resultado de uma tool, ligada pelo ``tool_call_id``."""
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}


class LLMProvider(Protocol):
    """Porta do modelo de linguagem: um adaptador OpenAI-compatível e um fake offline."""

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Chat com tool calling opcional."""
        ...

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Saída estruturada validada por ``schema``."""
        ...
