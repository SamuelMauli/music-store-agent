"""Cassetes (padrão VCR): grava as respostas do provedor real e as repete offline."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from emporio.agent.llm.types import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    Tier,
    ToolCall,
    Usage,
)


def request_key(kind: str, tier: str, messages: list[Message], extra: Any) -> str:
    """Hash estável do pedido (tipo, tier, mensagens e tools/schema)."""
    raw = json.dumps([kind, tier, messages, extra], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class Cassette:
    """Fila de respostas por chave de pedido, em JSONL (uma gravação por linha)."""

    def __init__(self, path: Path) -> None:
        """Abre o arquivo; ausente equivale a vazio."""
        self.path = path
        self._queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._queues[row["key"]].append(row)

    def pop(self, key: str) -> dict[str, Any] | None:
        """Próxima gravação do pedido, ou ``None`` se acabou."""
        queue = self._queues.get(key)
        return queue.popleft() if queue else None

    def append(self, row: dict[str, Any]) -> None:
        """Grava uma resposta ao final do arquivo."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class RecordingLLM:
    """Chama o provedor real e grava cada resposta no cassete."""

    def __init__(self, inner: LLMProvider, cassette: Cassette) -> None:
        """Cria o gravador."""
        self._inner = inner
        self._cassette = cassette

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Chat real, gravado."""
        response = self._inner.complete(messages, tier=tier, temperature=temperature, tools=tools)
        self._cassette.append(
            {
                "key": request_key("complete", tier, messages, tools),
                "text": response.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "args": c.args} for c in response.tool_calls
                ],
                "usage": response.usage.to_dict(),
            }
        )
        return response

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Saída estruturada real, gravada."""
        parsed, usage = self._inner.structured(messages, schema, tier=tier, temperature=temperature)
        self._cassette.append(
            {
                "key": request_key("structured", tier, messages, schema.__name__),
                "parsed": parsed.model_dump(mode="json"),
                "usage": usage.to_dict(),
            }
        )
        return parsed, usage


class ReplayLLM:
    """Repete as gravações sem rede; pedido sem gravação é erro (prompt mudou: regrave)."""

    def __init__(self, cassette: Cassette) -> None:
        """Cria o repetidor."""
        self._cassette = cassette

    def _next(self, key: str) -> dict[str, Any]:
        row = self._cassette.pop(key)
        if row is None:
            raise LLMError("pedido sem gravação no cassete: rode make eval-live para regravar")
        return row

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Resposta gravada do pedido."""
        row = self._next(request_key("complete", tier, messages, tools))
        calls = tuple(ToolCall(c["id"], c["name"], c["args"]) for c in row["tool_calls"])
        return LLMResponse(row["text"], calls, Usage.from_dict(row["usage"]))

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Saída estruturada gravada do pedido."""
        row = self._next(request_key("structured", tier, messages, schema.__name__))
        return schema.model_validate(row["parsed"]), Usage.from_dict(row["usage"])
