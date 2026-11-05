"""Serviço de conversa: única porta de entrada compartilhada por CLI e API."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from emporio.agent.graph import CompiledChat
from emporio.agent.llm.types import Usage
from emporio.agent.prompts import prompt_hash
from emporio.agent.state import ChatState

RECURSION_LIMIT = 25


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Resultado de um turno, com o que o ``/debug`` e o trace precisam."""

    answer: str
    session_id: str
    turn: int
    intents: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    handoff: bool = False
    blocked: bool = False
    latency_ms: int = 0
    request_id: str = ""
    prompt_hash: str = ""


class ChatService:
    """Executa turnos no grafo; o histórico vive no checkpointer (por ``thread_id``)."""

    def __init__(self, graph: CompiledChat) -> None:
        """Cria o serviço.

        Args:
            graph: Grafo compilado, com checkpointer quando houver memória.
        """
        self._graph = graph

    @staticmethod
    def _config(session_id: str, text: str) -> RunnableConfig:
        # O texto cru vai num dict: o LangGraph só copia primitivos para os metadados do
        # checkpoint, então a mensagem original (com PII) não é persistida.
        return {
            "configurable": {"thread_id": session_id, "inbound": {"text": text}},
            "recursion_limit": RECURSION_LIMIT,
        }

    @staticmethod
    def _result(state: dict[str, Any], session_id: str, started: float, rid: str) -> ChatResult:
        traces = state.get("tool_traces") or []
        sources = [*state.get("policy_sections", [])]
        for trace in traces:
            sources.extend(trace.get("sources", []))
        return ChatResult(
            answer=state.get("answer", ""),
            session_id=session_id,
            turn=int(state.get("turn", 0)),
            intents=list((state.get("route") or {}).get("intents", [])),
            tools=[t["name"] for t in traces],
            sources=list(dict.fromkeys(str(s) for s in sources)),
            usage=Usage.from_dict(state.get("usage")),
            handoff=bool(state.get("handoff")),
            blocked=bool(state.get("blocked")),
            latency_ms=int((time.perf_counter() - started) * 1000),
            request_id=rid,
            prompt_hash=prompt_hash(),
        )

    def chat(self, session_id: str, text: str, request_id: str | None = None) -> ChatResult:
        """Processa uma mensagem e devolve a resposta final."""
        started = time.perf_counter()
        state = self._graph.invoke(
            ChatState(session_id=session_id),
            self._config(session_id, text),
        )
        return self._result(dict(state), session_id, started, request_id or uuid.uuid4().hex)

    def stream(
        self, session_id: str, text: str, request_id: str | None = None
    ) -> Iterator[tuple[str, Any]]:
        """Emite ``("node", nome)`` por nó concluído e, no fim, ``("result", ChatResult)``."""
        started = time.perf_counter()
        rid = request_id or uuid.uuid4().hex
        for update in self._graph.stream(
            ChatState(session_id=session_id),
            self._config(session_id, text),
            stream_mode="updates",
        ):
            for node in update:
                yield "node", node
        snapshot = self._graph.get_state(self._config(session_id, text))
        yield "result", self._result(dict(snapshot.values), session_id, started, rid)

    def reset(self, session_id: str) -> None:
        """Apaga a memória da sessão."""
        saver = self._graph.checkpointer
        if saver is not None and hasattr(saver, "delete_thread"):
            saver.delete_thread(session_id)
