"""Serviço de conversa: única porta de entrada compartilhada por CLI e API."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from langchain_core.runnables import RunnableConfig

from emporio.agent.graph import CompiledChat
from emporio.agent.llm.types import Usage
from emporio.agent.prompts import prompt_hash
from emporio.agent.state import ChatState
from emporio.domain.pii import mask_pii
from emporio.guardrails.rate_limit import SlidingWindowLimiter
from emporio.observability.telemetry import NOOP, Telemetry, attributes_of
from emporio.observability.usage import UsageAccumulator

RECURSION_LIMIT = 25
log = logging.getLogger("emporio.chat")


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
    conversation_usage: Usage = field(default_factory=Usage)
    handoff: bool = False
    blocked: bool = False
    latency_ms: int = 0
    request_id: str = ""
    prompt_hash: str = ""


class ChatService:
    """Executa turnos no grafo; o histórico vive no checkpointer (por ``thread_id``)."""

    def __init__(
        self,
        graph: CompiledChat,
        limiter: SlidingWindowLimiter | None = None,
        ip_limiter: SlidingWindowLimiter | None = None,
        telemetry: Telemetry = NOOP,
        usage: UsageAccumulator | None = None,
    ) -> None:
        """Cria o serviço.

        Args:
            graph: Grafo compilado, com checkpointer quando houver memória.
            limiter: Limite de mensagens por sessão (``RateLimitedError`` ao exceder).
            ip_limiter: Limite de mensagens por IP de cliente.
            telemetry: Tracing; cada turno é um span ``invoke_agent`` (no-op por padrão).
            usage: Acumulador de tokens e custo por conversa.
        """
        self._graph = graph
        self._limiter = limiter
        self._ip_limiter = ip_limiter
        self._telemetry = telemetry
        self._usage = usage or UsageAccumulator()

    @staticmethod
    def _config(session_id: str, text: str, client_ip: str = "") -> RunnableConfig:
        # O texto cru vai num dict: o LangGraph só copia primitivos para os metadados do
        # checkpoint, então a mensagem original (com PII) não é persistida.
        return {
            "configurable": {"thread_id": session_id, "inbound": {"text": text, "ip": client_ip}},
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

    def _admit(self, session_id: str, text: str, client_ip: str) -> None:
        if self._ip_limiter is not None and client_ip:
            self._ip_limiter.check(client_ip)
        if self._limiter is not None:
            self._limiter.check(session_id)
        log.info("mensagem sessao=%s texto=%s", session_id, mask_pii(text)[:80])

    @staticmethod
    def _log(result: ChatResult) -> ChatResult:
        log.info(
            "turno sessao=%s turno=%s intents=%s tools=%s handoff=%s bloqueado=%s ms=%s tokens=%s",
            result.session_id,
            result.turn,
            result.intents,
            result.tools,
            result.handoff,
            result.blocked,
            result.latency_ms,
            result.usage.input_tokens + result.usage.output_tokens,
        )
        return result

    def _turn_span(self, session_id: str) -> Any:
        return self._telemetry.span(
            "invoke_agent Tom",
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "Tom",
                "gen_ai.conversation.id": session_id,
                "session_id": session_id,
            },
        )

    def _finish(self, span: Any, state: dict[str, Any], result: ChatResult) -> ChatResult:
        """Soma o turno à conversa, anota o span do turno e registra o log."""
        result = replace(
            result, conversation_usage=self._usage.add(result.session_id, result.usage)
        )
        products = [str(p["product_id"]) for p in state.get("last_products") or [] if result.tools]
        attributes = attributes_of(
            **{
                "prompt_hash": result.prompt_hash,
                "section_ids": ",".join(result.sources),
                "product_ids": ",".join(products),
                "gen_ai.usage.input_tokens": result.usage.input_tokens,
                "gen_ai.usage.output_tokens": result.usage.output_tokens,
                "emporio.cost_usd": result.usage.cost_usd,
                "emporio.conversation_cost_usd": result.conversation_usage.cost_usd,
                "emporio.turn": result.turn,
                "emporio.handoff": result.handoff,
                "emporio.blocked": result.blocked,
                "emporio.tools": ",".join(result.tools),
            }
        )
        span.set_attributes(attributes)
        return self._log(result)

    def chat(
        self, session_id: str, text: str, request_id: str | None = None, client_ip: str = ""
    ) -> ChatResult:
        """Processa uma mensagem e devolve a resposta final."""
        self._admit(session_id, text, client_ip)
        started = time.perf_counter()
        with self._turn_span(session_id) as span:
            state = dict(
                self._graph.invoke(
                    ChatState(session_id=session_id), self._config(session_id, text, client_ip)
                )
            )
            rid = request_id or uuid.uuid4().hex
            return self._finish(span, state, self._result(state, session_id, started, rid))

    def stream(
        self, session_id: str, text: str, request_id: str | None = None, client_ip: str = ""
    ) -> Iterator[tuple[str, Any]]:
        """Emite ``("node", nome)`` por nó concluído e, no fim, ``("result", ChatResult)``.

        O limite de taxa é checado já na chamada (antes do 1º evento), para a API poder
        responder 429 em vez de abrir um stream.
        """
        self._admit(session_id, text, client_ip)
        return self._stream(session_id, text, request_id, client_ip)

    def _stream(
        self, session_id: str, text: str, request_id: str | None, client_ip: str
    ) -> Iterator[tuple[str, Any]]:
        started = time.perf_counter()
        rid = request_id or uuid.uuid4().hex
        with self._turn_span(session_id) as span:
            for update in self._graph.stream(
                ChatState(session_id=session_id),
                self._config(session_id, text, client_ip),
                stream_mode="updates",
            ):
                for node in update:
                    yield "node", node
            snapshot = self._graph.get_state(self._config(session_id, text))
            state = dict(snapshot.values)
            yield "result", self._finish(span, state, self._result(state, session_id, started, rid))

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Estado atual da sessão (tool traces do último turno, rota, memória) para avaliação."""
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        return dict(self._graph.get_state(config).values)

    def reset(self, session_id: str) -> None:
        """Apaga a memória da sessão."""
        self._usage.reset(session_id)
        saver = self._graph.checkpointer
        if saver is not None and hasattr(saver, "delete_thread"):
            saver.delete_thread(session_id)
