"""Dependências compartilhadas pelos nós e utilitários pequenos."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from langchain_core.runnables import RunnableConfig

from emporio.agent.intents import RouteDecision
from emporio.agent.llm.types import LLMProvider, Usage
from emporio.agent.router import Router
from emporio.agent.state import ChatState
from emporio.agent.tools import ToolContext, ToolExecution, ToolRegistry
from emporio.domain.pii import mask_pii
from emporio.guardrails.attempts import AttemptStore, InMemoryAttemptStore
from emporio.ports import Clock, PolicyRetriever

MAX_TOOL_CALLS: Final = 4
HISTORY_WINDOW: Final = 12
TECH_FAILURE: Final = "Tive um problema técnico agora e não consegui concluir por aqui."


@dataclass(frozen=True, slots=True)
class VerifyPolicy:
    """Limites e validade da verificação de cliente."""

    ttl_seconds: float = 900.0
    max_per_target: int = 5
    max_per_ip: int = 10


@dataclass(frozen=True, slots=True)
class Deps:
    """Tudo que os nós usam, injetado pelo composition root."""

    llm: LLMProvider
    registry: ToolRegistry
    retriever: PolicyRetriever
    clock: Clock
    attempts: AttemptStore = field(default_factory=InMemoryAttemptStore)
    verify_policy: VerifyPolicy = field(default_factory=VerifyPolicy)
    wall: Callable[[], float] = time.time

    @property
    def router(self) -> Router:
        """Roteador sobre o LLM pequeno."""
        return Router(self.llm)


def inbound_text(config: RunnableConfig) -> str:
    """Mensagem crua do turno, passada fora do estado para não ir ao checkpoint."""
    inbound = (config.get("configurable") or {}).get("inbound") or {}
    return str(inbound.get("text", ""))


def inbound_ip(config: RunnableConfig) -> str:
    """IP do cliente do turno (vazio no CLI), passado fora do estado."""
    inbound = (config.get("configurable") or {}).get("inbound") or {}
    return str(inbound.get("ip", ""))


def tool_context(state: ChatState) -> ToolContext:
    """Contexto das tools: identidade verificada vem do estado, nunca do prompt."""
    return ToolContext(
        session_id=state.get("session_id", "anonymous"),
        verified_customer_id=state.get("verified_customer_id"),
        verification_attempts=state.get("verification_attempts", 0),
    )


def add_usage(state: ChatState, extra: Usage) -> dict[str, float | int]:
    """Soma ``extra`` ao consumo acumulado do turno."""
    return (Usage.from_dict(state.get("usage")) + extra).to_dict()


def trace_of(execution: ToolExecution) -> dict[str, Any]:
    """Registro da chamada para o estado e o ``/debug`` (argumentos com PII mascarada)."""
    payload = json.loads(execution.payload)
    args = json.loads(mask_pii(json.dumps(dict(execution.args), ensure_ascii=False, default=str)))
    return {
        "name": execution.name,
        "args": args,
        "status": payload.get("status", "error"),
        "payload": execution.payload,
        "sources": list(execution.sources),
        "ms": execution.duration_ms,
    }


def route_of(state: ChatState) -> RouteDecision:
    """Decisão de roteamento do turno."""
    return RouteDecision.model_validate(state.get("route") or {})


def failure(state: ChatState, reason: str) -> dict[str, Any]:
    """Falha do LLM: resposta segura e encaminhamento a um humano."""
    return {
        "answer": TECH_FAILURE,
        "handoff": True,
        "handoff_reason": reason,
        "consecutive_failures": state.get("consecutive_failures", 0) + 1,
    }
