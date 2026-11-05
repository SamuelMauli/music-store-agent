"""Nó ``agent``: laço ReAct limitado, só com as tools das intents roteadas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from emporio.agent.intents import POLICY_INTENTS, RouteDecision, tools_for
from emporio.agent.llm.types import (
    LLMError,
    Message,
    ToolCall,
    Usage,
    assistant_message,
    tool_message,
)
from emporio.agent.nodes.deps import (
    HISTORY_WINDOW,
    MAX_TOOL_CALLS,
    Deps,
    add_usage,
    failure,
    route_of,
    tool_context,
    trace_of,
)
from emporio.agent.prompts import render
from emporio.agent.state import ChatState
from emporio.agent.tools import ToolExecution
from emporio.agent.tools.formatting import when
from emporio.agent.tools.results import Ok
from emporio.policies.store import is_store_open, next_opening

NOT_ALLOWED = json.dumps(
    {"status": "error", "message": "Ferramenta indisponível para este assunto."}
)


def history(state: ChatState) -> list[Message]:
    """Janela das últimas mensagens (user/assistant) da conversa."""
    return list(state.get("messages", []))[-HISTORY_WINDOW:]


def policy_context(deps: Deps, state: ChatState, route: RouteDecision) -> list[dict[str, Any]]:
    if not any(i in POLICY_INTENTS for i in route.intents):
        return []
    snippets = deps.retriever.retrieve(state["user_text"], 3)
    return [{"section_id": s.section_id, "title": s.title, "text": s.text} for s in snippets]


def system_prompt(
    deps: Deps, state: ChatState, route: RouteDecision, policies: list[dict[str, Any]]
) -> str:
    """Prompt de sistema do turno (sem identificadores, só o primeiro nome)."""
    now = deps.clock.now()
    return render(
        "system",
        now_text=when(now),
        store_open=is_store_open(now),
        next_opening=when(next_opening(now)),
        customer_first_name=state.get("verified_first_name", ""),
        order_id=route.entities.order_id,
        summary=state.get("summary", ""),
        focus_products=state.get("last_products", []),
        route_intents=[i.value for i in route.intents],
        route_entities=json.dumps(
            route.entities.model_dump(exclude_none=True, mode="json"), ensure_ascii=False
        ),
        scope_notes=state.get("scope_notes", []),
        feedback=state.get("feedback", ""),
        policy_context=policies,
    )


def _products_of(executions: list[ToolExecution]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for execution in executions:
        if isinstance(execution.result, Ok):
            refs.extend(
                {"product_id": r.product_id, "name": r.name, "price": r.price}
                for r in execution.result.refs
            )
    return list({r["product_id"]: r for r in refs}.values())[:5]


def _needs_input(route: RouteDecision, traces: list[dict[str, Any]]) -> bool:
    """O agente terminou pedindo um dado ao cliente (canal, cidade, modelo...)."""
    if any(t["status"] == "ambiguous" for t in traces):
        return True
    e = route.entities
    called = {t["name"] for t in traces}
    missing_city = "shipping" in route.intents and not e.city and "estimate_shipping" not in called
    productless = any(
        i in route.intents for i in ("product_detail", "payment_simulation")
    ) and not (e.product_name or e.referenced_product_ids or traces)
    return missing_city or productless


def _remember(route: RouteDecision, traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not _needs_input(route, traces):
        return {"pending": {}}
    return {
        "pending": {
            "intents": [i.value for i in route.intents],
            "entities": route.entities.model_dump(exclude_none=True, mode="json"),
        }
    }


@dataclass(slots=True)
class _Loop:
    """Resultado acumulado do laço ReAct de um turno."""

    messages: list[Message]
    traces: list[dict[str, Any]]
    executions: list[ToolExecution] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    answer: str = ""


class ReactAgent:
    """Nó ``agent``: no máximo 4 tool calls por turno, só das intents roteadas."""

    def __init__(self, deps: Deps) -> None:
        """Cria o nó."""
        self._deps = deps

    def _run_tools(
        self, loop: _Loop, calls: tuple[ToolCall, ...], allowed: list[str], state: ChatState
    ) -> None:
        for call in calls:
            if call.name not in allowed:
                loop.messages.append(tool_message(call, NOT_ALLOWED))
                continue
            execution = self._deps.registry.call(call.name, call.args, tool_context(state))
            loop.executions.append(execution)
            loop.traces.append(trace_of(execution))
            loop.messages.append(tool_message(call, execution.payload))

    def _loop(self, loop: _Loop, allowed: list[str], state: ChatState) -> None:
        schemas = self._deps.registry.schemas(allowed) or None
        budget = MAX_TOOL_CALLS
        while True:
            response = self._deps.llm.complete(
                loop.messages, tier="medium", temperature=0.3, tools=schemas if budget else None
            )
            loop.usage += response.usage
            calls = response.tool_calls[:budget]
            if not calls:
                loop.answer = response.text.strip()
                return
            loop.messages.append(assistant_message(replace(response, tool_calls=calls)))
            self._run_tools(loop, calls, allowed, state)
            budget -= len(calls)

    def __call__(self, state: ChatState) -> dict[str, Any]:
        """Executa o turno do agente."""
        deps = self._deps
        route = route_of(state)
        policies = policy_context(deps, state, route)
        loop = _Loop(
            [
                {"role": "system", "content": system_prompt(deps, state, route, policies)},
                *history(state),
            ],
            list(state.get("tool_traces", [])),
        )
        try:
            self._loop(loop, tools_for(route.intents), state)
        except LLMError:
            reason = "falha no modelo de resposta"
            return {**failure(state, reason), "usage": add_usage(state, loop.usage)}
        if not loop.answer:
            return {**failure(state, "resposta vazia do modelo"), "tool_traces": loop.traces}
        updates: dict[str, Any] = {
            "answer": loop.answer,
            "tool_traces": loop.traces,
            "usage": add_usage(state, loop.usage),
            "policy_sections": [p["section_id"] for p in policies],
            "consecutive_failures": 0,
            **_remember(route, loop.traces),
        }
        if products := _products_of(loop.executions):
            updates["last_products"] = products
        return updates


def make_agent(deps: Deps) -> Any:
    """Cria o nó ``agent``."""
    agent = ReactAgent(deps)

    def node(state: ChatState) -> dict[str, Any]:
        return agent(state)

    return node
