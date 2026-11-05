"""Grafo LangGraph: input_guard -> router -> scope_gate -> verification -> agent -> respond."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from emporio.agent.nodes.deps import Deps
from emporio.agent.nodes.finish import make_handoff, respond
from emporio.agent.nodes.guard import make_output_guard
from emporio.agent.nodes.intake import make_input_guard, make_router, scope_node
from emporio.agent.nodes.memory import make_summarizer
from emporio.agent.nodes.react import make_agent
from emporio.agent.nodes.verify import make_verification
from emporio.agent.state import ChatState
from emporio.observability.telemetry import NOOP, Telemetry

type CompiledChat = CompiledStateGraph[ChatState, None, ChatState, ChatState]


def _after(next_node: str) -> Callable[[ChatState], str]:
    """Aresta condicional: encaminha a humano, encerra com resposta pronta ou segue."""

    def route(state: ChatState) -> str:
        if state.get("handoff"):
            return "handoff"
        return "respond" if state.get("answer") else next_node

    return route


def _handoff_or(next_node: str) -> Callable[[ChatState], str]:
    def route(state: ChatState) -> str:
        return "handoff" if state.get("handoff") else next_node

    return route


def _after_guard(state: ChatState) -> str:
    """Aprovada segue; sem resposta é pedido de regeneração; handoff encerra com chamado."""
    if state.get("handoff"):
        return "handoff"
    return "respond" if state.get("answer") else "agent"


def _traced(name: str, node: Callable[..., Any], telemetry: Telemetry) -> Callable[..., Any]:
    """Envolve o nó num span; ``wraps`` preserva a assinatura que o LangGraph inspeciona."""
    if not telemetry.enabled:
        return node

    @functools.wraps(node)
    def run(*args: Any, **kwargs: Any) -> Any:
        with telemetry.span(f"node {name}", **{"emporio.node": name}) as span:
            update = node(*args, **kwargs)
            if isinstance(update, dict) and update.get("handoff"):
                span.set_attribute("emporio.handoff", True)
            if isinstance(update, dict) and update.get("blocked"):
                span.set_attribute("emporio.blocked", True)
            return update

    return run


def build_graph(
    deps: Deps,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    telemetry: Telemetry = NOOP,
) -> CompiledChat:
    """Compila o grafo do agente.

    Args:
        deps: Dependências dos nós (LLM, tools, retriever, relógio).
        checkpointer: Persistência por ``thread_id`` (sessão).
        telemetry: Tracing; cada nó vira um span (no-op por padrão).

    Returns:
        Grafo compilado, pronto para ``invoke``/``stream``.
    """
    graph = StateGraph(ChatState)
    graph.add_node("input_guard", _traced("input_guard", make_input_guard(deps), telemetry))
    graph.add_node("router", _traced("router", make_router(deps), telemetry))
    graph.add_node("scope_gate", _traced("scope_gate", scope_node, telemetry))
    graph.add_node("verification", _traced("verification", make_verification(deps), telemetry))
    graph.add_node("agent", _traced("agent", make_agent(deps), telemetry))
    graph.add_node("output_guard", _traced("output_guard", make_output_guard(deps), telemetry))
    graph.add_node("handoff", _traced("handoff", make_handoff(deps), telemetry))
    graph.add_node("respond", _traced("respond", respond, telemetry))
    graph.add_edge(START, "input_guard")
    for node, following in (
        ("input_guard", "router"),
        ("router", "scope_gate"),
        ("scope_gate", "verification"),
        ("verification", "agent"),
    ):
        graph.add_conditional_edges(
            node,
            _after(following),
            {following: following, "handoff": "handoff", "respond": "respond"},
        )
    graph.add_conditional_edges(
        "agent", _handoff_or("output_guard"), {"output_guard": "output_guard", "handoff": "handoff"}
    )
    graph.add_conditional_edges(
        "output_guard", _after_guard, {"agent": "agent", "respond": "respond", "handoff": "handoff"}
    )
    graph.add_edge("handoff", "respond")
    graph.add_node("summarize", _traced("summarize", make_summarizer(deps), telemetry))
    graph.add_edge("respond", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer)
