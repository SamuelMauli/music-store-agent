"""Grafo LangGraph: input_guard -> router -> scope_gate -> verification -> agent -> respond."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from emporio.agent.nodes.deps import Deps
from emporio.agent.nodes.finish import make_handoff, respond
from emporio.agent.nodes.intake import input_guard, make_router, scope_node
from emporio.agent.nodes.react import make_agent
from emporio.agent.nodes.verify import make_verification
from emporio.agent.state import ChatState

type CompiledChat = CompiledStateGraph[ChatState, None, ChatState, ChatState]


def _after(next_node: str) -> Callable[[ChatState], str]:
    """Aresta condicional: encaminha a humano, encerra com resposta pronta ou segue."""

    def route(state: ChatState) -> str:
        if state.get("handoff"):
            return "handoff"
        return "respond" if state.get("answer") else next_node

    return route


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver[Any] | None = None) -> CompiledChat:
    """Compila o grafo do agente.

    Args:
        deps: Dependências dos nós (LLM, tools, retriever, relógio).
        checkpointer: Persistência por ``thread_id`` (sessão).

    Returns:
        Grafo compilado, pronto para ``invoke``/``stream``.
    """
    graph = StateGraph(ChatState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("router", make_router(deps))
    graph.add_node("scope_gate", scope_node)
    graph.add_node("verification", make_verification(deps))
    graph.add_node("agent", make_agent(deps))
    graph.add_node("handoff", make_handoff(deps))
    graph.add_node("respond", respond)
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
        "agent", _after("respond"), {"respond": "respond", "handoff": "handoff"}
    )
    graph.add_edge("handoff", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
