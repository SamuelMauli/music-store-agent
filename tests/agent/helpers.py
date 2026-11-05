"""Pequeno driver que reproduz um turno (router -> agente ReAct) sem o grafo."""

from __future__ import annotations

import json
from typing import Any

from emporio.agent.intents import Intent, RouteDecision, tools_for
from emporio.agent.llm.fake import FakeLLM
from emporio.agent.llm.types import assistant_message, tool_message
from emporio.agent.prompts import render
from emporio.agent.router import Router
from emporio.agent.tools import ToolContext, ToolRegistry


def system_prompt(route: RouteDecision, focus: list[dict[str, Any]] | None = None) -> str:
    return render(
        "system",
        now_text="quarta-feira, 25/03/2026 às 10:00",
        store_open=True,
        next_opening="",
        customer_first_name="",
        order_id=route.entities.order_id,
        summary="",
        focus_products=focus or [],
        route_intents=[i.value for i in route.intents],
        route_entities=json.dumps(route.entities.model_dump(exclude_none=True, mode="json")),
        scope_notes=[],
        feedback="",
        policy_context=[],
    )


def run_turn(
    fake: FakeLLM,
    registry: ToolRegistry,
    text: str,
    *,
    focus: list[dict[str, Any]] | None = None,
    ctx: ToolContext | None = None,
) -> tuple[str, list[str], RouteDecision]:
    route, _ = Router(fake).route(text, focus_products=focus or [])
    names = tools_for(route.intents)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(route, focus)},
        {"role": "user", "content": text},
    ]
    used: list[str] = []
    for _ in range(4):
        response = fake.complete(messages, tools=registry.schemas(list(names)) or None)
        if not response.tool_calls:
            return response.text, used, route
        messages.append(assistant_message(response))
        for call in response.tool_calls:
            result = registry.call(call.name, call.args, ctx or ToolContext())
            used.append(call.name)
            messages.append(tool_message(call, result.payload))
    raise AssertionError("o fake não parou de pedir tools")


__all__ = ["Intent", "run_turn", "system_prompt"]
