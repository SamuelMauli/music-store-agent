"""Nó ``output_guard``: valida a resposta do modelo; 1 regeneração, depois resposta segura."""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from emporio.agent.intents import Intent
from emporio.agent.nodes.deps import Deps, route_of
from emporio.agent.nodes.facts import with_return_verdict
from emporio.agent.nodes.react import policy_context
from emporio.agent.state import ChatState
from emporio.guardrails.output import (
    CITING_TOOLS,
    SAFE_ANSWER,
    check_answer,
    grounded_values,
)

log = logging.getLogger("emporio.guard")
MAX_REGENERATIONS: Final = 1


def _sources(deps: Deps, state: ChatState) -> list[str]:
    """Fontes confiáveis do turno: payloads e argumentos das tools, manual e contexto."""
    traces = state.get("tool_traces", [])
    policies = policy_context(deps, state, route_of(state))
    earlier = [m["content"] for m in state.get("messages", []) if m["role"] == "assistant"]
    return [
        *(t["payload"] for t in traces),
        *(json.dumps(t["args"], ensure_ascii=False) for t in traces),
        *(p["text"] for p in policies),
        *state.get("scope_notes", []),
        json.dumps(state.get("last_products", []), ensure_ascii=False),
        *earlier,
    ]


def make_output_guard(deps: Deps) -> Any:
    """Cria o nó que aprova, manda regenerar ou troca por resposta segura com handoff."""

    def output_guard(state: ChatState) -> dict[str, Any]:
        traces = state.get("tool_traces", [])
        pool = _sources(deps, state)
        verdict = check_answer(
            state.get("answer", ""),
            grounded=grounded_values(pool),
            sources=" ".join(pool),
            complaint=Intent.COMPLAINT in route_of(state).intents,
            needs_citation=any(t["name"] in CITING_TOOLS and t["status"] == "ok" for t in traces),
        )
        if verdict.ok:
            return {"answer": with_return_verdict(verdict.text, traces)}
        codes = ",".join(verdict.violations)
        log.warning("resposta reprovada: %s (regeneracao=%s)", codes, state.get("regenerations", 0))
        if state.get("regenerations", 0) < MAX_REGENERATIONS:
            return {
                "answer": "",
                "tool_traces": [t for t in traces if t["name"] == "open_ticket"],
                "regenerations": state.get("regenerations", 0) + 1,
                "feedback": " ".join(verdict.details),
            }
        return {
            "answer": SAFE_ANSWER,
            "blocked": True,
            "handoff": True,
            "handoff_reason": f"resposta reprovada pelas guardas ({codes})",
        }

    return output_guard
