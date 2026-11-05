"""Fim do turno: ``handoff`` (chamado + humano) e ``respond``."""

from __future__ import annotations

import json
import re
from typing import Any, Final

from emporio.agent.nodes.deps import Deps, tool_context, trace_of
from emporio.agent.state import ChatState
from emporio.policies.store import CONTACTS

GREETING: Final = (
    "Oi! Eu sou o Tom, assistente virtual da Empório da Música, e você pode pedir um "
    "atendente humano quando quiser. "
)
HANDOFF_TEXT: Final = (
    "Vou passar seu atendimento para a equipe da loja. Registrei o chamado {protocol} "
    "e retornamos em até {sla}. Se preferir, o telefone é {phone}."
)
HANDOFF_NO_TICKET: Final = (
    "Vou passar seu atendimento para a equipe da loja. Se preferir, fale agora no {phone}."
)
_TRANSPARENCY: Final = "assistente virtual"
_OPENING: Final = re.compile(r"^\s*(?:oi|olá|ola)[!,.]*\s*", re.IGNORECASE)


def make_handoff(deps: Deps) -> Any:
    """Nó ``handoff``: abre um chamado (idempotente) e avisa o cliente."""

    def handoff(state: ChatState) -> dict[str, Any]:
        reason = state.get("handoff_reason") or "encaminhamento solicitado"
        traces = list(state.get("tool_traces", []))
        existing = next(
            (t for t in traces if t["name"] == "open_ticket" and t["status"] == "ok"), None
        )
        if existing is not None:
            payload = json.loads(existing["payload"])
        else:
            execution = deps.registry.call(
                "open_ticket",
                {"kind": "human", "summary": f"Encaminhado a atendente: {reason}"},
                tool_context(state),
            )
            payload = json.loads(execution.payload)
            traces.append(trace_of(execution))
        data = payload.get("data") or {}
        if payload["status"] == "ok":
            text = HANDOFF_TEXT.format(
                protocol=data["protocol"], sla=data["sla"], phone=CONTACTS.phone
            )
        else:
            text = HANDOFF_NO_TICKET.format(phone=CONTACTS.phone)
        return {"answer": f"{state.get('answer', '')} {text}".strip(), "tool_traces": traces}

    return handoff


def respond(state: ChatState) -> dict[str, Any]:
    """Nó ``respond``: junta avisos e resposta, apresenta o Tom no 1º turno e fecha o turno."""
    answer = state.get("answer", "").strip()
    notes = [n for n in state.get("scope_notes", []) if n not in answer]
    text = "\n\n".join([*notes, answer]) if notes else answer
    if state.get("turn", 1) == 1 and _TRANSPARENCY not in text:
        text = GREETING + _OPENING.sub("", text, count=1)
    return {"answer": text, "messages": [{"role": "assistant", "content": text}]}
