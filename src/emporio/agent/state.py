"""Estado do grafo: só tipos JSON-friendly, para o checkpointer serializar sem surpresas."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

Message = dict[str, Any]


class ChatState(TypedDict, total=False):
    """Estado de uma conversa (``thread_id`` = sessão).

    Persiste entre turnos: ``messages``, ``summary``, ``summarized_upto``, ``pending``,
    ``verified_customer_id``, ``verified_first_name``, ``verification_attempts``,
    ``last_products``, ``turn``. O resto é do turno e é zerado no ``input_guard``.
    """

    session_id: str
    messages: Annotated[list[Message], operator.add]
    summary: str
    summarized_upto: int
    pending: dict[str, Any]
    verified_customer_id: int | None
    verified_first_name: str
    verification_attempts: int
    last_products: list[dict[str, Any]]
    turn: int
    # --- por turno ---
    user_text: str
    route: dict[str, Any]
    scope_notes: list[str]
    tool_traces: list[dict[str, Any]]
    policy_sections: list[str]
    answer: str
    handoff: bool
    handoff_reason: str
    usage: dict[str, float | int]
    blocked: bool
    regenerations: int
    feedback: str
    consecutive_failures: int


TURN_RESET: dict[str, Any] = {
    "route": {},
    "scope_notes": [],
    "tool_traces": [],
    "policy_sections": [],
    "answer": "",
    "handoff": False,
    "handoff_reason": "",
    "usage": {},
    "blocked": False,
    "regenerations": 0,
    "feedback": "",
}
