"""Nó ``verification``: pedido + e-mail ou telefone, com no máximo 3 tentativas."""

from __future__ import annotations

import json
from typing import Any, Final

from langchain_core.runnables import RunnableConfig

from emporio.agent.identity import Identity, extract_identity
from emporio.agent.intents import VERIFICATION_INTENTS, RouteDecision
from emporio.agent.nodes.deps import Deps, inbound_text, route_of, tool_context, trace_of
from emporio.agent.state import ChatState

MAX_ATTEMPTS: Final = 3
ASK_BOTH: Final = (
    "Para consultar seu pedido, preciso do número do pedido e do e-mail ou telefone "
    "usado no cadastro. Pode me passar?"
)
ASK_CONTACT: Final = (
    "Obrigado! Agora preciso do e-mail ou do telefone cadastrado na compra, "
    "para confirmar que o pedido é seu."
)
ASK_ORDER: Final = "Qual é o número do pedido?"
RETRY: Final = (
    "Não consegui confirmar com esses dados. Confere o número do pedido e o e-mail ou "
    "telefone do cadastro e me manda de novo?"
)


def _missing_text(order_id: int | None, identity: Identity) -> str:
    if order_id is None and identity.has_contact:
        return ASK_ORDER
    if order_id is not None and not identity.has_contact:
        return ASK_CONTACT
    return ASK_BOTH


def _without_verification(route: RouteDecision) -> RouteDecision:
    kept = [i for i in route.intents if i not in VERIFICATION_INTENTS]
    return route.model_copy(update={"intents": kept}).normalized()


def _pending(route: RouteDecision) -> dict[str, Any]:
    needed = [i.value for i in route.intents if i in VERIFICATION_INTENTS]
    return {
        "intents": needed,
        "entities": route.entities.model_dump(exclude_none=True, mode="json"),
    }


def _ask(state: ChatState, route: RouteDecision, text: str) -> dict[str, Any]:
    """Pede o dado que falta; se sobrar intent sem verificação, o agente segue com ela."""
    rest = _without_verification(route)
    base: dict[str, Any] = {"pending": _pending(route)}
    if not rest.intents:
        return {**base, "answer": text}
    notes = [*state.get("scope_notes", []), text]
    return {**base, "route": rest.model_dump(mode="json"), "scope_notes": notes}


def make_verification(deps: Deps) -> Any:
    """Nó ``verification``: libera dados de pedido só depois de confirmar o dono."""

    def verification(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
        route = route_of(state)
        if not route.needs_verification or state.get("verified_customer_id") is not None:
            return {}
        identity = extract_identity(inbound_text(config), allow_bare_order=True)
        order_id = identity.order_id or route.entities.order_id
        if order_id is None or not identity.has_contact:
            return _ask(state, route, _missing_text(order_id, identity))
        args: dict[str, Any] = {"order_id": order_id}
        args.update({"email": identity.email} if identity.email else {"phone": identity.phone})
        execution = deps.registry.call("verify_customer", args, tool_context(state))
        payload = json.loads(execution.payload)
        traces = [*state.get("tool_traces", []), trace_of(execution)]
        attempts = state.get("verification_attempts", 0)
        data = payload.get("data") or {}
        if payload["status"] == "ok" and data.get("verified"):
            entities = route.entities.model_copy(update={"order_id": order_id})
            return {
                "tool_traces": traces,
                "verified_customer_id": execution.state_updates["verified_customer_id"],
                "verified_first_name": data.get("first_name") or "",
                "route": route.model_copy(update={"entities": entities}).model_dump(mode="json"),
            }
        attempts += 1
        blocked = payload["status"] == "policy_blocked" or attempts >= MAX_ATTEMPTS
        updates: dict[str, Any] = {"tool_traces": traces, "verification_attempts": attempts}
        if blocked:
            return {
                **updates,
                "answer": "Não consegui confirmar seus dados por aqui.",
                "handoff": True,
                "handoff_reason": "verificação de cliente falhou",
            }
        return {**updates, **_ask(state, route, RETRY)}

    return verification
