"""Entrada do turno: ``input_guard``, ``router`` e ``scope_gate``."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from emporio.agent.identity import Identity, extract_identity, redact
from emporio.agent.intents import Entities, RouteDecision
from emporio.agent.llm.types import LLMError
from emporio.agent.nodes.deps import Deps, add_usage, failure, inbound_text, route_of
from emporio.agent.scope import scope_gate
from emporio.agent.state import TURN_RESET, ChatState
from emporio.guardrails.input import REFUSAL, normalize

log = logging.getLogger("emporio.guard")


def make_input_guard(deps: Deps) -> Any:
    """Nó ``input_guard``: expira a verificação vencida e prepara o turno."""

    def input_guard(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
        """Normaliza a mensagem, tira a PII, barra injeção óbvia e zera o que é do turno."""
        text = normalize(inbound_text(config))
        safe = redact(text)
        updates: dict[str, Any] = {
            **TURN_RESET,
            "user_text": safe,
            "messages": [{"role": "user", "content": safe}],
            "turn": state.get("turn", 0) + 1,
        }
        if state.get("verified_customer_id") is not None:
            age = deps.wall() - state.get("verified_at", 0.0)
            if age >= deps.verify_policy.ttl_seconds:  # sem verified_at (legado) = expirada
                updates.update(verified_customer_id=None, verified_first_name="", verified_at=0.0)
        verdict = deps.injection_guard.check(text)
        if verdict.blocked:
            log.warning(
                "injecao detectada na sessao %s (camada=%s, score=%s, %.1f ms): %s",
                state.get("session_id"),
                verdict.reason,
                "-" if verdict.score is None else f"{verdict.score:.3f}",
                verdict.elapsed_ms,
                safe[:80],
            )
            updates.update(answer=REFUSAL, blocked=True)
        elif verdict.degraded:
            log.warning("guarda de injecao degradada: so heuristica nesta mensagem")
        return updates

    return input_guard


_SHORT_REPLY = 6


def _from_pending(
    pending: dict[str, Any], identity: Identity, extra: Entities | None
) -> RouteDecision:
    """Retoma o que estava pendente com os dados que o cliente acabou de dar."""
    entities = Entities.model_validate(pending.get("entities") or {})
    if extra is not None:
        entities = entities.model_copy(update=extra.model_dump(exclude_none=True))
    if identity.order_id is not None:
        entities = entities.model_copy(update={"order_id": identity.order_id})
    return RouteDecision(intents=pending["intents"], entities=entities).normalized()


def _is_followup(pending: dict[str, Any], route: RouteDecision, text: str) -> bool:
    """Resposta curta a uma pergunta pendente, mesmo que o modelo a tenha classificado à parte."""
    if route.low_confidence or not route.intents:
        return True
    asked = {str(i) for i in pending.get("intents", [])}
    given = {i.value for i in route.intents}
    return len(text.split()) <= _SHORT_REPLY and not asked & given


def make_router(deps: Deps) -> Any:
    """Nó ``router``: classifica a mensagem ou retoma a pendência."""

    def router(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
        pending = state.get("pending") or {}
        identity = extract_identity(inbound_text(config), allow_bare_order=bool(pending))
        if pending and identity.any:
            route = _from_pending(pending, identity, None)
            return {"route": route.model_dump(mode="json")}
        try:
            route, usage = deps.router.route(
                state["user_text"],
                summary=state.get("summary", ""),
                focus_products=state.get("last_products", []),
                pending=", ".join(pending.get("intents", [])),
            )
        except LLMError:
            return failure(state, "falha no roteador")
        updates: dict[str, Any] = {"usage": add_usage(state, usage)}
        if pending and _is_followup(pending, route, state["user_text"]):
            route = _from_pending(pending, identity, route.entities)
        else:
            updates["pending"] = {}
            if identity.order_id is not None and route.entities.order_id is None:
                entities = route.entities.model_copy(update={"order_id": identity.order_id})
                route = route.model_copy(update={"entities": entities})
        updates["route"] = route.model_dump(mode="json")
        return updates

    return router


def scope_node(state: ChatState) -> dict[str, Any]:
    """Nó ``scope_gate``: recusas e esclarecimentos por template."""
    route = route_of(state)
    outcome = scope_gate(route)
    if outcome.answer:
        return {"answer": outcome.answer}
    updated = route.model_copy(update={"intents": outcome.intents}).normalized()
    return {"route": updated.model_dump(mode="json"), "scope_notes": outcome.notes}
