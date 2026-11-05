"""Roteador de intents: LLM pequeno, temperatura 0, saída estruturada."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from emporio.agent.intents import RouteDecision
from emporio.agent.llm.types import LLMProvider, Usage
from emporio.agent.prompts import render


class Router:
    """Classifica a mensagem em intents e entidades."""

    def __init__(self, llm: LLMProvider) -> None:
        """Cria o roteador.

        Args:
            llm: Provedor de LLM (o tier ``small`` é usado).
        """
        self._llm = llm

    def route(
        self,
        text: str,
        *,
        summary: str = "",
        focus_products: Sequence[dict[str, Any]] = (),
        order_id: int | None = None,
        pending: str = "",
    ) -> tuple[RouteDecision, Usage]:
        """Decide as intents da mensagem.

        Args:
            text: Mensagem já sanitizada e sem PII.
            summary: Resumo da conversa até aqui.
            focus_products: Produtos citados antes (``product_id`` e ``name``).
            order_id: Pedido em atendimento, se houver.
            pending: Intents que aguardam resposta do cliente (ex.: ``return_exchange``).

        Returns:
            Decisão normalizada (``needs_verification`` calculado pelo código) e o consumo.
        """
        system = render(
            "router",
            summary=summary,
            focus_products=list(focus_products),
            order_id=order_id,
            pending=pending,
        )
        decision, usage = self._llm.structured(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            RouteDecision,
            tier="small",
            temperature=0.0,
        )
        return decision.normalized(), usage
