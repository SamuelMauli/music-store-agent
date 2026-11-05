"""``open_ticket``: a única escrita do agente (CQRS leve, append-only, idempotente)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt

from emporio.agent.tools.base import Tool, ToolContext
from emporio.agent.tools.results import Ok, ToolResult
from emporio.domain.pii import mask_pii
from emporio.domain.ticket import SLA_TEXT, TicketKind
from emporio.ports import Clock, TicketRepository


class OpenTicketInput(BaseModel):
    """Dados do chamado."""

    kind: Literal["complaint", "privacy", "return", "human"] = Field(
        description="complaint = reclamação; privacy = LGPD; return = devolução; human = atendente."
    )
    summary: str = Field(
        min_length=5,
        max_length=500,
        description="Resumo objetivo do caso, sem e-mail, telefone ou CPF.",
    )
    order_id: PositiveInt | None = None


class TicketView(BaseModel):
    """Chamado aberto."""

    protocol: str
    sla: str
    already_opened: bool = False


class OpenTicket(Tool[OpenTicketInput, TicketView]):
    """Abre chamado para a equipe humana."""

    name = "open_ticket"
    description = (
        "Use para registrar reclamação, pedido de exclusão de dados (LGPD), devolução que exige "
        "a equipe ou pedido de atendente humano. Devolve o protocolo e o prazo de retorno "
        "(24 horas úteis): informe ambos ao cliente. É a única ferramenta que grava dados; "
        "chamar de novo com o mesmo resumo não duplica o chamado."
    )
    input_model = OpenTicketInput
    writes = True

    def __init__(self, tickets: TicketRepository, clock: Clock) -> None:
        self._tickets = tickets
        self._clock = clock

    def run(self, args: OpenTicketInput, ctx: ToolContext) -> ToolResult[TicketView]:
        """Grava o chamado com o resumo já sem PII."""
        ticket = self._tickets.open(
            ctx.session_id,
            TicketKind(args.kind),
            mask_pii(args.summary),
            args.order_id,
            self._clock.now(),
        )
        return Ok(
            TicketView(protocol=ticket.protocol, sla=SLA_TEXT, already_opened=not ticket.created),
            sources=(f"ticket:{ticket.protocol}", "§7.3"),
        )
