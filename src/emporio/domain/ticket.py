"""Chamados de suporte (contexto Suporte): a única escrita do agente."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

SLA_TEXT = "24 horas úteis"


class TicketKind(StrEnum):
    """Natureza do chamado."""

    COMPLAINT = "complaint"
    PRIVACY = "privacy"
    RETURN = "return"
    HUMAN = "human"


@dataclass(frozen=True, slots=True)
class Ticket:
    """Chamado aberto.

    Attributes:
        protocol: Protocolo ``EMP-AAAAMMDD-NNNN``.
        kind: Natureza do chamado.
        summary: Resumo sem PII.
        order_id: Pedido relacionado, se houver.
        session_id: Sessão de origem.
        created: Se este chamado foi criado agora (``False`` quando a chamada foi repetida).
    """

    protocol: str
    kind: TicketKind
    summary: str
    order_id: int | None
    session_id: str
    created: bool = True


def idempotency_key(session_id: str, kind: TicketKind, summary: str) -> str:
    """Hash de sessão + tipo + resumo: repetir a mesma chamada não duplica o chamado."""
    raw = f"{session_id}\0{kind.value}\0{' '.join(summary.lower().split())}"
    return sha256(raw.encode()).hexdigest()
