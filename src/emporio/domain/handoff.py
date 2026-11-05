"""Handoff humano: estado da conversa que o bot não responde mais e a fila de atendimento."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class HandoffState(StrEnum):
    """``waiting``: na fila, sem dono; ``assigned``: um operador assumiu."""

    WAITING = "waiting"
    ASSIGNED = "assigned"


class ClaimResult(StrEnum):
    """Resultado de um operador tentar assumir uma conversa."""

    OK = "ok"
    NOT_FOUND = "not_found"
    TAKEN = "taken"


@dataclass(frozen=True, slots=True)
class HandoffMessage:
    """Mensagem trocada durante o handoff (``in`` = cliente, ``out`` = operador)."""

    direction: Literal["in", "out"]
    text: str
    at: float


@dataclass(frozen=True, slots=True)
class HandoffConversation:
    """Conversa em atendimento humano.

    Attributes:
        id: Identificador opaco (HMAC com pepper do contato); é o que a API e os logs veem.
        phone: Contato real, guardado só para a entrega; nunca exposto pela API nem logado.
        state: Na fila ou assumida.
        operator: Nome informado por quem assumiu.
        entered_at: Epoch de entrada em handoff.
        expires_at: Epoch em que a conversa volta ao bot se ninguém agir.
        messages: Histórico do handoff, em ordem.
    """

    id: str
    phone: str
    state: HandoffState
    operator: str | None
    entered_at: float
    expires_at: float
    messages: tuple[HandoffMessage, ...] = field(default_factory=tuple)


def contact_id(pepper: str, phone: str) -> str:
    """Id opaco e estável do contato: HMAC-SHA256 com pepper (não reversível sem o pepper)."""
    return hmac.new(pepper.encode(), phone.encode(), hashlib.sha256).hexdigest()[:32]
