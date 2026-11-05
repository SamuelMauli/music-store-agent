"""Extração e redação de identidade (pedido, e-mail, telefone) da mensagem do cliente.

O texto cru nunca chega ao LLM, ao estado do grafo nem aos logs: o grafo guarda só a versão
redigida; a identidade é extraída do texto de entrada a cada turno, quando preciso.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from emporio.domain.pii import find_pii
from emporio.domain.text import normalize_text

CONTACT_PLACEHOLDER: Final = "[contato omitido]"
CPF_PLACEHOLDER: Final = "[cpf omitido]"
_ORDER: Final = re.compile(r"(?:pedido|ped)\s*(?:numero|num|n|no|#)?\s*(\d{1,6})\b|#(\d{1,6})\b")
_BARE_NUMBER: Final = re.compile(r"(?<![\d.,])(\d{1,6})(?![\d.,])")


@dataclass(frozen=True, slots=True)
class Identity:
    """Dados de identificação que o cliente informou na mensagem."""

    order_id: int | None = None
    email: str | None = None
    phone: str | None = None

    @property
    def has_contact(self) -> bool:
        """Há e-mail ou telefone."""
        return bool(self.email or self.phone)

    @property
    def any(self) -> bool:
        """O cliente informou algo útil à verificação."""
        return self.order_id is not None or self.has_contact


def redact(text: str) -> str:
    """Troca e-mail, telefone e CPF por marcadores neutros."""
    out, cursor = [], 0
    for match in find_pii(text):
        out.append(text[cursor : match.start])
        out.append(CPF_PLACEHOLDER if match.kind == "cpf" else CONTACT_PLACEHOLDER)
        cursor = match.end
    out.append(text[cursor:])
    return "".join(out)


def extract_identity(text: str, *, allow_bare_order: bool = False) -> Identity:
    """Lê pedido, e-mail e telefone do texto cru.

    Args:
        text: Mensagem como o cliente enviou.
        allow_bare_order: Aceita um número solto como pedido (resposta a "qual o pedido?").

    Returns:
        Identidade parcial; campos ausentes ficam ``None``.
    """
    email = phone = None
    for match in find_pii(text):
        if match.kind == "email" and email is None:
            email = match.value
        elif match.kind == "phone" and phone is None:
            phone = match.value
    rest = normalize_text(redact(text))
    found = _ORDER.search(rest)
    order_id = int(next(g for g in found.groups() if g)) if found else None
    if order_id is None and allow_bare_order:
        bare = _BARE_NUMBER.search(rest)
        order_id = int(bare.group(1)) if bare else None
    return Identity(order_id, email, phone)
