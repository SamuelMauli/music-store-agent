"""Detecção e mascaramento de PII (e-mail, telefone BR, CPF) em texto livre.

Usado por logs, traces e guardas: nada com PII crua sai do processo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

PiiKind = Literal["email", "phone", "cpf"]

EMAIL_RE: Final = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
CPF_RE: Final = re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)")
PHONE_RE: Final = re.compile(r"(?<![\d.])(?:\+?55[\s-]?)?\(?\d{2}\)?[\s-]?9?\d{4}[\s-]?\d{4}(?!\d)")


@dataclass(frozen=True, slots=True)
class PiiMatch:
    """Um trecho de PII encontrado.

    Attributes:
        kind: Tipo do dado pessoal.
        value: Texto original encontrado.
        start: Posição inicial no texto.
        end: Posição final (exclusiva).
    """

    kind: PiiKind
    value: str
    start: int
    end: int

    def masked(self) -> str:
        """Versão mascarada, compatível com ``Email.masked`` e ``Phone.masked``."""
        if self.kind == "email":
            user, _, domain = self.value.lower().partition("@")
            return f"{user[:1]}***@{domain}"
        digits = re.sub(r"\D", "", self.value)
        if self.kind == "cpf":
            return f"***.***.***-{digits[-2:]}"
        digits = digits[2:] if len(digits) in (12, 13) and digits.startswith("55") else digits
        ddd, number = digits[:2], digits[2:]
        return f"({ddd}) {'*' * (len(number) - 4)}-{number[-4:]}"


def find_pii(text: str) -> list[PiiMatch]:
    """Localiza PII sem sobreposição, em ordem de aparição.

    Args:
        text: Texto livre.

    Returns:
        Ocorrências de CPF, e-mail e telefone.
    """
    taken: list[PiiMatch] = []
    patterns: tuple[tuple[PiiKind, re.Pattern[str]], ...] = (
        ("cpf", CPF_RE),
        ("email", EMAIL_RE),
        ("phone", PHONE_RE),
    )
    for kind, pattern in patterns:
        for m in pattern.finditer(text):
            if not any(m.start() < t.end and t.start < m.end() for t in taken):
                taken.append(PiiMatch(kind, m.group(), m.start(), m.end()))
    return sorted(taken, key=lambda item: item.start)


def mask_pii(text: str) -> str:
    """Substitui cada PII pela forma mascarada.

    Args:
        text: Texto livre.

    Returns:
        O texto com e-mails, telefones e CPFs mascarados.
    """
    out, cursor = [], 0
    for match in find_pii(text):
        out.append(text[cursor : match.start])
        out.append(match.masked())
        cursor = match.end
    out.append(text[cursor:])
    return "".join(out)
