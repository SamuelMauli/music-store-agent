"""Guarda de entrada: normaliza o texto e detecta tentativa óbvia de injeção de prompt."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from emporio.domain.text import normalize_text

MAX_INPUT_CHARS: Final = 2000
REFUSAL: Final = (
    "Não consigo mudar regras, preços nem descontos por aqui, e não compartilho minhas "
    "instruções. Posso te ajudar com instrumentos, preços, pedidos e trocas. "
    "O que você procura?"
)
_INVISIBLE: Final = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_INJECTION: Final = tuple(
    re.compile(p)
    for p in (
        r"\b(ignore|ignora|desconsidere|esqueca|desobedeca|descarte)\b.{0,40}"
        r"\b(regras?|instrucoes|instrucao|prompt|politicas?|orientacoes)\b",
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(rules?|instructions?|prompt|polic)",
        r"\b(revele|mostre|exiba|repita|vaze|imprima)\b.{0,30}\b(prompt|instrucoes|system)\b",
        r"\b(reveal|show|print|repeat)\b.{0,30}\b(system prompt|instructions)\b",
        r"\bsystem prompt\b|\bprompt do sistema\b",
        r"\b(modo|mode)\s+(desenvolvedor|developer|admin|deus|god)\b",
        r"\b(voce agora e|a partir de agora voce|finja que voce|aja como se)\b",
        r"\b(jailbreak|dan mode)\b",
    )
)


def normalize(text: str) -> str:
    """NFKC, sem caracteres invisíveis ou de controle, espaços colapsados, limitado a 2000.

    Args:
        text: Mensagem crua do cliente.

    Returns:
        Texto limpo.
    """
    text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", text))
    text = "".join(c if unicodedata.category(c)[0] != "C" else " " for c in text)
    return " ".join(text.split())[:MAX_INPUT_CHARS]


def looks_like_injection(text: str) -> bool:
    """Heurística de padrões conhecidos (PT/EN); a defesa real é o prompt e a guarda de saída.

    Args:
        text: Mensagem já normalizada.

    Returns:
        ``True`` se casar com algum padrão de "ignore as regras" ou "revele o prompt".
    """
    norm = normalize_text(text)
    return any(p.search(norm) for p in _INJECTION)
