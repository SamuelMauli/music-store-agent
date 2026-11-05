"""Resolve texto livre ("violões", "sax") em categoria do catálogo."""

from __future__ import annotations

import re
from typing import Final

from emporio.domain.text import normalize_text

CATEGORY_NAMES: Final = {
    1: "Guitarras",
    2: "Baixos",
    3: "Baterias e Percussão",
    4: "Teclados e Pianos",
    5: "Violões",
    6: "Instrumentos de Sopro (Madeiras)",
    7: "Instrumentos de Sopro (Metais)",
    8: "Cordas Orquestrais",
    9: "Ukuleles",
}

# Ordem importa: "contrabaixo" e "viola" precisam ser testados antes de "baixo" e "violão".
_PATTERNS: Final = (
    (8, r"\b(violino|violinos|violoncelo|violoncelos|cello|contrabaixo|contrabaixos|violas?)\b"),
    (6, r"\b(flautas?|clarinetes?|saxofones?|saxes?|sax|oboes?|madeiras?)\b"),
    (7, r"\b(trompetes?|trombones?|tubas?|trompas?|metais|metal)\b"),
    (9, r"\b(ukuleles?|ukeleles?|ukulelês)\b"),
    (3, r"\b(baterias?|percussao|cajon|cajons)\b"),
    (4, r"\b(teclados?|pianos?|sintetizadores?|controladores?)\b"),
    (5, r"\b(violao|violoes)\b"),
    (2, r"\b(baixos?)\b"),
    (1, r"\b(guitarras?)\b"),
)


def resolve_category(text: str) -> int | None:
    """Mapeia texto livre para o ``category_id``.

    Args:
        text: Trecho do cliente, ex.: ``"violões"`` ou ``"saxofone"``.

    Returns:
        O ID da categoria ou ``None`` se nada casar.
    """
    needle = normalize_text(text)
    for category_id, pattern in _PATTERNS:
        if re.search(pattern, needle):
            return category_id
    return None
