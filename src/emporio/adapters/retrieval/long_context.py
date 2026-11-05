"""Estratégia 1: entregar o manual inteiro ao modelo (o manual cabe no contexto)."""

from __future__ import annotations

from emporio.domain.policy import PolicySection, PolicySnippet


def retrievable(sections: list[PolicySection]) -> list[PolicySection]:
    """Seções com corpo; títulos-pai vazios (§5, §6, §8) não carregam informação."""
    return [s for s in sections if s.text.strip()]


class LongContextRetriever:
    """Devolve todas as seções com corpo, na ordem do manual, ignorando a pergunta."""

    def __init__(self, sections: list[PolicySection]) -> None:
        """Pré-formata os trechos."""
        self._snippets = [PolicySnippet.from_section(s) for s in retrievable(sections)]

    def retrieve(self, query: str, k: int = 3) -> list[PolicySnippet]:
        """Todos os trechos; ``query`` e ``k`` não filtram nesta estratégia."""
        return list(self._snippets)
