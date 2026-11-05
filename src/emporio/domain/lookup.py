"""Resultado da busca de produto por nome livre (união etiquetada)."""

from __future__ import annotations

from dataclasses import dataclass

from emporio.domain.entities import CatalogEntry

MAX_OPTIONS = 3


@dataclass(frozen=True, slots=True)
class Found:
    """Um único produto corresponde ao que o cliente disse.

    Attributes:
        entry: O produto resolvido.
        score: Similaridade (0 a 100) que o elegeu.
    """

    entry: CatalogEntry
    score: float


@dataclass(frozen=True, slots=True)
class Ambiguous:
    """Mais de um produto empata: o agente deve perguntar qual o cliente quis.

    Attributes:
        options: Até :data:`MAX_OPTIONS` candidatos, em ordem estável de ID.
    """

    options: tuple[CatalogEntry, ...]


@dataclass(frozen=True, slots=True)
class NotFound:
    """Nenhum produto atinge o limiar de similaridade.

    Attributes:
        query: Texto original da consulta.
    """

    query: str


NameLookup = Found | Ambiguous | NotFound
