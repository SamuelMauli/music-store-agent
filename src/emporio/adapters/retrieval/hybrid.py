"""Estratégia 2: BM25 + embeddings densos fundidos por Reciprocal Rank Fusion."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from emporio.adapters.retrieval.long_context import retrievable
from emporio.domain.policy import PolicySection, PolicySnippet
from emporio.domain.text import normalize_text
from emporio.ports import Embedder, VectorIndex

RRF_K = 60
_WORD = re.compile(r"\w+")
_STOPWORDS = frozenset(
    [
        "a",
        "o",
        "as",
        "os",
        "um",
        "uma",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "por",
        "para",
        "com",
        "e",
        "ou",
        "que",
        "se",
        "eu",
        "meu",
        "minha",
        "é",
        "ao",
        "aos",
        "quero",
        "posso",
        "tem",
        "ter",
        "ser",
        "foi",
    ]
)
_STEM = 6


def tokenize(text: str) -> list[str]:
    """Tokens sem acento nem stopword, com radical de 6 letras (casa "devolver"/"devolução")."""
    words = _WORD.findall(normalize_text(text))
    return [w[:_STEM] for w in words if w not in _STOPWORDS]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: ``soma 1/(k + posição)``, posição a partir de 1.

    Empate de score resolvido pela ordem de primeira aparição (determinístico).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda item: -item[1])


class Bm25Ranker:
    """Ranking lexical por BM25 sobre o texto de busca de cada seção."""

    def __init__(self, sections: list[PolicySection]) -> None:
        """Indexa as seções com corpo."""
        self._ids = [s.section_id for s in sections]
        self._bm25 = BM25Okapi([tokenize(s.search_text) for s in sections])

    def rank(self, query: str) -> list[str]:
        """Ids com score positivo, do melhor ao pior."""
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self._ids)), key=lambda i: (-scores[i], i))
        return [self._ids[i] for i in order if scores[i] > 0]


class HybridRetriever:
    """Top-k por RRF entre o ranking BM25 e o ranking denso."""

    def __init__(
        self, sections: list[PolicySection], embedder: Embedder, index: VectorIndex
    ) -> None:
        """Indexa as seções com corpo no BM25 e no índice vetorial."""
        usable = retrievable(sections)
        self._by_id = {s.section_id: s for s in usable}
        self._bm25 = Bm25Ranker(usable)
        self._embedder = embedder
        self._index = index
        index.add([s.section_id for s in usable], embedder.embed([s.search_text for s in usable]))

    def rank(self, query: str) -> list[str]:
        """Ids ordenados por RRF (todos os que algum ranking enxergou)."""
        dense = [
            i for i, _ in self._index.search(self._embedder.embed([query])[0], len(self._by_id))
        ]
        return [doc_id for doc_id, _ in rrf([self._bm25.rank(query), dense])]

    def retrieve(self, query: str, k: int = 3) -> list[PolicySnippet]:
        """Os ``k`` trechos mais relevantes, já com a citação."""
        return [PolicySnippet.from_section(self._by_id[i]) for i in self.rank(query)[:k]]
