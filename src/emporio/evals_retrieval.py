"""Métricas de recuperação (recall@k e MRR) e o runner de ``make eval-retrieval``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from emporio.adapters.persistence import DuckDbPolicySectionRepository
from emporio.adapters.retrieval.embedders import FastEmbedEmbedder
from emporio.adapters.retrieval.hybrid import Bm25Ranker, HybridRetriever
from emporio.adapters.retrieval.long_context import retrievable
from emporio.adapters.retrieval.vector_index import InMemoryVectorIndex
from emporio.domain.policy import PolicySection
from emporio.ports import Embedder
from emporio.settings import load_settings

EVAL_FILE = Path("evals/retrieval.yaml")


class Ranker(Protocol):
    """Qualquer estratégia que ordene ids de seção para uma pergunta."""

    def rank(self, query: str) -> list[str]:
        """Ids do mais ao menos relevante."""
        ...


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Resultado de uma estratégia sobre o conjunto rotulado."""

    recall_at_3: float
    mrr: float
    questions: int


def load_questions(path: Path = EVAL_FILE) -> list[tuple[str, str]]:
    """Pares (pergunta, id da seção correta) do YAML."""
    return [(row["q"], str(row["section"])) for row in yaml.safe_load(path.read_text())]


def score(ranker: Ranker, questions: list[tuple[str, str]]) -> RetrievalScore:
    """Recall@3 e MRR de qualquer objeto com ``rank(query) -> list[str]``."""
    hits, reciprocal = 0, 0.0
    for query, expected in questions:
        ranking = ranker.rank(query)
        if expected in ranking[:3]:
            hits += 1
        if expected in ranking:
            reciprocal += 1.0 / (ranking.index(expected) + 1)
    total = len(questions)
    return RetrievalScore(hits / total, reciprocal / total, total)


def hybrid_for(sections: list[PolicySection], embedder: Embedder) -> HybridRetriever:
    """Híbrido com o embedder dado e índice em memória."""
    return HybridRetriever(sections, embedder, InMemoryVectorIndex())


def main() -> int:
    """Mede BM25 e híbrido (modelo real) e imprime recall@3 e MRR."""
    import duckdb  # noqa: PLC0415

    settings = load_settings()
    con = duckdb.connect(str(settings.db_path), read_only=True)
    sections = retrievable(DuckDbPolicySectionRepository(con).all())
    questions = load_questions()
    results = {
        "bm25": score(Bm25Ranker(sections), questions),
        "hybrid (real)": score(hybrid_for(sections, FastEmbedEmbedder()), questions),
    }
    print(f"{'estratégia':<16}{'recall@3':>10}{'MRR':>8}  (n={len(questions)})")
    for name, result in results.items():
        print(f"{name:<16}{result.recall_at_3:>10.3f}{result.mrr:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
