"""Fábrica da estratégia de recuperação, escolhida por ``EMPORIO_POLICY_RETRIEVER``."""

from __future__ import annotations

import duckdb

from emporio.adapters.persistence import DuckDbPolicySectionRepository
from emporio.adapters.retrieval.embedders import FastEmbedEmbedder
from emporio.adapters.retrieval.hybrid import HybridRetriever
from emporio.adapters.retrieval.long_context import LongContextRetriever
from emporio.adapters.retrieval.vector_index import InMemoryVectorIndex
from emporio.domain.policy import PolicySection
from emporio.ports import Embedder, PolicyRetriever
from emporio.settings import Settings


def build(
    settings: Settings,
    sections: list[PolicySection] | None = None,
    embedder: Embedder | None = None,
) -> PolicyRetriever:
    """Monta o retriever configurado.

    Args:
        settings: Configuração (``policy_retriever``, ``db_path``).
        sections: Seções já carregadas; se ``None``, lê do DuckDB.
        embedder: Embedder do modo híbrido; default é o modelo real do fastembed.

    Returns:
        ``LongContextRetriever`` ou ``HybridRetriever``.
    """
    if sections is None:
        con = duckdb.connect(str(settings.db_path), read_only=True)
        try:
            sections = DuckDbPolicySectionRepository(con).all()
        finally:
            con.close()
    if settings.policy_retriever == "long_context":
        return LongContextRetriever(sections)
    return HybridRetriever(sections, embedder or FastEmbedEmbedder(), InMemoryVectorIndex())
