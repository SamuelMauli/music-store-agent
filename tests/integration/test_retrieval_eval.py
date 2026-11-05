from __future__ import annotations

import duckdb
import pytest

from emporio.adapters.persistence import DuckDbPolicySectionRepository
from emporio.adapters.retrieval import retriever_factory
from emporio.adapters.retrieval.embedders import FakeEmbedder, FastEmbedEmbedder
from emporio.adapters.retrieval.hybrid import Bm25Ranker, HybridRetriever
from emporio.adapters.retrieval.long_context import LongContextRetriever, retrievable
from emporio.domain.policy import PolicySection
from emporio.evals_retrieval import hybrid_for, load_questions, score
from emporio.settings import Settings


@pytest.fixture
def sections(con: duckdb.DuckDBPyConnection) -> list[PolicySection]:
    return retrievable(DuckDbPolicySectionRepository(con).all())


def test_eval_set_is_labelled_with_real_sections(sections: list[PolicySection]) -> None:
    questions = load_questions()
    known = {s.section_id for s in sections}
    assert len(questions) >= 30
    assert {expected for _, expected in questions} <= known


def test_bm25_recall_and_mrr(sections: list[PolicySection]) -> None:
    result = score(Bm25Ranker(sections), load_questions())
    assert result.recall_at_3 >= 0.9
    assert result.mrr >= 0.85


def test_fake_hybrid_recall_and_mrr(sections: list[PolicySection]) -> None:
    result = score(hybrid_for(sections, FakeEmbedder()), load_questions())
    assert result.recall_at_3 >= 0.85
    assert result.mrr >= 0.75


def test_factory_reads_sections_from_duckdb(ingested: Settings) -> None:
    retriever = retriever_factory.build(
        ingested.model_copy(update={"policy_retriever": "long_context"})
    )
    assert isinstance(retriever, LongContextRetriever)
    snippets = retriever.retrieve("tanto faz")
    assert len(snippets) == 25
    assert snippets[0].formatted().startswith("[§1 Sobre a Empório da Música] ")
    hybrid = retriever_factory.build(
        ingested.model_copy(update={"policy_retriever": "hybrid"}), embedder=FakeEmbedder()
    )
    assert isinstance(hybrid, HybridRetriever)


@pytest.mark.live
def test_real_model_hybrid_recall(sections: list[PolicySection]) -> None:
    result = score(hybrid_for(sections, FastEmbedEmbedder()), load_questions())
    assert result.recall_at_3 >= 0.9
    assert result.mrr >= 0.8
