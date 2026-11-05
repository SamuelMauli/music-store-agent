from __future__ import annotations

import numpy as np
import pytest

from emporio.adapters.retrieval import retriever_factory
from emporio.adapters.retrieval.embedders import FakeEmbedder
from emporio.adapters.retrieval.hybrid import HybridRetriever, rrf, tokenize
from emporio.adapters.retrieval.long_context import LongContextRetriever
from emporio.adapters.retrieval.vector_index import InMemoryVectorIndex
from emporio.domain.policy import PolicySection, PolicySnippet
from emporio.settings import Settings


def section(sid: str, title: str, text: str, keywords: tuple[str, ...] = ()) -> PolicySection:
    return PolicySection(sid, title, None, 1, text, f"{sid}. {title}", keywords)


SECTIONS = [
    section("1", "Pai", ""),
    section("2", "Horário", "Abrimos de segunda a sexta das 9h às 18h."),
    section("4.1", "Arrependimento", "Devolução em sete dias com estorno.", ("reembolso",)),
    section("5.1", "Frete", "Frete grátis acima de R$ 500,00."),
    section("8", "Garantia", "   "),
]


def test_snippet_format_follows_citation_contract() -> None:
    snippet = PolicySnippet("4.1", "Arrependimento", "7 dias.")
    assert snippet.formatted() == "[§4.1 Arrependimento] 7 dias."


def test_long_context_returns_all_sections_with_body_in_order() -> None:
    retriever = LongContextRetriever(SECTIONS)
    ids = [s.section_id for s in retriever.retrieve("qualquer coisa", k=1)]
    assert ids == ["2", "4.1", "5.1"]


def test_rrf_rewards_agreement_and_uses_k60() -> None:
    fused = dict(rrf([["a", "b"], ["b", "c"]]))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert max(fused, key=lambda key: fused[key]) == "b"
    assert rrf([]) == []


def test_tokenize_strips_accents_stopwords_and_stems() -> None:
    assert tokenize("Quero devolver o violão") == ["devolv", "violao"]
    assert tokenize("devolução") == ["devolu"]


def test_vector_index_returns_nearest_first() -> None:
    index = InMemoryVectorIndex()
    assert index.search(np.array([1.0, 0.0], dtype=np.float32), 3) == []
    index.add(["x", "y"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    index.add(["z"], np.array([[0.6, 0.8]], dtype=np.float32))
    hits = index.search(np.array([0.0, 1.0], dtype=np.float32), 2)
    assert [i for i, _ in hits] == ["y", "z"]
    assert hits[0][1] == pytest.approx(1.0)
    assert index.search(np.array([0.0, 1.0], dtype=np.float32), 0) == []
    with pytest.raises(ValueError, match="tamanhos"):
        index.add(["a"], np.zeros((2, 2), dtype=np.float32))


def test_fake_embedder_is_deterministic_and_normalized() -> None:
    embedder = FakeEmbedder()
    first = embedder.embed(["frete grátis", "horário"])
    assert np.array_equal(first, embedder.embed(["frete grátis", "horário"]))
    assert np.linalg.norm(first[0]) == pytest.approx(1.0)
    assert not embedder.embed(["!!!"]).any()


def test_hybrid_finds_section_and_skips_empty_parents() -> None:
    retriever = HybridRetriever(SECTIONS, FakeEmbedder(), InMemoryVectorIndex())
    top = retriever.retrieve("quero reembolso da devolução", k=3)
    assert top[0].section_id == "4.1"
    assert top[0].formatted().startswith("[§4.1 Arrependimento] ")
    assert {s.section_id for s in top} <= {"2", "4.1", "5.1"}
    assert len(retriever.retrieve("frete", k=1)) == 1


def test_factory_selects_strategy_from_settings() -> None:
    long_ctx = retriever_factory.build(Settings(policy_retriever="long_context"), SECTIONS)
    hybrid = retriever_factory.build(
        Settings(policy_retriever="hybrid"), SECTIONS, embedder=FakeEmbedder()
    )
    assert isinstance(long_ctx, LongContextRetriever)
    assert isinstance(hybrid, HybridRetriever)


def test_policy_retriever_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMPORIO_POLICY_RETRIEVER", "hybrid")
    assert Settings().policy_retriever == "hybrid"
