from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pytest

from emporio.adapters.persistence import DuckDbPolicySectionRepository
from emporio.clock import FixedClock
from emporio.domain.policy import PolicySection
from emporio.ingestion.pipeline import run_pipeline
from emporio.ingestion.policy_load import corpus_hash
from emporio.ingestion.policy_parser import parse_policy_pdf
from emporio.settings import Settings

SNAPSHOT = Path(__file__).resolve().parents[1] / "snapshots" / "policy_sections.json"
KEY_SECTIONS = ["2", "3", "3.1", "4.1", "4.2", "4.3", "4.4", "5.1", "5.2", "7.3", "8.1", "9"]


@pytest.fixture
def sections(con: duckdb.DuckDBPyConnection) -> dict[str, PolicySection]:
    return {s.section_id: s for s in DuckDbPolicySectionRepository(con).all()}


def test_sections_match_snapshot(ingested: Settings) -> None:
    manual = parse_policy_pdf(ingested.policy_pdf)
    expected = json.loads(SNAPSHOT.read_text())
    actual = [[s.section_id, s.title, s.parent_id, s.page] for s in manual.sections]
    assert manual.version == expected["manual_version"]
    assert len(manual.sections) == expected["section_count"] == 28
    assert actual == expected["sections"]


def test_key_sections_are_present(sections: dict[str, PolicySection]) -> None:
    assert all(key in sections for key in KEY_SECTIONS)
    assert sections["4.1"].title == "Direito de Arrependimento (Compras Online)"
    assert sections["4.1"].parent_id == "4"
    assert sections["4.1"].breadcrumb.startswith("4. Política de Trocas e Devoluções › 4.1")


@pytest.mark.parametrize(
    ("section_id", "needles"),
    [
        ("2", ["Sábado: 09:00 às 13:00", "Domingo e Feriados: Fechado"]),
        ("3", ["Forma: PIX · Condições: Pagamento à vista · Observações: 5% de desconto"]),
        ("3.1", ["parcela mínima de R$ 80,00", "acima de R$ 2.000,00"]),
        ("4.1", ["7 (sete) dias corridos", "10 dias úteis"]),
        ("4.2", ["30 (trinta) dias corridos"]),
        ("5.1", ["R$ 500,00", "R$ 35,00", "1 a 3 dias úteis"]),
        ("5.2", ["Modalidade: SEDEX (Correios) · Prazo Estimado: 2 a 5 dias úteis"]),
        ("8.1", ["90 (noventa) dias"]),
    ],
)
def test_section_text_carries_the_numbers(
    sections: dict[str, PolicySection], section_id: str, needles: list[str]
) -> None:
    for needle in needles:
        assert needle in sections[section_id].text


def test_header_footer_and_page_numbers_are_removed(sections: dict[str, PolicySection]) -> None:
    corpus = "\n".join(s.text for s in sections.values())
    assert "Manual de Políticas e Procedimentos" not in corpus
    assert not re.search(r"^Página \d+$", corpus, flags=re.MULTILINE)
    assert "\u200b" not in corpus


def test_wrapped_lines_are_joined_into_paragraphs(sections: dict[str, PolicySection]) -> None:
    bullets = sections["4.1"].text.split("\n")
    assert len(bullets) == 4
    assert all(b.startswith("• ") for b in bullets)


def test_keywords_carry_customer_vocabulary(sections: dict[str, PolicySection]) -> None:
    assert {"devolver", "estorno", "reembolso"} <= {*sections["4.1"].keywords}
    assert "estorno" in sections["4.1"].search_text


def test_corpus_hash_is_persisted_and_stable(
    ingested: Settings, con: duckdb.DuckDBPyConnection
) -> None:
    manual = parse_policy_pdf(ingested.policy_pdf)
    stored = DuckDbPolicySectionRepository(con).corpus_hash()
    assert stored == corpus_hash(manual.sections)
    assert re.fullmatch(r"[0-9a-f]{64}", stored)
    rows = con.execute("SELECT DISTINCT corpus_sha256 FROM gold.v_policy_sections").fetchall()
    assert rows == [(stored,)]


def test_pipeline_reports_policies_and_is_idempotent(tmp_path: Path, ingested: Settings) -> None:
    settings = ingested.model_copy(update={"db_path": tmp_path / "again.duckdb"})
    first = run_pipeline(settings, FixedClock(settings.reference_datetime))
    second = run_pipeline(settings, FixedClock(settings.reference_datetime))
    assert first.policies == second.policies
    assert first.policies.sections == 28
    assert first.policies.manual_version == "2.1"


def test_corpus_hash_without_ingestion_raises() -> None:
    empty = duckdb.connect(":memory:")
    empty.execute("CREATE SCHEMA silver")
    empty.execute("CREATE TABLE silver.policy_corpus (corpus_sha256 VARCHAR)")
    with pytest.raises(LookupError, match="make ingest"):
        DuckDbPolicySectionRepository(empty).corpus_hash()
