from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from emporio.domain.policy import PolicySection
from emporio.ingestion.policy_parser import parse_policy_lines
from emporio.ingestion.policy_pdf import (
    PolicyParseError,
    RawLine,
    format_table_row,
    read_policy_lines,
)


def line(text: str, top: float, page: int = 1, kind: Literal["text", "row"] = "text") -> RawLine:
    return RawLine(page, top, text, kind, height=10.0)


COVER = [line("Versão 2.1 — Junho de 2025", 10)]


def test_format_table_row_two_columns_is_key_value() -> None:
    assert format_table_row(["Dia", "Horário"], ["Sábado", "09:00 às 13:00"]) == (
        "Sábado: 09:00 às 13:00"
    )


def test_format_table_row_many_columns_uses_header_pairs() -> None:
    row = format_table_row(
        ["Forma", "Condições", "Observações"], ["PIX", "à vista", "5%\nde desconto"]
    )
    assert row == "Forma: PIX · Condições: à vista · Observações: 5% de desconto"


def test_format_table_row_skips_empty_cells_and_handles_none() -> None:
    assert format_table_row(["A", "B", "C"], ["x", None, "z"]) == "A: x · C: z"


def test_parse_builds_hierarchy_and_breadcrumbs() -> None:
    manual = parse_policy_lines(
        [
            *COVER,
            line("1. Loja", 20),
            line("Texto da loja.", 40),
            line("1.1 Missão", 60),
            line("Missão em", 80),
            line("duas linhas.", 94),
        ]
    )
    top, sub = manual.sections
    assert (top.section_id, top.parent_id, top.text) == ("1", None, "Texto da loja.")
    assert (sub.section_id, sub.parent_id) == ("1.1", "1")
    assert sub.breadcrumb == "1. Loja › 1.1 Missão"
    assert sub.text == "Missão em duas linhas."
    assert manual.version == "2.1"


def test_numbered_list_item_is_not_mistaken_for_a_heading() -> None:
    manual = parse_policy_lines(
        [
            *COVER,
            line("1. Loja", 20),
            line("2. Fluxo", 40),
            line("1. Saudação: cumprimentar.", 60),
            line("2. Entendimento: ouvir.", 80),
            line("3. Garantia", 100),
        ]
    )
    assert [s.section_id for s in manual.sections] == ["1", "2", "3"]
    assert "1. Saudação" in manual.sections[1].text
    assert manual.sections[1].text.count("\n") == 1


def test_out_of_sequence_subsection_stays_in_body() -> None:
    manual = parse_policy_lines(
        [*COVER, line("1. Pagamento", 20), line("3.1 Solta", 40), line("1.1 Real", 60)]
    )
    assert [s.section_id for s in manual.sections] == ["1", "1.1"]
    assert "3.1 Solta" in manual.sections[0].text


def test_paragraph_breaks_on_marker_gap_row_and_page_change() -> None:
    manual = parse_policy_lines(
        [
            *COVER,
            line("1. Loja", 20),
            line("• item um que continua", 40),
            line("na segunda linha.", 54),
            line("• item dois", 70),
            line("Rótulo: texto", 100),
            line("Chave: valor", 100, kind="row"),
            line("fim da frase.", 700, page=1),
            line("começo da outra", 20, page=2),
            line("termina aqui.", 34, page=2),
            line("novo parágrafo sem ponto final e", 600, page=2),
            line("continua na página seguinte", 20, page=3),
        ]
    )
    assert manual.sections[0].text.split("\n") == [
        "• item um que continua na segunda linha.",
        "• item dois",
        "Rótulo: texto",
        "Chave: valor",
        "fim da frase.",
        "começo da outra termina aqui.",
        "novo parágrafo sem ponto final e continua na página seguinte",
    ]


def test_search_text_and_label() -> None:
    section = PolicySection(
        "4.1", "Arrependimento", "4", 4, "corpo", "4. Trocas › 4.1", ("devolver",)
    )
    assert section.label == "§4.1 Arrependimento"
    assert section.search_text == "4. Trocas › 4.1\ndevolver\ncorpo"


def test_missing_headings_and_version_raise() -> None:
    with pytest.raises(PolicyParseError, match="nenhuma seção"):
        parse_policy_lines(COVER)
    with pytest.raises(PolicyParseError, match="versão"):
        parse_policy_lines([line("1. Loja", 20)])


def test_unreadable_pdf_raises(tmp_path: Path) -> None:
    fake = tmp_path / "nao_e_pdf.pdf"
    fake.write_text("isto não é um PDF")
    with pytest.raises(PolicyParseError, match="não foi possível ler"):
        read_policy_lines(fake)
    with pytest.raises(PolicyParseError):
        read_policy_lines(tmp_path / "ausente.pdf")
