"""Leitura do PDF do manual: linhas limpas, sem cabeçalho/rodapé e com tabelas legíveis."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pdfplumber
from pdfplumber.page import Page

from emporio.domain.errors import DomainError

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"))
_MIN_REPEATS: Final = 3
_TWO_COLUMNS: Final = 2
_ROW_EPSILON: Final = 1e-3

LineKind = Literal["text", "row"]


class PolicyParseError(DomainError):
    """O PDF do manual não pôde ser lido ou não tem a estrutura esperada."""


@dataclass(frozen=True, slots=True)
class RawLine:
    """Linha do PDF já normalizada.

    Attributes:
        page: Página (1-based).
        top: Coordenada vertical, usada para decidir quebra de parágrafo.
        height: Altura da linha; o espaçamento normal entre linhas é proporcional a ela.
        text: Texto da linha.
        kind: ``"text"`` para linha corrida, ``"row"`` para linha de tabela já formatada.
    """

    page: int
    top: float
    text: str
    kind: LineKind = "text"
    height: float = 0.0


def _clean(text: str) -> str:
    return _SPACES.sub(" ", text.translate(_ZERO_WIDTH)).strip()


def format_table_row(header: list[str | None], row: list[str | None]) -> str:
    """Converte uma linha de tabela em pares chave-valor legíveis.

    Tabelas de 2 colunas viram ``"Chave: valor"``; as demais usam o cabeçalho:
    ``"Forma: PIX · Condições: Pagamento à vista"``.

    Args:
        header: Cabeçalho da tabela.
        row: Células da linha.

    Returns:
        Texto da linha em uma única linha.
    """
    cells = [_clean(c or "") for c in row]
    if len(cells) == _TWO_COLUMNS:
        return f"{cells[0]}: {cells[1]}"
    names = [_clean(h or "") for h in header]
    return " · ".join(f"{k}: {v}" for k, v in zip(names, cells, strict=False) if v)


def _inside(bboxes: list[tuple[float, float, float, float]], obj: dict[str, Any]) -> bool:
    cx = (float(obj["x0"]) + float(obj["x1"])) / 2
    cy = (float(obj["top"]) + float(obj["bottom"])) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)


def _page_lines(page: Page, number: int) -> list[RawLine]:
    tables = page.find_tables()
    bboxes = [t.bbox for t in tables]
    body = page.filter(lambda obj: not _inside(bboxes, obj)) if bboxes else page
    lines = [
        RawLine(
            number,
            float(line["top"]),
            _clean(line["text"]),
            height=float(line["bottom"]) - float(line["top"]),
        )
        for line in body.extract_text_lines()
        if _clean(line["text"])
    ]
    for table in tables:
        rows = table.extract()
        lines.extend(
            RawLine(number, table.bbox[1] + i * _ROW_EPSILON, format_table_row(rows[0], row), "row")
            for i, row in enumerate(rows[1:], start=1)
        )
    return sorted(lines, key=lambda line: line.top)


def _signature(text: str) -> str:
    return _DIGITS.sub("#", text)


def _repeated_edges(pages: list[list[RawLine]]) -> set[str]:
    """Assinaturas (números mascarados) que abrem ou fecham várias páginas."""
    counter: Counter[str] = Counter()
    for lines in pages:
        text_lines = [line for line in lines if line.kind == "text"]
        edges = {text_lines[0].text, text_lines[-1].text} if text_lines else set()
        counter.update(_signature(edge) for edge in edges)
    return {sig for sig, n in counter.items() if n >= _MIN_REPEATS}


def read_policy_lines(path: Path) -> list[RawLine]:
    """Extrai as linhas do manual removendo cabeçalho e rodapé repetidos.

    Args:
        path: Caminho do PDF (lido, nunca executado).

    Returns:
        Linhas em ordem de leitura; linhas de tabela já como pares chave-valor.

    Raises:
        PolicyParseError: Se o arquivo não existir ou não puder ser aberto como PDF.
    """
    try:
        with pdfplumber.open(path) as pdf:
            pages = [_page_lines(page, n) for n, page in enumerate(pdf.pages, start=1)]
    except Exception as exc:
        raise PolicyParseError(f"não foi possível ler o manual {path}: {exc}") from exc
    repeated = _repeated_edges(pages)
    return [
        line
        for lines in pages
        for line in lines
        if line.kind == "row" or _signature(line.text) not in repeated
    ]
