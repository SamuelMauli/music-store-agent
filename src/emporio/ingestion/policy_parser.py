"""Parser section-aware do manual: segmenta por numeração, nunca por tamanho fixo."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from emporio.domain.policy import PolicySection
from emporio.ingestion.policy_keywords import SECTION_KEYWORDS
from emporio.ingestion.policy_pdf import PolicyParseError, RawLine, read_policy_lines

_TOP: Final = re.compile(r"^(\d+)\.\s+(\S.*)$")
_SUB: Final = re.compile(r"^(\d+)\.(\d+)\s+(\S.*)$")
_MARKER: Final = re.compile(r"^(•|\d+\.\s)")
_VERSION: Final = re.compile(r"Vers[ãa]o\s+(\d+(?:\.\d+)*)")
_TERMINAL: Final = (".", ":", ";", "!", "?", ")")
# Linhas quebradas do mesmo parágrafo distam ~1,5x a altura; novo parágrafo, >= 1,9x.
_PARAGRAPH_GAP: Final = 1.7


@dataclass(frozen=True, slots=True)
class ParsedManual:
    """Resultado do parser.

    Attributes:
        version: Versão do manual, lida da capa (ex.: ``"2.1"``).
        sections: Seções em ordem de leitura.
    """

    version: str
    sections: tuple[PolicySection, ...]


@dataclass(slots=True)
class _Draft:
    section_id: str
    title: str
    page: int
    lines: list[RawLine] = field(default_factory=list)


def _heading(text: str, last_top: int, last_minor: int) -> tuple[str, str] | None:
    """Reconhece título numerado; a sequência esperada separa "1. Saudação" (item de lista)
    do título "8. Garantia", já que ambos casam com o mesmo padrão textual."""
    if sub := _SUB.match(text):
        major, minor = int(sub[1]), int(sub[2])
        if major == last_top and minor == last_minor + 1:
            return f"{major}.{minor}", sub[3]
    elif top := _TOP.match(text):
        if int(top[1]) == last_top + 1:
            return top[1], top[2]
    return None


def _segment(lines: list[RawLine]) -> tuple[list[RawLine], list[_Draft]]:
    preamble: list[RawLine] = []
    drafts: list[_Draft] = []
    last_top, last_minor = 0, 0
    for line in lines:
        found = _heading(line.text, last_top, last_minor) if line.kind == "text" else None
        if found is None:
            (drafts[-1].lines if drafts else preamble).append(line)
            continue
        section_id, title = found
        last_top, last_minor = (
            (int(section_id), 0) if "." not in section_id else (last_top, last_minor + 1)
        )
        drafts.append(_Draft(section_id, title, line.page))
    return preamble, drafts


def _starts_paragraph(prev: RawLine | None, line: RawLine) -> bool:
    if prev is None or line.kind == "row" or prev.kind == "row":
        return True
    if _MARKER.match(line.text):
        return True
    if prev.page != line.page:
        return prev.text.endswith(_TERMINAL)
    return line.top - prev.top > prev.height * _PARAGRAPH_GAP


def _join_paragraphs(lines: list[RawLine]) -> str:
    paragraphs: list[str] = []
    prev: RawLine | None = None
    for line in lines:
        if _starts_paragraph(prev, line):
            paragraphs.append(line.text)
        else:
            paragraphs[-1] = f"{paragraphs[-1]} {line.text}"
        prev = line
    return "\n".join(paragraphs)


def _build(drafts: list[_Draft]) -> tuple[PolicySection, ...]:
    titles = {d.section_id: f"{d.section_id}. {d.title}" for d in drafts}
    sections = []
    for draft in drafts:
        parent = draft.section_id.split(".")[0] if "." in draft.section_id else None
        label = titles[draft.section_id] if parent is None else f"{draft.section_id} {draft.title}"
        breadcrumb = label if parent is None else f"{titles[parent]} › {label}"
        sections.append(
            PolicySection(
                section_id=draft.section_id,
                title=draft.title,
                parent_id=parent,
                page=draft.page,
                text=_join_paragraphs(draft.lines),
                breadcrumb=breadcrumb,
                keywords=SECTION_KEYWORDS.get(draft.section_id, ()),
            )
        )
    return tuple(sections)


def parse_policy_lines(lines: list[RawLine]) -> ParsedManual:
    """Segmenta linhas já limpas em seções e subseções numeradas.

    Args:
        lines: Linhas em ordem de leitura (ver ``read_policy_lines``).

    Returns:
        Versão do manual e seções; o texto da capa só fornece a versão.

    Raises:
        PolicyParseError: Se não houver título numerado ou a versão não constar na capa.
    """
    preamble, drafts = _segment(lines)
    if not drafts:
        raise PolicyParseError("nenhuma seção numerada encontrada no manual")
    match = _VERSION.search(" ".join(line.text for line in preamble))
    if match is None:
        raise PolicyParseError("versão do manual não encontrada na capa")
    return ParsedManual(match[1], _build(drafts))


def parse_policy_pdf(path: Path) -> ParsedManual:
    """Lê o PDF e devolve as seções do manual.

    Args:
        path: Caminho do PDF.

    Returns:
        ``ParsedManual`` com versão e seções.
    """
    return parse_policy_lines(read_policy_lines(path))
