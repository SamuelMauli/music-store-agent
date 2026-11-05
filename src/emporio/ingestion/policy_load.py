"""Persistência do manual em DuckDB: ``silver.policy_sections`` e a view gold do agente."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from emporio.domain.policy import PolicySection
from emporio.ingestion.policy_parser import ParsedManual, parse_policy_pdf

_SECTIONS_DDL = """
CREATE TABLE silver.policy_sections (
  position INTEGER NOT NULL,
  section_id VARCHAR PRIMARY KEY,
  title VARCHAR NOT NULL,
  parent_id VARCHAR,
  page INTEGER NOT NULL,
  text VARCHAR NOT NULL,
  breadcrumb VARCHAR NOT NULL,
  keywords VARCHAR[] NOT NULL
)
"""

_CORPUS_DDL = """
CREATE TABLE silver.policy_corpus (
  corpus_sha256 VARCHAR NOT NULL,
  manual_version VARCHAR NOT NULL,
  source_file VARCHAR NOT NULL,
  section_count INTEGER NOT NULL
)
"""

# Uma única linha de corpus: a view anexa o hash a cada seção para o trace citar a versão.
_GOLD_VIEW = """
CREATE OR REPLACE VIEW gold.v_policy_sections AS
SELECT s.section_id, s.title, s.parent_id, s.page, s.text, s.breadcrumb, s.keywords,
       s.position, c.corpus_sha256, c.manual_version
FROM silver.policy_sections s CROSS JOIN silver.policy_corpus c
"""


@dataclass(frozen=True, slots=True)
class PolicyLoadResult:
    """Resumo da carga do manual.

    Attributes:
        sections: Quantidade de seções gravadas.
        corpus_sha256: Hash do corpus (conteúdo das seções, não do arquivo PDF).
        manual_version: Versão lida da capa.
    """

    sections: int
    corpus_sha256: str
    manual_version: str


def corpus_hash(sections: tuple[PolicySection, ...]) -> str:
    """Hash estável do conteúdo normativo (id, título, página e texto de cada seção).

    Args:
        sections: Seções em ordem de leitura.

    Returns:
        SHA-256 hexadecimal; muda só se o texto do manual mudar.
    """
    payload = [[s.section_id, s.title, s.parent_id, s.page, s.text] for s in sections]
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _write(con: duckdb.DuckDBPyConnection, manual: ParsedManual, digest: str, source: str) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    con.execute("DROP VIEW IF EXISTS gold.v_policy_sections")
    for table in ("policy_sections", "policy_corpus"):
        con.execute(f"DROP TABLE IF EXISTS silver.{table}")
    con.execute(_SECTIONS_DDL)
    con.execute(_CORPUS_DDL)
    con.executemany(
        "INSERT INTO silver.policy_sections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (i, s.section_id, s.title, s.parent_id, s.page, s.text, s.breadcrumb, list(s.keywords))
            for i, s in enumerate(manual.sections)
        ],
    )
    con.execute(
        "INSERT INTO silver.policy_corpus VALUES (?, ?, ?, ?)",
        [digest, manual.version, source, len(manual.sections)],
    )
    con.execute(_GOLD_VIEW)


def load_policy_sections(con: duckdb.DuckDBPyConnection, pdf_path: Path) -> PolicyLoadResult:
    """Faz o parse do manual e grava as seções (idempotente).

    Args:
        con: Conexão de escrita do banco medalhão.
        pdf_path: PDF do manual de políticas.

    Returns:
        Contagem de seções, hash do corpus e versão do manual.

    Raises:
        PolicyParseError: Se o PDF não puder ser lido ou segmentado.
    """
    manual = parse_policy_pdf(pdf_path)
    digest = corpus_hash(manual.sections)
    _write(con, manual, digest, pdf_path.name)
    return PolicyLoadResult(len(manual.sections), digest, manual.version)
