"""Leitura das seções do manual (gold ``v_policy_sections``)."""

from __future__ import annotations

import duckdb

from emporio.domain.policy import PolicySection

_SQL = (
    "SELECT section_id, title, parent_id, page, text, breadcrumb, keywords "
    "FROM gold.v_policy_sections ORDER BY position"
)


class DuckDbPolicySectionRepository:
    """Entrega as seções do manual na ordem de leitura."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        """Cria o repositório.

        Args:
            con: Conexão (leitura) com o banco medalhão.
        """
        self._con = con

    def all(self) -> list[PolicySection]:
        """Todas as seções, incluindo as de corpo vazio (títulos-pai).

        Returns:
            Seções em ordem de leitura.
        """
        return [
            PolicySection(sid, title, parent, page, text, crumb, tuple(keywords))
            for sid, title, parent, page, text, crumb, keywords in self._con.execute(
                _SQL
            ).fetchall()
        ]

    def corpus_hash(self) -> str:
        """Hash do corpus gravado na ingestão."""
        row = self._con.execute("SELECT corpus_sha256 FROM silver.policy_corpus").fetchone()
        if row is None:
            raise LookupError("corpus de políticas não carregado; rode `make ingest`")
        return str(row[0])
