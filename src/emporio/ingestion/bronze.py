"""Camada bronze: cópia fiel dos CSVs, tudo como texto, mais colunas de linhagem."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from emporio.ingestion.errors import IngestionError

TABLES = ("categories", "customers", "products", "promotions", "orders", "order_items")


def load_bronze(
    con: duckdb.DuckDBPyConnection, raw_dir: Path, ingested_at: datetime
) -> dict[str, int]:
    """Carrega cada CSV em ``bronze.<tabela>`` sem alterar nenhum valor.

    Acrescenta ``_source_file``, ``_ingested_at`` (UTC) e ``_row_hash`` (md5 da
    linha original), o que permite reprocessar e rastrear a origem de cada
    linha nas camadas seguintes.

    Args:
        con: Conexão DuckDB.
        raw_dir: Diretório com os CSVs imutáveis.
        ingested_at: Instante da ingestão (aware).

    Returns:
        Contagem de linhas por tabela.

    Raises:
        IngestionError: Se algum CSV estiver ausente ou malformado.
    """
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    stamp = ingested_at.astimezone(UTC).replace(tzinfo=None)
    counts: dict[str, int] = {}
    for table in TABLES:
        path = raw_dir / f"{table}.csv"
        if not path.is_file():
            raise IngestionError(f"arquivo de origem ausente: {path}")
        try:
            # `r::VARCHAR` serializa a linha inteira, então o hash cobre todas as colunas.
            con.execute(
                f"""
                CREATE OR REPLACE TABLE bronze.{table} AS
                SELECT r.*, ? AS _source_file, CAST(? AS TIMESTAMP) AS _ingested_at,
                       md5(r::VARCHAR) AS _row_hash
                FROM read_csv(?, header = true, all_varchar = true, delim = ',', quote = '"') AS r
                """,  # noqa: S608 - `table` vem da tupla constante TABLES
                [path.name, stamp, str(path)],
            )
        except duckdb.Error as exc:
            raise IngestionError(f"CSV malformado em {path.name}") from exc
        row = con.execute(f"SELECT count(*) FROM bronze.{table}").fetchone()  # noqa: S608
        counts[table] = int(row[0]) if row else 0
    return counts
