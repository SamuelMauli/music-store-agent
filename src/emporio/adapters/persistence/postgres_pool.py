"""Pool de conexões Postgres compartilhado pelo checkpointer e pelos chamados."""

from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool


def open_pool(
    url: str, *, max_size: int = 10, **conn_kwargs: Any
) -> ConnectionPool[Connection[DictRow]]:
    """Abre um pool psycopg configurado como o ``PostgresSaver`` do LangGraph exige.

    As conexões saem em ``autocommit`` (transações explícitas com ``conn.transaction()``),
    sem prepared statements (compatível com PgBouncer) e com linhas como ``dict``.

    Args:
        url: DSN ``postgresql://...``.
        max_size: Máximo de conexões do pool.
        **conn_kwargs: Parâmetros extras de conexão (ex.: ``options``).

    Returns:
        Pool aberto; feche com ``pool.close()``.
    """
    return ConnectionPool(
        url,
        max_size=max_size,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row, **conn_kwargs},
        open=True,
    )
