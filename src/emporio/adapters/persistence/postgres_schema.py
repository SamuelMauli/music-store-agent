"""DDL versionada do Postgres, aplicada no boot de forma idempotente e serializada.

Cada migração tem ``version`` crescente, SQL de subida e de descida. A tabela
``schema_version`` registra o que já rodou; o ``pg_advisory_xact_lock`` faz réplicas que
sobem juntas aplicarem uma de cada vez (a segunda encontra tudo aplicado e não faz nada).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg.rows import DictRow
    from psycopg_pool import ConnectionPool

_LOCK_ID = 0x454D504F  # "EMPO"


@dataclass(frozen=True, slots=True)
class Migration:
    """Uma versão do schema."""

    version: int
    name: str
    up: str
    down: str


_TICKETS_UP = """
CREATE TABLE IF NOT EXISTS tickets (
    protocol   TEXT PRIMARY KEY,
    idem_key   TEXT NOT NULL UNIQUE,
    kind       TEXT NOT NULL,
    summary    TEXT NOT NULL,
    order_id   BIGINT,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS ticket_counters (
    day TEXT PRIMARY KEY,
    n   INTEGER NOT NULL
);
CREATE OR REPLACE FUNCTION tickets_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'tickets são somente anexação' USING ERRCODE = 'integrity_constraint_violation';
END;
$$;
DROP TRIGGER IF EXISTS tickets_no_mutation ON tickets;
CREATE TRIGGER tickets_no_mutation BEFORE UPDATE OR DELETE ON tickets
    FOR EACH ROW EXECUTE FUNCTION tickets_append_only();
DROP TRIGGER IF EXISTS tickets_no_truncate ON tickets;
CREATE TRIGGER tickets_no_truncate BEFORE TRUNCATE ON tickets
    FOR EACH STATEMENT EXECUTE FUNCTION tickets_append_only();
"""

_TICKETS_DOWN = """
DROP TRIGGER IF EXISTS tickets_no_truncate ON tickets;
DROP TRIGGER IF EXISTS tickets_no_mutation ON tickets;
DROP FUNCTION IF EXISTS tickets_append_only();
DROP TABLE IF EXISTS ticket_counters;
DROP TABLE IF EXISTS tickets;
"""

MIGRATIONS: tuple[Migration, ...] = (Migration(1, "tickets", _TICKETS_UP, _TICKETS_DOWN),)

_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _locked(conn: Connection[DictRow]) -> set[int]:
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
    conn.execute(_VERSION_TABLE.encode(), prepare=False)
    return {row["version"] for row in conn.execute("SELECT version FROM schema_version")}


def migrate(pool: ConnectionPool[Connection[DictRow]]) -> list[int]:
    """Aplica as migrações pendentes numa transação.

    Args:
        pool: Pool aberto por :func:`emporio.adapters.persistence.postgres_pool.open_pool`.

    Returns:
        Versões aplicadas agora (vazio se o schema já estava em dia).
    """
    done: list[int] = []
    with pool.connection() as conn, conn.transaction():
        applied = _locked(conn)
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            conn.execute(migration.up.encode(), prepare=False)
            conn.execute(
                "INSERT INTO schema_version (version, name) VALUES (%s, %s)",
                (migration.version, migration.name),
            )
            done.append(migration.version)
    return done


def downgrade(pool: ConnectionPool[Connection[DictRow]], target: int = 0) -> list[int]:
    """Desfaz, da mais nova para a mais velha, as migrações acima de ``target``.

    Args:
        pool: Pool aberto.
        target: Versão a manter (``0`` desfaz tudo).

    Returns:
        Versões revertidas.
    """
    undone: list[int] = []
    with pool.connection() as conn, conn.transaction():
        applied = _locked(conn)
        for migration in reversed(MIGRATIONS):
            if migration.version <= target or migration.version not in applied:
                continue
            conn.execute(migration.down.encode(), prepare=False)
            conn.execute("DELETE FROM schema_version WHERE version = %s", (migration.version,))
            undone.append(migration.version)
    return undone
