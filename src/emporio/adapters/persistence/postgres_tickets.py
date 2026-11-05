"""Chamados em Postgres: somente anexação (trigger), idempotentes e seguros entre réplicas."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from psycopg.errors import UniqueViolation

from emporio.adapters.persistence.postgres_schema import migrate
from emporio.domain.ticket import Ticket, TicketKind, idempotency_key

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg.rows import DictRow
    from psycopg_pool import ConnectionPool

_SELECT = "SELECT protocol, kind, summary, order_id, session_id FROM tickets WHERE idem_key = %s"
_NEXT = (
    "INSERT INTO ticket_counters (day, n) VALUES (%s, 1) "
    "ON CONFLICT (day) DO UPDATE SET n = ticket_counters.n + 1 RETURNING n"
)
_INSERT = "INSERT INTO tickets VALUES (%s, %s, %s, %s, %s, %s, %s)"


class PostgresTicketRepository:
    """:class:`~emporio.ports.TicketRepository` compartilhado entre réplicas.

    UPDATE, DELETE e TRUNCATE são barrados por trigger. A idempotência vem da chave
    ``UNIQUE``: se duas réplicas abrem o mesmo chamado juntas, uma perde com
    ``UniqueViolation`` (e a transação dela, contador incluso, é desfeita) e relê o
    vencedor. O sequencial do dia é um ``UPSERT`` atômico em ``ticket_counters``.
    """

    def __init__(self, pool: ConnectionPool[Connection[DictRow]]) -> None:
        """Aplica as migrações pendentes e guarda o pool.

        Args:
            pool: Pool aberto (o repositório não o fecha; quem o abriu fecha).
        """
        self._pool = pool
        migrate(pool)

    def open(
        self,
        session_id: str,
        kind: TicketKind,
        summary: str,
        order_id: int | None,
        now: datetime,
    ) -> Ticket:
        """Abre o chamado, ou devolve o já aberto para a mesma chave."""
        key = idempotency_key(session_id, kind, summary)
        try:
            return self._try_open(key, session_id, kind, summary, order_id=order_id, now=now)
        except UniqueViolation:  # corrida com outra réplica: o vencedor já está gravado
            return self._try_open(key, session_id, kind, summary, order_id=order_id, now=now)

    def _try_open(
        self,
        key: str,
        session_id: str,
        kind: TicketKind,
        summary: str,
        *,
        order_id: int | None,
        now: datetime,
    ) -> Ticket:
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(_SELECT, (key,)).fetchone()
            if row:
                return Ticket(
                    row["protocol"],
                    TicketKind(row["kind"]),
                    row["summary"],
                    row["order_id"],
                    row["session_id"],
                    created=False,
                )
            stamp = now.strftime("%Y%m%d")
            counter = conn.execute(_NEXT, (stamp,)).fetchone()
            if counter is None:  # pragma: no cover - RETURNING sempre devolve a linha
                raise RuntimeError("contador de chamados não devolveu linha")
            protocol = f"EMP-{stamp}-{counter['n']:04d}"
            conn.execute(_INSERT, (protocol, key, kind.value, summary, order_id, session_id, now))
        return Ticket(protocol, kind, summary, order_id, session_id)
