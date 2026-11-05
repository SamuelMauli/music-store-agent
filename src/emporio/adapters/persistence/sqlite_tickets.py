"""Chamados em SQLite, somente anexação e idempotentes por hash de sessão + tipo + resumo."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from emporio.domain.ticket import Ticket, TicketKind, idempotency_key

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    protocol   TEXT PRIMARY KEY,
    idem_key   TEXT NOT NULL UNIQUE,
    kind       TEXT NOT NULL,
    summary    TEXT NOT NULL,
    order_id   INTEGER,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS tickets_no_update BEFORE UPDATE ON tickets
BEGIN SELECT RAISE(ABORT, 'tickets são somente anexação'); END;
CREATE TRIGGER IF NOT EXISTS tickets_no_delete BEFORE DELETE ON tickets
BEGIN SELECT RAISE(ABORT, 'tickets são somente anexação'); END;
"""


class SqliteTicketRepository:
    """:class:`~emporio.ports.TicketRepository` persistente (append-only)."""

    def __init__(self, path: Path | str) -> None:
        """Abre (e cria, se preciso) o banco de chamados.

        Args:
            path: Arquivo SQLite, ou ``:memory:``.
        """
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._con.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        """Fecha a conexão."""
        self._con.close()

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
        stamp = now.strftime("%Y%m%d")
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                row = self._con.execute(
                    "SELECT protocol, kind, summary, order_id, session_id "
                    "FROM tickets WHERE idem_key = ?",
                    (key,),
                ).fetchone()
                if row:
                    self._con.execute("COMMIT")
                    return Ticket(row[0], TicketKind(row[1]), row[2], row[3], row[4], created=False)
                (count,) = self._con.execute(
                    "SELECT COUNT(*) FROM tickets WHERE protocol LIKE ?", (f"EMP-{stamp}-%",)
                ).fetchone()
                protocol = f"EMP-{stamp}-{count + 1:04d}"
                self._con.execute(
                    "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (protocol, key, kind.value, summary, order_id, session_id, now.isoformat()),
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return Ticket(protocol, kind, summary, order_id, session_id)
