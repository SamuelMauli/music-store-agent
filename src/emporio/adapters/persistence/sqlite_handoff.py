"""Handoff humano e deduplicação de mensagens em SQLite (processo único)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Literal, cast

from emporio.domain.handoff import (
    ClaimResult,
    HandoffConversation,
    HandoffMessage,
    HandoffState,
)

DEDUP_TTL_SECONDS = 86_400.0  # a Meta reentrega por horas no pior caso; 24h cobre as repetições

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wa_conversation (
    id         TEXT PRIMARY KEY,
    phone      TEXT NOT NULL,
    state      TEXT NOT NULL,
    operator   TEXT,
    entered_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS wa_message (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    direction       TEXT NOT NULL,
    text            TEXT NOT NULL,
    at              REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS wa_message_conv ON wa_message (conversation_id, seq);
CREATE TABLE IF NOT EXISTS wa_seen (message_id TEXT PRIMARY KEY, seen_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS wa_seen_at ON wa_seen (seen_at);
"""

_SELECT = "SELECT id, phone, state, operator, entered_at, expires_at FROM wa_conversation"


class SqliteHandoffStore:
    """:class:`~emporio.ports.HandoffStore` e :class:`~emporio.ports.MessageDeduplicator`."""

    def __init__(self, path: Path | str) -> None:
        """Abre (e cria, se preciso) o banco.

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

    def _purge(self, conv_id: str) -> None:
        self._con.execute("DELETE FROM wa_message WHERE conversation_id = ?", (conv_id,))
        self._con.execute("DELETE FROM wa_conversation WHERE id = ?", (conv_id,))

    def _drop_expired(self, now: float) -> None:
        for (conv_id,) in self._con.execute(
            "SELECT id FROM wa_conversation WHERE expires_at <= ?", (now,)
        ).fetchall():
            self._purge(conv_id)

    def _exists(self, conv_id: str) -> bool:
        row = self._con.execute("SELECT 1 FROM wa_conversation WHERE id = ?", (conv_id,))
        return row.fetchone() is not None

    def _load(self, row: tuple[str, str, str, str | None, float, float]) -> HandoffConversation:
        conv_id, phone, state, operator, entered_at, expires_at = row
        messages = tuple(
            HandoffMessage(cast(Literal["in", "out"], direction), text, at)
            for direction, text, at in self._con.execute(
                "SELECT direction, text, at FROM wa_message WHERE conversation_id = ? ORDER BY seq",
                (conv_id,),
            )
        )
        return HandoffConversation(
            conv_id, phone, HandoffState(state), operator, entered_at, expires_at, messages
        )

    def enter(self, conv_id: str, phone: str, now: float, expires_at: float) -> bool:
        """Põe a conversa em handoff; ``False`` se já estava."""
        with self._lock:
            self._drop_expired(now)
            cursor = self._con.execute(
                "INSERT OR IGNORE INTO wa_conversation VALUES (?, ?, ?, NULL, ?, ?)",
                (conv_id, phone, HandoffState.WAITING.value, now, expires_at),
            )
            return cursor.rowcount == 1

    def is_human(self, conv_id: str, now: float) -> bool:
        """``True`` se a conversa está em handoff e não expirou."""
        with self._lock:
            self._drop_expired(now)
            return self._exists(conv_id)

    def append(self, conv_id: str, direction: str, text: str, now: float) -> None:
        """Anexa mensagem ao histórico (sem efeito se a conversa não existe)."""
        with self._lock:
            self._con.execute(
                "INSERT INTO wa_message (conversation_id, direction, text, at) "
                "SELECT ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM wa_conversation WHERE id = ?)",
                (conv_id, direction, text, now, conv_id),
            )

    def get(self, conv_id: str, now: float) -> HandoffConversation | None:
        """Conversa com o histórico, ou ``None``."""
        with self._lock:
            self._drop_expired(now)
            row = self._con.execute(_SELECT + " WHERE id = ?", (conv_id,)).fetchone()
            return self._load(row) if row else None

    def list_active(self, now: float) -> list[HandoffConversation]:
        """Conversas ativas, da mais antiga à mais recente."""
        with self._lock:
            self._drop_expired(now)
            rows = self._con.execute(_SELECT + " ORDER BY entered_at, id").fetchall()
            return [self._load(r) for r in rows]

    def claim(self, conv_id: str, operator: str, now: float, expires_at: float) -> ClaimResult:
        """Assume a conversa; ``TAKEN`` se outro operador é o dono."""
        with self._lock:
            self._drop_expired(now)
            cursor = self._con.execute(
                "UPDATE wa_conversation SET state = ?, operator = ?, expires_at = ? "
                "WHERE id = ? AND (operator IS NULL OR operator = ?)",
                (HandoffState.ASSIGNED.value, operator, expires_at, conv_id, operator),
            )
            if cursor.rowcount == 1:
                return ClaimResult.OK
            return ClaimResult.TAKEN if self._exists(conv_id) else ClaimResult.NOT_FOUND

    def extend(self, conv_id: str, operator: str, expires_at: float, now: float) -> bool:
        """Renova a expiração se ``operator`` é o dono."""
        with self._lock:
            self._drop_expired(now)
            cursor = self._con.execute(
                "UPDATE wa_conversation SET expires_at = ? WHERE id = ? AND operator = ?",
                (expires_at, conv_id, operator),
            )
            return cursor.rowcount == 1

    def release(self, conv_id: str) -> HandoffConversation | None:
        """Devolve ao bot: apaga conversa e histórico (minimização de dado pessoal)."""
        with self._lock:
            row = self._con.execute(_SELECT + " WHERE id = ?", (conv_id,)).fetchone()
            if row is None:
                return None
            released = self._load(row)
            self._purge(conv_id)
            return released

    def first_seen(self, message_id: str, now: float) -> bool:
        """``True`` na 1ª vez que o id aparece; o registro expira em 24h."""
        with self._lock:
            self._con.execute("DELETE FROM wa_seen WHERE seen_at < ?", (now - DEDUP_TTL_SECONDS,))
            cursor = self._con.execute(
                "INSERT OR IGNORE INTO wa_seen VALUES (?, ?)", (message_id, now)
            )
            return cursor.rowcount == 1
