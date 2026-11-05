# ruff: noqa: PLC0415  # imports do extra postgres ficam preguiçosos (CI sem extras)
"""Contrato de ``TicketRepository``: memória, SQLite e Postgres respondem igual."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from emporio.adapters.persistence.sqlite_tickets import SqliteTicketRepository
from emporio.domain.ticket import TicketKind
from emporio.ports import TicketRepository
from tests.contracts.conftest import integration

TZ = ZoneInfo("America/Campo_Grande")
NOW = datetime(2026, 3, 25, 10, 0, tzinfo=TZ)
TOMORROW = datetime(2026, 3, 26, 10, 0, tzinfo=TZ)


def test_protocol_is_sequential_per_day(ticket_repo: TicketRepository) -> None:
    first = ticket_repo.open("s1", TicketKind.COMPLAINT, "atraso", 7, NOW)
    second = ticket_repo.open("s1", TicketKind.HUMAN, "quero atendente", None, NOW)
    next_day = ticket_repo.open("s1", TicketKind.HUMAN, "outro", None, TOMORROW)
    assert (first.protocol, second.protocol) == ("EMP-20260325-0001", "EMP-20260325-0002")
    assert next_day.protocol == "EMP-20260326-0001"
    assert first.created
    assert (first.kind, first.order_id, first.session_id) == (TicketKind.COMPLAINT, 7, "s1")


def test_open_is_idempotent_by_session_kind_and_summary(ticket_repo: TicketRepository) -> None:
    first = ticket_repo.open("s1", TicketKind.COMPLAINT, "Atraso  no pedido", 7, NOW)
    again = ticket_repo.open("s1", TicketKind.COMPLAINT, "atraso no pedido", 7, NOW)
    other_session = ticket_repo.open("s2", TicketKind.COMPLAINT, "atraso no pedido", 7, NOW)
    other_kind = ticket_repo.open("s1", TicketKind.RETURN, "atraso no pedido", 7, NOW)
    assert again.protocol == first.protocol
    assert not again.created
    assert (again.kind, again.order_id) == (TicketKind.COMPLAINT, 7)
    assert len({first.protocol, other_session.protocol, other_kind.protocol}) == 3


def test_sqlite_table_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "t.sqlite"
    repo = SqliteTicketRepository(path)
    repo.open("s1", TicketKind.HUMAN, "oi", None, NOW)
    repo.close()
    raw = sqlite3.connect(path)
    with pytest.raises(sqlite3.DatabaseError, match="somente anexação"):
        raw.execute("UPDATE tickets SET summary = 'x'")
    with pytest.raises(sqlite3.DatabaseError, match="somente anexação"):
        raw.execute("DELETE FROM tickets")
    raw.close()


@integration
def test_postgres_table_is_append_only(postgres_pool: Any) -> None:
    import psycopg

    from emporio.adapters.persistence.postgres_tickets import PostgresTicketRepository

    PostgresTicketRepository(postgres_pool).open("s1", TicketKind.HUMAN, "oi", None, NOW)
    with postgres_pool.connection() as conn:
        for statement in (
            b"UPDATE tickets SET summary = 'x'",
            b"DELETE FROM tickets",
            b"TRUNCATE tickets",
        ):
            with pytest.raises(psycopg.Error, match="somente anexação"):
                conn.execute(statement)


@integration
def test_postgres_migrations_are_idempotent_and_reversible(postgres_pool: Any) -> None:
    from emporio.adapters.persistence.postgres_schema import MIGRATIONS, downgrade, migrate

    versions = [m.version for m in MIGRATIONS]
    assert migrate(postgres_pool) == versions
    assert migrate(postgres_pool) == []
    assert downgrade(postgres_pool) == versions[::-1]
    assert migrate(postgres_pool) == versions


@integration
def test_postgres_concurrent_replicas_share_one_protocol(postgres_pool: Any) -> None:
    from emporio.adapters.persistence.postgres_tickets import PostgresTicketRepository

    repo = PostgresTicketRepository(postgres_pool)
    with ThreadPoolExecutor(8) as pool:
        same = list(
            pool.map(lambda _: repo.open("s1", TicketKind.HUMAN, "igual", None, NOW), range(16))
        )
        many = list(
            pool.map(lambda i: repo.open(f"s{i}", TicketKind.HUMAN, "x", None, NOW), range(16))
        )
    assert len({t.protocol for t in same}) == 1
    assert sum(t.created for t in same) == 1
    assert len({t.protocol for t in many}) == 16
