from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from emporio.adapters.persistence.sqlite_tickets import SqliteTicketRepository
from emporio.agent.nodes.deps import HISTORY_WINDOW
from emporio.bootstrap import build_container
from emporio.domain.ticket import TicketKind
from emporio.settings import Settings
from tests.agent.conftest import fake_settings

NOW = datetime(2026, 3, 25, 10, 0, tzinfo=ZoneInfo("America/Campo_Grande"))


def test_protocol_is_sequential_per_day(tmp_path: Path) -> None:
    repo = SqliteTicketRepository(tmp_path / "t.sqlite")
    first = repo.open("s1", TicketKind.COMPLAINT, "atraso", 7, NOW)
    second = repo.open("s1", TicketKind.HUMAN, "quero atendente", None, NOW)
    assert first.protocol == "EMP-20260325-0001"
    assert second.protocol == "EMP-20260325-0002"
    assert first.created
    repo.close()


def test_open_is_idempotent_by_session_kind_and_summary(tmp_path: Path) -> None:
    repo = SqliteTicketRepository(tmp_path / "t.sqlite")
    first = repo.open("s1", TicketKind.COMPLAINT, "Atraso  no pedido", 7, NOW)
    again = repo.open("s1", TicketKind.COMPLAINT, "atraso no pedido", 7, NOW)
    other_session = repo.open("s2", TicketKind.COMPLAINT, "atraso no pedido", 7, NOW)
    assert again.protocol == first.protocol
    assert not again.created
    assert other_session.protocol != first.protocol
    repo.close()


def test_tickets_survive_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "t.sqlite"
    repo = SqliteTicketRepository(path)
    protocol = repo.open("s1", TicketKind.PRIVACY, "excluir dados", None, NOW).protocol
    repo.close()
    reopened = SqliteTicketRepository(path)
    assert reopened.open("s1", TicketKind.PRIVACY, "excluir dados", None, NOW).protocol == protocol
    reopened.close()


def test_table_is_append_only(tmp_path: Path) -> None:
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


def test_conversation_survives_a_restart(ingested: Settings, tmp_path: Path) -> None:
    settings = fake_settings(ingested, checkpoint_path=tmp_path / "ck.sqlite")
    first = build_container(settings)
    assert first.service.chat("m1", "Oi!").turn == 1
    first.close()
    second = build_container(settings)
    try:
        assert second.service.chat("m1", "Quais são as promoções ativas?").turn == 2
        second.service.reset("m1")
        assert second.service.chat("m1", "Oi!").turn == 1
    finally:
        second.close()


def test_old_messages_are_summarized_and_window_is_kept(ingested: Settings) -> None:
    container = build_container(fake_settings(ingested))
    try:
        for _ in range(8):
            container.service.chat("m2", "Quais são as promoções ativas?")
        graph = container.service._graph
        values = graph.get_state({"configurable": {"thread_id": "m2"}}).values
    finally:
        container.close()
    assert len(values["messages"]) == 16
    assert values["summarized_upto"] == 16 - HISTORY_WINDOW
    assert "promoções" in values["summary"]
