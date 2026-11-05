from __future__ import annotations

from datetime import datetime

import duckdb

from emporio.adapters.persistence.memory_tickets import InMemoryTicketRepository
from emporio.agent.tools import ToolContext
from emporio.agent.tools.results import Ok
from emporio.clock import FixedClock
from tests.agent.conftest import TZ, make_registry


def test_open_now_respects_national_holidays(
    con: duckdb.DuckDBPyConnection, tickets: InMemoryTicketRepository
) -> None:
    # quinta 02/04 à noite: fechada; sexta 03/04 é Sexta-feira Santa, então reabre sábado 04/04
    clock = FixedClock(datetime(2026, 4, 2, 19, 0, tzinfo=TZ))
    result = make_registry(con, clock, tickets).call(
        "get_store_info", {"topic": "open_now"}, ToolContext()
    )
    assert isinstance(result.result, Ok)
    assert result.result.value.open_now is False
    assert "04/04/2026" in (result.result.value.next_opening or "")


def test_open_now_on_the_holiday_itself(
    con: duckdb.DuckDBPyConnection, tickets: InMemoryTicketRepository
) -> None:
    clock = FixedClock(datetime(2026, 4, 3, 10, 0, tzinfo=TZ))
    result = make_registry(con, clock, tickets).call(
        "get_store_info", {"topic": "open_now"}, ToolContext()
    )
    assert isinstance(result.result, Ok)
    assert result.result.value.open_now is False
