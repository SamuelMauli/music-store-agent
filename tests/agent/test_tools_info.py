from __future__ import annotations

import json
from datetime import datetime

import duckdb

from emporio.adapters.persistence.memory_tickets import InMemoryTicketRepository
from emporio.agent.tools import ToolRegistry
from emporio.agent.tools.results import Ambiguous, Ok
from emporio.clock import FixedClock
from tests.agent.conftest import TZ, Caller, caller, make_registry


def _data(payload: str) -> dict[str, object]:
    return json.loads(payload)["data"]  # type: ignore[no-any-return]


def test_shipping_metro_by_total(call: Caller) -> None:
    free = call("estimate_shipping", city="campo grande", order_total=600)
    assert isinstance(free.result, Ok)
    assert free.result.value.fee == "grátis"
    paid = call("estimate_shipping", city="Campo Grande", order_total=499)
    assert isinstance(paid.result, Ok)
    assert paid.result.value.fee == "R$ 35,00"


def test_shipping_metro_without_total_states_the_rule(call: Caller) -> None:
    result = call("estimate_shipping", city="Campo Grande")
    assert isinstance(result.result, Ok)
    assert result.result.value.fee is None
    assert "R$ 500,00" in (result.result.value.rule or "")


def test_shipping_by_products_sums_effective_prices(call: Caller) -> None:
    result = call("estimate_shipping", city="Campo Grande", product_ids=[95])
    assert isinstance(result.result, Ok)
    assert result.result.value.fee == "grátis"


def test_shipping_other_city_never_quotes_a_price(call: Caller) -> None:
    result = call("estimate_shipping", city="Dourados", order_total=900)
    assert isinstance(result.result, Ok)
    view = result.result.value
    assert view.region == "outras_cidades"
    assert view.fee is None
    assert view.modes


def test_shipping_blank_city_asks(call: Caller) -> None:
    assert isinstance(call("estimate_shipping", city="  ").result, Ambiguous)


def test_store_info_topics(call: Caller) -> None:
    assert "Rua 14 de Maio" in str(_data(call("get_store_info", topic="address").payload))
    assert _data(call("get_store_info", topic="contacts").payload)["whatsapp"]
    assert _data(call("get_store_info", topic="hours").payload)["hours"]


def test_open_now_on_reference_day(call: Caller) -> None:
    assert _data(call("get_store_info", topic="open_now").payload)["open_now"] is True


def test_open_now_closed_on_sunday_reports_next_opening(
    con: duckdb.DuckDBPyConnection, tickets: InMemoryTicketRepository
) -> None:
    sunday = FixedClock(datetime(2026, 3, 29, 10, 0, tzinfo=TZ))
    registry = make_registry(con, sunday, tickets)
    data = _data(caller(registry)("get_store_info", topic="open_now").payload)
    assert data["open_now"] is False
    assert "30/03/2026" in str(data["next_opening"])


def test_search_policies_cites_sections(call: Caller) -> None:
    result = call("search_policies", query="garantia legal de quantos dias")
    assert isinstance(result.result, Ok)
    assert any(s.startswith("§") for s in result.sources)


def test_open_ticket_is_idempotent_and_masks_pii(
    registry: ToolRegistry, tickets: InMemoryTicketRepository
) -> None:
    args = {"kind": "complaint", "summary": "Cliente lucas@jmail.com reclama de atraso"}
    first = caller(registry, session_id="s1")("open_ticket", order_id=8, **args)
    again = caller(registry, session_id="s1")("open_ticket", order_id=8, **args)
    other = caller(registry, session_id="s2")("open_ticket", order_id=8, **args)
    assert isinstance(first.result, Ok)
    assert first.result.value.protocol == "EMP-20260325-0001"
    assert first.result.value.sla == "24 horas úteis"
    assert isinstance(again.result, Ok)
    assert again.result.value.protocol == "EMP-20260325-0001"
    assert again.result.value.already_opened
    assert isinstance(other.result, Ok)
    assert other.result.value.protocol == "EMP-20260325-0002"
    assert "lucas@jmail.com" not in first.payload
