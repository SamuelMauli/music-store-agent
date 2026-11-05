from __future__ import annotations

from datetime import date

import duckdb

from emporio.adapters.persistence import DuckDbOrderRepository
from emporio.domain.value_objects import OrderId
from emporio.policies.returns import (
    NotApplicableCause,
    ReturnDecision,
    ReturnNotApplicable,
    ReturnReason,
    SalesChannel,
    evaluate_return,
)


def test_gherkin_regret_order_7_on_day_four(con: duckdb.DuckDBPyConnection) -> None:
    order = DuckDbOrderRepository(con).get(OrderId.parse(7))
    assert order is not None
    assert order.estimated_delivery == date(2026, 2, 17)
    outcome = evaluate_return(order, ReturnReason.REGRET, SalesChannel.ONLINE, date(2026, 2, 21))
    assert isinstance(outcome, ReturnDecision)
    assert (outcome.day_of_window, outcome.window_days) == (4, 7)
    assert outcome.deadline == date(2026, 2, 24)
    assert outcome.eligible


def test_gherkin_order_8_in_transit(con: duckdb.DuckDBPyConnection) -> None:
    order = DuckDbOrderRepository(con).get(OrderId.parse(8))
    assert order is not None
    outcome = evaluate_return(order, ReturnReason.REGRET, SalesChannel.ONLINE, date(2026, 3, 25))
    assert isinstance(outcome, ReturnNotApplicable)
    assert outcome.cause is NotApplicableCause.IN_TRANSIT
