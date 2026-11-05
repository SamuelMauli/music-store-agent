"""Calendário de feriados integrado às políticas: reembolso em úteis, devolução em corridos."""

from __future__ import annotations

from datetime import date

from emporio.domain.calendar import HolidayCalendar
from emporio.domain.entities import Order
from emporio.domain.enums import OrderStatus, PaymentMethod
from emporio.domain.value_objects import Money, OrderId
from emporio.policies.returns import (
    ReturnDecision,
    ReturnReason,
    SalesChannel,
    evaluate_return,
)


def order(delivered: date) -> Order:
    return Order(
        order_id=OrderId.parse(7),
        customer_id=14,
        order_date=date(2026, 3, 1),
        status=OrderStatus.DELIVERED,
        total=Money("159.90"),
        payment_method=PaymentMethod.CREDIT_3X,
        estimated_delivery=delivered,
    )


def regret(today: date, calendar: HolidayCalendar | None = None) -> ReturnDecision:
    kwargs = {"calendar": calendar} if calendar else {}
    outcome = evaluate_return(
        order(date(2026, 3, 30)), ReturnReason.REGRET, SalesChannel.ONLINE, today, **kwargs
    )
    assert isinstance(outcome, ReturnDecision)
    return outcome


def test_refund_deadline_counts_business_days_skipping_good_friday() -> None:
    decision = regret(date(2026, 3, 31))
    # 10 úteis depois de ter 31/03: 01,02,(03 Sexta Santa),06,07,08,09,10,13,14,15
    assert decision.refund_by == date(2026, 4, 15)


def test_refund_deadline_with_carnival_calendar() -> None:
    # entrega 16/02 (Carnaval): 7 dias corridos seguem valendo, o reembolso pula 16 e 17
    delivered = date(2026, 2, 13)
    base = evaluate_return(
        order(delivered), ReturnReason.REGRET, SalesChannel.ONLINE, date(2026, 2, 13)
    )
    carnival = evaluate_return(
        order(delivered),
        ReturnReason.REGRET,
        SalesChannel.ONLINE,
        date(2026, 2, 13),
        calendar=HolidayCalendar(carnival=True),
    )
    assert isinstance(base, ReturnDecision)
    assert isinstance(carnival, ReturnDecision)
    assert base.refund_by == date(2026, 2, 27)
    assert carnival.refund_by == date(2026, 3, 3)
    assert base.deadline == carnival.deadline == date(2026, 2, 20)


def test_regret_window_stays_calendar_days_across_holidays() -> None:
    # entrega 30/03 + 7 dias CORRIDOS = 06/04, ainda que 03/04 seja Sexta Santa
    decision = regret(date(2026, 4, 6))
    assert decision.deadline == date(2026, 4, 6)
    assert decision.eligible
    assert decision.window_days == 7
    assert not regret(date(2026, 4, 7)).eligible


def test_no_refund_date_when_window_expired() -> None:
    assert regret(date(2026, 4, 20)).refund_by is None


def test_defect_and_preference_have_no_refund_date() -> None:
    for reason in (ReturnReason.DEFECT, ReturnReason.PREFERENCE):
        outcome = evaluate_return(
            order(date(2026, 3, 30)), reason, SalesChannel.ONLINE, date(2026, 3, 31)
        )
        assert isinstance(outcome, ReturnDecision)
        assert outcome.refund_by is None
