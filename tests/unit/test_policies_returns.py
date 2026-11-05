from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emporio.domain.entities import Order
from emporio.domain.enums import OrderStatus, PaymentMethod
from emporio.domain.value_objects import Money, OrderId
from emporio.policies.returns import (
    ChannelRequired,
    NotApplicableCause,
    ReturnDecision,
    ReturnItemTraits,
    ReturnNotApplicable,
    ReturnReason,
    SalesChannel,
    evaluate_return,
    non_returnable,
)

RECEIVED = date(2026, 2, 17)
ORDERED = date(2026, 2, 3)


def make_order(
    status: OrderStatus = OrderStatus.DELIVERED,
    delivery: date | None = RECEIVED,
    ordered: date = ORDERED,
) -> Order:
    return Order(
        order_id=OrderId.parse(7),
        customer_id=14,
        order_date=ordered,
        status=status,
        total=Money("159.90"),
        payment_method=PaymentMethod.CREDIT_3X,
        estimated_delivery=delivery,
    )


def decide(
    reason: ReturnReason,
    today: date,
    channel: SalesChannel = SalesChannel.ONLINE,
    order: Order | None = None,
    traits: tuple[ReturnItemTraits, ...] = (),
) -> ReturnDecision:
    outcome = evaluate_return(order or make_order(), reason, channel, today, traits)
    assert isinstance(outcome, ReturnDecision)
    return outcome


def test_gherkin_regret_day_four_of_seven() -> None:
    d = decide(ReturnReason.REGRET, date(2026, 2, 21))
    assert (d.eligible, d.rule) == (True, "4.1")
    assert (d.day_of_window, d.window_days, d.days_left) == (4, 7, 3)
    assert d.deadline == date(2026, 2, 24)
    assert d.refund == "mesma forma de pagamento original em até 10 dias úteis"
    assert d.freight == "por conta da loja"
    assert d.estimate_based
    assert "embalagem original" in d.requires


@pytest.mark.parametrize(
    ("today", "eligible", "days_left"),
    [
        (date(2026, 2, 17), True, 7),
        (date(2026, 2, 24), True, 0),
        (date(2026, 2, 25), False, 0),
        (date(2026, 2, 10), True, 14),
    ],
)
def test_regret_window_edges(today: date, eligible: bool, days_left: int) -> None:
    d = decide(ReturnReason.REGRET, today)
    assert (d.eligible, d.days_left) == (eligible, days_left)


def test_regret_unknown_channel_asks_the_customer() -> None:
    outcome = evaluate_return(
        make_order(), ReturnReason.REGRET, SalesChannel.UNKNOWN, date(2026, 2, 21)
    )
    assert outcome == ChannelRequired()
    assert "online" in outcome.question


def test_regret_in_store_purchase_is_not_eligible() -> None:
    d = decide(ReturnReason.REGRET, date(2026, 2, 21), SalesChannel.STORE)
    assert (d.eligible, d.rule, d.deadline) == (False, "4.1", None)


@pytest.mark.parametrize(
    ("status", "cause"),
    [
        (OrderStatus.SHIPPED, NotApplicableCause.IN_TRANSIT),
        (OrderStatus.CONFIRMED, NotApplicableCause.NOT_DELIVERED),
        (OrderStatus.PENDING, NotApplicableCause.NOT_DELIVERED),
        (OrderStatus.CANCELLED, NotApplicableCause.CANCELLED),
    ],
)
def test_undelivered_order_explains_window_starts_at_receipt(
    status: OrderStatus, cause: NotApplicableCause
) -> None:
    outcome = evaluate_return(
        make_order(status), ReturnReason.REGRET, SalesChannel.ONLINE, date(2026, 2, 21)
    )
    assert isinstance(outcome, ReturnNotApplicable)
    assert outcome.cause is cause
    if status is OrderStatus.SHIPPED:
        assert "recebimento" in outcome.explanation
        assert "recusar o recebimento na entrega" in outcome.options


def test_delivered_without_date_goes_to_a_human() -> None:
    outcome = evaluate_return(
        make_order(delivery=None), ReturnReason.REGRET, SalesChannel.ONLINE, date(2026, 2, 21)
    )
    assert isinstance(outcome, ReturnNotApplicable)
    assert outcome.cause is NotApplicableCause.MISSING_DELIVERY_DATE


@pytest.mark.parametrize(
    ("today", "rule", "eligible"),
    [
        (date(2026, 3, 5), "4.2", True),
        (date(2026, 3, 5 + 1), "warranty", True),
        (date(2026, 5, 18), "warranty", True),
        (date(2026, 5, 19), "warranty", False),
    ],
)
def test_defect_moves_from_exchange_to_legal_warranty(
    today: date, rule: str, eligible: bool
) -> None:
    d = decide(ReturnReason.DEFECT, today, SalesChannel.STORE)
    assert (d.rule, d.eligible) == (rule, eligible)


def test_defect_exchange_window_counts_from_purchase() -> None:
    d = decide(ReturnReason.DEFECT, date(2026, 2, 20))
    assert d.deadline == date(2026, 3, 5)
    assert d.window_days == 30


def test_defect_warranty_counts_90_days_from_receipt() -> None:
    d = decide(ReturnReason.DEFECT, date(2026, 4, 1))
    assert d.deadline == date(2026, 5, 18)
    assert "fabricante" in d.detail


def test_defect_outside_legal_warranty_points_to_manufacturer() -> None:
    d = decide(ReturnReason.DEFECT, date(2026, 8, 1))
    assert not d.eligible
    assert "§8.2" in d.detail


@pytest.mark.parametrize(
    ("today", "eligible"),
    [(date(2026, 2, 20), True), (date(2026, 2, 24), True), (date(2026, 2, 25), False)],
)
def test_preference_window(today: date, eligible: bool) -> None:
    d = decide(ReturnReason.PREFERENCE, today, SalesChannel.STORE)
    assert (d.rule, d.eligible) == ("4.3", eligible)
    assert "disponibilidade do produto desejado" in d.requires


@pytest.mark.parametrize(
    ("traits", "reason_text"),
    [
        (ReturnItemTraits(personalized=True), "personalização"),
        (ReturnItemTraits(final_sale=True), "venda final"),
        (ReturnItemTraits(mouthpiece=True), "boquilha"),
    ],
)
def test_non_returnable_items(traits: ReturnItemTraits, reason_text: str) -> None:
    assert reason_text in (non_returnable(traits) or "")
    for reason in (ReturnReason.REGRET, ReturnReason.PREFERENCE):
        d = decide(reason, date(2026, 2, 18), traits=(traits,))
        assert (d.eligible, d.rule) == (False, "4.4")


def test_returnable_item_has_no_exclusion() -> None:
    assert non_returnable(ReturnItemTraits()) is None


def test_defect_ignores_sale_exclusions() -> None:
    d = decide(ReturnReason.DEFECT, date(2026, 2, 20), traits=(ReturnItemTraits(final_sale=True),))
    assert d.eligible
    assert d.rule == "4.2"


@given(
    st.dates(min_value=date(2025, 10, 1), max_value=date(2026, 6, 30)),
    st.sampled_from(list(ReturnReason)),
)
def test_decision_invariants(today: date, reason: ReturnReason) -> None:
    d = decide(reason, today, SalesChannel.ONLINE)
    if d.eligible:
        assert d.deadline is not None
        assert d.days_left is not None
        assert d.days_left >= 0
        assert today <= d.deadline
    if d.window_days is not None and d.eligible:
        assert d.day_of_window is not None
        assert d.day_of_window <= d.window_days
