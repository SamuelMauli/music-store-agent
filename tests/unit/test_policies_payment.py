from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emporio.domain.value_objects import Money
from emporio.policies.payment import (
    MAX_INSTALLMENTS,
    combined_payment_allowed,
    installment_options,
    minimum_installment,
    pix_price,
)

prices = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100000"), places=2).map(Money)


@pytest.mark.parametrize(
    ("price", "expected"),
    [("2199.00", "2089.05"), ("100.00", "95.00"), ("429.90", "408.41"), ("0.01", "0.01")],
)
def test_pix_price_applies_five_percent(price: str, expected: str) -> None:
    assert pix_price(Money(price), has_promotion=False) == Money(expected)


def test_pix_price_is_none_with_promotion() -> None:
    assert pix_price(Money("2199"), has_promotion=True) is None


@given(prices)
def test_pix_never_coexists_with_promotion(price: Money) -> None:
    assert pix_price(price, has_promotion=True) is None
    discounted = pix_price(price, has_promotion=False)
    assert discounted is not None
    assert discounted <= price


@pytest.mark.parametrize(
    ("count", "minimum"),
    [(1, "50"), (3, "50"), (4, "80"), (6, "80"), (7, "100"), (12, "100")],
)
def test_minimum_installment_by_tier(count: int, minimum: str) -> None:
    assert minimum_installment(count) == Money(minimum)


@pytest.mark.parametrize("count", [0, 13, -1])
def test_minimum_installment_rejects_out_of_range(count: int) -> None:
    with pytest.raises(ValueError, match="fora de 1"):
        minimum_installment(count)


@pytest.mark.parametrize(
    ("total", "counts"),
    [
        ("2199.00", list(range(1, 13))),
        ("1200.00", list(range(1, 13))),
        ("1199.99", list(range(1, 13))),
        ("1199.90", list(range(1, 12))),
        ("600.00", list(range(1, 7))),
        ("560.00", list(range(1, 7))),
        ("479.99", list(range(1, 7))),
        ("479.90", list(range(1, 6))),
        ("300.00", list(range(1, 4))),
        ("159.90", [1, 2, 3]),
        ("100.00", [1, 2]),
        ("99.99", [1, 2]),
        ("99.98", [1]),
        ("40.00", [1]),
    ],
)
def test_installment_counts_follow_tiers(total: str, counts: list[int]) -> None:
    assert [o.count for o in installment_options(Money(total))] == counts


def test_golden_gd20_twelve_times() -> None:
    assert installment_options(Money("2199"))[-1].amount == Money("183.25")


@given(prices)
def test_every_installment_respects_tier_minimum(total: Money) -> None:
    for option in installment_options(total):
        assert 1 <= option.count <= MAX_INSTALLMENTS
        if option.count > 1:
            assert option.amount >= minimum_installment(option.count)


@given(prices)
def test_installments_are_contiguous_and_within_a_cent_of_total(total: Money) -> None:
    options = installment_options(total)
    assert [o.count for o in options][:1] == [1]
    for opt in options:
        assert abs(opt.amount.times(opt.count).amount - total.amount) <= Decimal("0.01") * opt.count


@pytest.mark.parametrize(
    ("total", "allowed"),
    [("2000.00", False), ("2000.01", True), ("1999.99", False), ("8732", True)],
)
def test_combined_payment_threshold(total: str, allowed: bool) -> None:
    assert combined_payment_allowed(Money(total)) is allowed
