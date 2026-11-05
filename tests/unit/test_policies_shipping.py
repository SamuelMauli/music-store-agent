from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emporio.domain.value_objects import Money
from emporio.policies.shipping import (
    CityRequired,
    MetroShipping,
    OutsideShipping,
    estimate_shipping,
    is_metro_city,
    metro_shipping_fee,
    needs_special_freight,
)

METRO = ["Campo Grande"]


@pytest.mark.parametrize(
    ("total", "fee"),
    [("0.01", "35"), ("499.99", "35"), ("500.00", "35"), ("500.01", "0"), ("8732", "0")],
)
def test_metro_fee_table(total: str, fee: str) -> None:
    assert metro_shipping_fee(Money(total)) == Money(fee)


@given(st.decimals(min_value=0, max_value=100000, places=2).map(Money))
def test_metro_fee_is_zero_or_flat(total: Money) -> None:
    assert metro_shipping_fee(total) in (Money("0"), Money("35"))


@pytest.mark.parametrize("city", ["Campo Grande", "campo grande", " CAMPO  GRANDE "])
def test_is_metro_city_ignores_case_and_spaces(city: str) -> None:
    assert is_metro_city(city, METRO)


def test_is_metro_city_ignores_accents_and_respects_config() -> None:
    assert is_metro_city("Sidrolândia", ["Sidrolandia"])
    assert not is_metro_city("Dourados", METRO)


def test_estimate_metro_paid_and_free() -> None:
    paid = estimate_shipping(Money("159.90"), "Campo Grande", METRO)
    free = estimate_shipping(Money("600"), "campo grande", METRO)
    assert paid == MetroShipping(fee=Money("35"), free=False)
    assert free == MetroShipping(fee=Money("0"), free=True)
    assert paid.delivery == "1 a 3 dias úteis"


def test_estimate_outside_lists_modes_without_a_price() -> None:
    est = estimate_shipping(Money("2199"), "Dourados", METRO)
    assert isinstance(est, OutsideShipping)
    assert [(m.name, m.deadline_text) for m in est.modes] == [
        ("PAC (Correios)", "5 a 12 dias úteis"),
        ("SEDEX (Correios)", "2 a 5 dias úteis"),
        ("Jadlog (.package)", "3 a 8 dias úteis"),
    ]
    assert all(m.tracking and m.insured for m in est.modes)
    assert not est.special_freight


@pytest.mark.parametrize("city", [None, "", "   "])
def test_estimate_without_city_asks(city: str | None) -> None:
    est = estimate_shipping(Money("100"), city, METRO)
    assert est == CityRequired()
    assert "cidade" in est.question


def test_estimate_flags_special_freight() -> None:
    est = estimate_shipping(Money("9000"), "Dourados", METRO, large_item=True)
    assert isinstance(est, OutsideShipping)
    assert est.special_freight


@pytest.mark.parametrize(
    ("name", "large"),
    [
        ("Bateria Yamaha Kit 1", True),
        ("Piano Digital Casio", True),
        ("Contrabaixo Acústico", True),
        ("Violão Takamine GD20", False),
    ],
)
def test_needs_special_freight(name: str, large: bool) -> None:
    assert needs_special_freight(name) is large
