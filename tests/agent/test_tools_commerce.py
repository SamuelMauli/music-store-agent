from __future__ import annotations

from emporio.agent.tools.results import NotFound, Ok
from tests.agent.conftest import Caller


def test_golden_four_active_promotions(call: Caller) -> None:
    result = call("list_active_promotions")
    assert isinstance(result.result, Ok)
    promos = {p.name.split()[0]: p for p in result.result.value.promotions}
    assert result.result.value.total == 4
    assert set(promos) == {"Kalani", "Ohana", "Crafter", "Taylor"}
    assert promos["Kalani"].discount == "10%"
    assert promos["Crafter"].discount == "18%"
    assert promos["Taylor"].discount == "8%"
    assert promos["Ohana"].discount == "20%"


def test_promotions_filter_by_category(call: Caller) -> None:
    result = call("list_active_promotions", category="ukulele")
    assert isinstance(result.result, Ok)
    assert {p.name.split()[0] for p in result.result.value.promotions} == {"Kalani", "Ohana"}


def test_golden_gd20_payment(call: Caller) -> None:
    result = call("simulate_payment", product_ids=[95])
    assert isinstance(result.result, Ok)
    out = result.result.value
    assert out.total == "R$ 2.199,00"
    assert out.pix_total == "R$ 2.089,05"
    assert out.max_installments == "12x de R$ 183,25"
    assert "1x de R$ 2.199,00" in out.installments
    assert out.combined_payment.startswith("permitida")


def test_payment_by_amount_below_threshold(call: Caller) -> None:
    result = call("simulate_payment", amount=120)
    assert isinstance(result.result, Ok)
    assert result.result.value.combined_payment.startswith("não permitida")
    assert result.result.value.installments[0] == "1x de R$ 120,00"


def test_payment_on_promoted_product_has_no_pix(call: Caller) -> None:
    result = call("simulate_payment", product_ids=[94])
    assert isinstance(result.result, Ok)
    assert result.result.value.pix_total is None
    assert result.result.value.pix_note


def test_payment_mixed_basket_flags_promoted_items(call: Caller) -> None:
    result = call("simulate_payment", product_ids=[94, 95])
    assert isinstance(result.result, Ok)
    assert result.result.value.pix_total is not None
    assert result.result.value.pix_note


def test_payment_needs_input_and_valid_products(call: Caller) -> None:
    assert isinstance(call("simulate_payment").result, NotFound)
    assert isinstance(call("simulate_payment", product_ids=[9999]).result, NotFound)
