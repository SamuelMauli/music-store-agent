from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emporio.domain.entities import CatalogEntry, Product
from emporio.domain.enums import ProductStatus
from emporio.domain.specifications import (
    HasElectronics,
    HasStringType,
    InCategory,
    InStock,
    MatchAll,
    OnPromotion,
    PriceAtLeast,
    PriceAtMost,
    Specification,
)
from emporio.domain.value_objects import Money


def entry(
    *,
    category_id: int = 5,
    price: str = "500",
    stock: int = 5,
    status: ProductStatus = ProductStatus.ACTIVE,
    discount: int | None = None,
    specs: dict[str, str | int | bool] | None = None,
) -> CatalogEntry:
    product = Product(
        product_id=1,
        name="x",
        category_id=category_id,
        description="",
        price=Money.of(price),
        stock_quantity=stock,
        status=status,
        specs=specs or {"strings": "nylon", "electronics": True},
    )
    effective = Money.of(price).apply_discount(discount) if discount else Money.of(price)
    return CatalogEntry(
        product,
        "cat",
        effective,
        discount_percent=Decimal(discount) if discount else None,
    )


def test_composition_matches_the_documented_example() -> None:
    spec = InCategory(5) & PriceAtMost(1000) & InStock()
    assert spec.is_satisfied_by(entry(price="999"))
    assert not spec.is_satisfied_by(entry(price="1000.01"))
    assert not spec.is_satisfied_by(entry(category_id=1))
    assert not spec.is_satisfied_by(entry(stock=0))
    assert not spec.is_satisfied_by(entry(status=ProductStatus.DISCONTINUED, stock=13))


def test_sql_is_parameterized_and_never_inlines_values() -> None:
    sql, params = (InCategory(5) & PriceAtMost(1000) & InStock()).to_sql()
    assert sql == "((category_id = ? AND effective_price <= ?) AND availability IN (?, ?))"
    assert params == [5, Decimal("1000.00"), "available", "low_stock"]
    assert "1000" not in sql


def test_hostile_string_value_stays_in_params() -> None:
    hostile = "nylon'; DROP TABLE products; --"
    sql, params = HasStringType(hostile).to_sql()
    assert sql == "string_type = ?"
    assert params == [hostile]
    assert "DROP" not in sql


def test_or_not_and_neutral_element() -> None:
    spec = InCategory(1) | ~InCategory(5)
    assert spec.is_satisfied_by(entry(category_id=1))
    assert spec.is_satisfied_by(entry(category_id=9))
    assert not spec.is_satisfied_by(entry(category_id=5))
    sql, params = spec.to_sql()
    assert sql == "(category_id = ? OR (NOT category_id = ?))"
    assert params == [1, 5]
    assert MatchAll().to_sql() == ("TRUE", [])
    assert MatchAll().is_satisfied_by(entry())


def test_price_bounds_use_effective_price() -> None:
    discounted = entry(price="1100", discount=10)  # efetivo R$ 990
    assert PriceAtMost("1000").is_satisfied_by(discounted)
    assert PriceAtLeast("990").is_satisfied_by(discounted)
    assert not PriceAtLeast("991").is_satisfied_by(discounted)
    assert PriceAtLeast(500).to_sql() == ("effective_price >= ?", [Decimal("500.00")])


def test_promotion_strings_and_electronics() -> None:
    assert OnPromotion().is_satisfied_by(entry(discount=10))
    assert not OnPromotion().is_satisfied_by(entry())
    assert OnPromotion().to_sql() == ("discount_percent IS NOT NULL", [])
    assert HasStringType("nylon").is_satisfied_by(entry())
    assert not HasStringType("steel").is_satisfied_by(entry())
    assert HasElectronics().is_satisfied_by(entry())
    assert HasElectronics(wanted=False).is_satisfied_by(entry(specs={"electronics": False}))
    assert not HasElectronics().is_satisfied_by(entry(specs={"body": "Ash"}))
    assert HasElectronics(wanted=False).to_sql() == ("has_electronics = ?", [False])


@pytest.mark.parametrize("stock", [1, 2, 3])
def test_in_stock_accepts_low_stock(stock: int) -> None:
    assert InStock().is_satisfied_by(entry(stock=stock))


@given(
    st.integers(min_value=1, max_value=5000),
    st.integers(min_value=1, max_value=5000),
    st.booleans(),
)
def test_and_is_conjunction_and_placeholders_match_params(
    price: int, ceiling: int, flag: bool
) -> None:
    a: Specification = PriceAtMost(ceiling)
    b: Specification = HasElectronics(wanted=flag)
    item = entry(price=str(price))
    assert (a & b).is_satisfied_by(item) == (a.is_satisfied_by(item) and b.is_satisfied_by(item))
    sql, params = (a & b | ~a).to_sql()
    assert sql.count("?") == len(params)
