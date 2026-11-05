from __future__ import annotations

from datetime import date
from decimal import Decimal

import duckdb

from emporio.adapters.persistence import (
    DuckDbCustomerRepository,
    DuckDbOrderRepository,
    DuckDbProductRepository,
    DuckDbPromotionRepository,
)
from emporio.domain.entities import Order
from emporio.domain.enums import Availability, OrderStatus
from emporio.domain.lookup import Ambiguous, Found, NotFound
from emporio.domain.specifications import InCategory, InStock, PriceAtMost
from emporio.domain.value_objects import Email, Money, OrderId

ACOUSTIC = 5


def test_golden_violoes_ate_mil_reais(con: duckdb.DuckDBPyConnection) -> None:
    repo = DuckDbProductRepository(con)
    spec = InCategory(ACOUSTIC) & PriceAtMost(Money("1000")) & InStock()
    found = repo.find(spec)
    assert len(found) == 12
    assert [e.effective_price for e in found] == sorted(e.effective_price for e in found)
    assert all(spec.is_satisfied_by(e) for e in found)
    assert found[0].product.product_id == 83
    assert found[-1].product.product_id == 112


def test_find_honours_limit(con: duckdb.DuckDBPyConnection) -> None:
    repo = DuckDbProductRepository(con)
    assert len(repo.find(InCategory(ACOUSTIC), limit=2)) == 2


def test_golden_takamine_gd20_by_id(con: duckdb.DuckDBPyConnection) -> None:
    entry = DuckDbProductRepository(con).get(95)
    assert entry is not None
    assert entry.effective_price == Money("2199.00")
    assert entry.product.stock_quantity == 5
    assert entry.pix_price == Money("2089.05")
    assert entry.availability is Availability.AVAILABLE
    assert entry.category == "Violões"  # nome da categoria vem do dado


def test_get_unknown_product_is_none(con: duckdb.DuckDBPyConnection) -> None:
    assert DuckDbProductRepository(con).get(9999) is None


def test_promoted_product_has_no_pix(con: duckdb.DuckDBPyConnection) -> None:
    entry = DuckDbProductRepository(con).get(127)
    assert entry is not None
    assert entry.pix_price is None
    assert entry.discount_percent == Decimal("10.00")
    assert entry.promo_name


def test_resolve_name_by_alias_variants(con: duckdb.DuckDBPyConnection) -> None:
    repo = DuckDbProductRepository(con)
    for query in ("GD20", "takamine gd 20", "Takamine GD20", "gd 20"):
        result = repo.resolve_name(query)
        assert isinstance(result, Found), query
        assert result.entry.product.product_id == 95


def test_resolve_generic_name_is_ambiguous_with_at_most_three(
    con: duckdb.DuckDBPyConnection,
) -> None:
    result = DuckDbProductRepository(con).resolve_name("dreadnought")
    assert isinstance(result, Ambiguous)
    assert [o.product.product_id for o in result.options] == [91, 92, 93]


def test_resolve_unknown_name(con: duckdb.DuckDBPyConnection) -> None:
    assert DuckDbProductRepository(con).resolve_name("theremin quântico") == NotFound(
        "theremin quântico"
    )


def test_resolve_survives_hostile_input(con: duckdb.DuckDBPyConnection) -> None:
    result = DuckDbProductRepository(con).resolve_name("'; DROP TABLE gold.v_catalog; --")
    assert isinstance(result, NotFound)
    assert DuckDbProductRepository(con).get(95) is not None


def test_order_with_items(con: duckdb.DuckDBPyConnection) -> None:
    order = DuckDbOrderRepository(con).get(OrderId(15))
    assert isinstance(order, Order)
    assert len(order.items) == 2


def test_overdue_shipped_order(con: duckdb.DuckDBPyConnection) -> None:
    order = DuckDbOrderRepository(con).get(OrderId(8))
    assert order is not None
    assert order.status is OrderStatus.SHIPPED
    assert order.estimated_delivery == date(2026, 2, 28)
    assert order.is_overdue(date(2026, 3, 25))


def test_order_total_is_the_official_one(con: duckdb.DuckDBPyConnection) -> None:
    order = DuckDbOrderRepository(con).get(OrderId(3))
    assert order is not None
    assert order.total == Money("3450")  # itens somam 3498, mas o total oficial prevalece


def test_unknown_order_is_none(con: duckdb.DuckDBPyConnection) -> None:
    assert DuckDbOrderRepository(con).get(OrderId(9999)) is None


def test_orders_of_customer_newest_first(con: duckdb.DuckDBPyConnection) -> None:
    orders = DuckDbOrderRepository(con).list_for_customer(
        next(iter(con.execute("SELECT customer_id FROM silver.orders LIMIT 1").fetchone() or [0]))
    )
    assert orders
    dates = [o.order_date for o in orders]
    assert dates == sorted(dates, reverse=True)


def test_customer_lookup_and_shared_email(con: duckdb.DuckDBPyConnection) -> None:
    repo = DuckDbCustomerRepository(con)
    first = repo.get(11)
    assert first is not None
    shared = repo.find_by_email(first.email)
    assert [c.customer_id for c in shared] == [11, 43]
    assert repo.get(9999) is None
    assert repo.find_by_email(Email("ninguem@example.com")) == []


def test_customer_repr_never_exposes_contacts(con: duckdb.DuckDBPyConnection) -> None:
    customer = DuckDbCustomerRepository(con).get(1)
    assert customer is not None
    text = repr(customer)
    assert customer.email.value not in text
    assert customer.phone.digits not in text


def test_promotions(con: duckdb.DuckDBPyConnection) -> None:
    repo = DuckDbPromotionRepository(con)
    assert [p.product_id for p in repo.active()] == [127, 121, 90, 94]
    assert [p.is_active for p in repo.for_product(93)] == [False, False]
    assert len(repo.for_product(93)) == 2
