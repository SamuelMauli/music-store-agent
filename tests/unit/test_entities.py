from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from emporio.domain.entities import (
    CatalogEntry,
    Customer,
    Order,
    OrderItem,
    Product,
    Promotion,
    availability_of,
)
from emporio.domain.enums import Availability, OrderStatus, PaymentMethod, ProductStatus
from emporio.domain.errors import InvalidValueError
from emporio.domain.value_objects import Email, Money, OrderId, Phone


def make_product(**overrides: object) -> Product:
    base: dict[str, object] = {
        "product_id": 95,
        "name": "Takamine GD20 Dreadnought Natural",
        "category_id": 5,
        "description": "Violão dreadnought",
        "price": Money.of("2199"),
        "stock_quantity": 5,
        "status": ProductStatus.ACTIVE,
        "specs": {"strings": "steel", "electronics": False},
    }
    base.update(overrides)
    return Product(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "stock", "expected"),
    [
        (ProductStatus.ACTIVE, 5, Availability.AVAILABLE),
        (ProductStatus.ACTIVE, 3, Availability.AVAILABLE),
        (ProductStatus.ACTIVE, 2, Availability.LOW_STOCK),
        (ProductStatus.ACTIVE, 1, Availability.LOW_STOCK),
        (ProductStatus.ACTIVE, 0, Availability.OUT_OF_STOCK),
        (ProductStatus.DISCONTINUED, 0, Availability.DISCONTINUED),
        # produto 136: descontinuado com estoque 13 não pode ser ofertado
        (ProductStatus.DISCONTINUED, 13, Availability.DISCONTINUED),
        (ProductStatus.COMING_SOON, 2, Availability.COMING_SOON),
    ],
)
def test_availability_status_wins_over_stock(
    status: ProductStatus, stock: int, expected: Availability
) -> None:
    assert availability_of(status, stock) is expected
    assert make_product(status=status, stock_quantity=stock).availability is expected


def test_product_invariants() -> None:
    with pytest.raises(InvalidValueError):
        make_product(stock_quantity=-1)
    with pytest.raises(InvalidValueError):
        make_product(price=Money.of(0))


def test_entities_are_immutable() -> None:
    product = make_product()
    with pytest.raises(AttributeError):
        product.stock_quantity = 1  # type: ignore[misc]


@pytest.mark.parametrize("pct", ["0", "-1", "100.5"])
def test_promotion_rejects_bad_discount(pct: str) -> None:
    with pytest.raises(InvalidValueError):
        Promotion(1, 95, Decimal(pct), "x", is_active=True)


def test_promotion_accepts_valid_discount() -> None:
    assert Promotion(12, 127, Decimal(10), "Oferta Especial", is_active=True).is_active


def test_order_item_requires_positive_quantity() -> None:
    with pytest.raises(InvalidValueError):
        OrderItem(1, 0)


def make_order(**overrides: object) -> Order:
    base: dict[str, object] = {
        "order_id": OrderId(8),
        "customer_id": 2,
        "order_date": date(2026, 2, 15),
        "status": OrderStatus.SHIPPED,
        "total": Money.of("349.90"),
        "payment_method": PaymentMethod.DEBIT,
        "tracking_code": "BRJL5544332BR",
        "estimated_delivery": date(2026, 2, 28),
    }
    base.update(overrides)
    return Order(**base)  # type: ignore[arg-type]


def test_order_overdue_when_shipped_and_estimate_passed() -> None:
    order = make_order()
    assert order.is_overdue(date(2026, 3, 25))
    assert not order.is_overdue(date(2026, 2, 28))
    assert not order.is_overdue(date(2026, 2, 20))


def test_order_not_overdue_if_not_shipped_or_no_estimate() -> None:
    assert not make_order(status=OrderStatus.DELIVERED).is_overdue(date(2026, 3, 25))
    assert not make_order(estimated_delivery=None).is_overdue(date(2026, 3, 25))


def test_order_ownership() -> None:
    order = make_order()
    assert order.belongs_to(2)
    assert not order.belongs_to(3)


def test_customer_repr_masks_pii() -> None:
    customer = Customer(
        1, "Lucas Mendes da Silva", Phone("(67) 99812-3456"), Email("lucas.mendes@jmail.com"), "CG"
    )
    text = repr(customer)
    assert "99812" not in text
    assert "lucas.mendes" not in text


def test_catalog_entry_exposes_availability() -> None:
    entry = CatalogEntry(make_product(stock_quantity=0), "Violões", Money.of("2199"))
    assert entry.availability is Availability.OUT_OF_STOCK
