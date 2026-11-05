"""Repositórios DuckDB: traduzem linhas da gold/silver em entidades de domínio.

Toda consulta é parametrizada; o único SQL composto vem de ``Specification.to_sql``,
que só emite fragmentos fixos com ``?`` (nunca texto do cliente).
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import duckdb

from emporio.adapters.persistence.fuzzy import pick, rank_candidates
from emporio.domain.entities import (
    CatalogEntry,
    Customer,
    Order,
    OrderItem,
    Product,
    Promotion,
)
from emporio.domain.enums import OrderStatus, PaymentMethod, ProductStatus
from emporio.domain.lookup import Ambiguous, Found, NameLookup, NotFound
from emporio.domain.specifications import Specification
from emporio.domain.text import normalize_text
from emporio.domain.value_objects import Email, Money, OrderId, Phone

_CATALOG_SQL = (
    "SELECT product_id, name, category_id, category, description, specs, price_brl, "
    "stock_quantity, status, discount_percent, promo_name, effective_price, pix_price "
    "FROM gold.v_catalog"
)


def _entry(row: tuple[Any, ...]) -> CatalogEntry:
    (product_id, name, category_id, category, description, specs, price, stock, status) = row[:9]
    discount, promo_name, effective, pix = row[9:]
    product = Product(
        product_id=product_id,
        name=name,
        category_id=category_id,
        description=description,
        price=Money(price),
        stock_quantity=stock,
        status=ProductStatus(status),
        specs=json.loads(specs),
    )
    return CatalogEntry(
        product=product,
        category=category,
        effective_price=Money(effective),
        pix_price=Money(pix) if pix is not None else None,
        discount_percent=discount,
        promo_name=promo_name,
    )


class DuckDbProductRepository:
    """:class:`~emporio.ports.ProductRepository` sobre ``gold.v_catalog``."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def get(self, product_id: int) -> CatalogEntry | None:
        """Busca por ID."""
        row = self._con.execute(f"{_CATALOG_SQL} WHERE product_id = ?", [product_id]).fetchone()
        return _entry(row) if row else None

    def find(self, spec: Specification, limit: int | None = None) -> list[CatalogEntry]:
        """Filtra por especificação, do mais barato ao mais caro."""
        fragment, params = spec.to_sql()
        sql = f"{_CATALOG_SQL} WHERE {fragment} ORDER BY effective_price, product_id"
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, limit]
        return [_entry(row) for row in self._con.execute(sql, params).fetchall()]

    def resolve_name(self, query: str) -> NameLookup:
        """Resolve nome livre por ``token_set_ratio >= 85`` sobre nome e aliases.

        Returns:
            ``Found`` com vencedor único, ``Ambiguous`` com até 3 empatados, ou
            ``NotFound``.
        """
        rows = self._con.execute("SELECT product_id, name_norm, aliases FROM gold.v_catalog")
        candidates = [
            (int(pid), str(norm), list(aliases)) for pid, norm, aliases in rows.fetchall()
        ]
        needle = normalize_text(query)
        exact = [pid for pid, norm, _ in candidates if norm == needle]
        chosen = pick(rank_candidates(query, candidates), exact)
        entries = [(self.get(pid), score) for pid, score in chosen]
        found = [(entry, score) for entry, score in entries if entry is not None]
        if not found:
            return NotFound(query)
        if len(found) == 1:
            entry, score = found[0]
            return Found(entry, score)
        return Ambiguous(tuple(entry for entry, _ in found))


class DuckDbOrderRepository:
    """:class:`~emporio.ports.OrderRepository` sobre ``gold.v_order_status``."""

    _SQL = (
        "SELECT order_id, customer_id, order_date, status, total_brl, payment_method, "
        "tracking_code, estimated_delivery, notes FROM gold.v_order_status"
    )

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def get(self, order_id: OrderId) -> Order | None:
        """Busca por número, com itens."""
        row = self._con.execute(f"{self._SQL} WHERE order_id = ?", [order_id.value]).fetchone()
        return self._build(row) if row else None

    def list_for_customer(self, customer_id: int) -> list[Order]:
        """Pedidos do cliente, do mais recente ao mais antigo."""
        rows = self._con.execute(
            f"{self._SQL} WHERE customer_id = ? ORDER BY order_date DESC, order_id DESC",
            [customer_id],
        ).fetchall()
        return [self._build(row) for row in rows]

    def _build(self, row: tuple[Any, ...]) -> Order:
        order_id, customer_id, order_date, status, total, payment, tracking, eta, notes = row
        items = self._con.execute(
            "SELECT product_id, quantity FROM silver.order_items WHERE order_id = ? "
            "ORDER BY line_no",
            [order_id],
        ).fetchall()
        return Order(
            order_id=OrderId(order_id),
            customer_id=customer_id,
            order_date=order_date,
            status=OrderStatus(status),
            total=Money(total),
            payment_method=PaymentMethod(payment),
            tracking_code=tracking,
            estimated_delivery=eta,
            notes=notes,
            items=tuple(OrderItem(product_id=p, quantity=q) for p, q in items),
        )


class DuckDbCustomerRepository:
    """:class:`~emporio.ports.CustomerRepository` sobre ``silver.customers``."""

    _SQL = "SELECT customer_id, name, phone, email, city FROM silver.customers"

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def get(self, customer_id: int) -> Customer | None:
        """Busca por ID."""
        row = self._con.execute(f"{self._SQL} WHERE customer_id = ?", [customer_id]).fetchone()
        return self._build(row) if row else None

    def find_by_email(self, email: Email) -> list[Customer]:
        """Todos os clientes que usam o e-mail (ele não é único)."""
        rows = self._con.execute(
            f"{self._SQL} WHERE email = ? ORDER BY customer_id", [email.value]
        ).fetchall()
        return [self._build(row) for row in rows]

    @staticmethod
    def _build(row: tuple[Any, ...]) -> Customer:
        customer_id, name, phone, email, city = row
        return Customer(
            customer_id=customer_id, name=name, phone=Phone(phone), email=Email(email), city=city
        )


class DuckDbPromotionRepository:
    """:class:`~emporio.ports.PromotionRepository` sobre ``silver.promotions``."""

    _SQL = (
        "SELECT promotion_id, product_id, discount_percent, description, is_active "
        "FROM silver.promotions"
    )

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def active(self) -> list[Promotion]:
        """Promoções ativas."""
        return self._run(f"{self._SQL} WHERE is_active ORDER BY promotion_id", [])

    def for_product(self, product_id: int) -> list[Promotion]:
        """Todas as promoções do produto."""
        return self._run(f"{self._SQL} WHERE product_id = ? ORDER BY promotion_id", [product_id])

    def _run(self, sql: str, params: list[object]) -> list[Promotion]:
        return [
            Promotion(
                promotion_id=pid,
                product_id=product_id,
                discount_percent=Decimal(discount),
                description=description,
                is_active=is_active,
            )
            for pid, product_id, discount, description, is_active in self._con.execute(
                sql, params
            ).fetchall()
        ]
