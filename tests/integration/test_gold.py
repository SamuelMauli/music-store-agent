from __future__ import annotations

from decimal import Decimal

import duckdb

from emporio.domain.entities import availability_of
from emporio.domain.enums import ProductStatus
from emporio.domain.value_objects import Money
from emporio.ingestion.gold import GOLD_VIEWS

PIX_DISCOUNT = Decimal(5)


def one(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[object, ...]:
    row = con.execute(sql).fetchone()
    assert row is not None
    return row


def test_all_gold_views_exist(con: duckdb.DuckDBPyConnection) -> None:
    names = {
        r[0]
        for r in con.execute(
            "SELECT view_name FROM duckdb_views() WHERE schema_name = 'gold'"
        ).fetchall()
    }
    assert set(GOLD_VIEWS) <= names


def test_catalog_has_one_row_per_product(con: duckdb.DuckDBPyConnection) -> None:
    assert one(con, "SELECT count(*), count(DISTINCT product_id) FROM gold.v_catalog") == (65, 65)


def test_golden_takamine_gd20(con: duckdb.DuckDBPyConnection) -> None:
    row = one(
        con,
        "SELECT price_brl, stock_quantity, effective_price, pix_price, availability "
        "FROM gold.v_catalog WHERE product_id = 95",
    )
    assert row == (Decimal("2199.00"), 5, Decimal("2199.00"), Decimal("2089.05"), "available")


def test_golden_violoes_ate_mil_reais(con: duckdb.DuckDBPyConnection) -> None:
    row = one(
        con,
        "SELECT count(*) FROM gold.v_catalog WHERE category_id = 5 AND effective_price <= 1000 "
        "AND availability IN ('available', 'low_stock')",
    )
    assert row == (12,)


def test_golden_four_active_promotions(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT product_id, discount_percent FROM gold.v_active_promotions ORDER BY 1"
    ).fetchall()
    assert rows == [
        (90, Decimal("18.00")),
        (94, Decimal("8.00")),
        (121, Decimal("20.00")),
        (127, Decimal("10.00")),
    ]


def test_promoted_product_has_no_pix_price(con: duckdb.DuckDBPyConnection) -> None:
    row = one(con, "SELECT effective_price, pix_price FROM gold.v_catalog WHERE product_id = 127")
    assert row[1] is None
    assert row[0] != one(con, "SELECT price_brl FROM silver.products WHERE product_id = 127")[0]


def test_effective_price_and_pix_match_python_money(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT product_id, price_brl, discount_percent, effective_price, pix_price "
        "FROM gold.v_catalog"
    ).fetchall()
    assert len(rows) == 65
    for product_id, price, discount, effective, pix in rows:
        base = Money(price)
        expected = base.apply_discount(discount) if discount is not None else base
        assert effective == expected.amount, product_id
        if discount is None:
            assert pix == base.apply_discount(PIX_DISCOUNT).amount, product_id
        else:
            assert pix is None, product_id


def test_availability_matches_python_rule(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT product_id, status, stock_quantity, availability FROM gold.v_catalog"
    ).fetchall()
    for product_id, status, stock, availability in rows:
        assert availability_of(ProductStatus(status), stock).value == availability, product_id


def test_discontinued_with_stock_is_not_offered(con: duckdb.DuckDBPyConnection) -> None:
    assert one(con, "SELECT availability FROM gold.v_catalog WHERE product_id = 136") == (
        "discontinued",
    )


def test_order_status_flags_total_divergence(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT order_id FROM gold.v_order_status WHERE NOT total_matches_items ORDER BY 1"
    ).fetchall()
    assert rows == [(3,), (20,)]
    assert one(con, "SELECT count(*) FROM gold.v_order_status") == (20,)
