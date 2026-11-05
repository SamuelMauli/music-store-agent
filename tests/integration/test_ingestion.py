from __future__ import annotations

import json
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from emporio.clock import FixedClock
from emporio.ingestion.bronze import load_bronze
from emporio.ingestion.errors import IngestionError
from emporio.ingestion.pipeline import run_pipeline
from emporio.settings import Settings
from tests.conftest import REFERENCE, ROOT


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> object:
    row = con.execute(sql).fetchone()
    assert row is not None
    return row[0]


def test_row_counts_match_the_profile(con: duckdb.DuckDBPyConnection) -> None:
    expected = {
        "categories": 9,
        "customers": 50,
        "products": 65,
        "promotions": 25,
        "orders": 20,
        "order_items": 22,
    }
    for table, count in expected.items():
        assert scalar(con, f"SELECT count(*) FROM bronze.{table}") == count
        assert scalar(con, f"SELECT count(*) FROM silver.{table}") == count
        assert scalar(con, f"SELECT count(*) FROM silver.quarantine_{table}") == 0


def test_bronze_is_a_faithful_text_copy_with_lineage(con: duckdb.DuckDBPyConnection) -> None:
    columns = {r[0]: r[1] for r in con.execute("DESCRIBE bronze.orders").fetchall()}
    assert {"_source_file", "_ingested_at", "_row_hash"} <= set(columns)
    assert columns["total_brl"] == "VARCHAR"
    assert scalar(con, "SELECT total_brl FROM bronze.orders WHERE order_id = '4'") == "799.9"
    assert scalar(con, "SELECT DISTINCT _source_file FROM bronze.orders") == "orders.csv"
    assert scalar(con, "SELECT count(DISTINCT _row_hash) FROM bronze.orders") == 20


def test_silver_types_and_enums(con: duckdb.DuckDBPyConnection) -> None:
    types = {r[0]: r[1] for r in con.execute("DESCRIBE silver.products").fetchall()}
    assert types["price_brl"] == "DECIMAL(12,2)"
    assert types["created_at"] == "DATE"
    assert types["status"] == "ENUM('active', 'discontinued', 'coming_soon')"
    assert scalar(con, "SELECT price_brl FROM silver.products WHERE product_id = 95") == Decimal(
        "2199.00"
    )
    assert scalar(con, "SELECT order_date FROM silver.orders WHERE order_id = 8") == date(
        2026, 2, 15
    )


def test_silver_specs_union_by_category(con: duckdb.DuckDBPyConnection) -> None:
    kinds = dict(
        con.execute(
            "SELECT category_id, any_value(specs_kind) FROM silver.products GROUP BY 1"
        ).fetchall()
    )
    assert kinds == {
        1: "guitar",
        2: "bass",
        3: "drums",
        4: "keyboard",
        5: "acoustic_guitar",
        9: "ukulele",
    }
    specs = json.loads(str(scalar(con, "SELECT specs FROM silver.products WHERE product_id = 139")))
    assert specs == {"shells": "Maple", "pieces": 5, "hardware": "included", "color": "Blue"}
    assert scalar(con, "SELECT string_type FROM silver.products WHERE product_id = 95") == "steel"
    assert scalar(con, "SELECT has_electronics FROM silver.products WHERE product_id = 107") is True


def test_name_norm_and_aliases(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        "SELECT name_norm, aliases FROM silver.products WHERE product_id = 95"
    ).fetchone()
    assert row is not None
    assert row[0] == "takamine gd20 dreadnought natural"
    assert {"gd20", "gd 20", "takamine gd20"} <= set(row[1])


def test_customer_contacts_are_normalized(con: duckdb.DuckDBPyConnection) -> None:
    assert scalar(con, "SELECT phone FROM silver.customers WHERE customer_id = 1") == "67998123456"
    assert scalar(con, "SELECT count(*) FROM silver.customers WHERE email <> lower(email)") == 0


def test_order_items_get_a_line_number(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT line_no FROM silver.order_items WHERE order_id = 15 ORDER BY 1"
    ).fetchall()
    assert rows == [(1,), (2,)]


def test_pipeline_is_idempotent(ingested: Settings, reference_clock: FixedClock) -> None:
    first = run_pipeline(ingested, reference_clock)
    second = run_pipeline(ingested, reference_clock)
    assert first == second


def append_rows(path: Path, *lines: str) -> None:
    text = path.read_text(encoding="utf-8")
    prefix = "" if text.endswith("\n") else "\n"
    path.write_text(text + prefix + "\n".join(lines) + "\n", encoding="utf-8")


def dirty_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    shutil.copytree(ROOT / "data" / "raw", raw)
    append_rows(
        raw / "products.csv",
        '500,0,Preço zero,5,desc,1,active,"{}",2024-01-01',
        '501,10,Categoria fantasma,77,desc,1,active,"{}",2024-01-01',
        '502,10,Specs quebradas,2,desc,1,active,"{""body"": ""Ash""}",2024-01-01',
        '95,2199,Duplicado do GD20,5,desc,1,active,"{}",2024-01-01',
    )
    append_rows(raw / "customers.csv", "900,Fulano,123,not-an-email,Campo Grande")
    append_rows(raw / "orders.csv", "800,900,2026-03-01,pending,100,pix,,,")
    append_rows(raw / "order_items.csv", "800,1,95", "1,1,500")
    return raw


def test_quarantine_captures_bad_rows_with_reasons(
    tmp_path: Path, reference_clock: FixedClock
) -> None:
    settings = Settings(
        db_path=tmp_path / "dirty.duckdb",
        raw_dir=dirty_raw(tmp_path),
        contracts_dir=ROOT / "data" / "contracts",
        reference_datetime=REFERENCE,
    )
    result = run_pipeline(settings, reference_clock)
    assert result.silver["products"].accepted == 65
    assert result.silver["products"].quarantined == 4
    assert result.silver["customers"].quarantined == 1
    assert result.silver["orders"].quarantined == 1  # pai (cliente 900) está em quarentena
    assert result.silver["order_items"].quarantined == 2  # pedido 800 e produto 500
    con = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        reasons = [
            r[0] for r in con.execute("SELECT reasons FROM silver.quarantine_products").fetchall()
        ]
        joined = " | ".join(reasons)
        for fragment in (
            "abaixo do mínimo",
            "referência inexistente",
            "specs inválidas",
            "duplicado",
        ):
            assert fragment in joined
        assert (
            scalar(con, "SELECT count(*) FROM silver.products WHERE name = 'Duplicado do GD20'")
            == 0
        )
        payload = json.loads(str(scalar(con, "SELECT row_json FROM silver.quarantine_customers")))
        assert payload["customer_id"] == "900"
        assert (
            scalar(
                con, "SELECT count(*) FROM silver.quarantine_customers WHERE _row_hash IS NOT NULL"
            )
            == 1
        )
    finally:
        con.close()


def test_missing_source_file_fails_loudly(tmp_path: Path) -> None:
    con = duckdb.connect()
    try:
        with pytest.raises(IngestionError, match="ausente"):
            load_bronze(con, tmp_path, REFERENCE)
    finally:
        con.close()


def test_malformed_csv_fails_loudly(tmp_path: Path) -> None:
    raw = dirty_raw(tmp_path)
    (raw / "customers.csv").write_text('customer_id,name\n1,"aberta\n', encoding="utf-8")
    con = duckdb.connect()
    try:
        with pytest.raises(IngestionError, match="malformado"):
            load_bronze(con, raw, REFERENCE)
    finally:
        con.close()
