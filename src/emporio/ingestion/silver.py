"""Camada silver: tipos corretos, enums, specs por categoria e quarentena com motivo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import duckdb

from emporio.domain.enums import OrderStatus, PaymentMethod, ProductStatus
from emporio.domain.errors import DomainError
from emporio.ingestion.contracts import RowValidator, contract_for
from emporio.ingestion.mappers import (
    Mapper,
    SilverRow,
    map_category,
    map_customer,
    map_order,
    map_order_item,
    map_product,
    map_promotion,
)


def _enum_sql(enum: type[StrEnum]) -> str:
    return ", ".join(f"'{member.value}'" for member in enum)


_TYPES_SQL = (
    f"CREATE TYPE silver.product_status AS ENUM ({_enum_sql(ProductStatus)})",
    f"CREATE TYPE silver.order_status AS ENUM ({_enum_sql(OrderStatus)})",
    f"CREATE TYPE silver.payment_method AS ENUM ({_enum_sql(PaymentMethod)})",
)

_QUARANTINE_COLUMNS = "_row_hash VARCHAR, row_json VARCHAR NOT NULL, reasons VARCHAR NOT NULL"


@dataclass(frozen=True)
class SilverTable:
    """Descrição de uma tabela silver: DDL, colunas e mapper."""

    name: str
    ddl: str
    columns: tuple[str, ...]
    mapper: Mapper


SILVER_TABLES: tuple[SilverTable, ...] = (
    SilverTable(
        "categories",
        "category_id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, description VARCHAR",
        ("category_id", "name", "description"),
        map_category,
    ),
    SilverTable(
        "customers",
        "customer_id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, phone VARCHAR NOT NULL, "
        "email VARCHAR NOT NULL, city VARCHAR",
        ("customer_id", "name", "phone", "email", "city"),
        map_customer,
    ),
    SilverTable(
        "products",
        "product_id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, name_norm VARCHAR NOT NULL, "
        "aliases VARCHAR[] NOT NULL, category_id INTEGER NOT NULL REFERENCES silver.categories, "
        "description VARCHAR, price_brl DECIMAL(12,2) NOT NULL CHECK (price_brl > 0), "
        "stock_quantity INTEGER NOT NULL CHECK (stock_quantity >= 0), "
        "status silver.product_status NOT NULL, specs_kind VARCHAR NOT NULL, "
        "specs JSON NOT NULL, string_type VARCHAR, has_electronics BOOLEAN, created_at DATE",
        (
            "product_id",
            "name",
            "name_norm",
            "aliases",
            "category_id",
            "description",
            "price_brl",
            "stock_quantity",
            "status",
            "specs_kind",
            "specs",
            "string_type",
            "has_electronics",
            "created_at",
        ),
        map_product,
    ),
    SilverTable(
        "promotions",
        "promotion_id INTEGER PRIMARY KEY, product_id INTEGER NOT NULL REFERENCES silver.products, "
        "discount_percent DECIMAL(5,2) NOT NULL, description VARCHAR, is_active BOOLEAN NOT NULL",
        ("promotion_id", "product_id", "discount_percent", "description", "is_active"),
        map_promotion,
    ),
    SilverTable(
        "orders",
        "order_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES silver.customers, "
        "order_date DATE NOT NULL, status silver.order_status NOT NULL, "
        "total_brl DECIMAL(12,2) NOT NULL, payment_method silver.payment_method NOT NULL, "
        "tracking_code VARCHAR, estimated_delivery DATE, notes VARCHAR",
        (
            "order_id",
            "customer_id",
            "order_date",
            "status",
            "total_brl",
            "payment_method",
            "tracking_code",
            "estimated_delivery",
            "notes",
        ),
        map_order,
    ),
    SilverTable(
        "order_items",
        "order_id INTEGER NOT NULL REFERENCES silver.orders, line_no INTEGER NOT NULL, "
        "product_id INTEGER NOT NULL REFERENCES silver.products, "
        "quantity INTEGER NOT NULL CHECK (quantity >= 1), PRIMARY KEY (order_id, line_no)",
        ("order_id", "line_no", "product_id", "quantity"),
        map_order_item,
    ),
)


@dataclass(frozen=True)
class TableResult:
    """Resultado de uma tabela na silver.

    Attributes:
        accepted: Linhas aceitas.
        quarantined: Linhas enviadas à quarentena.
    """

    accepted: int
    quarantined: int


def _fetch_dicts(con: duckdb.DuckDBPyConnection, table: str) -> list[dict[str, str | None]]:
    cursor = con.execute(f"SELECT * FROM bronze.{table} ORDER BY rowid")  # noqa: S608
    names = [col[0] for col in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _prepare_schema(con: duckdb.DuckDBPyConnection) -> None:
    # Camadas derivadas são recriadas do zero; gold depende de silver, então cai primeiro.
    con.execute("DROP SCHEMA IF EXISTS gold CASCADE")
    con.execute("DROP SCHEMA IF EXISTS silver CASCADE")
    con.execute("CREATE SCHEMA silver")
    for statement in _TYPES_SQL:
        con.execute(statement)
    for table in SILVER_TABLES:
        con.execute(f"CREATE TABLE silver.{table.name} ({table.ddl})")
        con.execute(f"CREATE TABLE silver.quarantine_{table.name} ({_QUARANTINE_COLUMNS})")


def _insert(con: duckdb.DuckDBPyConnection, table: SilverTable, rows: list[SilverRow]) -> None:
    if not rows:
        return
    cols = ", ".join(table.columns)
    marks = ", ".join("?" for _ in table.columns)
    con.executemany(
        f"INSERT INTO silver.{table.name} ({cols}) VALUES ({marks})",  # noqa: S608
        [[row[c] for c in table.columns] for row in rows],
    )


def _quarantine(
    con: duckdb.DuckDBPyConnection, table: str, rows: list[tuple[str | None, str, str]]
) -> None:
    if rows:
        con.executemany(f"INSERT INTO silver.quarantine_{table} VALUES (?, ?, ?)", rows)  # noqa: S608


def _process_table(
    con: duckdb.DuckDBPyConnection,
    table: SilverTable,
    contracts_dir: Path,
    accepted_refs: dict[str, set[object]],
) -> TableResult:
    contract = contract_for(table.name, contracts_dir)
    validator = RowValidator(contract.table, references=accepted_refs)
    accepted: list[SilverRow] = []
    rejected: list[tuple[str | None, str, str]] = []
    line_counter: dict[object, int] = {}
    for raw in _fetch_dicts(con, table.name):
        typed, reasons = validator.validate(raw)
        mapped: SilverRow = {}
        if not reasons:
            try:
                mapped = table.mapper(typed)
            except DomainError as exc:
                reasons = [str(exc)]
        if reasons:
            visible = {k: v for k, v in raw.items() if not k.startswith("_")}
            rejected.append(
                (raw.get("_row_hash"), json.dumps(visible, ensure_ascii=False), "; ".join(reasons))
            )
            continue
        if table.name == "order_items":
            mapped["line_no"] = line_counter[mapped["order_id"]] = (
                line_counter.get(mapped["order_id"], 0) + 1
            )
        accepted.append(mapped)
        for prop in contract.table.properties:
            if prop.unique:
                accepted_refs.setdefault(f"{table.name}.{prop.name}", set()).add(typed[prop.name])
    _insert(con, table, accepted)
    _quarantine(con, table.name, rejected)
    return TableResult(len(accepted), len(rejected))


def build_silver(con: duckdb.DuckDBPyConnection, contracts_dir: Path) -> dict[str, TableResult]:
    """Constrói a silver a partir da bronze, validando cada linha pelo contrato.

    Linhas que violam o contrato (tipo, obrigatório, enum, mínimo, unicidade,
    chave estrangeira, esquema de specs, contato inválido) vão para
    ``silver.quarantine_<tabela>`` com o motivo; as demais entram tipadas. Uma
    linha cujo pai foi para a quarentena também é quarentenada (FK).

    Args:
        con: Conexão DuckDB com a bronze carregada.
        contracts_dir: Diretório dos contratos ``<tabela>.yaml``.

    Returns:
        Contagem de aceitas e quarentenadas por tabela.

    Raises:
        IngestionError: Se algum contrato estiver ausente ou inválido.
    """
    _prepare_schema(con)
    accepted_refs: dict[str, set[object]] = {}
    return {
        table.name: _process_table(con, table, contracts_dir, accepted_refs)
        for table in SILVER_TABLES
    }
