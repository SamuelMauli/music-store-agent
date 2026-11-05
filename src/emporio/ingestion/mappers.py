"""Anti-Corruption Layer: traduz linhas validadas do CSV para o modelo silver."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import cast

from emporio.domain.text import build_aliases, normalize_text
from emporio.domain.value_objects import Email, Phone
from emporio.ingestion.specs import flatten_specs, parse_specs

Typed = Mapping[str, object]
SilverRow = dict[str, object]
Mapper = Callable[[Typed], SilverRow]

_KINDS_WITH_STRING_TYPE = {"acoustic_guitar", "ukulele"}


def map_category(row: Typed) -> SilverRow:
    """Categoria sem transformação além dos tipos."""
    return {
        "category_id": row["category_id"],
        "name": row["name"],
        "description": row["description"],
    }


def map_customer(row: Typed) -> SilverRow:
    """Cliente com telefone só em dígitos e e-mail minúsculo.

    Raises:
        InvalidValueError: Se telefone ou e-mail forem inválidos.
    """
    return {
        "customer_id": row["customer_id"],
        "name": row["name"],
        "phone": Phone(cast(str, row["phone"])).digits,
        "email": Email(cast(str, row["email"])).value,
        "city": row["city"],
    }


def map_product(row: Typed) -> SilverRow:
    """Produto com specs tipadas por categoria, nome normalizado e apelidos.

    Raises:
        IngestionError: Se as specs não casarem com o esquema da categoria.
    """
    specs = parse_specs(cast(int, row["category_id"]), cast(str, row["specs"]))
    flat = flatten_specs(specs)
    name_norm = normalize_text(cast(str, row["name"]))
    has_string_type = specs.kind in _KINDS_WITH_STRING_TYPE
    electronics = flat.get("electronics")
    return {
        "product_id": row["product_id"],
        "name": row["name"],
        "name_norm": name_norm,
        "aliases": build_aliases(name_norm),
        "category_id": row["category_id"],
        "description": row["description"],
        "price_brl": row["price_brl"],
        "stock_quantity": row["stock_quantity"],
        "status": row["status"],
        "specs_kind": specs.kind,
        "specs": json.dumps(flat, ensure_ascii=False),
        "string_type": flat.get("strings") if has_string_type else None,
        "has_electronics": electronics if isinstance(electronics, bool) else None,
        "created_at": row["created_at"],
    }


def map_promotion(row: Typed) -> SilverRow:
    """Promoção sem transformação além dos tipos."""
    return {
        "promotion_id": row["promotion_id"],
        "product_id": row["product_id"],
        "discount_percent": row["discount_percent"],
        "description": row["description"],
        "is_active": row["is_active"],
    }


def map_order(row: Typed) -> SilverRow:
    """Pedido; ``total_brl`` é mantido como veio (fonte do faturamento)."""
    return {
        "order_id": row["order_id"],
        "customer_id": row["customer_id"],
        "order_date": row["order_date"],
        "status": row["status"],
        "total_brl": row["total_brl"],
        "payment_method": row["payment_method"],
        "tracking_code": row["tracking_code"],
        "estimated_delivery": row["estimated_delivery"],
        "notes": row["notes"],
    }


def map_order_item(row: Typed) -> SilverRow:
    """Item de pedido (``line_no`` é atribuído pela camada silver)."""
    return {
        "order_id": row["order_id"],
        "product_id": row["product_id"],
        "quantity": row["quantity"],
    }
