"""Regras de qualidade dos dados (achados D1 a D14 do diagnóstico).

Cada regra lê a camada silver (e a bronze quando precisa do texto original),
devolve um :class:`Finding` e nunca altera dado. IDs afetados saem ordenados.
PII nunca entra no relatório (e-mails saem mascarados).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import duckdb

from emporio.domain.text import normalize_text
from emporio.domain.value_objects import Email

KNOWN_BRANDS = (
    "yamaha", "tagima", "giannini", "rozini", "shelby", "crafter", "martin", "taylor",
    "takamine", "kala", "ohana", "kalani", "gibson", "ibanez", "prs", "fender",
    "music man", "pearl", "tama", "korg", "nord", "roland",
)  # fmt: skip
# Modelos que identificam a marca sem citá-la (ex.: "Precision" é Fender).
MODEL_BRAND = {
    "precision": "fender",
    "jazz bass": "fender",
    "stratocaster": "fender",
    "telecaster": "fender",
    "les paul": "gibson",
    "stingray": "music man",
}
KNOWN_WOODS = ("poplar", "birch", "maple", "basswood", "mahogany")


@dataclass(frozen=True)
class Finding:
    """Resultado de uma regra de qualidade.

    Attributes:
        finding: Código do achado no diagnóstico (``D1`` a ``D14``).
        rule: Nome estável da regra.
        entity: Entidade afetada (``products``, ``orders``...).
        severity: ``info`` ou ``warning``.
        description: Explicação curta do achado.
        affected_ids: Identificadores afetados, ordenados.
        details: Dados de apoio (valores esperados/encontrados, sem PII).
    """

    finding: str
    rule: str
    entity: str
    severity: str
    description: str
    affected_ids: list[int]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa para o relatório JSON."""
        return {
            "finding": self.finding,
            "entity": self.entity,
            "severity": self.severity,
            "description": self.description,
            "count": len(self.affected_ids),
            "affected_ids": self.affected_ids,
            "details": self.details,
        }


@dataclass(frozen=True)
class QualityContext:
    """Entrada comum das regras.

    Attributes:
        con: Conexão DuckDB com bronze e silver prontas.
        today: Data de referência (do ``Clock``).
    """

    con: duckdb.DuckDBPyConnection
    today: date


Rule = Callable[[QualityContext], Finding]


def _ids(ctx: QualityContext, sql: str) -> list[int]:
    return sorted(int(row[0]) for row in ctx.con.execute(sql).fetchall())


def empty_categories(ctx: QualityContext) -> Finding:
    """D1: categorias sem nenhum produto."""
    ids = _ids(
        ctx,
        "SELECT category_id FROM silver.categories "
        "WHERE category_id NOT IN (SELECT category_id FROM silver.products)",
    )
    return Finding(
        "D1", "empty_categories", "categories", "warning",
        "categorias cadastradas sem produto: responder que não há no catálogo",
        ids,
    )  # fmt: skip


def _brands_in(text: str) -> set[str]:
    norm = normalize_text(text)
    found = {b for b in KNOWN_BRANDS if re.search(rf"\b{re.escape(b)}\b", norm)}
    found |= {brand for model, brand in MODEL_BRAND.items() if re.search(rf"\b{model}\b", norm)}
    return found


def _name_brand(name: str) -> str | None:
    norm = normalize_text(name)
    hits = [(m.start(), -len(b), b) for b in KNOWN_BRANDS if (m := re.search(rf"\b{b}\b", norm))]
    return min(hits)[2] if hits else None


def brand_consistency(ctx: QualityContext) -> Finding:
    """D2: marca do nome ausente da descrição, que cita outra marca."""
    rows = ctx.con.execute("SELECT product_id, name, description FROM silver.products").fetchall()
    ids: list[int] = []
    details: dict[str, Any] = {}
    for product_id, name, description in rows:
        brand, described = _name_brand(name), _brands_in(description or "")
        if brand and described and brand not in described:
            ids.append(product_id)
            details[str(product_id)] = {
                "name_brand": brand,
                "description_brands": sorted(described),
            }
    return Finding(
        "D2", "brand_consistency", "products", "warning",
        "nome e descrição citam marcas diferentes: usar nome + specs, omitir a descrição",
        sorted(ids), details,
    )  # fmt: skip


def _count_mismatch(description: str, specs: dict[str, Any]) -> dict[str, Any] | None:
    for noun, key in (("cordas", "strings"), ("teclas", "keys")):
        match = re.search(rf"(\d+)\s*{noun}", description)
        declared = specs.get(key)
        # Em violões `strings` é nylon/aço (texto): só compara quando é contagem.
        if match and isinstance(declared, int) and int(match.group(1)) != declared:
            return {"field": key, "description": int(match.group(1)), "specs": specs[key]}
    return None


def _wood_mismatch(description: str, specs: dict[str, Any]) -> dict[str, Any] | None:
    mentioned = [w for w in KNOWN_WOODS if w in normalize_text(description)]
    shells = str(specs.get("shells", "")).lower()
    if "shells" in specs and mentioned and shells not in mentioned:
        return {"field": "shells", "description": mentioned, "specs": specs["shells"]}
    return None


def specs_match_description(ctx: QualityContext) -> Finding:
    """D3: specs estruturadas contradizem a descrição (cordas, teclas, cascos)."""
    rows = ctx.con.execute("SELECT product_id, description, specs FROM silver.products").fetchall()
    ids: list[int] = []
    details: dict[str, Any] = {}
    for product_id, description, raw_specs in rows:
        specs = json.loads(raw_specs)
        mismatch = _count_mismatch(description or "", specs) or _wood_mismatch(
            description or "", specs
        )
        if mismatch:
            ids.append(product_id)
            details[str(product_id)] = mismatch
    return Finding(
        "D3", "specs_match_description", "products", "warning",
        "specs contradizem a descrição: as specs prevalecem",
        sorted(ids), details,
    )  # fmt: skip


def specs_format_variant(ctx: QualityContext) -> Finding:
    """D4: JSON de specs fora do formato compacto ou com números como texto."""
    rows = ctx.con.execute("SELECT product_id, specs FROM bronze.products").fetchall()
    ids: list[int] = []
    details: dict[str, Any] = {}
    for product_id, raw in rows:
        payload = json.loads(raw)
        compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        numeric_text = sorted(k for k, v in payload.items() if isinstance(v, str) and v.isdigit())
        if raw != compact:
            ids.append(int(product_id))
            details[str(product_id)] = {"spaced_json": True, "numeric_as_text": numeric_text}
    shapes = ctx.con.execute(
        "SELECT specs_kind, count(*) FROM silver.products GROUP BY 1 ORDER BY 1"
    ).fetchall()
    details["products_per_schema"] = dict(shapes)
    return Finding(
        "D4", "specs_format_variant", "products", "info",
        "esquema de specs varia por categoria e parte do lote usa formatação diferente",
        sorted(ids), details,
    )  # fmt: skip


def _status_stock(ctx: QualityContext, where: str) -> list[int]:
    return _ids(ctx, f"SELECT product_id FROM silver.products WHERE {where}")  # noqa: S608


def active_without_stock(ctx: QualityContext) -> Finding:
    """D5: produto ativo com estoque zero."""
    ids = _status_stock(ctx, "status = 'active' AND stock_quantity = 0")
    return Finding(
        "D5", "active_without_stock", "products", "warning",
        "ativo sem estoque: temporariamente indisponível, sugerir similares", ids,
    )  # fmt: skip


def discontinued_without_stock(ctx: QualityContext) -> Finding:
    """D6: produto descontinuado e sem estoque."""
    ids = _status_stock(ctx, "status = 'discontinued' AND stock_quantity = 0")
    return Finding(
        "D6", "discontinued_without_stock", "products", "info",
        "descontinuado sem estoque: fora do catálogo, sugerir sucessor", ids,
    )  # fmt: skip


def discontinued_has_stock(ctx: QualityContext) -> Finding:
    """D7: produto descontinuado que ainda tem estoque."""
    ids = _status_stock(ctx, "status = 'discontinued' AND stock_quantity > 0")
    stock = {
        str(r[0]): r[1]
        for r in ctx.con.execute(
            "SELECT product_id, stock_quantity FROM silver.products "
            "WHERE status = 'discontinued' AND stock_quantity > 0 ORDER BY 1"
        ).fetchall()
    }
    return Finding(
        "D7", "discontinued_has_stock", "products", "warning",
        "descontinuado com saldo: o status prevalece, não ofertar e sinalizar à loja",
        ids, {"stock_quantity": stock},
    )  # fmt: skip


def unreleased_in_orders(ctx: QualityContext) -> Finding:
    """D8: produto ``coming_soon`` presente em pedidos."""
    rows = ctx.con.execute(
        """
        SELECT o.order_id, o.status, p.product_id
        FROM silver.order_items i
        JOIN silver.orders o USING (order_id)
        JOIN silver.products p USING (product_id)
        WHERE p.status = 'coming_soon' ORDER BY 1
        """
    ).fetchall()
    details = {"orders": {str(o): {"status": s, "product_id": p} for o, s, p in rows}}
    return Finding(
        "D8", "unreleased_in_orders", "orders", "warning",
        "produto em lançamento aparece em pedido: inconsistência de dados",
        sorted({int(r[0]) for r in rows}), details,
    )  # fmt: skip


def order_total_mismatch(ctx: QualityContext) -> Finding:
    """D9: ``total_brl`` diferente da soma de itens a preço de tabela."""
    rows = ctx.con.execute(
        """
        SELECT o.order_id, o.total_brl, sum(i.quantity * p.price_brl) AS items_total
        FROM silver.orders o
        JOIN silver.order_items i USING (order_id)
        JOIN silver.products p USING (product_id)
        GROUP BY o.order_id, o.total_brl
        HAVING o.total_brl <> sum(i.quantity * p.price_brl)
        ORDER BY 1
        """
    ).fetchall()
    details = {
        str(o): {"total_brl": str(t), "items_total": str(s), "difference": str(s - t)}
        for o, t, s in rows
    }
    return Finding(
        "D9", "order_total_mismatch", "orders", "warning",
        "total do pedido difere da soma dos itens: informar o total do pedido, nunca recalcular",
        [int(r[0]) for r in rows], details,
    )  # fmt: skip


def duplicate_email(ctx: QualityContext) -> Finding:
    """D10: mesmo e-mail em clientes diferentes (e-mail mascarado no relatório)."""
    rows = ctx.con.execute(
        """
        SELECT email, list(customer_id ORDER BY customer_id)
        FROM silver.customers GROUP BY email HAVING count(*) > 1 ORDER BY 1
        """
    ).fetchall()
    ids = sorted(cid for _, cids in rows for cid in cids)
    details = {Email(email).masked(): cids for email, cids in rows}
    return Finding(
        "D10", "duplicate_email", "customers", "warning",
        "e-mail compartilhado: identidade exige pedido pertencente ao cliente", ids, details,
    )  # fmt: skip


def multiple_promotions(ctx: QualityContext) -> Finding:
    """D11a: produto com mais de uma promoção cadastrada."""
    rows = ctx.con.execute(
        """
        SELECT product_id, list(promotion_id ORDER BY promotion_id)
        FROM silver.promotions GROUP BY product_id HAVING count(*) > 1 ORDER BY 1
        """
    ).fetchall()
    return Finding(
        "D11", "multiple_promotions_per_product", "promotions", "info",
        "produtos com mais de uma promoção cadastrada",
        [int(r[0]) for r in rows], {"promotion_ids": {str(p): ids for p, ids in rows}},
    )  # fmt: skip


def multiple_active_promotions(ctx: QualityContext) -> Finding:
    """D11b: produto com mais de uma promoção ativa (aplica-se a de maior desconto)."""
    ids = _ids(
        ctx,
        "SELECT product_id FROM silver.promotions WHERE is_active "
        "GROUP BY product_id HAVING count(*) > 1",
    )
    return Finding(
        "D11", "multiple_active_promotions_per_product", "promotions", "warning",
        "mais de uma promoção ativa no produto: vale a de maior desconto", ids,
    )  # fmt: skip


def active_promotions(ctx: QualityContext) -> Finding:
    """D11c: produtos com promoção ativa (somente ``is_active``)."""
    ids = _ids(ctx, "SELECT DISTINCT product_id FROM silver.promotions WHERE is_active")
    return Finding(
        "D11", "active_promotions", "promotions", "info",
        "produtos com promoção ativa hoje (a flag é a única vigência)", ids,
    )  # fmt: skip


def promotions_without_validity(ctx: QualityContext) -> Finding:
    """D11d: promoções sem datas de vigência no cadastro."""
    columns = {r[0] for r in ctx.con.execute("DESCRIBE silver.promotions").fetchall()}
    has_dates = {"start_date", "end_date", "valid_from", "valid_until"} & columns
    ids = [] if has_dates else _ids(ctx, "SELECT promotion_id FROM silver.promotions")
    return Finding(
        "D11", "promotions_without_validity_dates", "promotions", "info",
        "promoções sem vigência: 'vigente hoje' é a flag is_active", ids,
    )  # fmt: skip


def shipped_overdue(ctx: QualityContext) -> Finding:
    """D12: pedido despachado cuja entrega estimada já passou."""
    rows = ctx.con.execute(
        "SELECT order_id, estimated_delivery FROM silver.orders "
        "WHERE status = 'shipped' AND estimated_delivery < ? ORDER BY 1",
        [ctx.today],
    ).fetchall()
    details = {
        str(o): {"estimated_delivery": est.isoformat(), "days_late": (ctx.today - est).days}
        for o, est in rows
    }
    return Finding(
        "D12", "shipped_overdue", "orders", "warning",
        "despachado com entrega estimada vencida: pedir desculpas e oferecer chamado",
        [int(r[0]) for r in rows], details | {"reference_date": ctx.today.isoformat()},
    )  # fmt: skip


def delivered_without_receipt_date(ctx: QualityContext) -> Finding:
    """D13: pedidos entregues sem data real de recebimento."""
    columns = {r[0] for r in ctx.con.execute("DESCRIBE silver.orders").fetchall()}
    has_receipt = bool({"delivered_at", "received_at"} & columns)
    ids = (
        []
        if has_receipt
        else _ids(ctx, "SELECT order_id FROM silver.orders WHERE status = 'delivered'")
    )
    return Finding(
        "D13", "delivered_without_receipt_date", "orders", "info",
        "sem data de recebimento: estimated_delivery serve de proxy declarado ao cliente",
        ids,
    )  # fmt: skip


_SHIPPING_COLUMNS = {
    "orders": ("address", "cep", "zip_code"),
    "products": ("weight", "weight_kg", "dimensions"),
}


def missing_shipping_attributes(ctx: QualityContext) -> Finding:
    """D14: dados necessários para cotar frete não existem na origem."""
    missing: dict[str, list[str]] = {}
    for table, wanted in _SHIPPING_COLUMNS.items():
        present = {r[0] for r in ctx.con.execute(f"DESCRIBE silver.{table}").fetchall()}
        if not present.intersection(wanted):
            missing[table] = ["address/cep"] if table == "orders" else ["weight", "dimensions"]
    ids = _ids(ctx, "SELECT order_id FROM silver.orders") if missing else []
    return Finding(
        "D14", "missing_shipping_attributes", "orders", "info",
        "sem endereço, CEP, peso ou dimensões: explicar modalidades e prazos, sem cotar valor",
        ids, {"missing_columns": missing},
    )  # fmt: skip


RULES: tuple[Rule, ...] = (
    empty_categories,
    brand_consistency,
    specs_match_description,
    specs_format_variant,
    active_without_stock,
    discontinued_without_stock,
    discontinued_has_stock,
    unreleased_in_orders,
    order_total_mismatch,
    duplicate_email,
    multiple_promotions,
    multiple_active_promotions,
    active_promotions,
    promotions_without_validity,
    shipped_overdue,
    delivered_without_receipt_date,
    missing_shipping_attributes,
)
