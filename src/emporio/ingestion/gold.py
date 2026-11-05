"""Camada gold: views somente leitura, única superfície visível ao agente."""

from __future__ import annotations

import duckdb

# A semântica de preço nasce aqui, em SQL, e é espelhada por Money/availability_of no
# domínio (o teste de paridade cobre os 65 produtos). Multiplicar por 0.01 mantém
# DECIMAL; dividir por 100 viraria DOUBLE e quebraria o centavo.
V_CATALOG = """
CREATE OR REPLACE VIEW gold.v_catalog AS
WITH promo AS (
  SELECT product_id,
         max(discount_percent) AS discount_percent,
         arg_max(description, discount_percent) AS promo_name
  FROM silver.promotions WHERE is_active GROUP BY product_id
)
SELECT p.product_id, p.name, p.name_norm, p.aliases, p.category_id, c.name AS category,
       p.description, p.specs, p.specs_kind, p.string_type, p.has_electronics,
       p.price_brl, p.stock_quantity, p.status,
       CASE
         WHEN p.status = 'discontinued' THEN 'discontinued'
         WHEN p.status = 'coming_soon'  THEN 'coming_soon'
         WHEN p.stock_quantity = 0      THEN 'out_of_stock'
         WHEN p.stock_quantity <= 2     THEN 'low_stock'
         ELSE 'available' END AS availability,
       promo.discount_percent, promo.promo_name,
       ROUND(p.price_brl * (1 - COALESCE(promo.discount_percent, 0) * 0.01), 2)
         AS effective_price,
       -- PIX não acumula com promoção (§6.2): sem preço PIX quando há promoção ativa
       CASE WHEN promo.product_id IS NULL THEN ROUND(p.price_brl * 0.95, 2) END AS pix_price
FROM silver.products p
JOIN silver.categories c USING (category_id)
LEFT JOIN promo USING (product_id)
"""

# Atraso não entra na view: depende da data de referência, que vem do Clock injetado.
V_ORDER_STATUS = """
CREATE OR REPLACE VIEW gold.v_order_status AS
WITH items AS (
  SELECT oi.order_id,
         count(*) AS item_lines,
         sum(oi.quantity) AS units,
         sum(oi.quantity * p.price_brl) AS items_total_brl
  FROM silver.order_items oi JOIN silver.products p USING (product_id)
  GROUP BY oi.order_id
)
SELECT o.order_id, o.customer_id, o.order_date, o.status, o.payment_method,
       o.total_brl, o.tracking_code, o.estimated_delivery, o.notes,
       COALESCE(i.item_lines, 0) AS item_lines,
       COALESCE(i.units, 0) AS units,
       i.items_total_brl,
       COALESCE(i.items_total_brl = o.total_brl, FALSE) AS total_matches_items
FROM silver.orders o LEFT JOIN items i USING (order_id)
"""

V_ACTIVE_PROMOTIONS = """
CREATE OR REPLACE VIEW gold.v_active_promotions AS
SELECT pr.promotion_id, pr.product_id, p.name AS product_name,
       pr.description AS promo_name, pr.discount_percent, p.price_brl,
       ROUND(p.price_brl * (1 - pr.discount_percent * 0.01), 2) AS promo_price
FROM silver.promotions pr JOIN silver.products p USING (product_id)
WHERE pr.is_active
"""

GOLD_VIEWS = ("v_catalog", "v_order_status", "v_active_promotions")


def build_gold(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Recria o schema gold a partir da silver.

    Args:
        con: Conexão com a silver pronta.

    Returns:
        Nomes das views criadas.
    """
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    for ddl in (V_CATALOG, V_ORDER_STATUS, V_ACTIVE_PROMOTIONS):
        con.execute(ddl)
    return list(GOLD_VIEWS)
