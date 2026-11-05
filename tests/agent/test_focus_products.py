"""Produtos em foco: o produto detalhado vem antes dos itens de listas do mesmo turno."""

from __future__ import annotations

from pydantic import BaseModel

from emporio.agent.nodes.react import _products_of
from emporio.agent.tools.base import ToolExecution
from emporio.agent.tools.results import Ok, ProductRef


class _Out(BaseModel):
    ok: bool = True


def _exec(name: str, *refs: ProductRef) -> ToolExecution:
    result = Ok(_Out(), refs=refs)
    return ToolExecution(name=name, args={}, result=result, payload="{}", duration_ms=1)


def test_detailed_product_comes_first_even_if_listed_later() -> None:
    promos = _exec("list_active_promotions", ProductRef(10, "Promo A"), ProductRef(11, "Promo B"))
    detail = _exec("get_product", ProductRef(95, "Tagima Woodstock", "R$ 899,00"))
    focus = _products_of([promos, detail])
    assert [p["product_id"] for p in focus] == [95, 10, 11]


def test_focus_is_deduplicated_and_capped() -> None:
    many = _exec("search_products", *(ProductRef(i, f"P{i}") for i in range(8)))
    again = _exec("get_product", ProductRef(3, "P3"))
    focus = _products_of([many, again])
    assert focus[0]["product_id"] == 3
    assert len(focus) == 5
    assert len({p["product_id"] for p in focus}) == 5
