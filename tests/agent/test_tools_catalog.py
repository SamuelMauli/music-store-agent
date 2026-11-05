from __future__ import annotations

import json

from emporio.agent.tools.results import Ambiguous, NotFound, Ok
from tests.agent.conftest import Caller


def data(execution_payload: str) -> dict[str, object]:
    return json.loads(execution_payload)["data"]  # type: ignore[no-any-return]


def test_golden_doze_violoes_ate_mil(call: Caller) -> None:
    cheapest = call("search_products", category="violões", max_price=1000)
    assert isinstance(cheapest.result, Ok)
    assert cheapest.result.value.total_matches == 12
    assert cheapest.result.value.items[0].name.startswith("Tagima Memphis AC-39")
    assert cheapest.result.value.items[0].price == "R$ 429,90"
    priciest = call("search_products", category="violão", max_price=1000, sort="price_desc")
    assert isinstance(priciest.result, Ok)
    assert priciest.result.value.items[0].name.startswith("Tagima TW-7")
    assert priciest.result.value.items[0].price == "R$ 999,00"
    names = " ".join(i.name for i in priciest.result.value.items)
    assert "GF-3D" not in names
    assert "SN-7C" not in names


def test_search_excludes_unavailable_unless_asked(call: Caller) -> None:
    default = call("search_products", category="violão", query="GF-3D")
    assert isinstance(default.result, Ok)
    assert default.result.value.total_matches == 0
    assert default.result.value.note
    wide = call("search_products", category="violão", query="GF-3D", include_unavailable=True)
    assert isinstance(wide.result, Ok)
    assert wide.result.value.total_matches == 1


def test_search_filters_strings_and_electronics(call: Caller) -> None:
    result = call("search_products", category="violão", strings="aço", electric=True, limit=5)
    assert isinstance(result.result, Ok)
    assert result.result.value.total_matches > 0
    assert len(result.result.value.items) <= 5


def test_search_unknown_category_is_not_found(call: Caller) -> None:
    assert isinstance(call("search_products", category="submarino").result, NotFound)


def test_search_empty_category_d1_has_no_items(call: Caller) -> None:
    result = call("search_products", category="saxofone")
    assert isinstance(result.result, Ok)
    assert result.result.value.total_matches == 0


def test_search_limit_is_capped_by_schema(call: Caller) -> None:
    assert json.loads(call("search_products", limit=50).payload)["status"] == "error"


def test_golden_takamine_gd20(call: Caller) -> None:
    result = call("get_product", name_or_id="takamine gd 20")
    assert isinstance(result.result, Ok)
    detail = result.result.value
    assert detail.price == "R$ 2.199,00"
    assert detail.stock_quantity == 5
    assert detail.promotion is None
    assert detail.pix_price == "R$ 2.089,05"
    assert detail.max_installments == "até 12x de R$ 183,25 sem juros"
    assert result.result.refs[0].product_id == 95


def test_get_product_by_id_and_promotion_has_no_pix(call: Caller) -> None:
    result = call("get_product", name_or_id="94")
    assert isinstance(result.result, Ok)
    detail = result.result.value
    assert detail.discount == "8%"
    assert detail.original_price == "R$ 5.999,00"
    assert detail.price == "R$ 5.519,08"
    assert detail.pix_price is None
    assert detail.pix_note


def test_get_product_does_not_expose_free_text_description(call: Caller) -> None:
    assert "description" not in call("get_product", name_or_id="95").payload


def test_get_product_ambiguous_and_not_found(call: Caller) -> None:
    assert isinstance(call("get_product", name_or_id="yamaha").result, Ambiguous)
    missing = call("get_product", name_or_id="zzzz inexistente")
    assert isinstance(missing.result, NotFound)
    assert isinstance(call("get_product", name_or_id="9999").result, NotFound)


def test_get_product_not_found_suggests_similar(call: Caller) -> None:
    result = call("get_product", name_or_id="takamine xyz999")
    assert isinstance(result.result, NotFound)
    assert result.result.similar


def test_compare_products(call: Caller) -> None:
    ok = call("compare_products", product_ids=[95, 94])
    assert isinstance(ok.result, Ok)
    assert ok.result.value.same_category
    mixed = call("compare_products", product_ids=[95, 121])
    assert isinstance(mixed.result, Ok)
    assert not mixed.result.value.same_category
    assert mixed.result.value.note
    assert isinstance(call("compare_products", product_ids=[95, 9999]).result, NotFound)
    assert json.loads(call("compare_products", product_ids=[95]).payload)["status"] == "error"


def test_find_alternatives_for_out_of_stock_product(call: Caller) -> None:
    result = call("find_alternatives", product_id=96)
    assert isinstance(result.result, Ok)
    items = result.result.value.items
    assert 1 <= len(items) <= 3
    assert all(i.product_id != 96 for i in items)
    assert all(i.availability in ("disponível", "últimas unidades") for i in items)
    assert isinstance(call("find_alternatives", product_id=9999).result, NotFound)


def test_find_alternatives_without_candidates(call: Caller) -> None:
    result = call("find_alternatives", product_id=94)  # Taylor 110e promocional, faixa estreita
    assert isinstance(result.result, Ok)
