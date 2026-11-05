from __future__ import annotations

import json

from emporio.agent.tools import ToolRegistry
from emporio.agent.tools.results import Ambiguous, NotFound, Ok, PolicyBlocked
from tests.agent.conftest import Caller


def test_quote_shipping_metro_by_cep(call: Caller) -> None:
    paid = call("quote_shipping", cep="79002-000", order_total=499)
    free = call("quote_shipping", cep="79002000", product_ids=[95])
    assert isinstance(paid.result, Ok)
    assert paid.result.value.fee == "R$ 35,00"
    assert paid.result.value.estimate is False
    assert paid.result.value.earliest == "26/03/2026"
    assert isinstance(free.result, Ok)
    assert free.result.value.fee == "grátis"


def test_quote_shipping_outside_is_a_marked_estimate(call: Caller) -> None:
    result = call("quote_shipping", cep="01310-100", weight_kg=2)
    assert isinstance(result.result, Ok)
    view = result.result.value
    assert view.estimate is True
    assert (view.uf, view.region) == ("SP", "sudeste")
    assert [o.mode for o in view.options] == [
        "PAC (Correios)",
        "SEDEX (Correios)",
        "Jadlog (.package)",
    ]
    assert view.options[0].price == "R$ 46,00"
    assert view.options[1].earliest == "27/03/2026"
    assert "ESTIMATIVA" in (view.note or "")
    assert "catálogo" in (view.weight_note or "")
    assert json.loads(result.payload)["data"]["estimate"] is True


def test_quote_shipping_asks_weight_when_missing(call: Caller) -> None:
    result = call("quote_shipping", cep="01310-100")
    assert isinstance(result.result, Ambiguous)
    assert "catálogo" in result.result.question


def test_quote_shipping_rejects_bad_input(call: Caller) -> None:
    for kwargs in (
        {"cep": "1234"},
        {"cep": "abcdefgh"},
        {"cep": "01310-100", "weight_kg": -1},
        {"cep": "01310-100", "weight_kg": 2, "surprise": 1},
    ):
        assert json.loads(call("quote_shipping", **kwargs).payload)["status"] == "error"
    assert isinstance(call("quote_shipping", cep="00100-000", weight_kg=1).result, NotFound)
    assert isinstance(call("quote_shipping", cep="01310-100", weight_kg=31).result, PolicyBlocked)


def test_quote_shipping_is_read_only_with_strict_schema(registry: ToolRegistry) -> None:
    assert not registry.get("quote_shipping").writes
    schema = registry.schemas(["quote_shipping"])[0]["function"]["parameters"]
    assert schema["additionalProperties"] is False
    assert "cep" in schema["required"]
