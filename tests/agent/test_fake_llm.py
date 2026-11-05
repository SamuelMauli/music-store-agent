from __future__ import annotations

import json

import duckdb
import pytest
from pydantic import BaseModel

from emporio.agent.intents import Intent
from emporio.agent.llm.fake import FakeLLM
from emporio.agent.llm.types import LLMError, Usage
from emporio.agent.prompts import SUMMARY_MARKER, render
from emporio.agent.tools import ToolContext, ToolRegistry
from tests.agent.helpers import run_turn


@pytest.fixture
def fake(con: duckdb.DuckDBPyConnection) -> FakeLLM:
    names = [r[0] for r in con.execute("select name from silver.products").fetchall()]
    return FakeLLM(names)


def intents(fake: FakeLLM, registry: ToolRegistry, text: str) -> list[Intent]:
    return list(dict.fromkeys(run_turn(fake, registry, text)[2].intents))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Quais violões vocês têm até R$ 1.000?", [Intent.CATALOG_SEARCH]),
        (
            "Quanto custa o Takamine GD20 e tem parcelamento?",
            [Intent.PRODUCT_DETAIL, Intent.PAYMENT_SIMULATION],
        ),
        ("Quais são as promoções ativas?", [Intent.ACTIVE_PROMOTIONS]),
        ("Vocês vendem cabo P10?", [Intent.OUT_OF_SCOPE_ACCESSORY]),
        ("Qual a capital da França?", [Intent.OUT_OF_SCOPE_GENERAL]),
        ("Quero falar com um atendente", [Intent.HUMAN_HANDOFF]),
        ("Onde fica a loja e vocês abrem hoje?", [Intent.STORE_INFO]),
        ("Quero devolver o pedido 7", [Intent.RETURN_EXCHANGE]),
    ],
)
def test_fake_router_classifies(
    fake: FakeLLM, registry: ToolRegistry, text: str, expected: list[Intent]
) -> None:
    assert intents(fake, registry, text) == expected


def test_fake_router_greeting_has_no_intents(fake: FakeLLM, registry: ToolRegistry) -> None:
    route = run_turn(fake, registry, "Oi, bom dia!")[2]
    assert route.intents == []
    assert not route.low_confidence


def test_fake_router_unclear_store_message_is_low_confidence(
    fake: FakeLLM, registry: ToolRegistry
) -> None:
    assert run_turn(fake, registry, "quero comprar um negócio aí")[2].low_confidence


def test_fake_router_multi_intent(fake: FakeLLM, registry: ToolRegistry) -> None:
    found = intents(fake, registry, "Quanto custa o GD20? E vocês abrem sábado?")
    assert found == [Intent.PRODUCT_DETAIL, Intent.STORE_INFO]


def test_gold_violoes_ate_1000(fake: FakeLLM, registry: ToolRegistry) -> None:
    text, tools, _ = run_turn(fake, registry, "Quais violões vocês têm até R$ 1.000?")
    assert tools == ["search_products"]
    assert "12" in text
    assert "R$ 429,90" in text


def test_gold_gd20_values_come_from_the_tool(fake: FakeLLM, registry: ToolRegistry) -> None:
    text, tools, _ = run_turn(fake, registry, "Quanto custa o Takamine GD20 no PIX e parcelado?")
    assert "get_product" in tools
    for value in ("R$ 2.199,00", "R$ 2.089,05", "12x de R$ 183,25"):
        assert value in text
    assert "Sem promoção ativa" in text


def test_gold_active_promotions(fake: FakeLLM, registry: ToolRegistry) -> None:
    text, tools, _ = run_turn(fake, registry, "Quais são as promoções ativas?")
    assert tools == ["list_active_promotions"]
    for name in ("Crafter HT-100", "Taylor 110e", "Ohana CK-20", "Kalani KAL-700T"):
        assert name in text


def test_payment_question_uses_tool_values(fake: FakeLLM, registry: ToolRegistry) -> None:
    text, tools, _ = run_turn(fake, registry, "Quanto fica o GD20 no PIX?")
    assert "simulate_payment" in tools or "get_product" in tools
    assert "R$ 2.089,05" in text


def test_reference_resolves_to_focus(fake: FakeLLM, registry: ToolRegistry) -> None:
    focus = [
        {"product_id": 83, "name": "Tagima Memphis AC-39", "price": "R$ 429,90"},
        {"product_id": 81, "name": "Yamaha C40", "price": "R$ 599,90"},
    ]
    text, tools, route = run_turn(fake, registry, "Quanto custa o mais barato?", focus=focus)
    assert route.entities.referenced_product_ids == [83]
    assert tools == ["get_product"]
    assert "Tagima Memphis" in text


def test_shipping_without_city_asks(fake: FakeLLM, registry: ToolRegistry) -> None:
    text, tools, _ = run_turn(fake, registry, "Qual o frete?")
    assert tools == []
    assert "cidade" in text


def test_return_flow_asks_channel_then_answers(fake: FakeLLM, registry_feb21: ToolRegistry) -> None:
    ctx = ToolContext(verified_customer_id=14)
    asked, _, _ = run_turn(fake, registry_feb21, "Quero devolver o pedido 7", ctx=ctx)
    assert "online ou na loja" in asked
    text, tools, _ = run_turn(
        fake, registry_feb21, "Quero devolver o pedido 7, comprei online", ctx=ctx
    )
    assert tools == ["check_return_eligibility"]
    assert "dia 4 de 7" in text
    assert "24/02/2026" in text


def test_complaint_opens_ticket(fake: FakeLLM, registry: ToolRegistry) -> None:
    text, tools, _ = run_turn(fake, registry, "Isso é um absurdo, vou reclamar do atraso")
    assert tools == ["open_ticket"]
    assert "EMP-" in text
    assert "sinto muito" in text


def test_structured_rejects_unknown_schema(fake: FakeLLM) -> None:
    class Other(BaseModel):
        required_field: int

    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "oi"}]
    with pytest.raises(LLMError):
        fake.structured(messages, Other)


def test_summary_is_produced_for_the_summary_prompt(fake: FakeLLM) -> None:
    prompt = render(
        "summary", previous="", messages=[{"role": "user", "content": "quero um violão"}]
    )
    assert prompt.startswith(SUMMARY_MARKER)
    response = fake.complete([{"role": "system", "content": prompt}], tier="small")
    assert "violão" in response.text


def test_usage_is_estimated_and_free(fake: FakeLLM, registry: ToolRegistry) -> None:
    route = run_turn(fake, registry, "Oi")[2]
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "oi"}]
    usage = fake.structured(messages, type(route))[1]
    assert isinstance(usage, Usage)
    assert usage.cost_usd == 0.0
    assert usage.calls == 1
    assert json.dumps(usage.to_dict())
