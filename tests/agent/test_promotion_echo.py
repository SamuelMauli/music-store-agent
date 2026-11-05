"""Promoção encerrada: o percentual citado pelo cliente só pode ser repetido para negá-lo."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import duckdb
import pytest
from pydantic import BaseModel

from emporio.agent.llm.fake import FakeLLM
from emporio.agent.llm.types import LLMResponse, Message, Usage
from emporio.agent.nodes.facts import with_promotion_status
from emporio.bootstrap import Container, build_container
from emporio.guardrails.output import check_answer, grounded_values
from emporio.settings import Settings
from tests.agent.conftest import Caller, fake_settings

QUESTION = "A Tagima Woodstock ainda está com 30% da Black Friday?"
GROUNDED = grounded_values(['{"price": "R$ 899,00"}'])


def check(answer: str, customer: str, *, ended: bool = True) -> list[str]:
    return check_answer(
        answer,
        grounded=GROUNDED,
        complaint=False,
        needs_citation=False,
        customer_text=customer,
        promotion_ended=ended,
    ).violations


def test_denial_of_the_percentage_asked_by_the_customer_is_accepted() -> None:
    assert check("A promoção de 30% da Black Friday foi encerrada.", QUESTION) == []
    assert check("Não, os 30% da Black Friday não estão mais valendo.", QUESTION) == []


@pytest.mark.parametrize(
    "answer",
    [
        "Sim, o violão continua com 30% da Black Friday.",
        "Está com 30% de desconto, vale até hoje.",
        "A promoção de 30% não foi encerrada.",
        "A promoção foi encerrada, mas você ainda leva 30% de desconto.",
    ],
)
def test_affirming_the_percentage_still_fails(answer: str) -> None:
    assert check(answer, QUESTION) == ["numero_sem_fonte"]


def test_not_a_vector_for_prices_or_other_percentages() -> None:
    assert check("O preço é R$ 10,00 e foi encerrado.", "o preço é R$ 10") == ["numero_sem_fonte"]
    assert check("A promoção de 45% foi encerrada.", QUESTION) == ["numero_sem_fonte"]


def test_needs_a_tool_saying_the_promotion_ended() -> None:
    assert check("A promoção de 30% foi encerrada.", QUESTION, ended=False) == ["numero_sem_fonte"]


def test_percentage_must_come_from_the_current_customer_message() -> None:
    assert check("A promoção de 30% foi encerrada.", "e o preço?") == ["numero_sem_fonte"]


def _trace(ended: list[str], promotion: str | None = None) -> dict[str, Any]:
    data = {"promotion": promotion, "ended_promotions": ended}
    return {"name": "get_product", "status": "ok", "payload": json.dumps({"data": data})}


def test_answer_is_made_explicit_when_the_tool_says_the_promotion_ended() -> None:
    out = with_promotion_status("O violão sai por R$ 899,00.", [_trace(["Black Friday"])])
    assert "A promoção Black Friday deste produto foi encerrada" in out
    said = "A Black Friday acabou, o violão sai por R$ 899,00."
    assert with_promotion_status(said, [_trace(["Black Friday"])]) == said
    assert with_promotion_status("ok", [_trace([])]) == "ok"
    assert with_promotion_status("ok", [_trace(["X"], promotion="Oferta")]) == "ok"
    wrong = "A promoção não foi encerrada."
    assert "foi encerrada e não está mais" in with_promotion_status(wrong, [_trace(["X"])])


def test_ended_promotion_note_goes_before_the_closing_question() -> None:
    answer = "O violão sai por R$ 899,00.\n\nPosso ajudar em mais alguma coisa?"
    out = with_promotion_status(answer, [_trace(["Black Friday"])])
    assert out == (
        "O violão sai por R$ 899,00. A promoção Black Friday deste produto foi encerrada "
        "e não está mais valendo.\n\nPosso ajudar em mais alguma coisa?"
    )
    long_question = "Quer que eu compare " + "este modelo com outros violões " * 4 + "?"
    tail = with_promotion_status(f"R$ 899,00.\n\n{long_question}", [_trace(["X"])])
    assert tail.endswith("não está mais valendo.")


def test_get_product_tells_about_ended_campaigns_without_their_percentage(call: Caller) -> None:
    payload = json.loads(call("get_product", name_or_id="91").payload)["data"]
    assert not payload.get("promotion")
    assert payload["ended_promotions"] == ["Black Friday"]
    assert "encerrada" in payload["promotion_note"]
    assert "30" not in json.dumps(payload)  # o percentual da campanha não vai ao modelo


def test_active_promotion_has_no_ended_note(call: Caller) -> None:
    payload = json.loads(call("get_product", name_or_id="90").payload)["data"]
    assert payload["promotion"]
    assert payload["ended_promotions"] == []


class Echo:
    """Fake que, na resposta final, ecoa o percentual do cliente como o modelo real fez."""

    def __init__(self, inner: FakeLLM, text: str) -> None:
        self.inner, self.text = inner, text

    def structured[T: BaseModel](
        self, messages: list[Message], schema: type[T], **kw: Any
    ) -> tuple[T, Usage]:
        return self.inner.structured(messages, schema, **kw)

    def complete(self, messages: list[Message], **kw: Any) -> LLMResponse:
        response = self.inner.complete(messages, **kw)
        return response if response.tool_calls else replace(response, text=self.text)


@pytest.fixture
def make(ingested: Settings, con: duckdb.DuckDBPyConnection) -> Iterator[Any]:
    built: list[Container] = []
    names = [r[0] for r in con.execute("select name from silver.products").fetchall()]

    def factory(text: str) -> Container:
        container = build_container(fake_settings(ingested), llm=Echo(FakeLLM(names), text))
        built.append(container)
        return container

    yield factory
    for container in built:
        container.close()


def test_e2e_denial_echoing_30_percent_is_delivered_not_handed_off(make: Any) -> None:
    container: Container = make(
        "Não, a Black Friday de 30% foi encerrada. O violão sai por R$ 899,00."
    )
    result = container.service.chat("p1", QUESTION)
    assert "get_product" in result.tools
    assert not result.handoff
    assert not result.blocked
    assert "30%" in result.answer
    assert "encerrada" in result.answer


def test_e2e_affirming_echo_is_still_rejected(make: Any) -> None:
    container: Container = make("Sim, ainda está com 30% da Black Friday por R$ 899,00.")
    result = container.service.chat("p2", QUESTION)
    assert result.blocked
    assert "30%" not in result.answer


def test_e2e_missing_notice_is_added_by_the_guard(make: Any) -> None:
    container: Container = make("O violão sai por R$ 899,00 hoje.")
    result = container.service.chat("p3", QUESTION)
    assert "foi encerrada" in result.answer
    assert not result.blocked
