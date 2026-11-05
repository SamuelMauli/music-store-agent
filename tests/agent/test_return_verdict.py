from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pytest
from pydantic import BaseModel

from emporio.agent.intents import Intent, tools_for
from emporio.agent.llm.fake import FakeLLM
from emporio.agent.llm.types import LLMResponse, Message, ToolCall, Usage
from emporio.agent.nodes.facts import with_return_verdict
from emporio.agent.prompts import render
from emporio.agent.tools.orders import CheckReturnInput
from emporio.bootstrap import Container, build_container
from emporio.settings import Settings
from tests.agent.conftest import fake_settings

TURNS = ["Quero devolver o pedido 7", "leticia.rocha@jmail.com", "comprei online"]
VAGUE = "Conforme §4.1, você tem até 24/02/2026 para devolver o pedido 7."


def _trace(data: dict[str, Any], name: str = "check_return_eligibility") -> dict[str, Any]:
    return {"name": name, "status": "ok", "payload": json.dumps({"status": "ok", "data": data})}


ELIGIBLE = {
    "eligible": True,
    "rule": "§4.1",
    "window": "dia 4 de 7 dias corridos",
    "deadline": "24/02/2026",
}


def test_verdict_is_prepended_when_the_model_omits_it() -> None:
    out = with_return_verdict(VAGUE, [_trace(ELIGIBLE)])
    assert out.startswith("Pela regra §4.1, dá para devolver. Você está no dia 4 de 7 dias")
    assert out.endswith(VAGUE)


def test_complete_model_answer_is_untouched() -> None:
    said = "Dá para devolver sim, você está no dia 4 de 7 dias corridos (§4.1)."
    assert with_return_verdict(said, [_trace(ELIGIBLE)]) == said


def test_window_missing_is_enough_to_add_the_sentence() -> None:
    assert "dia 4 de 7" in with_return_verdict("Pode devolver, §4.1.", [_trace(ELIGIBLE)])


def test_ineligible_verdict_is_stated() -> None:
    data = {"eligible": False, "rule": "§4.1", "window": None, "deadline": None}
    out = with_return_verdict("Conforme §4.1, o prazo acabou.", [_trace(data)])
    assert out.startswith("Pela regra §4.1, esse caso não é elegível")
    assert with_return_verdict("Não é elegível (§4.1).", [_trace(data)]) == "Não é elegível (§4.1)."


def test_other_tools_and_failed_checks_do_not_trigger_it() -> None:
    assert with_return_verdict("oi", [_trace(ELIGIBLE, "get_product")]) == "oi"
    failed = {"name": "check_return_eligibility", "status": "ambiguous", "payload": "{}"}
    assert with_return_verdict("oi", [failed]) == "oi"
    assert with_return_verdict("oi", []) == "oi"


def test_return_intent_cannot_open_tickets() -> None:
    assert "open_ticket" not in tools_for([Intent.RETURN_EXCHANGE])
    assert "open_ticket" in tools_for([Intent.HUMAN_HANDOFF])
    assert "open_ticket" in tools_for([Intent.COMPLAINT])


class Sloppy:
    """Fake que tenta abrir chamado e depois responde sem veredito nem janela."""

    def __init__(self, inner: FakeLLM) -> None:
        self.inner = inner

    def structured[T: BaseModel](
        self, messages: list[Message], schema: type[T], **kw: Any
    ) -> tuple[T, Usage]:
        return self.inner.structured(messages, schema, **kw)

    def complete(self, messages: list[Message], **kw: Any) -> LLMResponse:
        response = self.inner.complete(messages, **kw)
        if response.tool_calls:
            ticket = ToolCall("t1", "open_ticket", {"kind": "human", "summary": "devolução"})
            return replace(response, tool_calls=(*response.tool_calls, ticket))
        if "Pela regra" in response.text:
            return replace(response, text=VAGUE)
        return response


@pytest.fixture
def container(ingested: Settings, con: duckdb.DuckDBPyConnection) -> Iterator[Container]:
    names = [r[0] for r in con.execute("select name from silver.products").fetchall()]
    when = datetime(2026, 2, 21, 10, tzinfo=ZoneInfo("America/Campo_Grande"))
    settings = fake_settings(ingested, reference_datetime=when)
    built = build_container(settings, llm=Sloppy(FakeLLM(names)))
    yield built
    built.close()


def test_return_flow_states_verdict_from_tool_and_opens_no_ticket(container: Container) -> None:
    results = [container.service.chat("rv1", text) for text in TURNS]
    final = results[-1]
    assert "dá para devolver" in final.answer
    assert "dia 4 de 7" in final.answer
    assert "§4.1" in final.answer
    assert "check_return_eligibility" in final.tools
    for result in results:
        assert "open_ticket" not in result.tools
        assert "EMP-" not in result.answer
        assert not result.handoff


def test_reason_defaults_to_regret_and_prompt_never_asks_for_it() -> None:
    assert CheckReturnInput(order_id=7).reason == "regret"
    text = render(
        "system",
        now_text="x",
        store_open=True,
        next_opening="x",
        customer_first_name="",
        order_id=None,
        summary="",
        focus_products=[],
        route_intents=[],
        route_entities="",
        scope_notes=[],
        feedback="",
        policy_context=[],
    )
    assert "nunca pergunte o motivo" in text.lower()
