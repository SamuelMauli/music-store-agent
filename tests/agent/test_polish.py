from __future__ import annotations

import json
from types import SimpleNamespace

from emporio.agent.nodes.finish import GREETING, make_handoff, respond
from emporio.guardrails.output import check_answer


def test_respond_drops_the_model_greeting_after_the_intro() -> None:
    out = respond({"answer": "Oi! Temos 3 violões.", "turn": 1})["answer"]
    assert out == GREETING + "Temos 3 violões."


def test_markdown_markers_are_stripped() -> None:
    verdict = check_answer(
        "**Takamine** `GD20`", grounded=frozenset(), complaint=False, needs_citation=False
    )
    assert verdict.text == "Takamine GD20"


def test_invented_ticket_protocol_is_flagged() -> None:
    kwargs = {"grounded": frozenset(), "complaint": False, "needs_citation": False}
    fake = check_answer("Seu protocolo é EMP‑2026‑01123.", **kwargs, sources="")  # type: ignore[arg-type]
    real = check_answer(
        "Protocolo EMP‑20260221‑0001.",
        **kwargs,  # type: ignore[arg-type]
        sources="EMP-20260221-0001",
    )
    assert fake.violations == ["protocolo_inventado"]
    assert real.ok


def test_handoff_reuses_ticket_opened_in_the_turn() -> None:
    class Registry:
        def call(self, *_a: object, **_k: object) -> None:
            raise AssertionError("não deve abrir segundo chamado")

    payload = json.dumps(
        {"status": "ok", "data": {"protocol": "EMP-20260221-0001", "sla": "24 horas úteis"}}
    )
    trace = {"name": "open_ticket", "status": "ok", "payload": payload}
    deps = SimpleNamespace(registry=Registry())
    out = make_handoff(deps)({"answer": "x", "tool_traces": [trace]})  # type: ignore[arg-type]
    assert "EMP-20260221-0001" in out["answer"]
    assert out["tool_traces"] == [trace]
