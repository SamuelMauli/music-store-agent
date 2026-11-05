from __future__ import annotations

from emporio.agent.nodes.finish import GREETING, respond
from emporio.guardrails.output import check_answer


def test_respond_drops_the_model_greeting_after_the_intro() -> None:
    out = respond({"answer": "Oi! Temos 3 violões.", "turn": 1})["answer"]
    assert out == GREETING + "Temos 3 violões."


def test_markdown_markers_are_stripped() -> None:
    verdict = check_answer(
        "**Takamine** `GD20`", grounded=frozenset(), complaint=False, needs_citation=False
    )
    assert verdict.text == "Takamine GD20"
