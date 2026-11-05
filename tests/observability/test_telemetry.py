from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from rich.console import Console

from emporio.agent.llm.types import LLMError, LLMResponse, ToolCall, Usage
from emporio.bootstrap import Container, build_container
from emporio.interfaces.cli import debug_table
from emporio.observability import NOOP, UsageAccumulator, build_telemetry
from emporio.observability.tracing_llm import TracingLLM
from emporio.settings import ModelPrice, Settings
from tests.agent.conftest import fake_settings


class Memory(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: object) -> SpanExportResult:
        self.spans.extend(spans)  # type: ignore[call-overload]
        return SpanExportResult.SUCCESS


@pytest.fixture
def traced(ingested: Settings) -> Iterator[tuple[Container, Memory]]:
    memory = Memory()
    settings = fake_settings(ingested)
    container = build_container(settings, telemetry=build_telemetry(settings, memory))
    yield container, memory
    container.close()


def by_name(memory: Memory, prefix: str) -> list[ReadableSpan]:
    return [s for s in memory.spans if s.name.startswith(prefix)]


def test_turn_emits_genai_spans_for_nodes_tools_and_llm(
    traced: tuple[Container, Memory],
) -> None:
    container, memory = traced
    container.service.chat("t1", "Quais violões vocês têm até R$ 1.000?")
    turn = by_name(memory, "invoke_agent")[0]
    attrs = turn.attributes or {}
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.conversation.id"] == "t1"
    assert attrs["prompt_hash"]
    assert attrs["emporio.tools"] == "search_products"
    assert int(str(attrs["gen_ai.usage.input_tokens"])) > 0
    nodes = {s.name for s in by_name(memory, "node ")}
    assert {"node router", "node agent", "node respond"} <= nodes
    tool = by_name(memory, "execute_tool")[0]
    assert (tool.attributes or {})["gen_ai.tool.name"] == "search_products"
    chat = by_name(memory, "chat ")
    assert chat
    assert all((s.attributes or {})["gen_ai.request.model"] == "fake" for s in chat)
    assert all("gen_ai.usage.output_tokens" in (s.attributes or {}) for s in chat)
    # hierarquia: tudo pertence ao mesmo trace do turno
    assert {s.context.trace_id for s in memory.spans} == {turn.context.trace_id}  # type: ignore[union-attr]


def test_conversation_cost_accumulates_across_turns(traced: tuple[Container, Memory]) -> None:
    container, _ = traced
    first = container.service.chat("t2", "Quais são as promoções ativas?")
    second = container.service.chat("t2", "E o horário de funcionamento?")
    assert second.conversation_usage.calls == first.usage.calls + second.usage.calls
    assert second.conversation_usage.input_tokens == (
        first.usage.input_tokens + second.usage.input_tokens
    )
    container.service.reset("t2")
    third = container.service.chat("t2", "Quais são as promoções ativas?")
    assert third.conversation_usage == third.usage


def test_noop_by_default(ingested: Settings) -> None:
    container = build_container(fake_settings(ingested))
    try:
        assert container.telemetry is NOOP
        assert not container.telemetry.enabled
        result = container.service.chat("t3", "Quais são as promoções ativas?")
        assert result.conversation_usage == result.usage
    finally:
        container.close()


def test_jsonl_exporter_writes_one_span_per_line(ingested: Settings, tmp_path: Path) -> None:
    path = tmp_path / "t" / "traces.jsonl"
    settings = fake_settings(ingested, otel_exporter="jsonl", otel_jsonl_path=path)
    container = build_container(settings)
    container.service.chat("t4", "Quais são as promoções ativas?")
    container.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(r["name"].startswith("invoke_agent") for r in rows)
    assert all(r["context"]["trace_id"] for r in rows)


def test_console_exporter_is_built(ingested: Settings) -> None:
    telemetry = build_telemetry(fake_settings(ingested, otel_exporter="console"))
    assert telemetry.enabled
    telemetry.shutdown()


def test_otlp_without_package_fails_loudly(ingested: Settings) -> None:
    with pytest.raises(ValueError, match="otlp"):
        build_telemetry(fake_settings(ingested, otel_exporter="otlp"))


class Boom:
    def complete(self, *_: object, **__: object) -> LLMResponse:
        raise LLMError("x")

    def structured(self, *_: object, **__: object) -> object:
        raise LLMError("x")


def test_llm_error_marks_the_span(ingested: Settings) -> None:
    memory = Memory()
    settings = fake_settings(ingested)
    llm = TracingLLM(Boom(), build_telemetry(settings, memory), settings)  # type: ignore[arg-type]
    with pytest.raises(LLMError):
        llm.complete([{"role": "user", "content": "oi"}])
    with pytest.raises(LLMError):
        llm.structured([{"role": "user", "content": "oi"}], ToolCall)  # type: ignore[type-var]
    assert [(s.attributes or {})["error.type"] for s in memory.spans] == ["LLMError"] * 2
    assert all(s.status.status_code.name == "ERROR" for s in memory.spans)


def test_usage_accumulator_is_per_session() -> None:
    acc = UsageAccumulator()
    acc.add("a", Usage(10, 5, 0.1, 1))
    total = acc.add("a", Usage(1, 1, 0.2, 1))
    assert total == Usage(11, 6, pytest.approx(0.3), 2)  # type: ignore[arg-type]
    assert acc.total("b") == Usage()
    acc.reset("a")
    assert acc.total("a") == Usage()


def test_price_table_overrides_tier_price() -> None:
    s = Settings(llm_price_table={"m-x": ModelPrice(input=1.0, output=2.0)})
    assert s.price_for("m-x", "small") == (1.0, 2.0)
    assert s.price_for("other", "small") == (s.llm_price_small_input, s.llm_price_small_output)
    assert s.price_for("other", "medium") == (s.llm_price_medium_input, s.llm_price_medium_output)


def test_debug_panel_shows_conversation_cost(traced: tuple[Container, Memory]) -> None:
    container, _ = traced
    result = container.service.chat("t5", "Quais são as promoções ativas?")
    console = Console(record=True, width=100)
    console.print(debug_table(result))
    assert "conversa" in console.export_text()
