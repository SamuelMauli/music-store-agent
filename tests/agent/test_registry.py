from __future__ import annotations

import json
import logging
import time
from typing import Any

import pytest
from pydantic import BaseModel

from emporio.agent.tools import ToolContext, ToolRegistry
from emporio.agent.tools.base import Tool
from emporio.agent.tools.results import Ok, ToolResult
from tests.agent.conftest import caller

EXPECTED = {
    "search_products",
    "get_product",
    "compare_products",
    "find_alternatives",
    "list_active_promotions",
    "simulate_payment",
    "verify_customer",
    "get_order_status",
    "check_return_eligibility",
    "estimate_shipping",
    "get_store_info",
    "search_policies",
    "open_ticket",
}


def test_registry_has_exactly_the_13_tools(registry: ToolRegistry) -> None:
    assert set(registry.names) == EXPECTED
    assert len(registry.names) == 13


def test_only_open_ticket_writes(registry: ToolRegistry) -> None:
    assert registry.write_tools() == ["open_ticket"]


def test_schemas_are_openai_function_specs(registry: ToolRegistry) -> None:
    for schema in registry.schemas():
        assert schema["type"] == "function"
        assert schema["function"]["description"].count(" ") > 10
        assert schema["function"]["parameters"]["type"] == "object"
    subset = registry.schemas(["get_product"])
    assert [s["function"]["name"] for s in subset] == ["get_product"]


def test_unknown_tool_and_bad_arguments_become_tool_error(registry: ToolRegistry) -> None:
    ask = caller(registry)
    assert json.loads(ask("drop_tables").payload)["status"] == "error"
    bad = json.loads(ask("get_product").payload)
    assert bad["status"] == "error"
    assert "name_or_id" in bad["message"]


class _Empty(BaseModel):
    pass


class _Boom(Tool[_Empty, _Empty]):
    name = "boom"
    description = "falha"
    input_model = _Empty

    def run(self, args: _Empty, ctx: ToolContext) -> ToolResult[_Empty]:
        raise RuntimeError("segredo interno: senha=123")


class _Slow(Tool[_Empty, _Empty]):
    name = "slow"
    description = "demora"
    input_model = _Empty

    def run(self, args: _Empty, ctx: ToolContext) -> ToolResult[_Empty]:
        time.sleep(0.5)
        return Ok(_Empty())


def test_exceptions_are_converted_without_leaking_internals() -> None:
    execution = ToolRegistry([_Boom()]).call("boom", {}, ToolContext())
    assert json.loads(execution.payload)["status"] == "error"
    assert "senha" not in execution.payload


def test_slow_tool_hits_the_timeout() -> None:
    execution = ToolRegistry([_Slow()], timeout=0.05).call("slow", {}, ToolContext())
    assert json.loads(execution.payload)["status"] == "error"
    assert "demorou" in execution.payload


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicado"):
        ToolRegistry([_Boom(), _Boom()])


def test_tool_logs_mask_pii(registry: ToolRegistry, caplog: pytest.LogCaptureFixture) -> None:
    args: dict[str, Any] = {
        "order_id": 7,
        "email": "leticia.rocha@jmail.com",
        "phone": "(67) 99789-0123",
    }
    with caplog.at_level(logging.INFO, logger="emporio.tools"):
        registry.call("verify_customer", args, ToolContext())
    assert "verify_customer" in caplog.text
    assert "leticia.rocha" not in caplog.text
    assert "99789-0123" not in caplog.text
