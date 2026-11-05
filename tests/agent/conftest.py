from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pytest

from emporio.adapters.persistence import (
    DuckDbCustomerRepository,
    DuckDbOrderRepository,
    DuckDbPolicySectionRepository,
    DuckDbProductRepository,
    DuckDbPromotionRepository,
)
from emporio.adapters.persistence.memory_tickets import InMemoryTicketRepository
from emporio.adapters.retrieval.long_context import LongContextRetriever
from emporio.agent.tools import ToolContext, ToolExecution, ToolRegistry, build_registry
from emporio.clock import FixedClock
from emporio.ports import Clock
from emporio.settings import Settings

TZ = ZoneInfo("America/Campo_Grande")
ORDER_7_DAY = datetime(2026, 2, 21, 10, 0, tzinfo=TZ)  # cenário Gherkin de arrependimento

Caller = Callable[..., ToolExecution]


def fake_settings(base: Settings, **extra: Any) -> Settings:
    """Settings com LLM fake e SQLite em memória (nada vai ao disco)."""
    fields: dict[str, Any] = {
        "llm_provider": "fake",
        "checkpoint_path": Path(":memory:"),
        "tickets_path": Path(":memory:"),
        **extra,
    }
    return base.model_copy(update=fields)


@pytest.fixture
def tickets() -> InMemoryTicketRepository:
    return InMemoryTicketRepository()


def make_registry(
    con: duckdb.DuckDBPyConnection, clock: Clock, tickets: InMemoryTicketRepository
) -> ToolRegistry:
    return build_registry(
        products=DuckDbProductRepository(con),
        orders=DuckDbOrderRepository(con),
        customers=DuckDbCustomerRepository(con),
        promotions=DuckDbPromotionRepository(con),
        retriever=LongContextRetriever(DuckDbPolicySectionRepository(con).all()),
        tickets=tickets,
        clock=clock,
        metro_cities=["Campo Grande"],
    )


@pytest.fixture
def registry(
    con: duckdb.DuckDBPyConnection, reference_clock: FixedClock, tickets: InMemoryTicketRepository
) -> ToolRegistry:
    return make_registry(con, reference_clock, tickets)


@pytest.fixture
def registry_feb21(
    con: duckdb.DuckDBPyConnection, tickets: InMemoryTicketRepository
) -> ToolRegistry:
    return make_registry(con, FixedClock(ORDER_7_DAY), tickets)


def caller(registry: ToolRegistry, **ctx_fields: Any) -> Caller:
    ctx = ToolContext(**ctx_fields)

    def call(name: str, **args: Any) -> ToolExecution:
        return registry.call(name, args, ctx)

    return call


@pytest.fixture
def call(registry: ToolRegistry) -> Caller:
    return caller(registry)
