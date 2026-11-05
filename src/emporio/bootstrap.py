"""Composition root: liga adaptadores, tools, LLM e grafo a partir de ``Settings``."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import duckdb
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from emporio.adapters.persistence import (
    DuckDbCustomerRepository,
    DuckDbOrderRepository,
    DuckDbPolicySectionRepository,
    DuckDbProductRepository,
    DuckDbPromotionRepository,
)
from emporio.adapters.persistence.sqlite_tickets import SqliteTicketRepository
from emporio.adapters.retrieval import retriever_factory
from emporio.agent.graph import build_graph
from emporio.agent.llm import factory
from emporio.agent.llm.types import LLMProvider
from emporio.agent.nodes.deps import Deps
from emporio.agent.service import ChatService
from emporio.agent.tools import ToolRegistry, build_registry
from emporio.clock import FixedClock, SystemClock
from emporio.ports import Clock, TicketRepository
from emporio.settings import Settings


@dataclass(slots=True)
class Container:
    """Objetos de longa vida da aplicação."""

    service: ChatService
    registry: ToolRegistry
    connection: duckdb.DuckDBPyConnection
    closers: list[Callable[[], object]] = field(default_factory=list)

    def close(self) -> None:
        """Libera as conexões abertas (DuckDB e SQLite)."""
        for closer in self.closers:
            closer()
        self.connection.close()


def build_container(
    settings: Settings,
    *,
    llm: LLMProvider | None = None,
    clock: Clock | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    tickets: TicketRepository | None = None,
) -> Container:
    """Monta o serviço de conversa.

    Args:
        settings: Configuração.
        llm: Provedor já pronto (testes); default vem de ``settings``.
        clock: Relógio; default é o fixo de ``reference_datetime`` ou o do sistema.
        checkpointer: Memória por sessão; default é SQLite em ``checkpoint_path``.
        tickets: Repositório de chamados; default é SQLite em ``tickets_path``.

    Returns:
        Container com o ``ChatService`` pronto.
    """
    connection = duckdb.connect(str(settings.db_path), read_only=True)
    lock = threading.RLock()
    use_clock = clock or (
        SystemClock(settings.timezone)
        if settings.use_system_clock
        else FixedClock(settings.reference_datetime)
    )
    sections = DuckDbPolicySectionRepository(connection).all()
    retriever = retriever_factory.build(settings, sections)
    closers: list[Callable[[], object]] = []
    if tickets is None:
        sqlite_tickets = SqliteTicketRepository(settings.tickets_path)
        closers.append(sqlite_tickets.close)
        tickets = sqlite_tickets
    registry = build_registry(
        products=DuckDbProductRepository(connection),
        orders=DuckDbOrderRepository(connection),
        customers=DuckDbCustomerRepository(connection),
        promotions=DuckDbPromotionRepository(connection),
        retriever=retriever,
        tickets=tickets,
        clock=use_clock,
        metro_cities=settings.metro_cities,
        lock=lock,
    )
    if llm is None:
        names = [r[0] for r in connection.execute("select name from silver.products").fetchall()]
        llm = factory.build(settings, names)
    if checkpointer is None:
        if str(settings.checkpoint_path) != ":memory:":
            settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        memory = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
        closers.append(memory.close)
        checkpointer = SqliteSaver(memory)
    graph = build_graph(Deps(llm, registry, retriever, use_clock), checkpointer)
    return Container(ChatService(graph), registry, connection, closers)
