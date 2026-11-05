"""Composition root: liga adaptadores, tools, LLM e grafo a partir de ``Settings``."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import duckdb
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from emporio.adapters.persistence import (
    DuckDbCustomerRepository,
    DuckDbOrderRepository,
    DuckDbPolicySectionRepository,
    DuckDbProductRepository,
    DuckDbPromotionRepository,
)
from emporio.adapters.persistence.memory_tickets import InMemoryTicketRepository
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

    def close(self) -> None:
        """Libera a conexão com o DuckDB."""
        self.connection.close()


def build_container(
    settings: Settings,
    *,
    llm: LLMProvider | None = None,
    clock: Clock | None = None,
    checkpointer: BaseCheckpointSaver | None = None,  # type: ignore[type-arg]
    tickets: TicketRepository | None = None,
) -> Container:
    """Monta o serviço de conversa.

    Args:
        settings: Configuração.
        llm: Provedor já pronto (testes); default vem de ``settings``.
        clock: Relógio; default é o fixo de ``reference_datetime`` ou o do sistema.
        checkpointer: Memória por sessão; default é em processo.
        tickets: Repositório de chamados; default é em memória.

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
    repo = tickets or InMemoryTicketRepository()
    registry = build_registry(
        products=DuckDbProductRepository(connection),
        orders=DuckDbOrderRepository(connection),
        customers=DuckDbCustomerRepository(connection),
        promotions=DuckDbPromotionRepository(connection),
        retriever=retriever,
        tickets=repo,
        clock=use_clock,
        metro_cities=settings.metro_cities,
        lock=lock,
    )
    if llm is None:
        names = [r[0] for r in connection.execute("select name from silver.products").fetchall()]
        llm = factory.build(settings, names)
    graph = build_graph(Deps(llm, registry, retriever, use_clock), checkpointer or InMemorySaver())
    return Container(ChatService(graph), registry, connection)
