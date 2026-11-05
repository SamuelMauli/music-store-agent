"""Composition root: liga adaptadores, tools, LLM e grafo a partir de ``Settings``."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import duckdb
import httpx
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
from emporio.adapters.whatsapp import GraphApiSender
from emporio.agent.graph import build_graph
from emporio.agent.llm import factory
from emporio.agent.llm.types import LLMProvider
from emporio.agent.nodes.deps import Deps, VerifyPolicy
from emporio.agent.service import ChatService
from emporio.agent.tools import ToolRegistry, build_registry
from emporio.channels.whatsapp import WhatsAppChannel
from emporio.clock import FixedClock, SystemClock
from emporio.guardrails.attempts import AttemptStore, InMemoryAttemptStore, RedisAttemptStore
from emporio.guardrails.rate_limit import RedisSlidingWindowLimiter, SlidingWindowLimiter
from emporio.observability import Telemetry, build_telemetry
from emporio.observability.tracing_llm import TracingLLM
from emporio.ports import Clock, HandoffStore, MessageDeduplicator, RateLimiter, TicketRepository
from emporio.settings import Settings, secret_value

MISSING_POSTGRES = "state_backend=postgres exige o extra opcional: instale `emporio[postgres]`"
MISSING_REDIS = "redis_url exige o extra opcional: instale `emporio[redis]`"


@dataclass(slots=True)
class SharedState:
    """Estado escolhido por ``Settings``: checkpointer, chamados, limites e tentativas."""

    checkpointer: BaseCheckpointSaver[Any]
    tickets: TicketRepository
    limiter: RateLimiter
    ip_limiter: RateLimiter
    attempts: AttemptStore


def _postgres_state(
    settings: Settings, closers: list[Callable[[], object]]
) -> tuple[BaseCheckpointSaver[Any], TicketRepository]:
    """Checkpointer e chamados em Postgres, sobre um pool único (import preguiçoso)."""
    if settings.database_url is None:
        raise ValueError("state_backend=postgres exige database_url (EMPORIO_DATABASE_URL)")
    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415

        from emporio.adapters.persistence.postgres_pool import open_pool  # noqa: PLC0415
        from emporio.adapters.persistence.postgres_tickets import (  # noqa: PLC0415
            PostgresTicketRepository,
        )
    except ImportError as exc:
        raise RuntimeError(MISSING_POSTGRES) from exc
    pool = open_pool(settings.database_url.get_secret_value())
    closers.append(pool.close)
    saver = PostgresSaver(pool)
    saver.setup()
    return saver, PostgresTicketRepository(pool)


def _sqlite_state(
    settings: Settings, closers: list[Callable[[], object]]
) -> tuple[BaseCheckpointSaver[Any], TicketRepository]:
    """Checkpointer e chamados em SQLite (processo único)."""
    if str(settings.checkpoint_path) != ":memory:":
        settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    memory = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
    closers.append(memory.close)
    sqlite_tickets = SqliteTicketRepository(settings.tickets_path)
    closers.append(sqlite_tickets.close)
    return SqliteSaver(memory), sqlite_tickets


def _guard_state(
    settings: Settings, closers: list[Callable[[], object]]
) -> tuple[RateLimiter, RateLimiter, AttemptStore]:
    """Limites e tentativas: Redis se ``redis_url`` existir, senão memória do processo."""
    per_session, per_ip = settings.rate_limit_per_minute, settings.rate_limit_per_ip_per_minute
    if settings.redis_url is None:
        return (
            SlidingWindowLimiter(per_session, 60.0),
            SlidingWindowLimiter(per_ip, 60.0),
            InMemoryAttemptStore(settings.verify_lockout_seconds),
        )
    try:
        from redis import Redis  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(MISSING_REDIS) from exc
    client: Redis = Redis.from_url(settings.redis_url.get_secret_value())
    closers.append(client.close)
    return (
        RedisSlidingWindowLimiter(client, per_session, 60.0, prefix="emporio:rl:session:"),
        RedisSlidingWindowLimiter(client, per_ip, 60.0, prefix="emporio:rl:ip:"),
        RedisAttemptStore(client, settings.verify_lockout_seconds),
    )


def build_shared_state(
    settings: Settings,
    closers: list[Callable[[], object]],
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    tickets: TicketRepository | None = None,
) -> SharedState:
    """Escolhe os adaptadores de estado conforme ``state_backend`` e ``redis_url``.

    Args:
        settings: Configuração.
        closers: Lista que recebe o fechamento de cada recurso aberto (pool, conexões).
        checkpointer: Substitui o checkpointer escolhido (testes).
        tickets: Substitui o repositório de chamados escolhido (testes).

    Returns:
        Estado pronto para o grafo e o ``ChatService``.
    """
    if settings.state_backend == "postgres":
        if checkpointer is None or tickets is None:
            default_saver, default_tickets = _postgres_state(settings, closers)
            checkpointer, tickets = checkpointer or default_saver, tickets or default_tickets
    elif checkpointer is None or tickets is None:
        default_saver, default_tickets = _sqlite_state(settings, closers)
        checkpointer, tickets = checkpointer or default_saver, tickets or default_tickets
    limiter, ip_limiter, attempts = _guard_state(settings, closers)
    return SharedState(checkpointer, tickets, limiter, ip_limiter, attempts)


@dataclass(slots=True)
class Container:
    """Objetos de longa vida da aplicação."""

    service: ChatService
    telemetry: Telemetry
    registry: ToolRegistry
    connection: duckdb.DuckDBPyConnection
    closers: list[Callable[[], object]] = field(default_factory=list)

    def close(self) -> None:
        """Libera as conexões abertas (DuckDB e SQLite) e descarrega os spans."""
        for closer in self.closers:
            closer()
        self.telemetry.shutdown()
        self.connection.close()


def build_container(
    settings: Settings,
    *,
    llm: LLMProvider | None = None,
    clock: Clock | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    tickets: TicketRepository | None = None,
    telemetry: Telemetry | None = None,
    wall: Callable[[], float] | None = None,
) -> Container:
    """Monta o serviço de conversa.

    Args:
        settings: Configuração.
        llm: Provedor já pronto (testes); default vem de ``settings``.
        clock: Relógio; default é o fixo de ``reference_datetime`` ou o do sistema.
        checkpointer: Memória por sessão; default segue ``state_backend`` (SQLite ou Postgres).
        tickets: Repositório de chamados; default segue ``state_backend``.
        wall: Relógio de parede (epoch) da validade da verificação; default ``time.time``.
        telemetry: Tracing; default vem de ``settings.otel_exporter`` (``none`` = no-op).

    Returns:
        Container com o ``ChatService`` pronto.
    """
    telemetry = telemetry or build_telemetry(settings)
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
    state = build_shared_state(settings, closers, checkpointer=checkpointer, tickets=tickets)
    registry = build_registry(
        products=DuckDbProductRepository(connection),
        orders=DuckDbOrderRepository(connection),
        customers=DuckDbCustomerRepository(connection),
        promotions=DuckDbPromotionRepository(connection),
        retriever=retriever,
        tickets=state.tickets,
        clock=use_clock,
        metro_cities=settings.metro_cities,
        lock=lock,
    )
    if llm is None:
        names = [r[0] for r in connection.execute("select name from silver.products").fetchall()]
        llm = factory.build(settings, names)
    if telemetry.enabled:
        llm = TracingLLM(llm, telemetry, settings)
        registry.instrument(telemetry)
    deps = Deps(
        llm,
        registry,
        retriever,
        use_clock,
        attempts=state.attempts,
        wall=wall or time.time,
        verify_policy=VerifyPolicy(
            ttl_seconds=settings.verification_ttl_seconds,
            max_per_target=settings.verify_max_attempts_per_target,
            max_per_ip=settings.verify_max_attempts_per_ip,
        ),
    )
    graph = build_graph(deps, state.checkpointer, telemetry)
    return Container(
        ChatService(
            graph,
            state.limiter,
            state.ip_limiter,
            telemetry,
        ),
        telemetry,
        registry,
        connection,
        closers,
    )


def build_whatsapp_channel(
    settings: Settings,
    service: ChatService,
    closers: list[Callable[[], object]],
    *,
    transport: httpx.BaseTransport | None = None,
    now: Callable[[], float] = time.time,
) -> WhatsAppChannel:
    """Monta o canal WhatsApp (chame só com ``settings.whatsapp_enabled``).

    O handoff e a deduplicação vão no Redis se ``redis_url`` existir (valem entre réplicas),
    senão em SQLite. Os limites e o relógio ficam injetáveis para os testes.

    Args:
        settings: Configuração, com as variáveis do canal preenchidas.
        service: Serviço de conversa do agente.
        closers: Recebe o fechamento de cada recurso aberto.
        transport: Transporte httpx da Graph API (testes injetam um falso, sem rede).
        now: Relógio de parede (epoch).

    Returns:
        O canal pronto.
    """
    store: HandoffStore
    dedup: MessageDeduplicator
    limiter: RateLimiter
    per_minute = settings.whatsapp_rate_limit_per_minute
    if settings.redis_url is None:
        from emporio.adapters.persistence.sqlite_handoff import SqliteHandoffStore  # noqa: PLC0415

        sqlite_store = SqliteHandoffStore(settings.handoff_path)
        closers.append(sqlite_store.close)
        store = dedup = sqlite_store
        limiter = SlidingWindowLimiter(per_minute, 60.0)
    else:
        try:
            from redis import Redis  # noqa: PLC0415

            from emporio.adapters.persistence.redis_handoff import (  # noqa: PLC0415
                RedisHandoffStore,
            )
        except ImportError as exc:
            raise RuntimeError(MISSING_REDIS) from exc
        client: Redis = Redis.from_url(settings.redis_url.get_secret_value())
        closers.append(client.close)
        redis_store = RedisHandoffStore(client)
        store = dedup = redis_store
        limiter = RedisSlidingWindowLimiter(client, per_minute, 60.0, prefix="emporio:rl:wa:")
    sender = GraphApiSender(
        secret_value(settings.whatsapp_access_token),
        settings.whatsapp_phone_number_id or "",
        api_version=settings.whatsapp_api_version,
        transport=transport,
    )
    closers.append(sender.close)
    channel = WhatsAppChannel(
        service,
        sender,
        store,
        dedup,
        limiter,
        pepper=secret_value(settings.whatsapp_hash_pepper),
        max_message_chars=settings.max_message_chars,
        handoff_ttl_seconds=settings.whatsapp_handoff_ttl_seconds,
        workers=settings.whatsapp_workers,
        queue_size=settings.whatsapp_queue_size,
        now=now,
    )
    closers.insert(0, channel.close)  # 1º: o pool para antes de fechar store e sender
    return channel
