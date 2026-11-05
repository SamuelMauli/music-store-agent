# ruff: noqa: PLC0415  # imports do extra postgres ficam preguiçosos (CI sem extras)
"""Fábricas parametrizadas por backend: a mesma suíte de contrato roda em todas."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import fakeredis
import pytest
import redis

from emporio.adapters.persistence.memory_tickets import InMemoryTicketRepository
from emporio.adapters.persistence.sqlite_tickets import SqliteTicketRepository
from emporio.guardrails.attempts import AttemptStore, InMemoryAttemptStore, RedisAttemptStore
from emporio.guardrails.rate_limit import RedisSlidingWindowLimiter, SlidingWindowLimiter
from emporio.ports import RateLimiter, TicketRepository

DATABASE_URL = os.environ.get("EMPORIO_TEST_DATABASE_URL")
REDIS_URL = os.environ.get("EMPORIO_TEST_REDIS_URL")

needs_postgres = pytest.mark.skipif(not DATABASE_URL, reason="EMPORIO_TEST_DATABASE_URL ausente")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="EMPORIO_TEST_REDIS_URL ausente")
integration = pytest.mark.integration


class FakeTime:
    """Relógio manual compartilhado entre o limitador e o teste."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@dataclass(slots=True)
class Harness[T]:
    """Objeto sob teste mais o botão de "passou o tempo"."""

    subject: T
    advance: Callable[[float], None]


class PostgresSchema:
    """Cria um schema único, abre um pool com ``search_path`` nele e o derruba no fim."""

    def __enter__(self) -> Any:
        import psycopg

        from emporio.adapters.persistence.postgres_pool import open_pool

        assert DATABASE_URL
        self._name = f"t_{uuid.uuid4().hex[:12]}"
        with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
            admin.execute(f"CREATE SCHEMA {self._name}".encode())
        self._pool = open_pool(DATABASE_URL, options=f"-c search_path={self._name}")
        return self._pool

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        import psycopg

        self._pool.close()
        assert DATABASE_URL
        with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {self._name} CASCADE".encode())


@pytest.fixture
def postgres_pool() -> Iterator[Any]:
    """Pool isolado num schema descartável (só para testes Postgres)."""
    if not DATABASE_URL:
        pytest.skip("EMPORIO_TEST_DATABASE_URL ausente")
    with PostgresSchema() as pool:
        yield pool


TICKET_BACKENDS = [
    "memory",
    "sqlite",
    pytest.param("postgres", marks=[integration, needs_postgres]),
]


@pytest.fixture(params=TICKET_BACKENDS)
def ticket_repo(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[TicketRepository]:
    """Repositório de chamados limpo, de cada backend."""
    backend: str = request.param
    if backend == "memory":
        yield InMemoryTicketRepository()
    elif backend == "sqlite":
        repo = SqliteTicketRepository(tmp_path / "t.sqlite")
        yield repo
        repo.close()
    else:
        from emporio.adapters.persistence.postgres_tickets import PostgresTicketRepository

        with PostgresSchema() as pool:
            yield PostgresTicketRepository(pool)


def _redis_client(kind: str) -> redis.Redis:
    if kind == "fakeredis":
        return fakeredis.FakeRedis(server=fakeredis.FakeServer())
    assert REDIS_URL
    return redis.Redis.from_url(REDIS_URL)


REDIS_BACKENDS = [
    "memory",
    "fakeredis",
    pytest.param("redis", marks=[integration, needs_redis]),
]


@pytest.fixture(params=REDIS_BACKENDS)
def limiter_factory(
    request: pytest.FixtureRequest,
) -> Iterator[Callable[[int, float], Harness[RateLimiter]]]:
    """Cria limitadores ``(limite, janela)`` com relógio manual."""
    backend: str = request.param
    clients: list[redis.Redis] = []

    def build(limit: int, window: float) -> Harness[RateLimiter]:
        clock = FakeTime()

        def advance(seconds: float) -> None:
            clock.now += seconds

        if backend == "memory":
            return Harness(SlidingWindowLimiter(limit, window, monotonic=clock), advance)
        client = _redis_client(backend)
        clients.append(client)
        prefix = f"test:{uuid.uuid4().hex}:"
        limiter = RedisSlidingWindowLimiter(client, limit, window, prefix=prefix, clock=clock)
        return Harness(limiter, advance)

    yield build
    for client in clients:
        client.close()


@pytest.fixture(params=REDIS_BACKENDS)
def attempts_factory(
    request: pytest.FixtureRequest,
) -> Iterator[Callable[[float], Harness[AttemptStore]]]:
    """Cria stores de tentativas com TTL; no Redis o tempo passa de verdade."""
    backend: str = request.param
    clients: list[redis.Redis] = []

    def build(ttl: float) -> Harness[AttemptStore]:
        if backend == "memory":
            clock = FakeTime()

            def advance(seconds: float) -> None:
                clock.now += seconds

            return Harness(InMemoryAttemptStore(ttl, monotonic=clock), advance)
        client = _redis_client(backend)
        clients.append(client)
        store = RedisAttemptStore(client, ttl, prefix=f"test:{uuid.uuid4().hex}:")
        return Harness(store, time.sleep)

    yield build
    for client in clients:
        client.close()
