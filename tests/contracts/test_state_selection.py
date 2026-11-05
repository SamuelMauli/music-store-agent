# ruff: noqa: PLC0415  # imports do extra postgres ficam preguiçosos (CI sem extras)
"""Composition root: ``Settings`` escolhe os adaptadores de estado."""

from __future__ import annotations

import sys
from collections.abc import Callable

import fakeredis
import pytest
import redis
from pydantic import SecretStr

from emporio.adapters.persistence.sqlite_tickets import SqliteTicketRepository
from emporio.bootstrap import build_shared_state
from emporio.guardrails.attempts import InMemoryAttemptStore, RedisAttemptStore
from emporio.guardrails.rate_limit import RedisSlidingWindowLimiter, SlidingWindowLimiter
from emporio.settings import Settings
from tests.contracts.conftest import DATABASE_URL, integration, needs_postgres


@pytest.fixture
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("state")
    return Settings(checkpoint_path=root / "ck.sqlite", tickets_path=root / "t.sqlite")


def _close(closers: list[Callable[[], object]]) -> None:
    for closer in closers:
        closer()


def test_default_is_sqlite_and_memory(settings: Settings) -> None:
    closers: list[Callable[[], object]] = []
    state = build_shared_state(settings, closers)
    assert isinstance(state.tickets, SqliteTicketRepository)
    assert isinstance(state.limiter, SlidingWindowLimiter)
    assert isinstance(state.attempts, InMemoryAttemptStore)
    assert len(closers) == 2
    _close(closers)


def test_redis_url_moves_limits_and_attempts_to_redis(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_from_url(url: str, **_: object) -> redis.Redis:
        seen.append(url)
        return fakeredis.FakeRedis(server=fakeredis.FakeServer())

    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(fake_from_url))
    closers: list[Callable[[], object]] = []
    state = build_shared_state(
        settings.model_copy(update={"redis_url": SecretStr("redis://cache:6379/0")}), closers
    )
    assert isinstance(state.limiter, RedisSlidingWindowLimiter)
    assert isinstance(state.ip_limiter, RedisSlidingWindowLimiter)
    assert isinstance(state.attempts, RedisAttemptStore)
    assert seen == ["redis://cache:6379/0"]
    assert len(closers) == 3  # duas conexões sqlite + o cliente redis
    _close(closers)


def test_missing_redis_extra_is_a_clear_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "redis", None)
    with pytest.raises(RuntimeError, match=r"emporio\[redis\]"):
        build_shared_state(
            settings.model_copy(update={"redis_url": SecretStr("redis://x:6379/0")}), []
        )


def test_postgres_requires_database_url(settings: Settings) -> None:
    with pytest.raises(ValueError, match="database_url"):
        build_shared_state(settings.model_copy(update={"state_backend": "postgres"}), [])


def test_missing_postgres_extra_is_a_clear_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", None)
    postgres = settings.model_copy(
        update={"state_backend": "postgres", "database_url": SecretStr("postgresql://x/y")}
    )
    with pytest.raises(RuntimeError, match=r"emporio\[postgres\]"):
        build_shared_state(postgres, [])


@integration
@needs_postgres
def test_postgres_backend_wires_saver_and_tickets(settings: Settings) -> None:
    from langgraph.checkpoint.postgres import PostgresSaver

    from emporio.adapters.persistence.postgres_tickets import PostgresTicketRepository

    assert DATABASE_URL
    closers: list[Callable[[], object]] = []
    state = build_shared_state(
        settings.model_copy(
            update={"state_backend": "postgres", "database_url": SecretStr(DATABASE_URL)}
        ),
        closers,
    )
    assert isinstance(state.checkpointer, PostgresSaver)
    assert isinstance(state.tickets, PostgresTicketRepository)
    _close(closers)
