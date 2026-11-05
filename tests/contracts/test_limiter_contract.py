"""Contrato do limitador de taxa: memória, fakeredis e Redis real; mais concorrência."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest

from emporio.guardrails.rate_limit import RateLimitedError, RedisSlidingWindowLimiter
from emporio.ports import RateLimiter
from tests.contracts.conftest import FakeTime, Harness

Factory = Callable[[int, float], Harness[RateLimiter]]


def test_blocks_after_limit_and_reports_retry_after(limiter_factory: Factory) -> None:
    h = limiter_factory(2, 10.0)
    h.subject.check("a")
    h.advance(1)
    h.subject.check("a")
    with pytest.raises(RateLimitedError) as caught:
        h.subject.check("a")
    assert caught.value.retry_after == 10  # o 1º evento sai da janela em 9s (+1 de folga)


def test_keys_are_independent(limiter_factory: Factory) -> None:
    h = limiter_factory(1, 10.0)
    h.subject.check("a")
    h.subject.check("b")
    with pytest.raises(RateLimitedError):
        h.subject.check("a")


def test_window_slides(limiter_factory: Factory) -> None:
    h = limiter_factory(2, 10.0)
    h.subject.check("a")  # t=0
    h.advance(6)
    h.subject.check("a")  # t=6
    h.advance(5)  # t=11: o de t=0 saiu, o de t=6 ainda vale
    h.subject.check("a")
    with pytest.raises(RateLimitedError):
        h.subject.check("a")
    h.advance(11)
    h.subject.check("a")


def test_denied_events_do_not_extend_the_block(limiter_factory: Factory) -> None:
    h = limiter_factory(1, 10.0)
    h.subject.check("a")
    for _ in range(5):
        h.advance(1)
        with pytest.raises(RateLimitedError):
            h.subject.check("a")
    h.advance(5)  # t=10 desde o único evento aceito
    h.subject.check("a")


def test_redis_limiter_never_exceeds_limit_under_threads() -> None:
    client = fakeredis.FakeRedis(server=fakeredis.FakeServer())
    limit, threads, per_thread = 25, 8, 10
    limiter = RedisSlidingWindowLimiter(client, limit, 60.0, clock=FakeTime())

    def worker(_: int) -> int:
        accepted = 0
        for _call in range(per_thread):
            try:
                limiter.check("hot")
                accepted += 1
            except RateLimitedError:
                continue
        return accepted

    with ThreadPoolExecutor(threads) as pool:
        accepted = sum(pool.map(worker, range(threads)))
    assert 1 <= accepted <= limit


def test_redis_keys_hold_no_raw_identifier_and_expire() -> None:
    client = fakeredis.FakeRedis(server=fakeredis.FakeServer())
    RedisSlidingWindowLimiter(client, 3, 30.0, clock=FakeTime()).check("203.0.113.9")
    (name,) = client.keys("*")
    assert b"203.0.113.9" not in name
    assert 0 < client.pttl(name) <= 30_000
