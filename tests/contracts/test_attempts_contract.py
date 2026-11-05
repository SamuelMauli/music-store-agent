"""Contrato de ``AttemptStore``: mesmas regras de contagem e lockout em todos os backends."""

from __future__ import annotations

from collections.abc import Callable

from emporio.guardrails.attempts import AttemptStore
from tests.contracts.conftest import Harness

Factory = Callable[[float], Harness[AttemptStore]]


def test_counts_per_key(attempts_factory: Factory) -> None:
    store = attempts_factory(30).subject
    assert store.count("a") == 0
    assert store.register("a") == 1
    assert store.register("a") == 2
    assert store.register("b") == 1
    assert (store.count("a"), store.count("b")) == (2, 1)


def test_ttl_expires_the_counter(attempts_factory: Factory) -> None:
    h = attempts_factory(1)
    h.subject.register("a")
    h.subject.register("a")
    h.advance(1.2)
    assert h.subject.count("a") == 0
    assert h.subject.register("a") == 1  # recomeça do zero


def test_each_failure_renews_the_ttl(attempts_factory: Factory) -> None:
    h = attempts_factory(1.5)
    h.subject.register("a")
    h.advance(1.0)
    assert h.subject.register("a") == 2  # renova: expira 1,5s depois desta
    h.advance(1.0)
    assert h.subject.count("a") == 2  # sem a renovação a 1ª falha já teria vencido
    h.advance(1.0)
    assert h.subject.count("a") == 0
