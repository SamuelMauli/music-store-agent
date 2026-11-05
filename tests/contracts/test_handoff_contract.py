"""Contrato do estado de handoff e da deduplicação: SQLite e fakeredis (mesma suíte)."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from emporio.adapters.persistence.redis_handoff import RedisHandoffStore
from emporio.adapters.persistence.sqlite_handoff import DEDUP_TTL_SECONDS, SqliteHandoffStore
from emporio.domain.handoff import ClaimResult, HandoffState

NOW = 1_800_000_000.0  # epoch realista: o Redis expira por EXAT absoluto
TTL = 100.0


@pytest.fixture(params=["sqlite", "fakeredis"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    if request.param == "sqlite":
        sqlite_store = SqliteHandoffStore(tmp_path / "h.sqlite")
        yield sqlite_store
        sqlite_store.close()
    else:
        client = fakeredis.FakeRedis(server=fakeredis.FakeServer())
        yield RedisHandoffStore(client)
        client.close()


def test_enter_is_idempotent_and_marks_human(store: Any) -> None:
    assert not store.is_human("c1", NOW)
    assert store.enter("c1", "5567999990001", NOW, NOW + TTL)
    assert not store.enter("c1", "5567999990001", NOW + 1, NOW + 1 + TTL)
    assert store.is_human("c1", NOW + 1)
    assert not store.is_human("outra", NOW + 1)


def test_history_keeps_order_and_ignores_unknown_conversation(store: Any) -> None:
    store.append("fantasma", "in", "oi", NOW)
    assert store.get("fantasma", NOW) is None
    store.enter("c1", "5567999990001", NOW, NOW + TTL)
    store.append("c1", "in", "primeira", NOW + 1)
    store.append("c1", "out", "segunda", NOW + 2)
    conversation = store.get("c1", NOW + 3)
    assert conversation is not None
    assert [(m.direction, m.text) for m in conversation.messages] == [
        ("in", "primeira"),
        ("out", "segunda"),
    ]
    assert conversation.state is HandoffState.WAITING
    assert conversation.phone == "5567999990001"


def test_expiry_returns_conversation_to_the_bot_and_drops_history(store: Any) -> None:
    store.enter("c1", "5567999990001", NOW, NOW + TTL)
    store.append("c1", "in", "oi", NOW)
    assert store.is_human("c1", NOW + TTL - 1)
    assert not store.is_human("c1", NOW + TTL)
    assert store.get("c1", NOW + TTL) is None
    assert store.list_active(NOW + TTL) == []
    assert store.enter("c1", "5567999990001", NOW + TTL, NOW + 2 * TTL)  # nova entrada, limpa
    fresh = store.get("c1", NOW + TTL)
    assert fresh is not None
    assert fresh.messages == ()


def test_claim_rules(store: Any) -> None:
    assert store.claim("c1", "ana", NOW, NOW + TTL) is ClaimResult.NOT_FOUND
    store.enter("c1", "5567999990001", NOW, NOW + TTL)
    assert store.claim("c1", "ana", NOW + 1, NOW + 1 + TTL) is ClaimResult.OK
    assert store.claim("c1", "bia", NOW + 2, NOW + 2 + TTL) is ClaimResult.TAKEN
    assert store.claim("c1", "ana", NOW + 3, NOW + 3 + TTL) is ClaimResult.OK  # reassumir é ok
    conversation = store.get("c1", NOW + 4)
    assert conversation is not None
    assert conversation.state is HandoffState.ASSIGNED
    assert conversation.operator == "ana"
    assert conversation.expires_at == NOW + 3 + TTL


def test_extend_only_for_the_owner(store: Any) -> None:
    store.enter("c1", "5567999990001", NOW, NOW + TTL)
    assert not store.extend("c1", "ana", NOW + 500, NOW)  # ninguém assumiu
    store.claim("c1", "ana", NOW, NOW + TTL)
    assert not store.extend("c1", "bia", NOW + 500, NOW)
    assert store.extend("c1", "ana", NOW + 500, NOW)
    assert store.is_human("c1", NOW + 400)


def test_release_removes_everything_and_reports_what_existed(store: Any) -> None:
    assert store.release("c1", "ana", NOW) == (ClaimResult.NOT_FOUND, None)
    store.enter("c1", "5567999990001", NOW, NOW + TTL)
    store.append("c1", "in", "oi", NOW)
    result, released = store.release("c1", "ana", NOW)  # sem dono: qualquer operador devolve
    assert result is ClaimResult.OK
    assert released is not None
    assert released.phone == "5567999990001"
    assert [m.text for m in released.messages] == ["oi"]
    assert not store.is_human("c1", NOW)
    assert store.list_active(NOW) == []


def test_release_by_non_owner_is_refused_and_keeps_the_conversation(store: Any) -> None:
    store.enter("c1", "5567999990001", NOW, NOW + TTL)
    store.append("c1", "in", "oi", NOW)
    store.claim("c1", "ana", NOW, NOW + TTL)
    assert store.release("c1", "bia", NOW) == (ClaimResult.TAKEN, None)
    assert store.is_human("c1", NOW)
    assert [m.text for m in store.get("c1", NOW).messages] == ["oi"]
    assert store.release("c1", "ana", NOW)[0] is ClaimResult.OK


def test_concurrent_claim_and_release_never_free_a_taken_conversation(store: Any) -> None:
    """Corrida: bia assume enquanto ana (sem dono ainda) devolve; ou um ou outro, nunca os dois."""
    for i in range(30):
        cid = f"r{i}"
        store.enter(cid, "5567999990001", NOW, NOW + TTL)
        with ThreadPoolExecutor(max_workers=2) as pool:
            claim = pool.submit(store.claim, cid, "bia", NOW, NOW + TTL)
            release = pool.submit(store.release, cid, "ana", NOW)
        claimed, (released, _) = claim.result(), release.result()
        if released is ClaimResult.OK:  # ana apagou antes: bia não pode ter ficado com dono
            assert claimed is ClaimResult.NOT_FOUND
            assert not store.is_human(cid, NOW)
        else:  # bia assumiu antes: ana é recusada e a conversa segue de bia
            assert claimed is ClaimResult.OK
            assert released is ClaimResult.TAKEN
            assert store.get(cid, NOW).operator == "bia"


def test_list_active_orders_by_entry(store: Any) -> None:
    store.enter("b", "5567999990002", NOW + 5, NOW + 5 + TTL)
    store.enter("a", "5567999990001", NOW, NOW + TTL)
    assert [c.id for c in store.list_active(NOW + 6)] == ["a", "b"]


def test_first_seen_dedups_and_expires(store: Any) -> None:
    assert store.first_seen("wamid.1", NOW)
    assert not store.first_seen("wamid.1", NOW + 1)
    assert store.first_seen("wamid.2", NOW + 1)
    if isinstance(store, SqliteHandoffStore):  # no Redis o TTL é do servidor
        assert store.first_seen("wamid.1", NOW + DEDUP_TTL_SECONDS + 1)


def test_concurrent_claims_have_a_single_winner(store: Any) -> None:
    store.enter("c1", "5567999990001", NOW, NOW + TTL)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda n: store.claim("c1", f"op{n}", NOW, NOW + TTL), range(8)))
    assert results.count(ClaimResult.OK) == 1
    assert results.count(ClaimResult.TAKEN) == 7


def test_concurrent_first_seen_has_a_single_winner(store: Any) -> None:
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: store.first_seen("wamid.X", NOW), range(16)))
    assert results.count(True) == 1


def test_release_of_an_expired_conversation_is_not_found(store: Any) -> None:
    store.enter("c1", "5567999990001", NOW, NOW + TTL)
    assert store.release("c1", "ana", NOW + TTL + 1) == (ClaimResult.NOT_FOUND, None)
