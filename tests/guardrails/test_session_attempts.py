from __future__ import annotations

import pytest

from emporio.guardrails.attempts import InMemoryAttemptStore, hashed
from emporio.guardrails.session import InvalidSessionError, SessionSigner


def test_signer_roundtrip_and_entropy() -> None:
    signer = SessionSigner(b"s" * 32)
    token = signer.issue()
    session_id = signer.verify(token)
    assert token.startswith(session_id + ".")
    assert len(session_id) >= 22  # 16 bytes em base64url = 128 bits
    assert signer.issue() != token


@pytest.mark.parametrize("bad", ["", ".", "abc", "abc.", ".abc", "a.b.c"])
def test_signer_rejects_malformed(bad: str) -> None:
    with pytest.raises(InvalidSessionError):
        SessionSigner(b"s" * 32).verify(bad)


def test_signer_rejects_other_secret_and_short_secret() -> None:
    token = SessionSigner(b"a" * 32).issue()
    with pytest.raises(InvalidSessionError):
        SessionSigner(b"b" * 32).verify(token)
    with pytest.raises(ValueError, match="curto"):
        SessionSigner(b"short")


def test_attempt_store_counts_expires_and_is_bounded() -> None:
    now = [0.0]
    store = InMemoryAttemptStore(ttl=10, max_keys=3, monotonic=lambda: now[0])
    assert store.count("a") == 0
    assert store.register("a") == 1
    assert store.register("a") == 2
    now[0] = 9
    assert store.count("a") == 2
    now[0] = 19.5  # 10s depois da última falha
    assert store.count("a") == 0
    assert len(store) == 0
    for n in range(10):
        store.register(f"k{n}")
    assert len(store) <= 3


def test_hashed_keys_do_not_hold_the_contact() -> None:
    key = hashed("contact", "leticia.rocha@jmail.com")
    assert "leticia" not in key
    assert key == hashed("contact", "leticia.rocha@jmail.com")
