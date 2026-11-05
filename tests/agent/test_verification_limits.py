"""Verificação de cliente: limites por alvo e por IP (entre sessões) e validade da verificação."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from emporio.bootstrap import Container, build_container
from emporio.settings import Settings
from tests.agent.conftest import fake_settings

OWNER = "leticia.rocha@jmail.com"


def build(ingested: Settings, **extra: object) -> Container:
    return build_container(fake_settings(ingested, **extra))


@pytest.fixture
def container(ingested: Settings) -> Iterator[Container]:
    built = build(ingested, verify_max_attempts_per_target=2)
    yield built
    built.close()


def fail_once(container: Container, session: str, order: int, email: str, ip: str = "") -> bool:
    container.service.chat(session, f"Onde está meu pedido {order}?", client_ip=ip)
    return container.service.chat(session, email, client_ip=ip).handoff


def test_target_lock_survives_new_sessions(container: Container) -> None:
    """Trocar de sessão não zera o contador do pedido: o dono também fica barrado."""
    assert not fail_once(container, "s1", 7, "x1@x.com")
    fail_once(container, "s2", 7, "x2@x.com")  # 2ª falha no pedido 7 (limite do teste = 2)
    container.service.chat("s3", "Onde está meu pedido 7?")
    blocked = container.service.chat("s3", OWNER)
    assert blocked.handoff
    assert "get_order_status" not in blocked.tools
    assert "verify_customer" not in blocked.tools  # nem chega a consultar a base


def test_contact_lock_covers_enumeration_of_orders(container: Container) -> None:
    """Mesmo contato testado contra pedidos diferentes esgota o limite do contato."""
    fail_once(container, "c1", 7, "alvo@x.com")
    fail_once(container, "c2", 8, "alvo@x.com")
    container.service.chat("c3", "Onde está meu pedido 9?")
    assert container.service.chat("c3", "alvo@x.com").handoff


def test_ip_lock_blocks_even_with_fresh_targets(ingested: Settings) -> None:
    built = build(ingested, verify_max_attempts_per_ip=2)
    try:
        fail_once(built, "i1", 7, "a@x.com", ip="9.9.9.9")
        fail_once(built, "i2", 8, "b@x.com", ip="9.9.9.9")
        built.service.chat("i3", "Onde está meu pedido 9?", client_ip="9.9.9.9")
        assert built.service.chat("i3", "c@x.com", client_ip="9.9.9.9").handoff
        built.service.chat("i4", "Onde está meu pedido 7?", client_ip="8.8.8.8")
        other = built.service.chat("i4", OWNER, client_ip="8.8.8.8")
        assert "get_order_status" in other.tools  # outro IP não é afetado
    finally:
        built.close()


def test_verification_expires_after_ttl(ingested: Settings) -> None:
    now = [1_000.0]
    built = build_container(
        fake_settings(ingested, verification_ttl_seconds=60), wall=lambda: now[0]
    )
    try:
        built.service.chat("t1", "Onde está meu pedido 7?")
        assert "get_order_status" in built.service.chat("t1", OWNER).tools
        now[0] += 30
        assert built.service.snapshot("t1")["verified_customer_id"] == 14
        now[0] += 31
        again = built.service.chat("t1", "Onde está meu pedido 7?")
        assert "get_order_status" not in again.tools
        assert "e-mail" in again.answer  # pede a verificação de novo
        assert built.service.snapshot("t1")["verified_customer_id"] is None
    finally:
        built.close()
