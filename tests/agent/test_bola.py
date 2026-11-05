"""Teste BLOQUEANTE de BOLA: cliente verificado A nunca vê pedido de B."""

from __future__ import annotations

import pytest

from emporio.agent.tools import ToolRegistry
from emporio.agent.tools.results import NotFound, Unauthorized
from tests.agent.conftest import caller

ORDER_OF_B = 8  # pedido do cliente 2
CUSTOMER_A = 14
UNKNOWN_ORDER = 999

ORDER_TOOLS = [
    ("get_order_status", {}),
    ("check_return_eligibility", {"reason": "defect"}),
]


@pytest.mark.parametrize(("tool", "extra"), ORDER_TOOLS)
def test_verified_customer_cannot_read_someone_elses_order(
    registry: ToolRegistry, tool: str, extra: dict[str, str]
) -> None:
    ask = caller(registry, verified_customer_id=CUSTOMER_A)
    other = ask(tool, order_id=ORDER_OF_B, **extra)
    missing = ask(tool, order_id=UNKNOWN_ORDER, **extra)
    assert isinstance(other.result, Unauthorized)
    assert isinstance(missing.result, NotFound)
    assert other.payload == missing.payload


@pytest.mark.parametrize(("tool", "extra"), ORDER_TOOLS)
def test_denied_payload_leaks_nothing_about_the_order(
    registry: ToolRegistry, tool: str, extra: dict[str, str]
) -> None:
    payload = caller(registry, verified_customer_id=CUSTOMER_A)(
        tool, order_id=ORDER_OF_B, **extra
    ).payload
    for secret in ("28/02/2026", "enviado", "tracking", "R$"):
        assert secret not in payload


def test_every_order_tool_requires_a_verified_customer(registry: ToolRegistry) -> None:
    gated = [n for n in registry.names if registry.get(n).requires_verified_customer]
    assert sorted(gated) == ["check_return_eligibility", "get_order_status"]
    anonymous = caller(registry)
    for name in gated:
        payload = anonymous(name, order_id=ORDER_OF_B, reason="defect").payload
        assert "policy_blocked" in payload or "error" in payload
        assert "28/02/2026" not in payload
