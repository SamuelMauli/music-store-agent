"""Cartão de crédito é PII: detectado (com Luhn), mascarado e retirado antes do modelo."""

from __future__ import annotations

import pytest

from emporio.agent.identity import CARD_PLACEHOLDER, redact
from emporio.domain.pii import find_pii, mask_pii


@pytest.mark.parametrize(
    "pan", ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111", "378282246310005"]
)
def test_valid_card_is_found_and_masked(pan: str) -> None:
    text = f"meu cartão é {pan}, anota aí"
    assert [m.kind for m in find_pii(text)] == ["card"]
    assert pan not in mask_pii(text)
    assert mask_pii(text).endswith(f"**** **** **** {pan[-4:]}, anota aí")


def test_redact_removes_card_before_the_model() -> None:
    out = redact("CPF 123.456.789-09 e cartão 4111 1111 1111 1111")
    assert "4111" not in out
    assert CARD_PLACEHOLDER in out
    assert "[cpf omitido]" in out


@pytest.mark.parametrize(
    "text",
    [
        "4111 1111 1111 1112",  # falha no Luhn
        "R$ 2.199,00 em 12x de R$ 183,25",
        "pedido 7, rastreio BRJL5544332BR",
    ],
)
def test_non_card_numbers_are_left_alone(text: str) -> None:
    assert all(m.kind != "card" for m in find_pii(text))


def test_phone_is_still_a_phone() -> None:
    assert [m.kind for m in find_pii("me liga no (67) 99999-1234")] == ["phone"]
