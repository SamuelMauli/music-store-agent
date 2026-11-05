from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emporio.domain.text import build_aliases, normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Violão  Não", "violao nao"),
        ("  TAKAMINE GD20 ", "takamine gd20"),
        ("Teclado Sintetizador Korg", "teclado sintetizador korg"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


@given(st.text())
def test_normalize_is_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_aliases_for_gd20() -> None:
    aliases = build_aliases("takamine gd20 dreadnought natural")
    assert {"gd20", "gd 20", "takamine gd20", "takamine gd 20"} <= set(aliases)


def test_aliases_handle_hyphenated_models() -> None:
    aliases = build_aliases("tagima memphis ac-39 nylon natural")
    assert {"ac-39", "ac39", "ac 39", "tagima ac39", "memphis ac 39"} <= set(aliases)


def test_bare_numbers_are_not_aliases() -> None:
    aliases = build_aliases("shelby sn-7c 7 cordas nylon natural")
    assert "7" not in aliases
    assert "sn7c" in aliases


def test_name_without_model_token_has_no_aliases() -> None:
    assert build_aliases("bateria acustica yamaha studio") == []
