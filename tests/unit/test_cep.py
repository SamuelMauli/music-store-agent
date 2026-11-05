from __future__ import annotations

from itertools import pairwise

import pytest

from emporio.domain.cep import (
    REGION_BY_UF,
    UF_RANGES,
    format_cep,
    normalize_cep,
    region_of_uf,
    uf_of_cep,
)
from emporio.domain.errors import InvalidValueError

# Única lacuna aceita: bloco 789xx, antigo de RO, que hoje não pertence a nenhuma UF.
KNOWN_GAP = (78900000, 78999999)


def test_ranges_cover_the_cep_space_without_overlap_except_the_known_gap() -> None:
    ordered = sorted(UF_RANGES, key=lambda r: r[1])
    assert ordered[0][1] == 1000000
    assert ordered[-1][2] == 99999999
    gaps = []
    for (_, _, prev_high), (_, low, high) in pairwise(ordered):
        assert low <= high
        assert low > prev_high  # sem sobreposição
        if low != prev_high + 1:
            gaps.append((prev_high + 1, low - 1))
    assert gaps == [KNOWN_GAP]


def test_known_gap_is_unknown() -> None:
    assert uf_of_cep("78900-000") is None
    assert uf_of_cep("78999-999") is None
    assert uf_of_cep("78899-999") == "MT"
    assert uf_of_cep("79000-000") == "MS"


def test_every_uf_range_has_a_region() -> None:
    assert {uf for uf, _, _ in UF_RANGES} == set(REGION_BY_UF)
    assert len(REGION_BY_UF) == 27


@pytest.mark.parametrize(
    ("cep", "uf"),
    [
        ("01310-100", "SP"),  # Av. Paulista
        ("20040020", "RJ"),
        ("79002-000", "MS"),
        ("70040-010", "DF"),
        ("72800-000", "GO"),  # início da faixa de GO dentro do miolo do DF
        ("73000-000", "DF"),
        ("69301-000", "RR"),
        ("69400-000", "AM"),
        ("76800-000", "RO"),
        ("90010-000", "RS"),
        ("99999-999", "RS"),
    ],
)
def test_uf_of_cep(cep: str, uf: str) -> None:
    assert uf_of_cep(cep) == uf


def test_uf_of_cep_outside_ranges() -> None:
    assert uf_of_cep("00000-000") is None
    assert uf_of_cep("00999999") is None


@pytest.mark.parametrize("bad", ["", "1234567", "123456789", "79002-00", "abcde-fgh", "79 002000"])
def test_invalid_format_is_rejected(bad: str) -> None:
    with pytest.raises(InvalidValueError):
        normalize_cep(bad)


def test_format_and_region() -> None:
    assert normalize_cep(" 79002-000 ") == "79002000"
    assert format_cep("79002000") == "79002-000"
    assert region_of_uf("MS") == "centro_oeste"
    assert region_of_uf("BA") == "nordeste"
