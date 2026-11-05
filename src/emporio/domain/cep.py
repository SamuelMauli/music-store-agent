"""CEP brasileiro: validação de formato e resolução de UF e região pelas faixas dos Correios.

Fonte das faixas: tabela pública "Faixas de CEP por UF" dos Correios (Guia Postal / e-DNE),
transcrita à mão aqui, sem sobreposição. Lacuna documentada: 78900-000 a 78999-999 (bloco
antigo de RO, hoje fora da faixa vigente 76800-000 a 76999-999) resolve para ``None``.
``tests/unit/test_cep.py`` verifica a continuidade das demais faixas, mas isso não substitui
conferir contra a base oficial (DNE) antes de uso em produção. A UF vem só da faixa: o CEP
não diz o município.
"""

from __future__ import annotations

import re
from typing import Final

from emporio.domain.errors import InvalidValueError

CEP_PATTERN: Final = re.compile(r"^\d{5}-?\d{3}$")

# (UF, primeiro CEP, último CEP) em ordem crescente. UFs partidas aparecem em mais de uma linha.
UF_RANGES: Final[tuple[tuple[str, int, int], ...]] = (
    ("SP", 1000000, 19999999),
    ("RJ", 20000000, 28999999),
    ("ES", 29000000, 29999999),
    ("MG", 30000000, 39999999),
    ("BA", 40000000, 48999999),
    ("SE", 49000000, 49999999),
    ("PE", 50000000, 56999999),
    ("AL", 57000000, 57999999),
    ("PB", 58000000, 58999999),
    ("RN", 59000000, 59999999),
    ("CE", 60000000, 63999999),
    ("PI", 64000000, 64999999),
    ("MA", 65000000, 65999999),
    ("PA", 66000000, 68899999),
    ("AP", 68900000, 68999999),
    ("AM", 69000000, 69299999),
    ("RR", 69300000, 69399999),
    ("AM", 69400000, 69899999),
    ("AC", 69900000, 69999999),
    ("DF", 70000000, 72799999),
    ("GO", 72800000, 72999999),
    ("DF", 73000000, 73699999),
    ("GO", 73700000, 76799999),
    ("RO", 76800000, 76999999),
    ("TO", 77000000, 77999999),
    ("MT", 78000000, 78899999),
    ("MS", 79000000, 79999999),
    ("PR", 80000000, 87999999),
    ("SC", 88000000, 89999999),
    ("RS", 90000000, 99999999),
)

REGION_BY_UF: Final = {
    **dict.fromkeys(("AC", "AM", "AP", "PA", "RO", "RR", "TO"), "norte"),
    **dict.fromkeys(("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"), "nordeste"),
    **dict.fromkeys(("DF", "GO", "MT", "MS"), "centro_oeste"),
    **dict.fromkeys(("ES", "MG", "RJ", "SP"), "sudeste"),
    **dict.fromkeys(("PR", "RS", "SC"), "sul"),
}


def normalize_cep(value: str) -> str:
    """Valida o formato e devolve os 8 dígitos.

    Args:
        value: CEP como ``79002-000`` ou ``79002000``.

    Returns:
        Os 8 dígitos, sem hífen.

    Raises:
        InvalidValueError: Se não tiver exatamente 8 dígitos (hífen opcional após o 5º).
    """
    text = value.strip()
    if not CEP_PATTERN.match(text):
        raise InvalidValueError("CEP deve ter 8 dígitos, ex.: 79002-000")
    return text.replace("-", "")


def format_cep(digits: str) -> str:
    """``79002000`` -> ``79002-000``."""
    return f"{digits[:5]}-{digits[5:]}"


def uf_of_cep(cep: str) -> str | None:
    """UF do CEP pelas faixas dos Correios.

    Args:
        cep: CEP em qualquer formato aceito por :func:`normalize_cep`.

    Returns:
        Sigla da UF, ou ``None`` se o CEP está fora das faixas (ex.: ``00000-000``).

    Raises:
        InvalidValueError: Se o formato for inválido.
    """
    number = int(normalize_cep(cep))
    return next((uf for uf, low, high in UF_RANGES if low <= number <= high), None)


def region_of_uf(uf: str) -> str:
    """Região do Brasil da UF: norte, nordeste, centro_oeste, sudeste ou sul."""
    return REGION_BY_UF[uf]
