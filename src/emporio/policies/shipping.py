"""Frete e entregas (§5): região metropolitana com taxa fixa, demais cidades sem cotação."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Final

from emporio.domain.text import normalize_text
from emporio.domain.value_objects import Money

FREE_SHIPPING_ABOVE: Final = Money.of("500")
METRO_FLAT_FEE: Final = Money.of("35")
METRO_DELIVERY: Final = "1 a 3 dias úteis"
CITY_QUESTION: Final = "Para qual cidade é a entrega?"
LARGE_ITEM_KEYWORDS: Final = ("bateria", "piano", "contrabaixo")


@dataclass(frozen=True, slots=True)
class ShippingMode:
    """Modalidade de envio para fora da região metropolitana (§5.2).

    Attributes:
        name: Nome da modalidade.
        min_days: Prazo mínimo em dias úteis.
        max_days: Prazo máximo em dias úteis.
        tracking: Se há rastreamento.
        insured: Se o seguro está incluído.
    """

    name: str
    min_days: int
    max_days: int
    tracking: bool = True
    insured: bool = True

    @property
    def deadline_text(self) -> str:
        """Prazo legível, ex.: ``"2 a 5 dias úteis"``."""
        return f"{self.min_days} a {self.max_days} dias úteis"


OUTSIDE_MODES: Final = (
    ShippingMode("PAC (Correios)", 5, 12),
    ShippingMode("SEDEX (Correios)", 2, 5),
    ShippingMode("Jadlog (.package)", 3, 8),
)


@dataclass(frozen=True, slots=True)
class MetroShipping:
    """Entrega por motoboy próprio na região metropolitana (§5.1).

    Attributes:
        fee: Taxa de entrega (zero quando grátis).
        free: Se o frete é grátis.
        delivery: Prazo em texto.
    """

    fee: Money
    free: bool
    delivery: str = METRO_DELIVERY


@dataclass(frozen=True, slots=True)
class OutsideShipping:
    """Envio para outras cidades: modalidades e prazos, sem valor (D14).

    Attributes:
        modes: Modalidades disponíveis.
        quote_note: Por que o valor não é informado aqui.
        special_freight: Item de grande porte, com cotação individual.
    """

    modes: tuple[ShippingMode, ...]
    quote_note: str
    special_freight: bool


@dataclass(frozen=True, slots=True)
class CityRequired:
    """Sem cidade informada não dá para escolher a regra de frete.

    Attributes:
        question: Pergunta a fazer ao cliente.
    """

    question: str = CITY_QUESTION


ShippingEstimate = MetroShipping | OutsideShipping | CityRequired


def is_metro_city(city: str, metro_cities: Collection[str]) -> bool:
    """Compara cidade ignorando acentos e caixa.

    Args:
        city: Cidade informada.
        metro_cities: Cidades da região metropolitana (``Settings.metro_cities``).

    Returns:
        ``True`` se a cidade pertence à região metropolitana configurada.
    """
    wanted = normalize_text(city)
    return any(normalize_text(c) == wanted for c in metro_cities)


def metro_shipping_fee(order_total: Money) -> Money:
    """Taxa de entrega na região metropolitana.

    Implementa §5.1 — grátis acima de R$ 500,00, senão taxa fixa de R$ 35,00. O
    manual não cobre exatamente R$ 500,00; adota-se "acima" como estritamente maior.

    Args:
        order_total: Valor do pedido.

    Returns:
        ``Money`` zero quando grátis, R$ 35,00 caso contrário.
    """
    return Money.of(0) if order_total > FREE_SHIPPING_ABOVE else METRO_FLAT_FEE


def needs_special_freight(product_name: str) -> bool:
    """Instrumento de grande porte (bateria acústica, piano digital, contrabaixo).

    Implementa §5.2 — frete especial com cotação individual.

    Args:
        product_name: Nome do produto.

    Returns:
        ``True`` se o nome indica item de grande porte.
    """
    name = normalize_text(product_name)
    return any(word in name for word in LARGE_ITEM_KEYWORDS)


def estimate_shipping(
    order_total: Money,
    city: str | None,
    metro_cities: Collection[str],
    *,
    large_item: bool = False,
) -> ShippingEstimate:
    """Estima frete conforme a cidade, sem inventar valor fora da região (D14).

    Implementa §5.1 e §5.2.

    Args:
        order_total: Valor do pedido.
        city: Cidade de entrega, se informada.
        metro_cities: Cidades da região metropolitana.
        large_item: Se o pedido tem item de grande porte.

    Returns:
        ``MetroShipping``, ``OutsideShipping`` ou ``CityRequired``.
    """
    if city is None or not city.strip():
        return CityRequired()
    if is_metro_city(city, metro_cities):
        fee = metro_shipping_fee(order_total)
        return MetroShipping(fee=fee, free=fee.amount == 0)
    return OutsideShipping(
        modes=OUTSIDE_MODES,
        quote_note="o valor é calculado pelo CEP de destino, peso e dimensões do produto",
        special_freight=large_item,
    )
