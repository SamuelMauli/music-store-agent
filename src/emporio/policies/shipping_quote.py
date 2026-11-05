"""Cotação de frete por CEP e peso: regra do manual na região metropolitana, estimativa fora.

Região metropolitana (§5.1) usa a regra oficial do manual (``metro_shipping_fee``). Fora dela,
o manual não traz valores (D14), então a cotação vem de ``shipping_estimates.yaml``, uma
PREMISSA configurável e versionada; o resultado nunca é apresentado como preço oficial.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from emporio.domain.calendar import DEFAULT_CALENDAR, HolidayCalendar, add_business_days
from emporio.domain.cep import normalize_cep, region_of_uf, uf_of_cep
from emporio.domain.errors import InvalidValueError
from emporio.domain.value_objects import Money
from emporio.policies.shipping import METRO_DELIVERY, OUTSIDE_MODES, metro_shipping_fee

ESTIMATES_PATH: Final = Path(__file__).with_name("shipping_estimates.yaml")
METRO_MIN_DAYS: Final = 1
METRO_MAX_DAYS: Final = 3
ZONES: Final = ("ms", "centro_oeste", "sudeste", "sul", "nordeste", "norte")


class _ZoneRates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    days: tuple[int, int]
    price: tuple[Decimal, ...]


class ShippingEstimates(BaseModel):
    """Tabela de estimativas carregada do YAML (premissa, não preço oficial)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    origin_uf: str
    metro_cep_ranges: tuple[tuple[int, int], ...]
    weight_bands_kg: tuple[Decimal, ...]
    modes: dict[str, dict[str, _ZoneRates]]

    @model_validator(mode="after")
    def _consistent(self) -> ShippingEstimates:
        manual = {m.name: m for m in OUTSIDE_MODES}
        if set(self.modes) != set(manual):
            raise ValueError(f"modalidades devem ser exatamente {sorted(manual)}")
        if list(self.weight_bands_kg) != sorted(set(self.weight_bands_kg)):
            raise ValueError("weight_bands_kg deve ser estritamente crescente")
        for name, zones in self.modes.items():
            if set(zones) != set(ZONES):
                raise ValueError(f"{name}: zonas devem ser exatamente {list(ZONES)}")
            for zone, rates in zones.items():
                low, high = rates.days
                if not manual[name].min_days <= low <= high <= manual[name].max_days:
                    raise ValueError(f"{name}/{zone}: prazo fora da faixa do manual (§5.2)")
                if len(rates.price) != len(self.weight_bands_kg):
                    raise ValueError(f"{name}/{zone}: um preço por faixa de peso")
        return self


@cache
def load_estimates(path: Path = ESTIMATES_PATH) -> ShippingEstimates:
    """Carrega e valida a tabela de estimativas (cacheada por caminho).

    Args:
        path: YAML da tabela; o default é o versionado junto do código.

    Returns:
        Tabela validada.

    Raises:
        pydantic.ValidationError: Se o arquivo violar o esquema ou os prazos do manual.
    """
    return ShippingEstimates.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class QuoteOption:
    """Uma modalidade estimada para o destino.

    Attributes:
        mode: Nome da modalidade, como no manual.
        price: Valor estimado.
        min_days: Prazo mínimo em dias úteis.
        max_days: Prazo máximo em dias úteis.
        earliest: Data mais cedo prevista (feriados considerados).
        latest: Data mais tarde prevista.
        tracking: Se há rastreamento.
        insured: Se o seguro está incluído.
    """

    mode: str
    price: Money
    min_days: int
    max_days: int
    earliest: date
    latest: date
    tracking: bool
    insured: bool


@dataclass(frozen=True, slots=True)
class MetroQuote:
    """Entrega na região metropolitana pela regra oficial do manual (§5.1).

    Attributes:
        uf: UF de destino (a da loja: a metrópole fica na UF de origem).
        fee: Taxa de entrega; ``None`` quando o valor do pedido não é conhecido.
        free: Se o frete é grátis (só com valor do pedido conhecido).
        delivery: Prazo em texto.
        earliest: Data mais cedo prevista.
        latest: Data mais tarde prevista.
    """

    uf: str
    fee: Money | None
    free: bool
    delivery: str
    earliest: date
    latest: date


@dataclass(frozen=True, slots=True)
class OutsideQuote:
    """Estimativa por modalidade fora da região metropolitana (premissa, não oficial).

    Attributes:
        uf: UF de destino.
        zone: Zona de preço usada.
        options: Uma opção por modalidade.
    """

    uf: str
    zone: str
    options: tuple[QuoteOption, ...]


@dataclass(frozen=True, slots=True)
class OverweightQuote:
    """Peso acima da última faixa: frete especial com cotação individual (§5.2).

    Attributes:
        max_kg: Maior peso estimado pela tabela.
    """

    max_kg: Decimal


@dataclass(frozen=True, slots=True)
class UnknownCep:
    """CEP bem formado, mas fora das faixas dos Correios."""


ShippingQuote = MetroQuote | OutsideQuote | OverweightQuote | UnknownCep


def is_metro_cep(cep: str, table: ShippingEstimates) -> bool:
    """CEP dentro das faixas atendidas por motoboy (premissa do YAML)."""
    number = int(normalize_cep(cep))
    return any(low <= number <= high for low, high in table.metro_cep_ranges)


def quote_shipping(
    cep: str,
    weight_kg: Decimal | None,
    today: date,
    *,
    order_total: Money | None = None,
    calendar: HolidayCalendar = DEFAULT_CALENDAR,
    table: ShippingEstimates | None = None,
) -> ShippingQuote:
    """Cota o frete de um CEP: regra do manual na metrópole, estimativa fora dela.

    As datas contam os dias úteis a partir de ``today``, supondo postagem hoje.

    Args:
        cep: CEP de destino (8 dígitos, hífen opcional).
        weight_kg: Peso total; obrigatório fora da região metropolitana.
        today: Data de referência, vinda do ``Clock``.
        order_total: Valor do pedido, para o frete grátis acima de R$ 500 na metrópole.
        calendar: Feriados para converter prazo em data.
        table: Tabela de estimativas; default é o YAML versionado.

    Returns:
        ``MetroQuote``, ``OutsideQuote``, ``OverweightQuote`` ou ``UnknownCep``.

    Raises:
        InvalidValueError: CEP mal formado, ou peso ausente/não positivo fora da metrópole.
    """
    rates = table or load_estimates()
    if is_metro_cep(cep, rates):
        fee = metro_shipping_fee(order_total) if order_total is not None else None
        return MetroQuote(
            uf=rates.origin_uf,
            fee=fee,
            free=fee is not None and fee.amount == 0,
            delivery=METRO_DELIVERY,
            earliest=add_business_days(today, METRO_MIN_DAYS, calendar),
            latest=add_business_days(today, METRO_MAX_DAYS, calendar),
        )
    uf = uf_of_cep(cep)
    if uf is None:
        return UnknownCep()
    if weight_kg is None or weight_kg <= 0:
        raise InvalidValueError("peso em kg é obrigatório e deve ser maior que zero")
    band = next((i for i, top in enumerate(rates.weight_bands_kg) if weight_kg <= top), None)
    if band is None:
        return OverweightQuote(max_kg=rates.weight_bands_kg[-1])
    zone = "ms" if uf == rates.origin_uf else region_of_uf(uf)
    options = []
    for mode in OUTSIDE_MODES:
        zone_rates = rates.modes[mode.name][zone]
        low, high = zone_rates.days
        options.append(
            QuoteOption(
                mode=mode.name,
                price=Money.of(zone_rates.price[band]),
                min_days=low,
                max_days=high,
                earliest=add_business_days(today, low, calendar),
                latest=add_business_days(today, high, calendar),
                tracking=mode.tracking,
                insured=mode.insured,
            )
        )
    return OutsideQuote(uf=uf, zone=zone, options=tuple(options))
