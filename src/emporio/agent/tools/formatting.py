"""Formatação PT-BR de valores que as tools entregam prontos ao LLM."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Final

from emporio.domain.enums import Availability, OrderStatus
from emporio.domain.value_objects import Money

WEEKDAYS: Final = (
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
)

AVAILABILITY_LABEL: Final = {
    Availability.AVAILABLE: "disponível",
    Availability.LOW_STOCK: "últimas unidades",
    Availability.OUT_OF_STOCK: "sem estoque no momento",
    Availability.DISCONTINUED: "descontinuado (fora do catálogo)",
    Availability.COMING_SOON: "em breve",
}

ORDER_STATUS_LABEL: Final = {
    OrderStatus.PENDING: "aguardando pagamento",
    OrderStatus.CONFIRMED: "confirmado, em separação",
    OrderStatus.SHIPPED: "enviado, em trânsito",
    OrderStatus.DELIVERED: "entregue",
    OrderStatus.CANCELLED: "cancelado",
}


def brl(value: Money) -> str:
    """``Money`` no padrão ``R$ 1.299,00``."""
    return value.format_brl()


def percent(value: Decimal) -> str:
    """Percentual sem zeros supérfluos: ``12%``, ``12,5%``."""
    text = f"{value.normalize():f}".replace(".", ",")
    return f"{text}%"


def day(value: date) -> str:
    """Data ``dd/mm/aaaa``."""
    return value.strftime("%d/%m/%Y")


def weekday_name(value: date) -> str:
    """Dia da semana em português, independente de locale."""
    return WEEKDAYS[value.weekday()]


def when(moment: datetime) -> str:
    """Instante como ``sábado, 28/03/2026 às 09:00``."""
    return f"{weekday_name(moment.date())}, {day(moment.date())} às {moment:%H:%M}"
