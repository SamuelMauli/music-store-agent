"""Dados da loja (§1.2, §7) e horário de funcionamento (§2)."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Final
from zoneinfo import ZoneInfo

from emporio.domain.errors import InvalidValueError

STORE_TZ: Final = ZoneInfo("America/Campo_Grande")
SPECIAL_DATES_NOTE: Final = (
    "Em datas especiais o horário pode ser estendido; avisos nas redes sociais e no WhatsApp."
)
_SEARCH_DAYS: Final = 14


@dataclass(frozen=True, slots=True)
class StoreContacts:
    """Endereço e canais da loja.

    O manual traz dois números: (67) 3341-4444 na capa e em §1.2 ("Telefone /
    WhatsApp") e (67) 3321-4500 em §7 (canal de WhatsApp). Vale a cláusula
    específica: WhatsApp é o do §7 e telefone é o da capa (P1). O e-mail é a
    forma sem acento (P2), a única válida em DNS.

    Attributes:
        name: Nome fantasia.
        address: Endereço completo (§1.2).
        phone: Telefone fixo da capa.
        whatsapp: Número do canal de WhatsApp (§7).
        email: E-mail de contato.
    """

    name: str = "Empório da Música"
    address: str = "Rua 14 de Maio, 3200 — Centro, Campo Grande - MS, CEP 79202-333"
    phone: str = "(67) 3341-4444"
    whatsapp: str = "(67) 3321-4500"
    email: str = "contato@emporiodamusica.com.br"


CONTACTS: Final = StoreContacts()

# weekday(): segunda=0 ... domingo=6. Domingo ausente = fechado.
_HOURS: Final[dict[int, tuple[time, time]]] = {
    **{day: (time(9), time(18)) for day in range(5)},
    5: (time(9), time(13)),
}


def _local(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise InvalidValueError("horário da loja exige datetime com fuso")
    return now.astimezone(STORE_TZ)


def _opening_hours(day: date, holidays: Collection[date]) -> tuple[time, time] | None:
    return None if day in holidays else _HOURS.get(day.weekday())


def is_store_open(now: datetime, holidays: Collection[date] = ()) -> bool:
    """Diz se a loja está aberta (abre às 09:00 inclusive, fecha no horário final exclusivo).

    Implementa §2 — Horário de Funcionamento: seg-sex 09:00-18:00, sáb 09:00-13:00,
    domingo e feriados fechado. O manual não traz calendário de feriados; quem
    chama pode injetá-los em ``holidays``.

    Args:
        now: Instante a testar (aware); é convertido para o fuso da loja.
        holidays: Datas de feriado (loja fechada).

    Returns:
        ``True`` se aberta.

    Raises:
        InvalidValueError: Se ``now`` for naive.
    """
    local = _local(now)
    hours = _opening_hours(local.date(), holidays)
    return hours is not None and hours[0] <= local.time() < hours[1]


def next_opening(now: datetime, holidays: Collection[date] = ()) -> datetime:
    """Próxima abertura estritamente depois de ``now``.

    Implementa §2. Com a loja aberta, devolve a abertura do próximo dia útil.

    Args:
        now: Instante de referência (aware).
        holidays: Datas de feriado.

    Returns:
        Instante aware, no fuso da loja.

    Raises:
        InvalidValueError: Se ``now`` for naive ou não houver abertura em 14 dias.
    """
    local = _local(now)
    for offset in range(_SEARCH_DAYS):
        day = local.date() + timedelta(days=offset)
        hours = _opening_hours(day, holidays)
        if hours is None:
            continue
        opening = datetime.combine(day, hours[0], tzinfo=STORE_TZ)
        if opening > local:
            return opening
    raise InvalidValueError("sem abertura nos próximos 14 dias; confira os feriados informados")


def opening_hours_text() -> str:
    """Horário padrão em texto, com o aviso de datas especiais (P8)."""
    return (
        "Segunda a sexta, 09:00 às 18:00; sábado, 09:00 às 13:00; domingo e feriados, "
        f"fechado. {SPECIAL_DATES_NOTE}"
    )
