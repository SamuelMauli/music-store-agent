"""Calendário de feriados nacionais do Brasil e aritmética de dias úteis (stdlib pura).

Dia útil aqui é segunda a sexta que não seja feriado nacional, a convenção usada por
Correios e transportadoras para prazo de entrega e pelo manual (§4.1, §5) para reembolso.
Sábado não conta, embora a loja abra (§2).

Feriados móveis saem da Páscoa (algoritmo gregoriano anônimo, Meeus/Jones/Butcher):
Sexta-feira Santa (Páscoa - 2) e Corpus Christi (Páscoa + 60). Carnaval (segunda e terça,
Páscoa - 48 e - 47) é ponto facultativo, não feriado: só vira dia não útil se o calendário
for criado com ``carnival=True``. Feriados estaduais e municipais não entram; quem chama
os injeta em ``extra`` (ex.: aniversário de Campo Grande).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cache
from typing import Final

# (mês, dia, nome, primeiro ano em vigor). 20/11 é feriado nacional desde a Lei 14.759/2023
# (Dia Nacional de Zumbi e da Consciência Negra), que vale a partir de 2024.
_FIXED: Final = (
    (1, 1, "Confraternização Universal", 1),
    (4, 21, "Tiradentes", 1),
    (5, 1, "Dia do Trabalho", 1),
    (9, 7, "Independência do Brasil", 1),
    (10, 12, "Nossa Senhora Aparecida", 1),
    (11, 2, "Finados", 1),
    (11, 15, "Proclamação da República", 1),
    (11, 20, "Consciência Negra", 2024),
    (12, 25, "Natal", 1),
)
_SATURDAY: Final = 5


def easter_sunday(year: int) -> date:
    """Domingo de Páscoa (calendário gregoriano).

    Args:
        year: Ano, ex.: ``2026``.

    Returns:
        Data da Páscoa no ano.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


@cache
def _holidays_of(year: int, carnival: bool) -> dict[date, str]:
    found = {date(year, m, d): name for m, d, name, since in _FIXED if year >= since}
    easter = easter_sunday(year)
    found[easter - timedelta(days=2)] = "Sexta-feira Santa"
    found[easter + timedelta(days=60)] = "Corpus Christi"
    if carnival:
        found[easter - timedelta(days=48)] = "Carnaval (segunda)"
        found[easter - timedelta(days=47)] = "Carnaval (terça)"
    return found


@dataclass(frozen=True, slots=True)
class HolidayCalendar:
    """Feriados nacionais mais datas extras injetadas.

    Attributes:
        carnival: Trata segunda e terça de Carnaval como dia não útil (ponto facultativo).
        extra: Feriados locais ou datas de recesso a somar aos nacionais.
    """

    carnival: bool = False
    extra: frozenset[date] = field(default_factory=frozenset)

    def holidays(self, year: int) -> dict[date, str]:
        """Feriados do ano, data -> nome (extras entram como ``"Feriado local"``)."""
        found = dict(_holidays_of(year, self.carnival))
        for day in self.extra:
            if day.year == year:
                found.setdefault(day, "Feriado local")
        return found

    def is_holiday(self, day: date) -> bool:
        """Se ``day`` é feriado neste calendário."""
        return day in self.holidays(day.year)

    def holiday_name(self, day: date) -> str | None:
        """Nome do feriado, ou ``None`` se ``day`` não é feriado."""
        return self.holidays(day.year).get(day)

    def holidays_between(self, start: date, end: date) -> frozenset[date]:
        """Feriados em ``[start, end]``, para APIs que recebem uma coleção de datas."""
        years = range(start.year, end.year + 1)
        return frozenset(d for y in years for d in self.holidays(y) if start <= d <= end)


DEFAULT_CALENDAR: Final = HolidayCalendar()


def is_business_day(day: date, calendar: HolidayCalendar = DEFAULT_CALENDAR) -> bool:
    """Segunda a sexta que não seja feriado.

    Args:
        day: Data a avaliar.
        calendar: Calendário de feriados.

    Returns:
        ``True`` se for dia útil.
    """
    return day.weekday() < _SATURDAY and not calendar.is_holiday(day)


def add_business_days(start: date, days: int, calendar: HolidayCalendar = DEFAULT_CALENDAR) -> date:
    """Soma (ou subtrai, se negativo) dias úteis.

    O dia ``start`` nunca conta: com ``days=1`` o resultado é o próximo dia útil depois
    dele, mesmo quando ``start`` cai em sábado, domingo ou feriado. Com ``days=0``
    devolve ``start`` sem ajuste.

    Args:
        start: Data de partida.
        days: Quantidade de dias úteis a andar.
        calendar: Calendário de feriados.

    Returns:
        Data do ``days``-ésimo dia útil depois (ou antes) de ``start``.
    """
    step = 1 if days >= 0 else -1
    current, left = start, abs(days)
    while left:
        current += timedelta(days=step)
        if is_business_day(current, calendar):
            left -= 1
    return current


def business_days_between(
    start: date, end: date, calendar: HolidayCalendar = DEFAULT_CALENDAR
) -> int:
    """Dias úteis em ``(start, end]``: exclui ``start``, inclui ``end``.

    É o inverso de :func:`add_business_days`: se ``end`` é dia útil,
    ``add_business_days(start, business_days_between(start, end)) == end``. Negativo
    quando ``end`` é anterior a ``start``.

    Args:
        start: Data inicial (não conta).
        end: Data final (conta, se for útil).
        calendar: Calendário de feriados.

    Returns:
        Quantidade de dias úteis no intervalo.
    """
    if end < start:
        return -business_days_between(end, start, calendar)
    span = (end - start).days
    return sum(is_business_day(start + timedelta(days=i), calendar) for i in range(1, span + 1))
