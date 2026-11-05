from __future__ import annotations

from datetime import date

import pytest

from emporio.domain.calendar import (
    DEFAULT_CALENDAR,
    HolidayCalendar,
    add_business_days,
    business_days_between,
    easter_sunday,
    is_business_day,
)

CARNIVAL = HolidayCalendar(carnival=True)


@pytest.mark.parametrize(
    ("year", "easter"),
    [
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
        (2027, date(2027, 3, 28)),
        (2000, date(2000, 4, 23)),
        (2038, date(2038, 4, 25)),  # Páscoa mais tardia possível do século
    ],
)
def test_easter_known_dates(year: int, easter: date) -> None:
    assert easter_sunday(year) == easter


def test_movable_holidays_2026() -> None:
    names = DEFAULT_CALENDAR.holidays(2026)
    assert names[date(2026, 4, 3)] == "Sexta-feira Santa"
    assert names[date(2026, 6, 4)] == "Corpus Christi"
    assert date(2026, 2, 16) not in names  # Carnaval é facultativo por padrão
    assert CARNIVAL.holiday_name(date(2026, 2, 16)) == "Carnaval (segunda)"
    assert CARNIVAL.holiday_name(date(2026, 2, 17)) == "Carnaval (terça)"


def test_consciencia_negra_only_from_2024() -> None:
    assert not DEFAULT_CALENDAR.is_holiday(date(2023, 11, 20))
    assert DEFAULT_CALENDAR.holiday_name(date(2024, 11, 20)) == "Consciência Negra"
    assert DEFAULT_CALENDAR.is_holiday(date(2026, 11, 20))


def test_all_fixed_national_holidays_2026() -> None:
    fixed = {(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (11, 20), (12, 25)}
    assert fixed <= {(d.month, d.day) for d in DEFAULT_CALENDAR.holidays(2026)}


def test_is_business_day() -> None:
    assert is_business_day(date(2026, 3, 25))  # quarta
    assert not is_business_day(date(2026, 3, 28))  # sábado
    assert not is_business_day(date(2026, 3, 29))  # domingo
    assert not is_business_day(date(2026, 4, 3))  # Sexta-feira Santa
    assert not is_business_day(date(2026, 11, 20))  # sexta, Consciência Negra
    assert is_business_day(date(2026, 2, 16))
    assert not is_business_day(date(2026, 2, 16), CARNIVAL)


def test_extra_local_holiday() -> None:
    cal = HolidayCalendar(extra=frozenset({date(2026, 8, 26)}))
    assert cal.holiday_name(date(2026, 8, 26)) == "Feriado local"
    assert not is_business_day(date(2026, 8, 26), cal)
    assert is_business_day(date(2026, 8, 26))


def test_add_business_days_skips_weekend_and_holidays() -> None:
    assert add_business_days(date(2026, 3, 25), 0) == date(2026, 3, 25)
    assert add_business_days(date(2026, 3, 25), 3) == date(2026, 3, 30)  # qua -> seg
    # quinta 02/04 + 1 pula Sexta Santa (03/04), sábado e domingo
    assert add_business_days(date(2026, 4, 2), 1) == date(2026, 4, 6)
    # partindo de sábado, o 1º dia útil é a segunda
    assert add_business_days(date(2026, 3, 28), 1) == date(2026, 3, 30)
    assert add_business_days(date(2026, 3, 30), -1) == date(2026, 3, 27)
    assert add_business_days(date(2026, 4, 6), -1) == date(2026, 4, 2)


def test_add_business_days_crosses_year_end() -> None:
    # 24/12 qui + 5: 25/12 Natal, 26 sex(1), 29 seg(2), 30(3), 31(4), 01/01 feriado, 02/01(5)
    assert add_business_days(date(2026, 12, 24), 5) == date(2027, 1, 4)


def test_ten_business_days_refund_window() -> None:
    # 10 dias úteis de 2026-03-25: 26,27,30,31,01/04,02/04, (03/04 feriado), 06,07,08,09
    assert add_business_days(date(2026, 3, 25), 10) == date(2026, 4, 9)


def test_business_days_between_and_inverse() -> None:
    assert business_days_between(date(2026, 4, 2), date(2026, 4, 6)) == 1
    assert business_days_between(date(2026, 3, 25), date(2026, 3, 25)) == 0
    assert business_days_between(date(2026, 4, 6), date(2026, 4, 2)) == -1
    assert business_days_between(date(2026, 3, 28), date(2026, 3, 29)) == 0
    assert business_days_between(date(2026, 3, 25), date(2026, 4, 9)) == 10
    start = date(2026, 12, 20)
    for n in range(0, 15):
        end = add_business_days(start, n)
        assert business_days_between(start, end) == n


def test_holidays_between() -> None:
    got = DEFAULT_CALENDAR.holidays_between(date(2026, 12, 20), date(2027, 1, 5))
    assert got == {date(2026, 12, 25), date(2027, 1, 1)}
