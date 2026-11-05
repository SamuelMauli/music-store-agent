from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emporio.domain.errors import InvalidValueError
from emporio.policies.store import (
    CONTACTS,
    STORE_TZ,
    is_store_open,
    next_opening,
    opening_hours_text,
)


def at(day: str, hour: int, minute: int = 0, tz: ZoneInfo = STORE_TZ) -> datetime:
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00").replace(tzinfo=tz)


@pytest.mark.parametrize(
    ("moment", "open_"),
    [
        (at("2026-03-25", 10), True),  # quarta
        (at("2026-03-25", 8, 59), False),
        (at("2026-03-25", 9), True),
        (at("2026-03-25", 17, 59), True),
        (at("2026-03-25", 18), False),
        (at("2026-03-27", 12), True),  # sexta
        (at("2026-03-28", 9), True),  # sábado
        (at("2026-03-28", 12, 59), True),
        (at("2026-03-28", 13), False),
        (at("2026-03-28", 15), False),
        (at("2026-03-29", 10), False),  # domingo
    ],
)
def test_is_store_open_table(moment: datetime, open_: bool) -> None:
    assert is_store_open(moment) is open_


def test_is_store_open_converts_timezone() -> None:
    utc = at("2026-03-25", 13, tz=ZoneInfo("UTC"))  # 09:00 em Campo Grande (UTC-4)
    assert is_store_open(utc)
    assert not is_store_open(at("2026-03-25", 12, tz=ZoneInfo("UTC")))


def test_holidays_close_the_store() -> None:
    assert not is_store_open(at("2026-03-25", 10), holidays=[date(2026, 3, 25)])


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(InvalidValueError, match="fuso"):
        is_store_open(datetime(2026, 3, 25, 10))  # noqa: DTZ001


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (at("2026-03-25", 8), at("2026-03-25", 9)),
        (at("2026-03-25", 9), at("2026-03-26", 9)),
        (at("2026-03-25", 19), at("2026-03-26", 9)),
        (at("2026-03-27", 18), at("2026-03-28", 9)),
        (at("2026-03-28", 13), at("2026-03-30", 9)),
        (at("2026-03-29", 10), at("2026-03-30", 9)),
    ],
)
def test_next_opening_table(now: datetime, expected: datetime) -> None:
    assert next_opening(now) == expected


def test_next_opening_skips_holidays() -> None:
    assert next_opening(at("2026-03-25", 19), holidays=[date(2026, 3, 26)]) == at("2026-03-27", 9)


def test_next_opening_gives_up_when_everything_is_a_holiday() -> None:
    days = [date(2026, 3, d) for d in range(25, 32)] + [date(2026, 4, d) for d in range(1, 15)]
    with pytest.raises(InvalidValueError, match="14 dias"):
        next_opening(at("2026-03-25", 19), holidays=days)


BASE = at("2025-10-01", 0)


@given(st.integers(min_value=0, max_value=60 * 24 * 450).map(lambda m: BASE + timedelta(minutes=m)))
def test_next_opening_is_in_the_future_and_open(now: datetime) -> None:
    nxt = next_opening(now)
    assert nxt > now
    assert is_store_open(nxt)
    assert not is_store_open(nxt.replace(minute=0, hour=8))


def test_contacts_resolve_manual_conflicts() -> None:
    assert CONTACTS.phone == "(67) 3341-4444"
    assert CONTACTS.whatsapp == "(67) 3321-4500"
    assert CONTACTS.email == "contato@emporiodamusica.com.br"
    assert "3200" in CONTACTS.address
    assert CONTACTS.email.isascii()


def test_opening_hours_text_mentions_special_dates() -> None:
    text = opening_hours_text()
    assert "09:00 às 18:00" in text
    assert "redes sociais" in text
