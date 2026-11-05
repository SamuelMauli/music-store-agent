from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from emporio.domain.calendar import HolidayCalendar, is_business_day
from emporio.domain.errors import InvalidValueError
from emporio.domain.value_objects import Money
from emporio.policies.shipping import OUTSIDE_MODES
from emporio.policies.shipping_quote import (
    ESTIMATES_PATH,
    MetroQuote,
    OutsideQuote,
    OverweightQuote,
    UnknownCep,
    load_estimates,
    quote_shipping,
)

TODAY = date(2026, 3, 25)  # quarta


def test_shipped_table_is_valid_and_within_manual_ranges() -> None:
    table = load_estimates()
    assert table.version == 1
    for mode in OUTSIDE_MODES:
        for rates in table.modes[mode.name].values():
            assert mode.min_days <= rates.days[0] <= rates.days[1] <= mode.max_days
            assert list(rates.price) == sorted(rates.price)  # mais pesado nunca é mais barato


def test_metro_follows_manual_rule() -> None:
    free = quote_shipping("79002-000", None, TODAY, order_total=Money.of("600"))
    paid = quote_shipping("79002000", None, TODAY, order_total=Money.of("499"))
    edge = quote_shipping("79002000", None, TODAY, order_total=Money.of("500"))
    unknown = quote_shipping("79002000", None, TODAY)
    assert isinstance(free, MetroQuote)
    assert free.free
    assert free.fee == Money.of(0)
    assert isinstance(paid, MetroQuote)
    assert paid.fee == Money.of(35)
    assert isinstance(edge, MetroQuote)
    assert edge.fee == Money.of(35)  # "acima" = estritamente maior
    assert isinstance(unknown, MetroQuote)
    assert unknown.fee is None
    assert not unknown.free
    assert (free.earliest, free.latest) == (date(2026, 3, 26), date(2026, 3, 30))


def test_outside_quote_prices_dates_and_bands() -> None:
    quote = quote_shipping("01310-100", Decimal("2.5"), TODAY)
    assert isinstance(quote, OutsideQuote)
    assert (quote.uf, quote.zone) == ("SP", "sudeste")
    by_mode = {o.mode: o for o in quote.options}
    assert set(by_mode) == {m.name for m in OUTSIDE_MODES}
    assert by_mode["PAC (Correios)"].price == Money.of(46)  # faixa 1-5 kg
    sedex = by_mode["SEDEX (Correios)"]
    assert (sedex.min_days, sedex.max_days) == (2, 4)
    assert (sedex.earliest, sedex.latest) == (date(2026, 3, 27), date(2026, 3, 31))
    for option in quote.options:
        assert is_business_day(option.earliest)
        assert is_business_day(option.latest)


def test_band_edges() -> None:
    def pac(kg: str) -> Money:
        quote = quote_shipping("01310-100", Decimal(kg), TODAY)
        assert isinstance(quote, OutsideQuote)
        return quote.options[0].price

    assert pac("0.1") == Money.of(32)
    assert pac("1") == Money.of(32)  # limite inclusivo
    assert pac("1.01") == Money.of(46)
    assert pac("30") == Money.of(120)


def test_same_state_uses_ms_zone_and_far_region_costs_more() -> None:
    ms = quote_shipping("79200-000", Decimal(1), TODAY)  # interior de MS
    norte = quote_shipping("69900-000", Decimal(1), TODAY)
    assert isinstance(ms, OutsideQuote)
    assert isinstance(norte, OutsideQuote)
    assert ms.zone == "ms"
    assert norte.zone == "norte"
    assert norte.options[0].price > ms.options[0].price


def test_holidays_push_the_forecast_dates() -> None:
    # quinta 02/04 + 2 dias úteis: pula Sexta Santa 03/04 e o fim de semana
    quote = quote_shipping("01310-100", Decimal(1), date(2026, 4, 2))
    assert isinstance(quote, OutsideQuote)
    assert quote.options[1].earliest == date(2026, 4, 7)
    carnival = quote_shipping(
        "01310-100", Decimal(1), date(2026, 2, 13), calendar=HolidayCalendar(carnival=True)
    )
    plain = quote_shipping("01310-100", Decimal(1), date(2026, 2, 13))
    assert isinstance(carnival, OutsideQuote)
    assert isinstance(plain, OutsideQuote)
    assert carnival.options[1].earliest > plain.options[1].earliest


def test_overweight_unknown_cep_and_missing_weight() -> None:
    assert quote_shipping("01310-100", Decimal("30.5"), TODAY) == OverweightQuote(Decimal(30))
    assert isinstance(quote_shipping("00100-000", Decimal(1), TODAY), UnknownCep)
    with pytest.raises(InvalidValueError):
        quote_shipping("01310-100", None, TODAY)
    with pytest.raises(InvalidValueError):
        quote_shipping("01310-100", Decimal(0), TODAY)
    with pytest.raises(InvalidValueError):
        quote_shipping("1234", Decimal(1), TODAY)


def _write(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    data = yaml.safe_load(ESTIMATES_PATH.read_text(encoding="utf-8"))
    mutate(data)
    target = tmp_path / "t.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    return target


def test_table_rejects_deadline_outside_manual(tmp_path: Path) -> None:
    def bad(d: dict[str, Any]) -> None:
        d["modes"]["SEDEX (Correios)"]["norte"]["days"] = [4, 9]

    with pytest.raises(ValidationError, match="fora da faixa do manual"):
        load_estimates(_write(tmp_path, bad))


def test_table_rejects_missing_zone_and_price_count(tmp_path: Path) -> None:
    def no_zone(d: dict[str, Any]) -> None:
        del d["modes"]["PAC (Correios)"]["sul"]

    def short(d: dict[str, Any]) -> None:
        d["modes"]["PAC (Correios)"]["sul"]["price"] = [1, 2]

    with pytest.raises(ValidationError, match="zonas"):
        load_estimates(_write(tmp_path, no_zone))
    with pytest.raises(ValidationError, match="um preço por faixa"):
        load_estimates(_write(tmp_path, short))
