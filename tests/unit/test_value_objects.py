from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emporio.clock import FixedClock, SystemClock
from emporio.domain.errors import InvalidValueError
from emporio.domain.value_objects import Email, Money, OrderId, Phone
from emporio.settings import Settings

cents = st.integers(min_value=0, max_value=10_000_000).map(lambda c: Decimal(c) / 100)


class TestMoney:
    def test_quantizes_to_cents_half_up(self) -> None:
        assert Money.of("10.005").amount == Decimal("10.01")
        assert Money.of(5).amount == Decimal("5.00")

    def test_rejects_float_negative_and_garbage(self) -> None:
        with pytest.raises(InvalidValueError):
            Money(0.1)  # type: ignore[arg-type]  # o teste prova que float é recusado
        with pytest.raises(InvalidValueError):
            Money.of("-1")
        with pytest.raises(InvalidValueError):
            Money.of("abc")
        with pytest.raises(InvalidValueError):
            Money.of("Infinity")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1299", "R$ 1.299,00"),
            ("0.5", "R$ 0,50"),
            ("1234567.89", "R$ 1.234.567,89"),
            ("2089.05", "R$ 2.089,05"),
        ],
    )
    def test_format_brl(self, value: str, expected: str) -> None:
        assert Money.of(value).format_brl() == expected
        assert str(Money.of(value)) == expected

    def test_pix_and_installment_golden_values(self) -> None:
        price = Money.of("2199")
        assert price.apply_discount(5) == Money.of("2089.05")
        assert price.divided_by(12) == Money.of("183.25")

    def test_discount_bounds_and_parts(self) -> None:
        with pytest.raises(InvalidValueError):
            Money.of(10).apply_discount(101)
        with pytest.raises(InvalidValueError):
            Money.of(10).divided_by(0)

    def test_subtraction_cannot_go_negative(self) -> None:
        assert Money.of(5) - Money.of(2) == Money.of(3)
        with pytest.raises(InvalidValueError):
            Money.of(1) - Money.of(2)

    def test_repr_is_exact(self) -> None:
        assert repr(Money.of("7")) == "Money('7.00')"

    @given(cents, cents)
    def test_addition_commutative_and_exact(self, a: Decimal, b: Decimal) -> None:
        assert Money(a) + Money(b) == Money(b) + Money(a)
        assert (Money(a) + Money(b)).amount == a + b

    @given(cents, st.integers(min_value=0, max_value=100))
    def test_discount_never_increases_price(self, a: Decimal, pct: int) -> None:
        assert Money(a).apply_discount(pct) <= Money(a)

    @given(cents, st.integers(min_value=0, max_value=50))
    def test_times_matches_decimal(self, a: Decimal, n: int) -> None:
        assert Money(a).times(n).amount == a * n


class TestOrderId:
    @pytest.mark.parametrize("raw", [7, "7", "#7", "pedido 7"])
    def test_parse(self, raw: str | int) -> None:
        assert OrderId.parse(raw) == OrderId(7)
        assert str(OrderId(7)) == "#7"

    @pytest.mark.parametrize("raw", ["sem numero", "", 0, -3])
    def test_invalid(self, raw: str | int) -> None:
        with pytest.raises(InvalidValueError):
            OrderId.parse(raw)

    def test_bool_is_not_an_id(self) -> None:
        with pytest.raises(InvalidValueError):
            OrderId(True)


class TestEmail:
    def test_normalizes_and_masks(self) -> None:
        email = Email("  Lucas.Mendes@JMail.com ")
        assert email.value == "lucas.mendes@jmail.com"
        assert email.masked() == "l***@jmail.com"

    def test_repr_and_str_never_leak(self) -> None:
        email = Email("lucas.mendes@jmail.com")
        assert "lucas.mendes" not in repr(email)
        assert "lucas.mendes" not in str(email)
        assert "lucas.mendes" not in f"{email!r} {email}"

    @pytest.mark.parametrize("raw", ["", "sem-arroba", "a@b", "a b@c.com"])
    def test_invalid(self, raw: str) -> None:
        with pytest.raises(InvalidValueError):
            Email(raw)

    def test_equality_after_normalization(self) -> None:
        assert Email("A@b.com") == Email("a@b.com")


class TestPhone:
    def test_formats_and_masks(self) -> None:
        phone = Phone("(67) 99812-3456")
        assert phone.digits == "67998123456"
        assert phone.formatted() == "(67) 99812-3456"
        assert phone.masked() == "(67) *****-3456"

    def test_accepts_country_code_and_landline(self) -> None:
        assert Phone("+55 67 99812-3456") == Phone("67998123456")
        assert Phone("(67) 3341-4444").masked() == "(67) ****-4444"

    def test_repr_and_str_never_leak(self) -> None:
        phone = Phone("(67) 99812-3456")
        assert "99812" not in repr(phone)
        assert "99812" not in str(phone)

    @pytest.mark.parametrize("raw", ["123", "", "abc", "6799812345678901"])
    def test_invalid(self, raw: str) -> None:
        with pytest.raises(InvalidValueError):
            Phone(raw)


class TestClockAndSettings:
    def test_fixed_clock_requires_timezone(self) -> None:
        with pytest.raises(InvalidValueError):
            FixedClock(datetime(2026, 3, 25))  # noqa: DTZ001 - naive de propósito

    def test_fixed_clock_is_constant(self) -> None:
        instant = datetime(2026, 3, 25, 10, tzinfo=ZoneInfo("America/Campo_Grande"))
        clock = FixedClock(instant)
        assert clock.now() == clock.now() == instant

    def test_system_clock_is_aware_in_store_timezone(self) -> None:
        now = SystemClock().now()
        assert now.tzinfo is not None
        assert now.utcoffset() == datetime.now(ZoneInfo("America/Campo_Grande")).utcoffset()

    def test_settings_reference_is_aware(self) -> None:
        assert Settings().reference_datetime.tzinfo is not None

    def test_settings_localizes_naive_reference(self) -> None:
        settings = Settings(reference_datetime=datetime(2026, 3, 25, 9))  # noqa: DTZ001
        assert settings.reference_datetime.utcoffset() is not None
        assert settings.reference_datetime.hour == 9
