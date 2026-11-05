"""Value objects imutáveis do domínio: dinheiro, identificadores e contatos."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Self

from emporio.domain.errors import InvalidValueError

_CENT = Decimal("0.01")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _to_decimal(value: Decimal | int | str) -> Decimal:
    # float é recusado de propósito: 0.1 + 0.2 já nasce errado.
    if isinstance(value, float):
        raise InvalidValueError("Money não aceita float; use Decimal, int ou str")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise InvalidValueError(f"valor monetário inválido: {value!r}") from exc


@dataclass(frozen=True, slots=True, order=True)
class Money:
    """Valor em reais (BRL) com precisão de centavos.

    O montante é sempre um ``Decimal`` quantizado em 2 casas (arredondamento
    comercial, meio para cima) e nunca negativo.

    Attributes:
        amount: Montante em reais.
    """

    amount: Decimal

    def __post_init__(self) -> None:
        quantized = _to_decimal(self.amount)
        if not quantized.is_finite():
            raise InvalidValueError("Money precisa ser finito")
        quantized = quantized.quantize(_CENT, rounding=ROUND_HALF_UP)
        if quantized < 0:
            raise InvalidValueError("Money não pode ser negativo")
        object.__setattr__(self, "amount", quantized)

    @classmethod
    def of(cls, value: Decimal | int | str) -> Self:
        """Cria um Money a partir de Decimal, int ou str (nunca float).

        Args:
            value: Montante em reais.

        Returns:
            Money quantizado em centavos.

        Raises:
            InvalidValueError: Se o valor for inválido, negativo ou float.
        """
        return cls(_to_decimal(value))

    def __add__(self, other: Money) -> Money:
        return Money(self.amount + other.amount)

    def __sub__(self, other: Money) -> Money:
        return Money(self.amount - other.amount)

    def times(self, factor: Decimal | int) -> Money:
        """Multiplica por um fator (ex.: quantidade de itens)."""
        return Money(self.amount * Decimal(factor))

    def apply_discount(self, percent: Decimal | int) -> Money:
        """Aplica desconto percentual.

        Usa ``* 0.01`` em vez de ``/ 100`` para espelhar a view gold, onde a
        divisão promoveria o valor a DOUBLE.

        Args:
            percent: Percentual entre 0 e 100.

        Returns:
            Preço após o desconto, em centavos.

        Raises:
            InvalidValueError: Se o percentual estiver fora de 0..100.
        """
        pct = Decimal(percent)
        if not Decimal(0) <= pct <= Decimal(100):
            raise InvalidValueError("desconto deve estar entre 0 e 100")
        return Money(self.amount * (1 - pct * _CENT))

    def divided_by(self, parts: int) -> Money:
        """Divide em ``parts`` parcelas iguais (arredondamento comercial)."""
        if parts < 1:
            raise InvalidValueError("número de parcelas deve ser >= 1")
        return Money(self.amount / parts)

    def format_brl(self) -> str:
        """Formata no padrão PT-BR, ex.: ``R$ 1.299,00``."""
        grouped = f"{self.amount:,.2f}"
        swapped = grouped.replace(",", "\0").replace(".", ",").replace("\0", ".")
        return f"R$ {swapped}"

    def __str__(self) -> str:
        return self.format_brl()

    def __repr__(self) -> str:
        return f"Money('{self.amount}')"


@dataclass(frozen=True, slots=True, order=True)
class OrderId:
    """Identificador positivo de pedido.

    Attributes:
        value: Número do pedido.
    """

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or self.value < 1:
            raise InvalidValueError("OrderId deve ser inteiro positivo")

    @classmethod
    def parse(cls, raw: str | int) -> OrderId:
        """Aceita ``7``, ``"7"``, ``"#7"`` ou ``"pedido 7"``.

        Raises:
            InvalidValueError: Se não houver um número de pedido válido.
        """
        if isinstance(raw, int):
            return cls(raw)
        match = re.search(r"\d+", raw)
        if match is None:
            raise InvalidValueError(f"número de pedido inválido: {raw!r}")
        return cls(int(match.group()))

    def __str__(self) -> str:
        return f"#{self.value}"


@dataclass(frozen=True, slots=True, repr=False)
class Email:
    """E-mail normalizado (minúsculo) com mascaramento para logs.

    Attributes:
        value: Endereço em minúsculas.
    """

    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip().lower()
        if not _EMAIL_RE.match(normalized):
            raise InvalidValueError("e-mail inválido")
        object.__setattr__(self, "value", normalized)

    def masked(self) -> str:
        """Retorna ``l***@dominio`` sem expor o usuário."""
        user, _, domain = self.value.partition("@")
        return f"{user[0]}***@{domain}"

    def __str__(self) -> str:
        return self.masked()

    def __repr__(self) -> str:
        return f"Email({self.masked()!r})"


@dataclass(frozen=True, slots=True, repr=False)
class Phone:
    """Telefone brasileiro (DDD + número) armazenado só com dígitos.

    Attributes:
        digits: DDD e número, 10 ou 11 dígitos, sem o código do país.
    """

    digits: str

    def __post_init__(self) -> None:
        digits = re.sub(r"\D", "", self.digits)
        if len(digits) in (12, 13) and digits.startswith("55"):
            digits = digits[2:]
        if len(digits) not in (10, 11):
            raise InvalidValueError("telefone deve ter DDD e 8 ou 9 dígitos")
        object.__setattr__(self, "digits", digits)

    def formatted(self) -> str:
        """Formata como ``(67) 99812-3456``."""
        ddd, number = self.digits[:2], self.digits[2:]
        return f"({ddd}) {number[:-4]}-{number[-4:]}"

    def masked(self) -> str:
        """Mantém DDD e os 4 últimos dígitos: ``(67) *****-3456``."""
        ddd, number = self.digits[:2], self.digits[2:]
        return f"({ddd}) {'*' * (len(number) - 4)}-{number[-4:]}"

    def __str__(self) -> str:
        return self.masked()

    def __repr__(self) -> str:
        return f"Phone({self.masked()!r})"
