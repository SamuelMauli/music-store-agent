"""Specification pattern: filtros componíveis compiláveis para SQL parametrizado.

Cada especificação avalia em memória (``is_satisfied_by``) e compila para um
fragmento SQL com marcadores ``?``. Valores nunca são concatenados; nomes de
coluna são constantes do código, nunca entrada do usuário.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from emporio.domain.entities import CatalogEntry
from emporio.domain.enums import Availability
from emporio.domain.value_objects import Money

SqlFragment = tuple[str, list[object]]


class Specification(ABC):
    """Critério booleano sobre uma :class:`CatalogEntry`."""

    @abstractmethod
    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Avalia o critério em memória."""

    @abstractmethod
    def to_sql(self) -> SqlFragment:
        """Compila para ``(fragmento, parâmetros)`` sobre a view ``gold.v_catalog``."""

    def __and__(self, other: Specification) -> Specification:
        return AndSpec(self, other)

    def __or__(self, other: Specification) -> Specification:
        return OrSpec(self, other)

    def __invert__(self) -> Specification:
        return NotSpec(self)


@dataclass(frozen=True, slots=True)
class AndSpec(Specification):
    """Conjunção de duas especificações."""

    left: Specification
    right: Specification

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Verdadeira se ambas forem."""
        return self.left.is_satisfied_by(entry) and self.right.is_satisfied_by(entry)

    def to_sql(self) -> SqlFragment:
        """Compila como ``(a AND b)``."""
        (lsql, lparams), (rsql, rparams) = self.left.to_sql(), self.right.to_sql()
        return f"({lsql} AND {rsql})", [*lparams, *rparams]


@dataclass(frozen=True, slots=True)
class OrSpec(Specification):
    """Disjunção de duas especificações."""

    left: Specification
    right: Specification

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Verdadeira se alguma for."""
        return self.left.is_satisfied_by(entry) or self.right.is_satisfied_by(entry)

    def to_sql(self) -> SqlFragment:
        """Compila como ``(a OR b)``."""
        (lsql, lparams), (rsql, rparams) = self.left.to_sql(), self.right.to_sql()
        return f"({lsql} OR {rsql})", [*lparams, *rparams]


@dataclass(frozen=True, slots=True)
class NotSpec(Specification):
    """Negação de uma especificação."""

    inner: Specification

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Verdadeira se a interna for falsa."""
        return not self.inner.is_satisfied_by(entry)

    def to_sql(self) -> SqlFragment:
        """Compila como ``(NOT a)``."""
        sql, params = self.inner.to_sql()
        return f"(NOT {sql})", params


@dataclass(frozen=True, slots=True)
class MatchAll(Specification):
    """Aceita tudo; elemento neutro da conjunção."""

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Sempre verdadeira."""
        return True

    def to_sql(self) -> SqlFragment:
        """Compila como ``TRUE``."""
        return "TRUE", []


@dataclass(frozen=True, slots=True)
class InCategory(Specification):
    """Produto da categoria informada."""

    category_id: int

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Compara ``category_id``."""
        return entry.product.category_id == self.category_id

    def to_sql(self) -> SqlFragment:
        """Compila como ``category_id = ?``."""
        return "category_id = ?", [self.category_id]


def _money(value: Money | Decimal | int | str) -> Money:
    return value if isinstance(value, Money) else Money.of(value)


@dataclass(frozen=True, slots=True, init=False)
class PriceAtMost(Specification):
    """Preço efetivo (o que o cliente paga, já com promoção) até o teto, inclusive."""

    ceiling: Money

    def __init__(self, ceiling: Money | Decimal | int | str) -> None:
        """Define o teto.

        Args:
            ceiling: Valor máximo em reais.
        """
        object.__setattr__(self, "ceiling", _money(ceiling))

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Compara ``effective_price <= teto``."""
        return entry.effective_price <= self.ceiling

    def to_sql(self) -> SqlFragment:
        """Compila como ``effective_price <= ?``."""
        return "effective_price <= ?", [self.ceiling.amount]


@dataclass(frozen=True, slots=True, init=False)
class PriceAtLeast(Specification):
    """Preço efetivo a partir do piso, inclusive."""

    floor: Money

    def __init__(self, floor: Money | Decimal | int | str) -> None:
        """Define o piso.

        Args:
            floor: Valor mínimo em reais.
        """
        object.__setattr__(self, "floor", _money(floor))

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Compara ``effective_price >= piso``."""
        return entry.effective_price >= self.floor

    def to_sql(self) -> SqlFragment:
        """Compila como ``effective_price >= ?``."""
        return "effective_price >= ?", [self.floor.amount]


@dataclass(frozen=True, slots=True)
class InStock(Specification):
    """Ofertável agora: ativo e com estoque (``available`` ou ``low_stock``)."""

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Verdadeira para disponibilidade ofertável."""
        return entry.availability in (Availability.AVAILABLE, Availability.LOW_STOCK)

    def to_sql(self) -> SqlFragment:
        """Compila como ``availability IN (?, ?)``."""
        return "availability IN (?, ?)", [
            Availability.AVAILABLE.value,
            Availability.LOW_STOCK.value,
        ]


@dataclass(frozen=True, slots=True)
class OnPromotion(Specification):
    """Produto com promoção ativa."""

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Verdadeira se há desconto ativo."""
        return entry.discount_percent is not None

    def to_sql(self) -> SqlFragment:
        """Compila como ``discount_percent IS NOT NULL``."""
        return "discount_percent IS NOT NULL", []


@dataclass(frozen=True, slots=True)
class HasStringType(Specification):
    """Violão/ukulele com cordas de ``nylon`` ou ``steel`` (aço)."""

    string_type: str

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Compara o tipo de corda das specs."""
        return str(entry.product.specs.get("strings", "")) == self.string_type

    def to_sql(self) -> SqlFragment:
        """Compila como ``string_type = ?``."""
        return "string_type = ?", [self.string_type]


@dataclass(frozen=True, slots=True)
class HasElectronics(Specification):
    """Instrumento com (ou sem) captação eletrônica."""

    wanted: bool = True

    def is_satisfied_by(self, entry: CatalogEntry) -> bool:
        """Compara a flag ``electronics`` das specs."""
        return entry.product.specs.get("electronics") is self.wanted

    def to_sql(self) -> SqlFragment:
        """Compila como ``has_electronics = ?``."""
        return "has_electronics = ?", [self.wanted]
