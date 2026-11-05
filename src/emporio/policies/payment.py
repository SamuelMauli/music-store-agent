"""Políticas de pagamento do manual (§3 e §6.2), como funções puras sobre ``Money``."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from emporio.domain.value_objects import Money

PIX_DISCOUNT_PERCENT: Final = Decimal("5")
COMBINED_PAYMENT_THRESHOLD: Final = Money.of("2000")
MAX_INSTALLMENTS: Final = 12

# (primeira parcela, última parcela, parcela mínima). A tabela do §3 fala em R$ 100 para
# todo o crédito, mas o §3.1 detalha por faixa e prevalece (P3). "Exceto abaixo de R$ 50"
# no §3.1 é lido como parcela mínima de R$ 50 até 3x.
_TIERS: Final = (
    (1, 3, Money.of("50")),
    (4, 6, Money.of("80")),
    (7, MAX_INSTALLMENTS, Money.of("100")),
)


@dataclass(frozen=True, slots=True)
class Installment:
    """Uma opção de parcelamento no crédito, sem juros.

    Attributes:
        count: Número de parcelas.
        amount: Valor de cada parcela, arredondado ao centavo.
    """

    count: int
    amount: Money


def pix_price(price: Money, *, has_promotion: bool) -> Money | None:
    """Preço no PIX: 5% sobre o preço de tabela, nunca sobre preço promocional.

    Implementa §3 (PIX, 5% de desconto) e §6.2 (o desconto de PIX não se aplica
    sobre preços já promocionais).

    Args:
        price: Preço de tabela do produto.
        has_promotion: Se há promoção ativa sobre o produto.

    Returns:
        O preço com 5% off, ou ``None`` quando há promoção (PIX não acumula).
    """
    if has_promotion:
        return None
    return price.apply_discount(PIX_DISCOUNT_PERCENT)


def minimum_installment(count: int) -> Money:
    """Parcela mínima da faixa em que ``count`` cai.

    Implementa §3.1 — Regras de Parcelamento (faixas 1-3x, 4-6x e 7-12x).

    Args:
        count: Número de parcelas, de 1 a 12.

    Returns:
        A parcela mínima aplicável.

    Raises:
        ValueError: Se ``count`` estiver fora de 1..12.
    """
    for first, last, minimum in _TIERS:
        if first <= count <= last:
            return minimum
    raise ValueError(f"parcelamento fora de 1..{MAX_INSTALLMENTS}x: {count}")


def installment_options(total: Money) -> list[Installment]:
    """Lista os parcelamentos válidos no crédito, todos sem juros.

    Implementa §3.1 — Regras de Parcelamento. A compra em 1x (à vista no cartão) é
    sempre possível: o piso de R$ 50 só restringe o parcelamento a partir de 2x.

    Args:
        total: Valor a pagar no cartão.

    Returns:
        Opções em ordem crescente de parcelas; cada parcela respeita o mínimo da
        sua faixa.
    """
    options = [Installment(1, total)]
    for count in range(2, MAX_INSTALLMENTS + 1):
        amount = total.divided_by(count)
        if amount >= minimum_installment(count):
            options.append(Installment(count, amount))
    return options


def combined_payment_allowed(total: Money) -> bool:
    """Combinação de formas de pagamento (ex.: PIX + cartão) só acima de R$ 2.000.

    Implementa §3.1 — "para compras acima de R$ 2.000,00" (estritamente acima).

    Args:
        total: Valor total da compra.

    Returns:
        ``True`` se ``total`` for maior que R$ 2.000,00.
    """
    return total > COMBINED_PAYMENT_THRESHOLD
