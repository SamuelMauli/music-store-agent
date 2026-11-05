"""Trocas e devoluções (§4) e garantia legal (§8.1) como regras determinísticas."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import StrEnum
from typing import Final, Literal

from emporio.domain.entities import Order
from emporio.domain.enums import OrderStatus

REGRET_DAYS: Final = 7
DEFECT_DAYS: Final = 30
PREFERENCE_DAYS: Final = 7
LEGAL_WARRANTY_DAYS: Final = 90
REFUND_TERMS: Final = "mesma forma de pagamento original em até 10 dias úteis"
CHANNEL_QUESTION: Final = "A compra foi feita online ou na loja?"

Rule = Literal["4.1", "4.2", "4.3", "4.4", "warranty"]


class ReturnReason(StrEnum):
    """Motivo declarado pelo cliente."""

    REGRET = "regret"
    DEFECT = "defect"
    PREFERENCE = "preference"


class SalesChannel(StrEnum):
    """Canal da compra; os pedidos não o registram (P6), então ``UNKNOWN`` é comum."""

    ONLINE = "online"
    STORE = "store"
    UNKNOWN = "unknown"


class NotApplicableCause(StrEnum):
    """Por que não há decisão de devolução ainda."""

    NOT_DELIVERED = "not_delivered"
    IN_TRANSIT = "in_transit"
    CANCELLED = "cancelled"
    MISSING_DELIVERY_DATE = "missing_delivery_date"


@dataclass(frozen=True, slots=True)
class ReturnItemTraits:
    """Características do item que o excluem de troca (§4.4).

    Attributes:
        personalized: Personalização ou ajuste sob encomenda (setup, regulagem).
        final_sale: Liquidação com aviso explícito de "venda final".
        mouthpiece: Boquilha de instrumento de sopro (higiene).
    """

    personalized: bool = False
    final_sale: bool = False
    mouthpiece: bool = False


@dataclass(frozen=True, slots=True)
class ReturnDecision:
    """Decisão de elegibilidade com datas calculadas.

    Attributes:
        eligible: Se o pedido está dentro das condições da regra.
        rule: Cláusula aplicada (``"warranty"`` é a garantia legal do §8.1).
        deadline: Último dia do prazo, quando existe.
        days_left: Dias restantes até o prazo (0 no último dia); ``None`` sem prazo.
        day_of_window: Quantos dias corridos já se passaram desde o marco inicial.
        window_days: Tamanho do prazo, em dias corridos.
        requires: Condições para aceitar o produto.
        refund: Forma e prazo do reembolso, quando aplicável.
        freight: Quem paga o frete de devolução, quando o manual define.
        estimate_based: ``True`` se o marco inicial é a entrega *estimada* (D13).
        detail: Orientação em texto para o agente explicar ao cliente.
    """

    eligible: bool
    rule: Rule
    deadline: date | None = None
    days_left: int | None = None
    day_of_window: int | None = None
    window_days: int | None = None
    requires: tuple[str, ...] = ()
    refund: str | None = None
    freight: str | None = None
    estimate_based: bool = False
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ChannelRequired:
    """Estado ambíguo (P6): o arrependimento só vale online e o pedido não diz o canal.

    Attributes:
        question: Pergunta a fazer ao cliente antes de concluir.
    """

    question: str = CHANNEL_QUESTION


@dataclass(frozen=True, slots=True)
class ReturnNotApplicable:
    """Ainda não é possível decidir (o prazo conta do recebimento).

    Attributes:
        cause: Motivo estruturado.
        explanation: Texto de apoio para a resposta.
        options: Caminhos que o agente pode oferecer.
    """

    cause: NotApplicableCause
    explanation: str
    options: tuple[str, ...] = ()


ReturnOutcome = ReturnDecision | ChannelRequired | ReturnNotApplicable


def non_returnable(traits: ReturnItemTraits) -> str | None:
    """Verifica a lista de itens não elegíveis para troca.

    Implementa §4.4 — Itens Não Elegíveis para Troca.

    Args:
        traits: Características do item.

    Returns:
        O motivo da exclusão, ou ``None`` se o item pode ser trocado.
    """
    if traits.personalized:
        return "instrumento com personalização ou ajuste sob encomenda"
    if traits.final_sale:
        return 'produto de liquidação com aviso explícito de "venda final"'
    if traits.mouthpiece:
        return "boquilha de instrumento de sopro (higiene)"
    return None


def _window(start: date, today: date, days: int, rule: Rule) -> ReturnDecision:
    """Janela em dias corridos: o último dia do prazo (``start + days``) ainda vale."""
    deadline = start + timedelta(days=days)
    eligible = today <= deadline
    return ReturnDecision(
        eligible=eligible,
        rule=rule,
        deadline=deadline,
        days_left=(deadline - today).days if eligible else 0,
        day_of_window=max((today - start).days, 0),
        window_days=days,
    )


def _received(order: Order) -> date | ReturnNotApplicable:
    if order.status is OrderStatus.DELIVERED:
        if order.estimated_delivery is None:
            return ReturnNotApplicable(
                NotApplicableCause.MISSING_DELIVERY_DATE,
                "o pedido consta como entregue mas sem data de entrega; encaminhar a um atendente",
                ("abrir chamado",),
            )
        return order.estimated_delivery
    if order.status is OrderStatus.CANCELLED:
        return ReturnNotApplicable(
            NotApplicableCause.CANCELLED, "o pedido foi cancelado, não há produto a devolver"
        )
    if order.status is OrderStatus.SHIPPED:
        return ReturnNotApplicable(
            NotApplicableCause.IN_TRANSIT,
            "o prazo de 7 dias corridos começa no recebimento do produto",
            ("recusar o recebimento na entrega", "abrir chamado"),
        )
    return ReturnNotApplicable(
        NotApplicableCause.NOT_DELIVERED,
        "o pedido ainda não foi entregue; o prazo começa no recebimento do produto",
        ("abrir chamado",),
    )


def _regret(received: date, today: date, channel: SalesChannel) -> ReturnOutcome:
    """Implementa §4.1 — Direito de Arrependimento (compras online)."""
    if channel is SalesChannel.UNKNOWN:
        return ChannelRequired()
    if channel is SalesChannel.STORE:
        return ReturnDecision(
            eligible=False,
            rule="4.1",
            detail="o direito de arrependimento vale só para compras online (§4.1)",
        )
    return replace(
        _window(received, today, REGRET_DAYS, "4.1"),
        requires=("embalagem original", "sem sinais de uso", "todos os acessórios e manuais"),
        refund=REFUND_TERMS,
        freight="por conta da loja",
        estimate_based=True,
    )


def _defect(order: Order, received: date, today: date) -> ReturnDecision:
    """Implementa §4.2 (troca em 30 dias da compra) e §8.1 (garantia legal de 90 dias)."""
    in_store_window = _window(order.order_date, today, DEFECT_DAYS, "4.2")
    if in_store_window.eligible:
        return replace(
            in_store_window,
            detail="defeito de fabricação: troca em até 30 dias corridos após a compra",
        )
    legal = _window(received, today, LEGAL_WARRANTY_DAYS, "warranty")
    return replace(
        legal,
        requires=("defeito de fabricação, não mau uso, queda, umidade ou modificação",),
        estimate_based=True,
        detail=(
            "passou o prazo de troca de 30 dias: acionar a garantia com o fabricante, "
            "que a loja pode intermediar"
            if legal.eligible
            else "fora da garantia legal de 90 dias; ver a garantia do fabricante (§8.2)"
        ),
    )


def _preference(received: date, today: date) -> ReturnDecision:
    """Implementa §4.3 — Trocas por Preferência (7 dias, sujeito a disponibilidade)."""
    return replace(
        _window(received, today, PREFERENCE_DAYS, "4.3"),
        requires=("perfeito estado", "embalagem original", "disponibilidade do produto desejado"),
        estimate_based=True,
        detail="diferença de valor cobrada ou reembolsada conforme o caso",
    )


def evaluate_return(
    order: Order,
    reason: ReturnReason,
    channel: SalesChannel,
    today: date,
    traits: Sequence[ReturnItemTraits] = (),
) -> ReturnOutcome:
    """Decide devolução ou troca aplicando §4.1 a §4.4 e a garantia do §8.1.

    O manual não registra a data de recebimento: ``estimated_delivery`` é o proxy
    (D13) e a decisão sai marcada com ``estimate_based``. O prazo de preferência
    (§4.3) não diz de quando conta; adota-se o recebimento, como no §4.1. A
    exclusão do §4.4 vale para arrependimento e preferência; defeito segue para a
    garantia, que é direito do consumidor.

    Args:
        order: Pedido consultado.
        reason: Motivo do cliente.
        channel: Canal da compra (``UNKNOWN`` gera ``ChannelRequired`` no arrependimento).
        today: Data de referência, vinda do ``Clock``.
        traits: Características dos itens do pedido, para a regra do §4.4.

    Returns:
        ``ReturnDecision``, ``ChannelRequired`` ou ``ReturnNotApplicable``.
    """
    received = _received(order)
    if isinstance(received, ReturnNotApplicable):
        return received
    if reason is ReturnReason.DEFECT:
        return _defect(order, received, today)
    excluded = next((why for t in traits if (why := non_returnable(t))), None)
    if excluded is not None:
        return ReturnDecision(eligible=False, rule="4.4", detail=f"não elegível: {excluded}")
    if reason is ReturnReason.REGRET:
        return _regret(received, today, channel)
    return _preference(received, today)
