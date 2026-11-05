"""Tools de pedido (contexto Pedidos): verificação de identidade, status e devolução.

Anti-BOLA: pedido de outro cliente devolve ``Unauthorized``, que serializa
exatamente como ``NotFound``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt

from emporio.agent.tools.base import MAX_VERIFICATION_ATTEMPTS, Tool, ToolContext
from emporio.agent.tools.formatting import ORDER_STATUS_LABEL, brl, day
from emporio.agent.tools.results import (
    ORDER_NOT_FOUND_HINT,
    Ambiguous,
    NotFound,
    Ok,
    PolicyBlocked,
    ToolResult,
    Unauthorized,
)
from emporio.domain.entities import Order
from emporio.domain.errors import InvalidValueError
from emporio.domain.text import normalize_text
from emporio.domain.value_objects import Email, OrderId, Phone
from emporio.policies.returns import (
    ChannelRequired,
    ReturnDecision,
    ReturnItemTraits,
    ReturnNotApplicable,
    ReturnReason,
    SalesChannel,
    evaluate_return,
)
from emporio.ports import Clock, CustomerRepository, OrderRepository, ProductRepository

PAYMENT_LABEL = {
    "pix": "PIX",
    "boleto": "boleto",
    "debit": "débito",
    "credit_3x": "crédito em até 3x",
    "credit_6x": "crédito em até 6x",
    "credit_12x": "crédito em até 12x",
}


class VerifyCustomerInput(BaseModel):
    """Dados que o cliente informa para provar que o pedido é dele."""

    order_id: PositiveInt = Field(description="Número do pedido informado pelo cliente.")
    email: str | None = Field(default=None, description="E-mail do cadastro.")
    phone: str | None = Field(default=None, description="Telefone do cadastro, com DDD.")


class VerifyCustomerOutput(BaseModel):
    """Resultado da verificação; o ID do cliente fica só no estado do grafo."""

    verified: bool
    first_name: str | None = None
    message: str | None = None


class VerifyCustomer(Tool[VerifyCustomerInput, VerifyCustomerOutput]):
    """Confere pedido + e-mail ou telefone (LGPD e prevenção de BOLA)."""

    name = "verify_customer"
    description = (
        "Use quando o cliente informar número do pedido e e-mail OU telefone, para liberar "
        "dados de pedido. Falha de verificação não diz se o pedido existe. Máximo de 3 "
        "tentativas por sessão. Não use para dados de catálogo."
    )
    input_model = VerifyCustomerInput

    def __init__(self, orders: OrderRepository, customers: CustomerRepository) -> None:
        self._orders = orders
        self._customers = customers

    def run(self, args: VerifyCustomerInput, ctx: ToolContext) -> ToolResult[VerifyCustomerOutput]:
        """Compara o contato informado com o cadastro do dono do pedido."""
        if ctx.verification_attempts >= MAX_VERIFICATION_ATTEMPTS:
            return PolicyBlocked(
                "limite de 3 tentativas de verificação atingido", ("atendente humano",)
            )
        if not args.email and not args.phone:
            return Ambiguous("Peça o e-mail ou o telefone do cadastro.")
        order = self._orders.get(OrderId(args.order_id))
        customer = self._customers.get(order.customer_id) if order else None
        if customer is None or not _contact_matches(customer.email, customer.phone, args):
            return Ok(
                VerifyCustomerOutput(
                    verified=False,
                    message="Não consegui confirmar com esses dados. Peça para conferir.",
                )
            )
        return Ok(
            VerifyCustomerOutput(verified=True, first_name=customer.name.split()[0]),
            sources=(f"order:{args.order_id}",),
            state_updates={"verified_customer_id": customer.customer_id},
        )


def _contact_matches(email: Email, phone: Phone, args: VerifyCustomerInput) -> bool:
    try:
        if args.email and Email(args.email) == email:
            return True
        return bool(args.phone) and Phone(args.phone or "") == phone
    except InvalidValueError:
        return False


class OrderItemView(BaseModel):
    """Item do pedido."""

    name: str
    quantity: int


class OrderStatusView(BaseModel):
    """Status do pedido em PT-BR."""

    order_id: int
    status: str
    order_date: str
    total: str
    payment: str
    tracking_code: str | None = None
    estimated_delivery: str | None = None
    is_delayed: bool
    delay_note: str | None = None
    items: list[OrderItemView]


class GetOrderStatusInput(BaseModel):
    """Pedido consultado."""

    order_id: PositiveInt = Field(description="Número do pedido informado pelo cliente.")


class GetOrderStatus(Tool[GetOrderStatusInput, OrderStatusView]):
    """Status, rastreio e previsão de entrega de um pedido do cliente verificado."""

    name = "get_order_status"
    description = (
        "Use quando o cliente perguntar onde está, quando chega ou qual o status de um pedido. "
        "Requer cliente verificado. Não use para preço ou catálogo."
    )
    input_model = GetOrderStatusInput
    requires_verified_customer = True

    def __init__(self, orders: OrderRepository, products: ProductRepository, clock: Clock) -> None:
        self._orders = orders
        self._products = products
        self._clock = clock

    def run(self, args: GetOrderStatusInput, ctx: ToolContext) -> ToolResult[OrderStatusView]:
        """Devolve o status só se o pedido pertencer ao cliente verificado."""
        order = self._orders.get(OrderId(args.order_id))
        if order is None:
            return NotFound(ORDER_NOT_FOUND_HINT)
        if ctx.verified_customer_id is None or not order.belongs_to(ctx.verified_customer_id):
            return Unauthorized()
        today = self._clock.now().date()
        delayed = order.is_overdue(today)
        late = (
            (today - order.estimated_delivery).days if delayed and order.estimated_delivery else 0
        )
        items = [
            OrderItemView(name=self._name(i.product_id), quantity=i.quantity) for i in order.items
        ]
        return Ok(
            OrderStatusView(
                order_id=order.order_id.value,
                status=ORDER_STATUS_LABEL[order.status],
                order_date=day(order.order_date),
                total=brl(order.total),
                payment=PAYMENT_LABEL[order.payment_method.value],
                tracking_code=order.tracking_code,
                estimated_delivery=day(order.estimated_delivery)
                if order.estimated_delivery
                else None,
                is_delayed=delayed,
                delay_note=f"atrasado há {late} dia(s) em relação à previsão" if delayed else None,
                items=items,
            ),
            sources=(f"order:{order.order_id.value}",),
        )

    def _name(self, product_id: int) -> str:
        entry = self._products.get(product_id)
        return entry.product.name if entry else f"produto {product_id}"


class CheckReturnInput(BaseModel):
    """Pedido e motivo da devolução ou troca."""

    order_id: PositiveInt
    reason: Literal["regret", "defect", "preference"] = Field(
        description="regret = arrependimento; defect = defeito; preference = trocar por outro."
    )
    channel: Literal["online", "store"] | None = Field(
        default=None, description="Onde comprou. Se o cliente não disse, deixe vazio e pergunte."
    )


class ReturnView(BaseModel):
    """Decisão de devolução com datas já calculadas."""

    eligible: bool
    rule: str
    window: str | None = None
    deadline: str | None = None
    days_left: str | None = None
    requires: list[str] = Field(default_factory=list)
    refund: str | None = None
    freight: str | None = None
    estimate_note: str | None = None
    detail: str = ""


class CheckReturnEligibility(Tool[CheckReturnInput, ReturnView]):
    """Elegibilidade de arrependimento (§4.1), defeito (§4.2), preferência (§4.3) e §4.4."""

    name = "check_return_eligibility"
    description = (
        "Use quando o cliente quiser devolver, trocar ou reclamar de defeito de um pedido. "
        "Requer cliente verificado. Se o canal da compra (online ou loja) não foi dito, não "
        "invente: a ferramenta pedirá. Não use para status de entrega."
    )
    input_model = CheckReturnInput
    requires_verified_customer = True

    def __init__(self, orders: OrderRepository, products: ProductRepository, clock: Clock) -> None:
        self._orders = orders
        self._products = products
        self._clock = clock

    def run(self, args: CheckReturnInput, ctx: ToolContext) -> ToolResult[ReturnView]:
        """Aplica as políticas de §4 com o ``Clock`` injetado."""
        order = self._orders.get(OrderId(args.order_id))
        if order is None:
            return NotFound(ORDER_NOT_FOUND_HINT)
        if ctx.verified_customer_id is None or not order.belongs_to(ctx.verified_customer_id):
            return Unauthorized()
        outcome = evaluate_return(
            order,
            ReturnReason(args.reason),
            SalesChannel(args.channel) if args.channel else SalesChannel.UNKNOWN,
            self._clock.now().date(),
            self._traits(order),
        )
        sources = (f"order:{order.order_id.value}",)
        match outcome:
            case ChannelRequired(question=question):
                return Ambiguous(question, ("online", "na loja"))
            case ReturnNotApplicable(explanation=explanation, options=options):
                return PolicyBlocked(explanation, options, "4.1")
            case ReturnDecision():
                return Ok(_view(outcome), sources=(*sources, f"§{outcome.rule}"))
        raise AssertionError("ReturnOutcome inexaustivo")  # pragma: no cover

    def _traits(self, order: Order) -> list[ReturnItemTraits]:
        notes = normalize_text(order.notes or "")
        shared = ReturnItemTraits(
            personalized="personaliz" in notes or "sob encomenda" in notes,
            final_sale="venda final" in notes,
        )
        traits = []
        for item in order.items:
            entry = self._products.get(item.product_id)
            mouthpiece = bool(entry and "boquilha" in normalize_text(entry.product.name))
            traits.append(
                ReturnItemTraits(shared.personalized, shared.final_sale, mouthpiece=mouthpiece)
            )
        return traits or [shared]


def _view(decision: ReturnDecision) -> ReturnView:
    window = (
        f"dia {decision.day_of_window} de {decision.window_days} dias corridos"
        if decision.window_days is not None
        else None
    )
    return ReturnView(
        eligible=decision.eligible,
        rule=f"§{decision.rule}" if decision.rule != "warranty" else "§8.1 garantia legal",
        window=window,
        deadline=day(decision.deadline) if decision.deadline else None,
        days_left=_days_left(decision),
        requires=list(decision.requires),
        refund=decision.refund,
        freight=decision.freight,
        estimate_note=(
            "prazo contado da entrega estimada, pois o pedido não registra a data de recebimento"
            if decision.estimate_based
            else None
        ),
        detail=decision.detail,
    )


def _days_left(decision: ReturnDecision) -> str | None:
    if decision.days_left is None or not decision.eligible:
        return None
    return "último dia do prazo" if decision.days_left == 0 else f"restam {decision.days_left} dias"
