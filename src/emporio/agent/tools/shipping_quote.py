"""Tool ``quote_shipping``: cotação de frete por CEP e peso (leitura pura, sem efeito colateral)."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from emporio.agent.tools.base import Tool, ToolContext
from emporio.agent.tools.formatting import brl, day
from emporio.agent.tools.results import Ambiguous, NotFound, Ok, PolicyBlocked, ToolResult
from emporio.domain.calendar import DEFAULT_CALENDAR, HolidayCalendar
from emporio.domain.cep import format_cep, normalize_cep, region_of_uf, uf_of_cep
from emporio.domain.errors import InvalidValueError
from emporio.domain.value_objects import Money
from emporio.policies.shipping import FREE_SHIPPING_ABOVE, METRO_FLAT_FEE, needs_special_freight
from emporio.policies.shipping_quote import (
    MetroQuote,
    OutsideQuote,
    OverweightQuote,
    UnknownCep,
    quote_shipping,
)
from emporio.ports import Clock, ProductRepository

WEIGHT_QUESTION = (
    "Qual o peso do pedido embalado, em kg? O catálogo da loja não tem o peso dos produtos, "
    "então vale o peso que o cliente informar."
)
WEIGHT_NOTE = "peso informado pelo cliente (o catálogo da loja não traz peso dos produtos)"
ESTIMATE_NOTE = (
    "ESTIMATIVA da loja, não é o preço oficial dos Correios nem da Jadlog; o valor final é "
    "calculado no fechamento. Datas contam dias úteis (sem feriados nacionais) supondo "
    "postagem hoje."
)
SPECIAL_NOTE = "item de grande porte: pode exigir frete especial com cotação individual"


class QuoteShippingInput(BaseModel):
    """CEP de destino e peso (ou valor do pedido, na região metropolitana)."""

    model_config = ConfigDict(extra="forbid")

    cep: str = Field(
        pattern=r"^\d{5}-?\d{3}$", description="CEP de destino com 8 dígitos, ex.: 79002-000."
    )
    weight_kg: Decimal | None = Field(
        default=None,
        gt=0,
        le=1000,
        description="Peso total em kg informado pelo cliente; obrigatório fora de Campo Grande.",
    )
    order_total: Decimal | None = Field(
        default=None, ge=0, description="Valor do pedido em reais (frete grátis acima de R$ 500)."
    )
    product_ids: list[int] | None = Field(
        default=None, max_length=10, description="Produtos do pedido: somam o valor e checam porte."
    )

    @field_validator("cep", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class QuoteOptionView(BaseModel):
    """Uma modalidade estimada."""

    mode: str
    price: str
    deadline: str
    earliest: str
    latest: str
    tracking: bool
    insured: bool


class QuoteView(BaseModel):
    """Cotação de frete já formatada em PT-BR."""

    cep: str
    uf: str
    region: str
    estimate: bool
    fee: str | None = None
    delivery: str | None = None
    earliest: str | None = None
    latest: str | None = None
    rule: str | None = None
    weight_kg: str | None = None
    weight_note: str | None = None
    options: list[QuoteOptionView] = Field(default_factory=list)
    note: str | None = None


class QuoteShipping(Tool[QuoteShippingInput, QuoteView]):
    """Cotação de frete por CEP e peso (§5.1 na metrópole; estimativa fora dela)."""

    name = "quote_shipping"
    description = (
        "Use quando o cliente der um CEP (8 dígitos) e quiser valor e prazo de frete ou a data "
        "prevista de chegada. Campo Grande: regra do manual (grátis acima de R$ 500,00, senão "
        "R$ 35,00). Outros CEPs: exige o peso em kg dito pelo cliente (o catálogo não tem peso) "
        "e devolve ESTIMATIVA por PAC, SEDEX e Jadlog com datas previstas (estimate=true; nunca "
        "apresente como preço oficial). Só tem a cidade, sem CEP? Use estimate_shipping."
    )
    input_model = QuoteShippingInput

    def __init__(
        self,
        products: ProductRepository,
        clock: Clock,
        calendar: HolidayCalendar = DEFAULT_CALENDAR,
    ) -> None:
        self._products = products
        self._clock = clock
        self._calendar = calendar

    def run(self, args: QuoteShippingInput, ctx: ToolContext) -> ToolResult[QuoteView]:
        """Valida o CEP, resolve UF e cota pela regra do manual ou pela tabela de estimativas."""
        try:
            cep = normalize_cep(args.cep)
        except InvalidValueError as exc:
            return Ambiguous(str(exc))
        total, large = self._basket(args)
        try:
            quote = quote_shipping(
                cep,
                args.weight_kg,
                self._clock.now().date(),
                order_total=total,
                calendar=self._calendar,
            )
        except InvalidValueError:
            return Ambiguous(WEIGHT_QUESTION)
        sources = ("§5.1" if isinstance(quote, MetroQuote) else "§5.2",)
        uf = uf_of_cep(cep) or ""
        match quote:
            case UnknownCep():
                return NotFound("CEP fora das faixas dos Correios. Peça para conferir o CEP.")
            case OverweightQuote(max_kg=max_kg):
                return PolicyBlocked(
                    f"acima de {max_kg} kg o frete é especial, com cotação individual da equipe",
                    ("falar com a equipe pelo WhatsApp da loja",),
                    "5.2",
                )
            case MetroQuote():
                return Ok(self._metro_view(quote, cep, uf), sources=sources)
            case OutsideQuote():
                return Ok(self._outside_view(quote, cep, args.weight_kg, large), sources=sources)
        raise AssertionError("ShippingQuote inexaustivo")  # pragma: no cover

    def _basket(self, args: QuoteShippingInput) -> tuple[Money | None, bool]:
        if args.order_total is not None:
            return Money.of(args.order_total), False
        if not args.product_ids:
            return None, False
        entries = [e for pid in args.product_ids if (e := self._products.get(pid))]
        total = sum((e.effective_price for e in entries), Money.of(0))
        return total, any(needs_special_freight(e.product.name) for e in entries)

    @staticmethod
    def _metro_view(quote: MetroQuote, cep: str, uf: str) -> QuoteView:
        return QuoteView(
            cep=format_cep(cep),
            uf=uf,
            region="metropolitana",
            estimate=False,  # regra oficial do manual, não estimativa
            fee=None if quote.fee is None else "grátis" if quote.free else brl(quote.fee),
            delivery=quote.delivery,
            earliest=day(quote.earliest),
            latest=day(quote.latest),
            rule=(
                f"frete grátis acima de {brl(FREE_SHIPPING_ABOVE)}; abaixo disso, "
                f"{brl(METRO_FLAT_FEE)}"
                if quote.fee is None
                else None
            ),
        )

    @staticmethod
    def _outside_view(
        quote: OutsideQuote, cep: str, weight_kg: Decimal | None, large: bool
    ) -> QuoteView:
        return QuoteView(
            cep=format_cep(cep),
            uf=quote.uf,
            region=region_of_uf(quote.uf),
            estimate=True,
            weight_kg=f"{weight_kg:f}" if weight_kg is not None else None,
            weight_note=WEIGHT_NOTE,
            options=[
                QuoteOptionView(
                    mode=o.mode,
                    price=brl(o.price),
                    deadline=f"{o.min_days} a {o.max_days} dias úteis",
                    earliest=day(o.earliest),
                    latest=day(o.latest),
                    tracking=o.tracking,
                    insured=o.insured,
                )
                for o in quote.options
            ],
            note=ESTIMATE_NOTE + (f"; {SPECIAL_NOTE}" if large else ""),
        )
