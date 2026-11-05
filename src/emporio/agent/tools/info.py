"""Tools de políticas e dados da loja (contexto Políticas): frete, loja e manual."""

from __future__ import annotations

from collections.abc import Collection
from datetime import timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from emporio.agent.tools.base import Tool, ToolContext
from emporio.agent.tools.formatting import brl, when
from emporio.agent.tools.results import Ambiguous, NotFound, Ok, ToolResult
from emporio.domain.calendar import DEFAULT_CALENDAR, HolidayCalendar
from emporio.domain.value_objects import Money
from emporio.policies.shipping import (
    FREE_SHIPPING_ABOVE,
    METRO_DELIVERY,
    METRO_FLAT_FEE,
    CityRequired,
    MetroShipping,
    OutsideShipping,
    estimate_shipping,
    is_metro_city,
    needs_special_freight,
)
from emporio.policies.store import (
    CONTACTS,
    SPECIAL_DATES_NOTE,
    is_store_open,
    next_opening,
    opening_hours_text,
)
from emporio.ports import Clock, PolicyRetriever, ProductRepository

POLICY_TOP_K = 3
NEXT_OPENING_SCAN = 15


class ShippingModeView(BaseModel):
    """Modalidade de envio para fora da região metropolitana."""

    name: str
    deadline: str
    tracking: bool
    insured: bool


class ShippingView(BaseModel):
    """Estimativa de frete."""

    region: Literal["metropolitana", "outras_cidades"]
    fee: str | None = None
    delivery: str | None = None
    rule: str | None = None
    modes: list[ShippingModeView] = Field(default_factory=list)
    note: str | None = None


class EstimateShippingInput(BaseModel):
    """Cidade e valor do pedido."""

    city: str = Field(description="Cidade de entrega informada pelo cliente.")
    order_total: Decimal | None = Field(default=None, ge=0, description="Valor do pedido em reais.")
    product_ids: list[int] | None = Field(
        default=None, max_length=10, description="Produtos do pedido, para somar e checar porte."
    )


class EstimateShipping(Tool[EstimateShippingInput, ShippingView]):
    """Frete por cidade (§5.1 e §5.2)."""

    name = "estimate_shipping"
    description = (
        "Use quando o cliente perguntar frete, prazo de entrega ou envio para uma cidade. "
        "Região metropolitana de Campo Grande: grátis acima de R$ 500,00, senão R$ 35,00. "
        "Outras cidades: modalidades e prazos, sem valor (nunca invente cotação). "
        "Não use para rastrear pedido."
    )
    input_model = EstimateShippingInput

    def __init__(
        self,
        products: ProductRepository,
        metro_cities: Collection[str],
    ) -> None:
        self._products = products
        self._metro = list(metro_cities)

    def run(self, args: EstimateShippingInput, ctx: ToolContext) -> ToolResult[ShippingView]:
        """Escolhe a regra de frete pela cidade, sem inventar valor fora da região."""
        total, large = self._basket(args)
        if not args.city.strip():
            return Ambiguous(CityRequired().question)
        if is_metro_city(args.city, self._metro) and total is None:
            return Ok(
                ShippingView(
                    region="metropolitana",
                    delivery=METRO_DELIVERY,
                    rule=(
                        f"frete grátis acima de {brl(FREE_SHIPPING_ABOVE)}; "
                        f"abaixo disso, {brl(METRO_FLAT_FEE)}"
                    ),
                ),
                sources=("§5.1",),
            )
        estimate = estimate_shipping(total or Money.of(0), args.city, self._metro, large_item=large)
        match estimate:
            case MetroShipping(fee=fee, free=free, delivery=delivery):
                return Ok(
                    ShippingView(
                        region="metropolitana",
                        fee="grátis" if free else brl(fee),
                        delivery=delivery,
                    ),
                    sources=("§5.1",),
                )
            case OutsideShipping(modes=modes, quote_note=note, special_freight=special):
                return Ok(
                    ShippingView(
                        region="outras_cidades",
                        modes=[
                            ShippingModeView(
                                name=m.name,
                                deadline=m.deadline_text,
                                tracking=m.tracking,
                                insured=m.insured,
                            )
                            for m in modes
                        ],
                        note=note
                        + (
                            "; item de grande porte tem frete especial com cotação individual"
                            if special
                            else ""
                        ),
                    ),
                    sources=("§5.2",),
                )
            case CityRequired(question=question):
                return Ambiguous(question)
        raise AssertionError("ShippingEstimate inexaustivo")  # pragma: no cover

    def _basket(self, args: EstimateShippingInput) -> tuple[Money | None, bool]:
        if args.order_total is not None:
            return Money.of(args.order_total), False
        if not args.product_ids:
            return None, False
        entries = [e for pid in args.product_ids if (e := self._products.get(pid))]
        total = sum((e.effective_price for e in entries), Money.of(0))
        return total, any(needs_special_freight(e.product.name) for e in entries)


class StoreInfoInput(BaseModel):
    """Assunto sobre a loja."""

    topic: Literal["address", "hours", "contacts", "open_now"] = Field(
        description="address, hours, contacts ou open_now (está aberta agora?)."
    )


class StoreInfoView(BaseModel):
    """Dados da loja."""

    topic: str
    address: str | None = None
    hours: str | None = None
    phone: str | None = None
    whatsapp: str | None = None
    email: str | None = None
    open_now: bool | None = None
    now: str | None = None
    next_opening: str | None = None


class GetStoreInfo(Tool[StoreInfoInput, StoreInfoView]):
    """Endereço, contatos, horário e se está aberta agora."""

    name = "get_store_info"
    description = (
        "Use para endereço, telefone/WhatsApp, e-mail, horário de funcionamento ou 'está aberto "
        "agora?'. Não use para políticas de troca, frete ou pagamento."
    )
    input_model = StoreInfoInput

    def __init__(self, clock: Clock, calendar: HolidayCalendar = DEFAULT_CALENDAR) -> None:
        self._clock = clock
        self._calendar = calendar

    def run(self, args: StoreInfoInput, ctx: ToolContext) -> ToolResult[StoreInfoView]:
        """Monta a resposta do tópico a partir de §1.2, §2 e §7."""
        now = self._clock.now()
        today = now.date()
        # feriados nacionais fecham a loja (§2); janela cobre a busca de 14 dias de next_opening
        holidays = self._calendar.holidays_between(today, today + timedelta(days=NEXT_OPENING_SCAN))
        match args.topic:
            case "address":
                return Ok(
                    StoreInfoView(topic="address", address=CONTACTS.address), sources=("§1.2",)
                )
            case "contacts":
                return Ok(
                    StoreInfoView(
                        topic="contacts",
                        phone=CONTACTS.phone,
                        whatsapp=CONTACTS.whatsapp,
                        email=CONTACTS.email,
                    ),
                    sources=("§1.2", "§7"),
                )
            case "hours":
                return Ok(StoreInfoView(topic="hours", hours=opening_hours_text()), sources=("§2",))
            case "open_now":
                is_open = is_store_open(now, holidays)
                return Ok(
                    StoreInfoView(
                        topic="open_now",
                        open_now=is_open,
                        now=when(now),
                        next_opening=None if is_open else when(next_opening(now, holidays)),
                        hours=SPECIAL_DATES_NOTE,
                    ),
                    sources=("§2",),
                )
        raise AssertionError("tópico inexaustivo")  # pragma: no cover


class SearchPoliciesInput(BaseModel):
    """Pergunta sobre regras da loja."""

    query: str = Field(min_length=3, max_length=300, description="Pergunta sobre política da loja.")


class PolicySnippetView(BaseModel):
    """Trecho citável do manual."""

    section: str
    title: str
    text: str


class SearchPoliciesOutput(BaseModel):
    """Trechos relevantes do manual."""

    sections: list[PolicySnippetView]


class SearchPolicies(Tool[SearchPoliciesInput, SearchPoliciesOutput]):
    """Busca no manual de políticas (garantia, trocas, promoções, privacidade...)."""

    name = "search_policies"
    description = (
        "Use para regras da loja que as outras ferramentas não cobrem: garantia (§8), trocas "
        "(§4), promoções (§6), privacidade (§9). Devolve trechos com o número da seção para "
        "citar. Se nada cobrir o caso, diga que vai confirmar com a equipe."
    )
    input_model = SearchPoliciesInput

    def __init__(self, retriever: PolicyRetriever) -> None:
        self._retriever = retriever

    def run(self, args: SearchPoliciesInput, ctx: ToolContext) -> ToolResult[SearchPoliciesOutput]:
        """Recupera os trechos pela estratégia configurada."""
        snippets = self._retriever.retrieve(args.query, k=POLICY_TOP_K)
        if not snippets:
            return NotFound("O manual não cobre esse assunto. Ofereça abrir um chamado.")
        return Ok(
            SearchPoliciesOutput(
                sections=[
                    PolicySnippetView(section=f"§{s.section_id}", title=s.title, text=s.text)
                    for s in snippets
                ]
            ),
            sources=tuple(f"§{s.section_id}" for s in snippets),
        )
