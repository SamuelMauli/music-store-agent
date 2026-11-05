"""Tools de promoção e pagamento (Policy-as-Code §3, §6.2)."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field

from emporio.agent.tools.base import Tool, ToolContext
from emporio.agent.tools.categories import resolve_category
from emporio.agent.tools.formatting import brl, percent
from emporio.agent.tools.results import NotFound, Ok, ToolResult
from emporio.agent.tools.views import ref_of
from emporio.domain.entities import CatalogEntry
from emporio.domain.value_objects import Money
from emporio.policies.payment import (
    COMBINED_PAYMENT_THRESHOLD,
    combined_payment_allowed,
    installment_options,
    pix_price,
)
from emporio.ports import ProductRepository, PromotionRepository


class PromotionView(BaseModel):
    """Promoção ativa com preço original, percentual e preço final (§6.2)."""

    product_id: int
    name: str
    campaign: str
    original_price: str
    discount: str
    final_price: str


class ListPromotionsInput(BaseModel):
    """Filtro opcional por categoria."""

    category: str | None = Field(default=None, description="Categoria em texto livre.")


class ListPromotionsOutput(BaseModel):
    """Promoções ativas."""

    promotions: list[PromotionView]
    total: int
    note: str = "PIX não acumula com preço promocional (§6.2)."


class ListActivePromotions(Tool[ListPromotionsInput, ListPromotionsOutput]):
    """Promoções vigentes hoje."""

    name = "list_active_promotions"
    description = (
        "Use quando o cliente perguntar por promoções, ofertas ou descontos vigentes, ou se um "
        "produto está em promoção. Mostra apenas promoções ativas, com preço original, % e "
        "preço final. Nunca prometa desconto que esta ferramenta não listou."
    )
    input_model = ListPromotionsInput

    def __init__(self, promotions: PromotionRepository, products: ProductRepository) -> None:
        self._promotions = promotions
        self._products = products

    def run(self, args: ListPromotionsInput, ctx: ToolContext) -> ToolResult[ListPromotionsOutput]:
        """Cruza promoções ativas com o catálogo para obter preços."""
        wanted = resolve_category(args.category) if args.category else None
        entries: list[CatalogEntry] = []
        for promo in self._promotions.active():
            entry = self._products.get(promo.product_id)
            if entry is not None and (wanted is None or entry.product.category_id == wanted):
                entries.append(entry)
        views = [
            PromotionView(
                product_id=e.product.product_id,
                name=e.product.name,
                campaign=e.promo_name or "",
                original_price=brl(e.product.price),
                discount=percent(e.discount_percent or Decimal(0)),
                final_price=brl(e.effective_price),
            )
            for e in entries
        ]
        return Ok(
            ListPromotionsOutput(promotions=views, total=len(views)),
            sources=tuple(f"product:{e.product.product_id}" for e in entries),
            refs=tuple(ref_of(e) for e in entries),
        )


class SimulatePaymentInput(BaseModel):
    """Valor ou produtos a simular."""

    amount: Decimal | None = Field(
        default=None,
        gt=0,
        le=1_000_000,
        description="Valor total em reais, se não houver produtos.",
    )
    product_ids: list[int] | None = Field(
        default=None, min_length=1, max_length=10, description="IDs dos produtos da compra."
    )


class SimulatePaymentOutput(BaseModel):
    """Formas de pagamento para o total."""

    total: str
    pix_total: str | None = None
    pix_note: str | None = None
    installments: list[str]
    max_installments: str
    combined_payment: str


class SimulatePayment(Tool[SimulatePaymentInput, SimulatePaymentOutput]):
    """PIX, parcelas válidas e combinação de pagamentos."""

    name = "simulate_payment"
    description = (
        "Use para calcular PIX (5% off sobre preço de tabela), parcelas sem juros válidas e se "
        "pode combinar formas de pagamento (acima de R$ 2.000,00). Informe product_ids (de "
        "get_product) OU amount. Não calcule parcelas de cabeça: use esta ferramenta."
    )
    input_model = SimulatePaymentInput

    def __init__(self, products: ProductRepository) -> None:
        self._products = products

    def run(
        self, args: SimulatePaymentInput, ctx: ToolContext
    ) -> ToolResult[SimulatePaymentOutput]:
        """Aplica §3, §3.1 e §6.2 ao total."""
        if args.product_ids:
            entries = [self._products.get(pid) for pid in args.product_ids]
            if any(e is None for e in entries):
                return NotFound("Algum dos códigos informados não existe no catálogo.")
            found = [e for e in entries if e is not None]
            total = sum((e.effective_price for e in found), Money.of(0))
            pix_total, note = _pix_for(found)
            refs = tuple(ref_of(e) for e in found)
        elif args.amount is not None:
            total = Money.of(args.amount)
            pix_total, note = pix_price(total, has_promotion=False), None
            refs = ()
        else:
            return NotFound("Informe os produtos ou o valor para simular.")
        options = installment_options(total)
        best = options[-1]
        combined = combined_payment_allowed(total)
        return Ok(
            SimulatePaymentOutput(
                total=brl(total),
                pix_total=brl(pix_total) if pix_total else None,
                pix_note=note,
                installments=[f"{o.count}x de {brl(o.amount)}" for o in options],
                max_installments=f"{best.count}x de {brl(best.amount)}",
                combined_payment=(
                    f"permitida (compra acima de {brl(COMBINED_PAYMENT_THRESHOLD)})"
                    if combined
                    else f"não permitida (só acima de {brl(COMBINED_PAYMENT_THRESHOLD)})"
                ),
            ),
            sources=("§3", "§3.1", "§6.2", *(f"product:{r.product_id}" for r in refs)),
            refs=refs,
        )


def _pix_for(entries: list[CatalogEntry]) -> tuple[Money | None, str | None]:
    """PIX só sobre itens sem promoção; itens promocionais pagam o preço promocional."""
    if all(e.discount_percent is not None for e in entries):
        return None, "PIX não acumula com promoção (§6.2)"
    total = Money.of(0)
    mixed = False
    for e in entries:
        if e.discount_percent is None:
            discounted = pix_price(e.effective_price, has_promotion=False)
            total = total + (discounted or e.effective_price)
        else:
            total = total + e.effective_price
            mixed = True
    return total, "itens em promoção ficam sem desconto de PIX (§6.2)" if mixed else None
