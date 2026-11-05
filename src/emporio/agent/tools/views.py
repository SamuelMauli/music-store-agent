"""Saídas compactas e já formatadas que as tools entregam ao LLM."""

from __future__ import annotations

from pydantic import BaseModel, Field

from emporio.agent.tools.formatting import AVAILABILITY_LABEL, brl, percent
from emporio.agent.tools.results import ProductRef
from emporio.domain.entities import CatalogEntry
from emporio.policies.payment import installment_options


class ProductCard(BaseModel):
    """Produto em lista: o suficiente para escolher."""

    product_id: int
    name: str
    price: str = Field(description="Preço que o cliente paga, já com promoção.")
    original_price: str | None = None
    discount: str | None = None
    availability: str

    @classmethod
    def from_entry(cls, entry: CatalogEntry) -> ProductCard:
        """Constrói o cartão a partir da linha do catálogo."""
        promoted = entry.discount_percent is not None
        return cls(
            product_id=entry.product.product_id,
            name=entry.product.name,
            price=brl(entry.effective_price),
            original_price=brl(entry.product.price) if promoted else None,
            discount=percent(entry.discount_percent) if entry.discount_percent else None,
            availability=AVAILABILITY_LABEL[entry.availability],
        )


class ProductDetail(ProductCard):
    """Produto com preço PIX, parcelamento e especificações.

    A descrição livre do cadastro não é exposta: pode contradizer as specs (D2).
    """

    category: str
    stock_quantity: int
    promotion: str | None = None
    pix_price: str | None = None
    pix_note: str | None = None
    max_installments: str
    specs: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_entry(cls, entry: CatalogEntry) -> ProductDetail:
        """Constrói o detalhe, com preço PIX e parcelas calculados pelo Policy-as-Code."""
        card = ProductCard.from_entry(entry)
        best = installment_options(entry.effective_price)[-1]
        return cls(
            **card.model_dump(),
            category=entry.category,
            stock_quantity=entry.product.stock_quantity,
            promotion=entry.promo_name,
            pix_price=brl(entry.pix_price) if entry.pix_price else None,
            pix_note=None if entry.pix_price else "PIX não acumula com promoção (§6.2)",
            max_installments=f"até {best.count}x de {brl(best.amount)} sem juros",
            specs={k: _spec(v) for k, v in entry.product.specs.items()},
        )


def _spec(value: str | int | bool) -> str:
    if isinstance(value, bool):
        return "sim" if value else "não"
    return str(value)


def ref_of(entry: CatalogEntry) -> ProductRef:
    """Referência curta do produto para a memória de "esse" e "o mais barato"."""
    return ProductRef(entry.product.product_id, entry.product.name, brl(entry.effective_price))
