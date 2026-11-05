"""Tools de catálogo: busca, detalhe, comparação e alternativas (contexto Catálogo)."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from emporio.agent.tools.base import Tool, ToolContext
from emporio.agent.tools.categories import CATEGORY_NAMES, resolve_category
from emporio.agent.tools.results import (
    Ambiguous,
    NotFound,
    Ok,
    ProductRef,
    ToolResult,
)
from emporio.agent.tools.views import ProductCard, ProductDetail, ref_of
from emporio.domain.entities import CatalogEntry
from emporio.domain.lookup import Ambiguous as NameAmbiguous
from emporio.domain.lookup import Found
from emporio.domain.specifications import (
    HasElectronics,
    HasStringType,
    InCategory,
    InStock,
    MatchAll,
    PriceAtLeast,
    PriceAtMost,
    Specification,
)
from emporio.domain.text import normalize_text
from emporio.domain.value_objects import Money
from emporio.ports import ProductRepository

MAX_RESULTS = 5
MAX_SIMILAR = 3
ALTERNATIVE_BAND = Decimal("0.30")


class SearchProductsInput(BaseModel):
    """Filtros de busca no catálogo."""

    category: str | None = Field(
        default=None, description="Categoria em texto livre: violão, guitarra, baixo, ukulele..."
    )
    min_price: Decimal | None = Field(default=None, ge=0, description="Preço mínimo em reais.")
    max_price: Decimal | None = Field(default=None, gt=0, description="Preço máximo em reais.")
    strings: Literal["nylon", "aço", "steel"] | None = Field(
        default=None, description="Tipo de corda (violão e ukulele)."
    )
    electric: bool | None = Field(default=None, description="Com captação eletrônica?")
    query: str | None = Field(default=None, description="Palavras do nome, ex.: marca ou modelo.")
    include_unavailable: bool = Field(
        default=False, description="Incluir sem estoque e descontinuados (padrão: não)."
    )
    sort: Literal["price_asc", "price_desc"] = "price_asc"
    limit: int = Field(default=MAX_RESULTS, ge=1, le=MAX_RESULTS)


class SearchProductsOutput(BaseModel):
    """Resultado da busca."""

    items: list[ProductCard]
    total_matches: int
    note: str | None = None


class SearchProducts(Tool[SearchProductsInput, SearchProductsOutput]):
    """Busca produtos por categoria, faixa de preço, cordas, eletrônica ou nome."""

    name = "search_products"
    description = (
        "Use para listar ou recomendar produtos por categoria, faixa de preço, tipo de corda, "
        "captação eletrônica ou nome. Devolve até 5 itens do mais barato ao mais caro e o total "
        "de resultados (total_matches). Só traz produtos à venda, salvo include_unavailable. "
        "Não use para o preço de um modelo específico (use get_product)."
    )
    input_model = SearchProductsInput

    def __init__(self, products: ProductRepository) -> None:
        self._products = products

    def run(self, args: SearchProductsInput, ctx: ToolContext) -> ToolResult[SearchProductsOutput]:
        """Compila os filtros em uma Specification e consulta o catálogo."""
        spec: Specification = MatchAll()
        if args.category:
            category_id = resolve_category(args.category)
            if category_id is None:
                return NotFound("Categoria não reconhecida. A loja vende " + _categories_text())
            spec = spec & InCategory(category_id)
        spec = _price_filters(spec, args)
        if args.strings:
            spec = spec & HasStringType("steel" if args.strings in ("aço", "steel") else "nylon")
        if args.electric is not None:
            spec = spec & HasElectronics(args.electric)
        if not args.include_unavailable:
            spec = spec & InStock()
        found = self._products.find(spec)
        if args.query:
            tokens = normalize_text(args.query).split()
            found = [e for e in found if all(t in normalize_text(e.product.name) for t in tokens)]
        if args.sort == "price_desc":
            found = list(reversed(found))
        shown = found[: args.limit]
        note = None if found else "Nenhum produto com esses filtros no catálogo."
        return Ok(
            SearchProductsOutput(
                items=[ProductCard.from_entry(e) for e in shown],
                total_matches=len(found),
                note=note,
            ),
            sources=tuple(f"product:{e.product.product_id}" for e in shown),
            refs=tuple(ref_of(e) for e in shown),
        )


def _price_filters(spec: Specification, args: SearchProductsInput) -> Specification:
    if args.min_price is not None:
        spec = spec & PriceAtLeast(Money.of(args.min_price))
    if args.max_price is not None:
        spec = spec & PriceAtMost(Money.of(args.max_price))
    return spec


def _categories_text() -> str:
    return ", ".join(name for cid, name in CATEGORY_NAMES.items() if cid in (1, 2, 3, 4, 5, 9))


class GetProductInput(BaseModel):
    """Identificação do produto."""

    name_or_id: str = Field(
        description="Nome ou ID do produto, mesmo aproximado: 'GD20', 'takamine gd 20', '95'."
    )


class GetProduct(Tool[GetProductInput, ProductDetail]):
    """Detalhe de um produto com preço, estoque, promoção, PIX e parcelas."""

    name = "get_product"
    description = (
        "Use quando o cliente perguntar preço, estoque, promoção, PIX, parcelas ou ficha técnica "
        "de UM modelo, mesmo com nome aproximado. Se houver mais de um candidato, devolve "
        "ambiguous: pergunte qual. Não use para listar vários produtos (use search_products)."
    )
    input_model = GetProductInput

    def __init__(self, products: ProductRepository) -> None:
        self._products = products

    def run(self, args: GetProductInput, ctx: ToolContext) -> ToolResult[ProductDetail]:
        """Resolve por ID ou por nome com fuzzy match."""
        text = args.name_or_id.strip()
        if text.isdigit():
            entry = self._products.get(int(text))
            return _detail(entry) if entry else NotFound("Não há produto com esse código.")
        match self._products.resolve_name(text):
            case Found(entry=entry):
                return _detail(entry)
            case NameAmbiguous(options=options):
                return Ambiguous(
                    "Qual destes modelos o cliente quis?",
                    tuple(f"{e.product.name} ({e.product.product_id})" for e in options),
                )
            case _:
                return NotFound(
                    "Não encontrei esse modelo no catálogo.",
                    similar=self._similar(text),
                )

    def _similar(self, text: str) -> tuple[ProductRef, ...]:
        tokens = [t for t in normalize_text(text).split() if len(t) >= 3]
        close = [
            e
            for e in self._products.find(InStock())
            if any(t in normalize_text(e.product.name) for t in tokens)
        ]
        return tuple(ref_of(e) for e in close[:MAX_SIMILAR])


def _detail(entry: CatalogEntry) -> ToolResult[ProductDetail]:
    return Ok(
        ProductDetail.from_entry(entry),
        sources=(f"product:{entry.product.product_id}",),
        refs=(ref_of(entry),),
    )


class CompareProductsInput(BaseModel):
    """Produtos a comparar."""

    product_ids: list[int] = Field(min_length=2, max_length=3, description="IDs de 2 a 3 produtos.")


class CompareProductsOutput(BaseModel):
    """Comparação lado a lado."""

    products: list[ProductDetail]
    same_category: bool
    note: str | None = None


class CompareProducts(Tool[CompareProductsInput, CompareProductsOutput]):
    """Tabela de especificações e preços de 2 a 3 produtos."""

    name = "compare_products"
    description = (
        "Use para comparar 2 ou 3 produtos já identificados (por product_id) lado a lado. "
        "Avisa quando são de categorias diferentes. Não use para descobrir produtos."
    )
    input_model = CompareProductsInput

    def __init__(self, products: ProductRepository) -> None:
        self._products = products

    def run(
        self, args: CompareProductsInput, ctx: ToolContext
    ) -> ToolResult[CompareProductsOutput]:
        """Busca cada produto e monta a tabela."""
        entries = [self._products.get(pid) for pid in dict.fromkeys(args.product_ids)]
        if any(e is None for e in entries) or len(entries) < 2:
            return NotFound("Algum dos códigos informados não existe no catálogo.")
        found = [e for e in entries if e is not None]
        same = len({e.product.category_id for e in found}) == 1
        return Ok(
            CompareProductsOutput(
                products=[ProductDetail.from_entry(e) for e in found],
                same_category=same,
                note=None if same else "Categorias diferentes: a comparação é limitada.",
            ),
            sources=tuple(f"product:{e.product.product_id}" for e in found),
            refs=tuple(ref_of(e) for e in found),
        )


class FindAlternativesInput(BaseModel):
    """Produto de referência."""

    product_id: int
    k: int = Field(default=3, ge=1, le=3)


class FindAlternativesOutput(BaseModel):
    """Alternativas disponíveis."""

    items: list[ProductCard]
    reference_price: str
    note: str | None = None


class FindAlternatives(Tool[FindAlternativesInput, FindAlternativesOutput]):
    """Alternativas em estoque, da mesma categoria e a ±30% do preço (§7.3)."""

    name = "find_alternatives"
    description = (
        "Use quando um produto estiver sem estoque, descontinuado ou caro demais, para sugerir "
        "até 3 alternativas disponíveis da mesma categoria, a ±30% do preço (§7.3). "
        "Não use quando o produto já está disponível e o cliente só quer o preço."
    )
    input_model = FindAlternativesInput

    def __init__(self, products: ProductRepository) -> None:
        self._products = products

    def run(
        self, args: FindAlternativesInput, ctx: ToolContext
    ) -> ToolResult[FindAlternativesOutput]:
        """Filtra por categoria, estoque e faixa; ordena pela proximidade de preço."""
        base = self._products.get(args.product_id)
        if base is None:
            return NotFound("Não há produto com esse código.")
        reference = base.effective_price
        spec = (
            InCategory(base.product.category_id)
            & InStock()
            & PriceAtLeast(reference.times(1 - ALTERNATIVE_BAND))
            & PriceAtMost(reference.times(1 + ALTERNATIVE_BAND))
        )
        candidates = [
            e for e in self._products.find(spec) if e.product.product_id != args.product_id
        ]
        candidates.sort(
            key=lambda e: (abs(e.effective_price.amount - reference.amount), e.product.product_id)
        )
        chosen = candidates[: args.k]
        return Ok(
            FindAlternativesOutput(
                items=[ProductCard.from_entry(e) for e in chosen],
                reference_price=reference.format_brl(),
                note=None if chosen else "Sem alternativas disponíveis nessa faixa de preço.",
            ),
            sources=tuple(f"product:{e.product.product_id}" for e in chosen),
            refs=tuple(ref_of(e) for e in chosen),
        )
