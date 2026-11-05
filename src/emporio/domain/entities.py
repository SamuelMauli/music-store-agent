"""Entidades do domínio: catálogo, pedidos, clientes e promoções."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from emporio.domain.enums import Availability, OrderStatus, PaymentMethod, ProductStatus
from emporio.domain.errors import InvalidValueError
from emporio.domain.value_objects import Email, Money, OrderId, Phone

LOW_STOCK_THRESHOLD = 2

SpecValue = str | int | bool


def availability_of(status: ProductStatus, stock_quantity: int) -> Availability:
    """Deriva a disponibilidade; o status prevalece sobre o estoque.

    Um produto descontinuado com estoque (caso real do produto 136) continua
    indisponível para oferta.

    Args:
        status: Situação comercial do cadastro.
        stock_quantity: Unidades em estoque.

    Returns:
        A disponibilidade, idêntica à regra da view gold ``v_catalog``.
    """
    if status is ProductStatus.DISCONTINUED:
        return Availability.DISCONTINUED
    if status is ProductStatus.COMING_SOON:
        return Availability.COMING_SOON
    if stock_quantity == 0:
        return Availability.OUT_OF_STOCK
    if stock_quantity <= LOW_STOCK_THRESHOLD:
        return Availability.LOW_STOCK
    return Availability.AVAILABLE


@dataclass(frozen=True, slots=True)
class Product:
    """Produto do catálogo.

    Attributes:
        product_id: Identificador do produto.
        name: Nome comercial.
        category_id: Categoria do produto.
        description: Texto livre; pode contradizer ``specs``, que prevalecem.
        price: Preço de tabela.
        stock_quantity: Unidades em estoque.
        status: Situação comercial.
        specs: Especificações estruturadas (fonte confiável).
        created_at: Data de cadastro.
    """

    product_id: int
    name: str
    category_id: int
    description: str
    price: Money
    stock_quantity: int
    status: ProductStatus
    specs: Mapping[str, SpecValue] = field(default_factory=dict)
    created_at: date | None = None

    def __post_init__(self) -> None:
        if self.stock_quantity < 0:
            raise InvalidValueError("estoque não pode ser negativo")
        if self.price.amount <= 0:
            raise InvalidValueError("preço deve ser positivo")

    @property
    def availability(self) -> Availability:
        """Disponibilidade derivada de status e estoque."""
        return availability_of(self.status, self.stock_quantity)


@dataclass(frozen=True, slots=True)
class Promotion:
    """Promoção sobre um produto (sem datas de vigência: vale a flag ``is_active``).

    Attributes:
        promotion_id: Identificador da promoção.
        product_id: Produto em promoção.
        discount_percent: Desconto em pontos percentuais (0 < p <= 100).
        description: Nome da campanha.
        is_active: Se a promoção está vigente hoje.
    """

    promotion_id: int
    product_id: int
    discount_percent: Decimal
    description: str
    is_active: bool

    def __post_init__(self) -> None:
        if not Decimal(0) < self.discount_percent <= Decimal(100):
            raise InvalidValueError("desconto deve estar entre 0 (exclusivo) e 100")


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Linha da view gold ``v_catalog``: produto já com preço efetivo e disponibilidade.

    Attributes:
        product: O produto.
        category: Nome da categoria.
        effective_price: Preço após promoção ativa (o que o cliente paga).
        pix_price: Preço no PIX (5% off); ``None`` quando há promoção, pois o PIX
            não acumula (§6.2).
        discount_percent: Desconto da promoção ativa, se houver.
        promo_name: Campanha da promoção ativa, se houver.
    """

    product: Product
    category: str
    effective_price: Money
    pix_price: Money | None = None
    discount_percent: Decimal | None = None
    promo_name: str | None = None

    @property
    def availability(self) -> Availability:
        """Disponibilidade do produto."""
        return self.product.availability


@dataclass(frozen=True, slots=True)
class Customer:
    """Cliente. Contatos são value objects mascarados em logs.

    Attributes:
        customer_id: Identificador do cliente.
        name: Nome completo.
        phone: Telefone.
        email: E-mail (não é único: caso real dos clientes 11 e 43).
        city: Cidade.
    """

    customer_id: int
    name: str
    phone: Phone
    email: Email
    city: str


@dataclass(frozen=True, slots=True)
class OrderItem:
    """Item de pedido.

    Attributes:
        product_id: Produto comprado.
        quantity: Quantidade (>= 1).
    """

    product_id: int
    quantity: int

    def __post_init__(self) -> None:
        if self.quantity < 1:
            raise InvalidValueError("quantidade deve ser >= 1")


@dataclass(frozen=True, slots=True)
class Order:
    """Pedido. ``total`` é a fonte do faturamento e nunca é recalculado.

    Attributes:
        order_id: Número do pedido.
        customer_id: Dono do pedido.
        order_date: Data do pedido.
        status: Situação atual.
        total: Total oficial do pedido.
        payment_method: Forma de pagamento.
        tracking_code: Código de rastreio, quando despachado.
        estimated_delivery: Entrega estimada (único proxy de recebimento).
        notes: Observações.
        items: Itens do pedido.
    """

    order_id: OrderId
    customer_id: int
    order_date: date
    status: OrderStatus
    total: Money
    payment_method: PaymentMethod
    tracking_code: str | None = None
    estimated_delivery: date | None = None
    notes: str | None = None
    items: tuple[OrderItem, ...] = ()

    def is_overdue(self, today: date) -> bool:
        """Indica pedido despachado cuja entrega estimada já venceu.

        Args:
            today: Data de referência (vem do ``Clock``).

        Returns:
            ``True`` se ``shipped`` e ``estimated_delivery < today``.
        """
        return (
            self.status is OrderStatus.SHIPPED
            and self.estimated_delivery is not None
            and self.estimated_delivery < today
        )

    def belongs_to(self, customer_id: int) -> bool:
        """Verifica posse do pedido (prevenção de BOLA)."""
        return self.customer_id == customer_id
