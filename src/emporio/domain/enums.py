"""Enumerações do domínio."""

from __future__ import annotations

from enum import StrEnum


class ProductStatus(StrEnum):
    """Situação comercial do produto no cadastro."""

    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    COMING_SOON = "coming_soon"


class Availability(StrEnum):
    """Disponibilidade derivada de status e estoque (o status prevalece sobre o estoque)."""

    AVAILABLE = "available"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"
    COMING_SOON = "coming_soon"


class OrderStatus(StrEnum):
    """Ciclo de vida do pedido."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentMethod(StrEnum):
    """Formas de pagamento presentes nos pedidos."""

    PIX = "pix"
    BOLETO = "boleto"
    DEBIT = "debit"
    CREDIT_3X = "credit_3x"
    CREDIT_6X = "credit_6x"
    CREDIT_12X = "credit_12x"
