"""Portas (Protocols) pelas quais o domínio fala com o mundo externo."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from emporio.domain.entities import CatalogEntry, Customer, Order, Promotion
from emporio.domain.lookup import NameLookup
from emporio.domain.policy import PolicySnippet
from emporio.domain.specifications import Specification
from emporio.domain.value_objects import Email, OrderId


class Clock(Protocol):
    """Fonte de "agora" injetada, para que prazos sejam determinísticos."""

    def now(self) -> datetime:
        """Retorna o instante atual como datetime *aware*."""
        ...


class ProductRepository(Protocol):
    """Leitura do catálogo (gold ``v_catalog``), sempre com preço efetivo."""

    def get(self, product_id: int) -> CatalogEntry | None:
        """Busca por ID; ``None`` se não existir."""
        ...

    def find(self, spec: Specification, limit: int | None = None) -> list[CatalogEntry]:
        """Filtra por especificação, do mais barato ao mais caro (desempate por ID)."""
        ...

    def resolve_name(self, query: str) -> NameLookup:
        """Resolve nome livre: ``Found``, ``Ambiguous`` (até 3) ou ``NotFound``."""
        ...


class OrderRepository(Protocol):
    """Leitura de pedidos com seus itens."""

    def get(self, order_id: OrderId) -> Order | None:
        """Busca por número; ``None`` se não existir."""
        ...

    def list_for_customer(self, customer_id: int) -> list[Order]:
        """Pedidos do cliente, do mais recente ao mais antigo."""
        ...


class CustomerRepository(Protocol):
    """Leitura de clientes."""

    def get(self, customer_id: int) -> Customer | None:
        """Busca por ID; ``None`` se não existir."""
        ...

    def find_by_email(self, email: Email) -> list[Customer]:
        """E-mail não é único (clientes 11 e 43): devolve todos os que o usam."""
        ...


class PromotionRepository(Protocol):
    """Leitura de promoções."""

    def active(self) -> list[Promotion]:
        """Promoções com ``is_active``, ordenadas por ID."""
        ...

    def for_product(self, product_id: int) -> list[Promotion]:
        """Todas as promoções do produto (ativas ou não), ordenadas por ID."""
        ...


class PolicyRetriever(Protocol):
    """Estratégia de recuperação de trechos do manual (Strategy)."""

    def retrieve(self, query: str, k: int = 3) -> list[PolicySnippet]:
        """Trechos relevantes à pergunta, do mais ao menos relevante."""
        ...


class Embedder(Protocol):
    """Transforma textos em vetores densos de norma 1."""

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Matriz ``(len(texts), dim)`` normalizada em L2."""
        ...


class VectorIndex(Protocol):
    """Índice de vetores por similaridade de cosseno."""

    def add(self, ids: list[str], vectors: NDArray[np.float32]) -> None:
        """Indexa ``vectors`` (norma 1) sob os ``ids``."""
        ...

    def search(self, vector: NDArray[np.float32], k: int) -> list[tuple[str, float]]:
        """Os ``k`` ids mais próximos com a similaridade, do maior ao menor."""
        ...
