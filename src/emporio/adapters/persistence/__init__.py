"""Repositórios sobre DuckDB (camada gold, somente leitura)."""

from emporio.adapters.persistence.duckdb_repositories import (
    DuckDbCustomerRepository,
    DuckDbOrderRepository,
    DuckDbProductRepository,
    DuckDbPromotionRepository,
)

__all__ = [
    "DuckDbCustomerRepository",
    "DuckDbOrderRepository",
    "DuckDbProductRepository",
    "DuckDbPromotionRepository",
]
