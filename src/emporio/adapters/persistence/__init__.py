"""Repositórios sobre DuckDB (camada gold, somente leitura)."""

from emporio.adapters.persistence.duckdb_repositories import (
    DuckDbCustomerRepository,
    DuckDbOrderRepository,
    DuckDbProductRepository,
    DuckDbPromotionRepository,
)
from emporio.adapters.persistence.policy_repository import DuckDbPolicySectionRepository

__all__ = [
    "DuckDbCustomerRepository",
    "DuckDbOrderRepository",
    "DuckDbPolicySectionRepository",
    "DuckDbProductRepository",
    "DuckDbPromotionRepository",
]
