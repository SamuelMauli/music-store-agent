"""Orquestração do pipeline medalhão (bronze → silver)."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from emporio.ingestion.bronze import load_bronze
from emporio.ingestion.silver import TableResult, build_silver
from emporio.ports import Clock
from emporio.settings import Settings


@dataclass(frozen=True)
class IngestionResult:
    """Resumo de uma execução do pipeline.

    Attributes:
        bronze: Linhas carregadas por tabela na bronze.
        silver: Aceitas e quarentenadas por tabela na silver.
    """

    bronze: dict[str, int]
    silver: dict[str, TableResult]


def run_pipeline(settings: Settings, clock: Clock) -> IngestionResult:
    """Executa a ingestão completa e grava em ``settings.db_path``.

    O pipeline é idempotente: bronze e silver são recriadas a cada execução.

    Args:
        settings: Caminhos de origem, contratos e destino.
        clock: Relógio usado para carimbar ``_ingested_at``.

    Returns:
        Contagens por camada e tabela.
    """
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.db_path))
    try:
        bronze = load_bronze(con, settings.raw_dir, clock.now())
        silver = build_silver(con, settings.contracts_dir)
    finally:
        con.close()
    return IngestionResult(bronze=bronze, silver=silver)
