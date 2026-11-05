"""Orquestração do pipeline medalhão (bronze → silver + relatório de qualidade)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb

from emporio.ingestion.bronze import load_bronze
from emporio.ingestion.gold import build_gold
from emporio.ingestion.policy_load import PolicyLoadResult, load_policy_sections
from emporio.ingestion.quality import build_quality_report, write_quality_report
from emporio.ingestion.silver import TableResult, build_silver
from emporio.ports import Clock
from emporio.settings import Settings


@dataclass(frozen=True)
class IngestionResult:
    """Resumo de uma execução do pipeline.

    Attributes:
        bronze: Linhas carregadas por tabela na bronze.
        silver: Aceitas e quarentenadas por tabela na silver.
        quality: Relatório de qualidade (regras D1 a D14).
        policies: Seções do manual gravadas, com hash do corpus.
    """

    bronze: dict[str, int]
    silver: dict[str, TableResult]
    quality: dict[str, Any]
    policies: PolicyLoadResult


def run_pipeline(settings: Settings, clock: Clock) -> IngestionResult:
    """Executa a ingestão completa e grava em ``settings.db_path``.

    O pipeline é idempotente: bronze e silver são recriadas a cada execução e o
    relatório de qualidade é regravado em ``settings.quality_report_path``.

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
        build_gold(con)
        policies = load_policy_sections(con, settings.policy_pdf)
        quality = build_quality_report(con, settings.reference_datetime.date())
    finally:
        con.close()
    write_quality_report(quality, settings.quality_report_path)
    return IngestionResult(bronze=bronze, silver=silver, quality=quality, policies=policies)
