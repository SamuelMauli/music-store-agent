"""Ponto de entrada de ``make ingest``: ``python -m emporio.ingestion``."""

from __future__ import annotations

import sys

from emporio.clock import SystemClock
from emporio.domain.errors import DomainError
from emporio.ingestion.pipeline import run_pipeline
from emporio.ingestion.quality import format_quality_report
from emporio.settings import load_settings


def main() -> int:
    """Roda a ingestão e imprime o resumo por tabela.

    Returns:
        Código de saída do processo (0 em sucesso).
    """
    settings = load_settings()
    try:
        result = run_pipeline(settings, SystemClock(settings.timezone))
    except DomainError as exc:
        print(f"falha na ingestão: {exc}", file=sys.stderr)
        return 1
    print(f"banco: {settings.db_path}")
    for table, count in result.bronze.items():
        silver = result.silver[table]
        print(
            f"  {table:<12} bronze={count:>3}  silver={silver.accepted:>3}  "
            f"quarentena={silver.quarantined}"
        )
    policies = result.policies
    print(
        f"  {'policies':<12} manual v{policies.manual_version}: {policies.sections} seções "
        f"(sha256 {policies.corpus_sha256[:12]})"
    )
    print(format_quality_report(result.quality))
    print(f"relatório: {settings.quality_report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
