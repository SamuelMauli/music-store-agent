"""Relatório de qualidade: executa as regras D1 a D14 e serializa o resultado."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from emporio.ingestion.quality_rules import RULES, QualityContext


def build_quality_report(con: duckdb.DuckDBPyConnection, today: date) -> dict[str, Any]:
    """Executa todas as regras e monta o relatório determinístico.

    O relatório não contém carimbo de tempo de execução: com o mesmo dado e a
    mesma data de referência, a saída é idêntica (base do teste de snapshot).

    Args:
        con: Conexão com bronze e silver prontas.
        today: Data de referência.

    Returns:
        Dicionário serializável com a contagem e os IDs afetados por regra.
    """
    ctx = QualityContext(con, today)
    findings = [rule(ctx) for rule in RULES]
    return {
        "reference_date": today.isoformat(),
        "summary": {
            "rules_evaluated": len(findings),
            "rules_with_findings": sum(1 for f in findings if f.affected_ids),
        },
        "rules": {f.rule: f.to_dict() for f in findings},
    }


def write_quality_report(report: dict[str, Any], path: Path) -> None:
    """Grava o relatório em JSON legível (UTF-8, chaves em ordem de execução)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_quality_report(report: dict[str, Any]) -> str:
    """Renderiza o relatório como texto curto para o terminal."""
    summary = report["summary"]
    lines = [
        f"Relatório de qualidade (referência {report['reference_date']}): "
        f"{summary['rules_with_findings']}/{summary['rules_evaluated']} regras com achados"
    ]
    for name, rule in report["rules"].items():
        ids = ", ".join(str(i) for i in rule["affected_ids"])
        shown = ids if rule["count"] <= 12 else f"{ids[:60]}..."
        head = f"  {rule['finding']:<3} {name:<40} {rule['severity']:<7}"
        lines.append(f"{head} n={rule['count']:<3} [{shown}]")
    return "\n".join(lines)
