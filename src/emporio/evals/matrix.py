"""Matriz custo x qualidade por família de provedor, lida de ``evals/reports/live-*.json``.

Só entra o que os relatórios contêm (nada estimado). A "versão do código" é o ``prompt_hash``
gravado no relatório (os relatórios não trazem o SHA do git). Por padrão a linha da família é o
relatório mais recente; os demais saem como histórico.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emporio.evals.runner import REPORT_DIR

MATRIX_PATH = REPORT_DIR / "matrix.json"
NOT_RUN = "not run (no API key)"
_HOST_NOISE = {"api", "www", "com", "br", "io", "ai"}


def family_of(report: dict[str, Any]) -> str:
    """Família do provedor: prefixo das specs (``google:...``) ou, em relatório antigo sem
    prefixo, o host do endpoint (``api.groq.com`` -> ``groq``)."""
    models = report.get("models") or {}
    prefixes = {m.split(":", 1)[0] for m in models.values() if ":" in m}
    if prefixes:
        return "+".join(sorted(prefixes))
    provider = str(report.get("provider", "desconhecido"))
    parts = [p for p in provider.replace("(", ".").replace(")", "").split(".") if p]
    return next((p for p in parts if p not in _HOST_NOISE), provider)


def row_of(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Uma linha da matriz a partir de um relatório ao vivo."""
    s = report["summary"]
    models = report.get("models") or {}
    cost = s.get("cost_usd")
    passed = s.get("passed")
    return {
        "family": family_of(report),
        "report": path.name,
        "generated_at": report.get("generated_at"),
        "prompt_hash": report.get("prompt_hash"),
        "router_model": models.get("small"),
        "answer_model": models.get("medium"),
        "cases": s.get("cases"),
        "passed": passed,
        "stale": s.get("stale", 0),
        "tool_accuracy": (s.get("tool_accuracy") or {}).get("rate"),
        "groundedness": (s.get("groundedness") or {}).get("rate"),
        "security": (s.get("security") or {}).get("rate"),
        "p50_ms": (s.get("latency_ms") or {}).get("p50"),
        "p95_ms": (s.get("latency_ms") or {}).get("p95"),
        "cost_per_conversation_usd": s.get("cost_per_conversation_usd"),
        "cost_total_usd": cost,
        "usd_per_passed_case": round(cost / passed, 6) if cost is not None and passed else None,
    }


def build_matrix(report_dir: Path, expected: dict[str, str]) -> dict[str, Any]:
    """Agrupa os relatórios por família; o mais recente é ``latest``, o resto ``history``."""
    rows: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("live-*.json")):
        rows.append(row_of(path, json.loads(path.read_text(encoding="utf-8"))))
    families: dict[str, Any] = {}
    for fam in sorted({r["family"] for r in rows}):
        ordered = sorted(
            (r for r in rows if r["family"] == fam),
            key=lambda r: str(r["generated_at"]),
            reverse=True,
        )
        families[fam] = {"latest": ordered[0], "history": ordered[1:]}
    return {
        "families": families,
        "not_run": {f: why for f, why in sorted(expected.items()) if f not in families},
        "note": (
            "valores copiados dos relatórios ao vivo; prompt_hash = versão do código/prompt "
            "gravada no relatório; US$/caso aprovado = cost_total / casos aprovados; "
            "famílias com prompt_hash diferente rodaram versões diferentes do código, "
            "a comparação entre elas é aproximada"
        ),
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _usd(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"${value:.{digits}f}"


def _row_cells(r: dict[str, Any]) -> list[str]:
    latency = "-" if r["p50_ms"] is None else f"{r['p50_ms']}/{r['p95_ms']}"
    return [
        r["family"],
        f"`{r['router_model']}`",
        f"`{r['answer_model']}`",
        str(r["generated_at"])[:16].replace("T", " "),
        f"`{r['prompt_hash']}`",
        f"{r['passed']}/{r['cases']}",
        _pct(r["tool_accuracy"]),
        _pct(r["groundedness"]),
        _pct(r["security"]),
        latency,
        _usd(r["cost_per_conversation_usd"], 6),
        _usd(r["cost_total_usd"]),
        _usd(r["usd_per_passed_case"], 6),
    ]


_HEADER = [
    "Família",
    "Router",
    "Answer",
    "Data (UTC)",
    "Prompt",
    "Casos",
    "Tool",
    "Ground.",
    "Security",
    "p50/p95 ms",
    "US$/conversa",
    "US$ total",
    "US$/caso aprovado",
]


def _table(rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(_HEADER) + " |",
        "|" + "|".join("---" for _ in _HEADER) + "|",
        *("| " + " | ".join(cells) + " |" for cells in rows),
    ]


def render_markdown(matrix: dict[str, Any]) -> str:
    """Tabela principal (último relatório por família), famílias não rodadas e histórico."""
    fams = matrix["families"]
    rows = [_row_cells(f["latest"]) for f in fams.values()]
    rows += [[fam, why, *["-"] * (len(_HEADER) - 2)] for fam, why in matrix["not_run"].items()]
    lines = ["## Matriz custo x qualidade (último relatório ao vivo por família)", ""]
    lines += _table(rows)
    history = [_row_cells(r) for f in fams.values() for r in f["history"]]
    if history:
        lines += ["", "### Histórico (demais rodadas)", "", *_table(history)]
    return "\n".join(lines) + "\n"


def _parse_expect(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for value in values:
        name, _, why = value.partition("=")
        expected[name.strip()] = why.strip() or NOT_RUN
    return expected


def main(argv: list[str]) -> int:
    """``python -m emporio.evals matrix``: imprime o markdown e grava ``matrix.json``."""
    parser = argparse.ArgumentParser(prog="emporio.evals matrix", description=__doc__)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    parser.add_argument("--out", type=Path, default=MATRIX_PATH)
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="FAMILIA[=motivo]",
        help=f"família esperada; sem relatório sai como {NOT_RUN!r} (repita)",
    )
    args = parser.parse_args(argv)
    matrix = build_matrix(args.reports, _parse_expect(args.expect))
    if not matrix["families"]:
        raise SystemExit(f"nenhum live-*.json em {args.reports}")
    print(render_markdown(matrix), end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nmatriz: {args.out}")
    return 0
