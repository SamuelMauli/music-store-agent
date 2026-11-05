"""CLI das avaliações: ``python -m emporio.evals --mode fake|replay|live``.

Subcomandos: ``matrix`` (custo x qualidade por família) e ``judge-calibrate`` (juiz x rótulos).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from emporio.agent.llm import factory
from emporio.agent.llm.types import LLMProvider
from emporio.evals import calibration, matrix
from emporio.evals.cases import SUITES, Case, load_cases
from emporio.evals.runner import (
    CASSETTE_PATH,
    REPORT_DIR,
    STALE_MESSAGE,
    CaseResult,
    Evaluator,
    RunOptions,
    build_report,
)
from emporio.settings import Settings, load_settings

MIN_TOOL_ACCURACY = 0.95
MIN_GROUNDEDNESS = 0.98


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="emporio.evals", description=__doc__)
    parser.add_argument("--mode", choices=["fake", "replay", "live"], default="fake")
    parser.add_argument("--suite", action="append", choices=SUITES, help="repita; padrão: todas")
    parser.add_argument("--case", action="append", help="roda só estes ids")
    parser.add_argument("--live-only", action="store_true", help="só casos com live: true")
    parser.add_argument(
        "--judge", action="store_true", help="LLM-juiz (exige EMPORIO_EVAL_JUDGE_MODEL)"
    )
    parser.add_argument(
        "--sleep", type=float, help="segundos entre turnos (padrão EMPORIO_EVAL_SLEEP)"
    )
    parser.add_argument("--cassette", type=Path, default=CASSETTE_PATH)
    parser.add_argument("--out", type=Path, help="grava o relatório JSON neste arquivo")
    parser.add_argument("--strict", action="store_true", help="falha se algum caso falhar (live)")
    return parser.parse_args(argv)


def _judge_llm(settings: Settings) -> LLMProvider:
    if not settings.eval_judge_model:
        raise SystemExit("--judge exige EMPORIO_EVAL_JUDGE_MODEL (família diferente do gerador).")
    judge_settings = settings.model_copy(update={"llm_model_small": settings.eval_judge_model})
    return factory.build(judge_settings)


def _print(results: list[CaseResult], summary: dict[str, Any]) -> None:
    for r in results:
        mark = "STALE" if r.stale else "ok  " if r.passed else "FAIL"
        print(f"{mark} {r.suite}/{r.id}")
        for failure in r.failures:
            print(f"       - {failure}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def gate(summary: dict[str, Any], results: list[CaseResult]) -> list[str]:
    """Violações do piso de qualidade do CI (tool, groundedness, PII/BOLA, qualquer caso)."""
    results = [r for r in results if not r.stale]
    problems = [f"caso reprovado: {r.suite}/{r.id}" for r in results if not r.passed]
    tool = cast("dict[str, Any]", summary["tool_accuracy"])["rate"]
    grounded = cast("dict[str, Any]", summary["groundedness"])["rate"]
    if tool is not None and tool < MIN_TOOL_ACCURACY:
        problems.append(f"acurácia de tool {tool:.1%} < {MIN_TOOL_ACCURACY:.0%}")
    if grounded is not None and grounded < MIN_GROUNDEDNESS:
        problems.append(f"groundedness {grounded:.1%} < {MIN_GROUNDEDNESS:.0%}")
    return problems


def _cases(args: argparse.Namespace) -> list[Case]:
    cases = [c for suite in args.suite or SUITES for c in load_cases(suite)]
    if args.case:
        cases = [c for c in cases if c.id in args.case]
    return cases


def main(argv: list[str] | None = None) -> int:
    """Roda a avaliação e devolve o código de saída."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "matrix":
        return matrix.main(argv[1:])
    if argv and argv[0] == "judge-calibrate":
        return calibration.main(argv[1:])
    args = _parse(argv)
    settings = load_settings()
    live_like = args.mode != "fake"
    if args.judge and not live_like:
        raise SystemExit("--judge só faz sentido com --mode live ou replay.")
    options = RunOptions(
        mode=args.mode,
        live_only=args.live_only or live_like,
        sleep=settings.eval_sleep if args.sleep is None else args.sleep,
        judge=args.judge,
        cassette=args.cassette,
    )
    if args.mode == "live":
        args.cassette.unlink(missing_ok=True)
    judge = _judge_llm(settings) if args.judge else None
    started = time.monotonic()
    evaluator = Evaluator(settings, options, judge_llm=judge)
    results = evaluator.run(_cases(args))
    if evaluator.skipped:
        print(f"puladros no modo fake (fake: false): {', '.join(evaluator.skipped)}")
    report = build_report(results, settings, options, time.monotonic() - started)
    _print(results, report["summary"])
    out = args.out
    if out is None and args.mode == "live":
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = REPORT_DIR / f"live-{stamp}.json"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"relatório: {out}")
    stale = [r for r in results if r.stale]
    if stale:
        print(
            f"AVISO: {STALE_MESSAGE} ({len(stale)} de {len(results)} casos sem gravação para o "
            f"prompt atual {report['prompt_hash']}); esses casos foram marcados stale e não "
            "entram no resultado.",
            file=sys.stderr,
        )
    problems = gate(report["summary"], results)
    for problem in problems:
        print(f"GATE: {problem}", file=sys.stderr)
    return 1 if problems and (args.mode != "live" or args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
