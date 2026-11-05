"""Executa os casos (fake, cassete ou provedor real) e monta o relatório de métricas."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import duckdb

from emporio.adapters.persistence import DuckDbPolicySectionRepository
from emporio.agent.llm import factory
from emporio.agent.llm.types import LLMError, LLMProvider, Usage
from emporio.agent.prompts import prompt_hash
from emporio.bootstrap import build_container
from emporio.domain.pii import mask_pii
from emporio.evals.cases import Case
from emporio.evals.cassette import Cassette, RecordingLLM, ReplayLLM
from emporio.evals.checks import Outcome, TurnRecord, evaluate
from emporio.evals.judge import JudgeScore, judge
from emporio.settings import Settings

Mode = Literal["fake", "live", "replay"]
CASSETTE_PATH = Path("evals/cassettes/live.jsonl")
REPORT_DIR = Path("evals/reports")


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Parâmetros de uma execução.

    Attributes:
        mode: ``fake`` (offline), ``replay`` (cassete) ou ``live`` (provedor real, grava cassete).
        live_only: Roda só os casos marcados ``live: true`` (obrigatório em replay e live).
        sleep: Segundos de espera entre turnos (limite de tokens por minuto do plano gratuito).
        judge: Pontua tom, clareza e completude com o LLM-juiz (só fora do modo fake).
        cassette: Arquivo do cassete.
    """

    mode: Mode = "fake"
    live_only: bool = False
    sleep: float = 0.0
    judge: bool = False
    cassette: Path = CASSETTE_PATH


@dataclass(slots=True)
class CaseResult:
    """Resultado de um caso, serializável no relatório."""

    id: str
    suite: str
    tags: list[str]
    passed: bool
    failures: list[str]
    tool_ok: bool | None
    grounded_ok: bool | None
    turns: list[dict[str, Any]] = field(default_factory=list)
    judge: dict[str, Any] | None = None


def percentile(values: list[int], pct: float) -> int:
    """Percentil por posto mais próximo (``0`` para lista vazia)."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def _product_names(settings: Settings) -> list[str]:
    connection = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        return [r[0] for r in connection.execute("select name from silver.products").fetchall()]
    finally:
        connection.close()


def _policy_texts(settings: Settings) -> list[str]:
    connection = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        return [s.text for s in DuckDbPolicySectionRepository(connection).all()]
    finally:
        connection.close()


def _record(turn: TurnRecord) -> dict[str, Any]:
    return {
        "user": mask_pii(turn.user),
        "answer": turn.answer,
        "intents": turn.intents,
        "tools": [t["name"] for t in turn.traces],
        "handoff": turn.handoff,
        "blocked": turn.blocked,
        "latency_ms": turn.latency_ms,
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "cost_usd": round(turn.cost_usd, 6),
        "llm_calls": turn.calls,
    }


class Evaluator:
    """Roda casos contra o agente real montado a partir das settings."""

    def __init__(
        self,
        settings: Settings,
        options: RunOptions,
        sleep: Callable[[float], None] = time.sleep,
        judge_llm: LLMProvider | None = None,
    ) -> None:
        """Cria o avaliador.

        Args:
            settings: Configuração base (banco, modelos, preços).
            options: Modo, filtro, throttle e juiz.
            sleep: Função de espera (injetável nos testes).
            judge_llm: Provedor do juiz; ``None`` desliga o juiz.
        """
        self._base = settings.model_copy(
            update={
                "llm_provider": "fake" if options.mode == "fake" else settings.llm_provider,
                "checkpoint_path": Path(":memory:"),
                "tickets_path": Path(":memory:"),
                "rate_limit_per_minute": 10_000,
                "otel_exporter": settings.otel_exporter,
            }
        )
        self._options = options
        self._sleep = sleep
        self._judge_llm = judge_llm
        self._cassette = Cassette(options.cassette)
        self._turns_done = 0
        self.skipped: list[str] = []
        self._policy = _policy_texts(self._base)

    def _llm(self, settings: Settings) -> LLMProvider:
        mode = self._options.mode
        if mode == "replay":
            return ReplayLLM(self._cassette)
        llm = factory.build(settings, _product_names(settings))
        return RecordingLLM(llm, self._cassette) if mode == "live" else llm

    def _throttle(self) -> None:
        if self._turns_done and self._options.sleep > 0:
            self._sleep(self._options.sleep)
        self._turns_done += 1

    def run_case(self, case: Case) -> CaseResult:
        """Executa a conversa do caso e avalia as expectativas."""
        settings = self._base
        if case.reference_date is not None:
            when = datetime.combine(case.reference_date, datetime.min.time()).replace(
                hour=10, tzinfo=settings.reference_datetime.tzinfo
            )
            settings = settings.model_copy(update={"reference_datetime": when})
        container = build_container(settings, llm=self._llm(settings))
        session = f"eval-{case.id}-{uuid.uuid4().hex[:6]}"
        records: list[TurnRecord] = []
        error = ""
        try:
            for text in case.turns:
                self._throttle()
                result = container.service.chat(session, text)
                traces = container.service.snapshot(session).get("tool_traces") or []
                records.append(
                    TurnRecord(
                        text,
                        result.answer,
                        result.intents,
                        list(traces),
                        result.handoff,
                        result.blocked,
                        result.latency_ms,
                        result.usage.input_tokens,
                        result.usage.output_tokens,
                        result.usage.cost_usd,
                        result.usage.calls,
                    )
                )
        except (LLMError, RuntimeError, KeyError) as exc:
            error = f"erro na execução: {type(exc).__name__}: {exc}"
        finally:
            container.close()
        return self._result(case, records, error)

    def _result(self, case: Case, records: list[TurnRecord], error: str) -> CaseResult:
        if error or not records:
            outcome = Outcome(failures=[error or "sem turnos"])
        else:
            outcome = evaluate(case, records, self._policy)
        score = self._judge(case, records, outcome) if records else None
        return CaseResult(
            case.id,
            case.suite,
            case.tags,
            outcome.passed,
            outcome.failures,
            outcome.tool_ok,
            outcome.grounded_ok,
            [_record(r) for r in records],
            score,
        )

    def _judge(
        self, case: Case, records: list[TurnRecord], outcome: Outcome
    ) -> dict[str, Any] | None:
        if self._judge_llm is None:
            return None
        self._throttle()
        try:
            score, usage = judge(self._judge_llm, records[-1].user, records[-1].answer)
        except LLMError as exc:
            return {"error": type(exc).__name__}
        return self._judged(case, outcome, score, usage)

    @staticmethod
    def _judged(case: Case, outcome: Outcome, score: JudgeScore, usage: Usage) -> dict[str, Any]:
        minimum = case.expect.judge_min
        if minimum is not None and score.mean < minimum:
            outcome.failures.append(f"juiz: média {score.mean:.2f} < {minimum}")
        return {**score.model_dump(), "mean": round(score.mean, 2), "cost_usd": usage.cost_usd}

    def run(self, cases: list[Case]) -> list[CaseResult]:
        """Roda os casos selecionados pelas opções, na ordem."""
        selected = [c for c in cases if c.live] if self._options.live_only else cases
        if self._options.mode == "fake":
            self.skipped = [c.id for c in selected if not c.fake]
            selected = [c for c in selected if c.fake]
        return [self.run_case(c) for c in selected]


def _rate(flags: list[bool]) -> dict[str, Any]:
    return {
        "n": len(flags),
        "passed": sum(flags),
        "rate": round(sum(flags) / len(flags), 4) if flags else None,
    }


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    """Métricas agregadas: aprovação, tool, groundedness, segurança, latência, tokens e custo."""
    turns = [t for r in results for t in r.turns]
    latencies = [int(t["latency_ms"]) for t in turns]
    cost = sum(float(t["cost_usd"]) for t in turns)
    security = [r.passed for r in results if {"bola", "pii", "injection"} & set(r.tags)]
    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / len(results), 4) if results else None,
        "tool_accuracy": _rate([r.tool_ok for r in results if r.tool_ok is not None]),
        "groundedness": _rate([r.grounded_ok for r in results if r.grounded_ok is not None]),
        "security": _rate(security),
        "turns": len(turns),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies, default=0),
        },
        "tokens": {
            "input": sum(int(t["input_tokens"]) for t in turns),
            "output": sum(int(t["output_tokens"]) for t in turns),
        },
        "llm_calls": sum(int(t["llm_calls"]) for t in turns),
        "cost_usd": round(cost, 6),
        "cost_per_conversation_usd": round(cost / len(results), 6) if results else None,
    }


def build_report(
    results: list[CaseResult], settings: Settings, options: RunOptions, wall_seconds: float
) -> dict[str, Any]:
    """Relatório completo (JSON-friendly) com resumo global, por suíte e por caso."""
    suites = sorted({r.suite for r in results})
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": options.mode,
        "prompt_hash": prompt_hash(),
        "provider": "fake" if options.mode == "fake" else urlparse(settings.llm_base_url).hostname,
        "models": {"small": settings.llm_model_small, "medium": settings.llm_model_medium},
        "throttle_seconds": options.sleep,
        "wall_seconds": round(wall_seconds, 1),
        "summary": summarize(results),
        "by_suite": {s: summarize([r for r in results if r.suite == s]) for s in suites},
        "cases": [asdict(r) for r in results],
    }
