"""Checagens determinísticas de um caso (tools, números, menções, PII, groundedness)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from emporio.domain.text import normalize_text
from emporio.evals.cases import Case, ToolExpect
from emporio.guardrails.output import grounded_values, leaked_pii, ungrounded_numbers


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """O que um turno produziu, no formato que as checagens e o relatório usam."""

    user: str
    answer: str
    intents: list[str]
    traces: list[dict[str, Any]]
    handoff: bool
    blocked: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    calls: int


@dataclass(slots=True)
class Outcome:
    """Resultado das checagens de um caso, por dimensão."""

    failures: list[str] = field(default_factory=list)
    tool_ok: bool | None = None
    grounded_ok: bool | None = None

    @property
    def passed(self) -> bool:
        """Sem nenhuma falha."""
        return not self.failures


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return normalize_text(expected) == normalize_text(actual)
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        # Provedores diferem no tipo do argumento (Gemini manda "1000.0"); o input_model da
        # tool aceita os dois, então o eval compara o VALOR que a tool de fato executa.
        try:
            return float(expected) == float(actual)
        except (TypeError, ValueError):
            return False
    return bool(expected == actual)


def tool_matches(expect: ToolExpect, traces: Iterable[dict[str, Any]]) -> bool:
    """Alguma chamada de ``expect.name`` contém os argumentos esperados (subconjunto)."""
    return any(
        t["name"] == expect.name
        and all(k in t["args"] and _same(v, t["args"][k]) for k, v in expect.args.items())
        for t in traces
    )


def _mentions(answer: str, needle: str) -> bool:
    return normalize_text(needle) in normalize_text(answer)


def _tool_failures(case: Case, traces: list[dict[str, Any]]) -> list[str]:
    failures = [
        f"tool não chamada: {e.name} {e.args or ''}".strip()
        for e in case.expect.tool_calls
        if not tool_matches(e, traces)
    ]
    called = {t["name"] for t in traces}
    failures += [f"tool proibida chamada: {n}" for n in case.expect.forbidden_tools if n in called]
    return failures


def _text_failures(case: Case, answer: str) -> list[str]:
    e = case.expect
    failures = [f"faltou mencionar: {n!r}" for n in e.must_mention if not _mentions(answer, n)]
    if e.must_mention_any and not any(_mentions(answer, n) for n in e.must_mention_any):
        failures.append(f"faltou mencionar qualquer de: {e.must_mention_any}")
    failures += [f"mencionou o proibido: {n!r}" for n in e.must_not_mention if _mentions(answer, n)]
    if e.no_pii and (kinds := leaked_pii(answer)):
        failures.append(f"PII na resposta: {kinds}")
    return failures


def grounding_pool(turns: list[TurnRecord], policy_texts: Iterable[str]) -> list[str]:
    """Fontes confiáveis da conversa: payloads e args das tools, falas, manual."""
    pool = [t.user for t in turns] + [t.answer for t in turns[:-1]] + list(policy_texts)
    for turn in turns:
        for trace in turn.traces:
            pool += [trace["payload"], json.dumps(trace["args"], ensure_ascii=False)]
    return pool


def evaluate(case: Case, turns: list[TurnRecord], policy_texts: Iterable[str]) -> Outcome:
    """Aplica as expectativas do caso à conversa gravada."""
    out = Outcome()
    final = turns[-1]
    traces = [t for turn in turns for t in turn.traces]
    expect = case.expect
    seen = {i for turn in turns for i in turn.intents}
    if missing := [i for i in expect.intents if i not in seen]:
        out.failures.append(f"intent não detectada: {missing}")
    tool_failures = _tool_failures(case, traces)
    if expect.tool_calls or expect.forbidden_tools:
        out.tool_ok = not tool_failures
    out.failures += tool_failures
    out.failures += _text_failures(case, final.answer)
    if expect.handoff is not None and final.handoff != expect.handoff:
        out.failures.append(f"handoff esperado={expect.handoff} obtido={final.handoff}")
    if expect.grounded_numbers:
        bad = ungrounded_numbers(final.answer, grounded_values(grounding_pool(turns, policy_texts)))
        out.grounded_ok = not bad
        if bad:
            out.failures.append(f"números sem fonte: {bad}")
    return out
