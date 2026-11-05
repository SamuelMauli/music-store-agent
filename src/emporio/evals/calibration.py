"""Calibração do LLM-juiz contra rótulos de referência (``evals/judge_labels.yaml``).

O juiz dá notas de 1 a 5 (``JudgeScore``); os rótulos são binários por critério (adequado ou
não). Para cada critério varre-se o limiar (nota >= limiar = adequado), mede-se a concordância
(acurácia, kappa de Cohen, matriz de confusão) e escolhe-se o limiar de maior kappa.

Modos:
    ``fake``   heurística determinística de superfície (NÃO é LLM): só exercita o pipeline offline.
    ``replay`` repete notas gravadas por ``live`` (cassete JSON: pergunta+resposta+rubrica).
    ``live``   chama o juiz real (``EMPORIO_EVAL_JUDGE_MODEL``) e grava o cassete.

Os rótulos atuais são rascunho de IA (``reviewed_by_human: false``): o relatório repete o aviso.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from emporio.agent.llm import factory
from emporio.agent.llm.types import LLMProvider, Usage
from emporio.evals.judge import RUBRIC, JudgeScore, judge
from emporio.settings import load_settings

LABELS_PATH = Path("evals/judge_labels.yaml")
CASSETTE_PATH = Path("evals/cassettes/judge-calibration.json")
CRITERIA = ("tone", "clarity", "completeness")
THRESHOLDS = (2, 3, 4, 5)  # limiar 1 aprovaria tudo: kappa indefinido

JudgeFn = Callable[[str, str], tuple[JudgeScore, Usage]]


class Labels(BaseModel):
    """Rótulo binário por critério (``True`` = adequado)."""

    model_config = ConfigDict(extra="forbid")

    tone: bool
    clarity: bool
    completeness: bool


class LabeledItem(BaseModel):
    """Par (pergunta, resposta) com o rótulo de referência e sua proveniência."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: dict[str, Any]
    question: str
    answer: str
    labels: Labels
    note: str = ""
    labeler: str
    reviewed_by_human: bool


def load_labels(path: Path = LABELS_PATH) -> list[LabeledItem]:
    """Carrega e valida os rótulos; ids duplicados são erro."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = [LabeledItem.model_validate(row) for row in raw["items"]]
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"ids duplicados em {path}")
    return items


# --- métricas ------------------------------------------------------------------------------


def kappa(truth: Sequence[object], pred: Sequence[object]) -> float | None:
    """Kappa de Cohen entre dois conjuntos de rótulos categóricos.

    ``(po - pe) / (1 - pe)``, com ``po`` a concordância observada e ``pe`` a esperada pelo acaso
    (produto das marginais). Devolve ``None`` quando indefinido (sem itens ou ``pe == 1``: os
    dois avaliadores usam uma só categoria, igual).
    """
    if len(truth) != len(pred):
        raise ValueError("listas de tamanhos diferentes")
    n = len(truth)
    if n == 0:
        return None
    observed = sum(a == b for a, b in zip(truth, pred, strict=True)) / n
    ca, cb = Counter(truth), Counter(pred)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def confusion(truth: Sequence[bool], pred: Sequence[bool]) -> dict[str, int]:
    """Matriz de confusão (positivo = adequado): tp, fp, fn, tn."""
    pairs = list(zip(truth, pred, strict=True))
    return {
        "tp": sum(t and p for t, p in pairs),
        "fp": sum(not t and p for t, p in pairs),
        "fn": sum(t and not p for t, p in pairs),
        "tn": sum(not t and not p for t, p in pairs),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _at_threshold(truth: list[bool], scores: list[int], threshold: int) -> dict[str, Any]:
    pred = [s >= threshold for s in scores]
    return {
        "threshold": threshold,
        "accuracy": round(sum(t == p for t, p in zip(truth, pred, strict=True)) / len(truth), 4),
        "kappa": _round(kappa(truth, pred)),
        "confusion": confusion(truth, pred),
    }


def best_threshold(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    """Limiar de maior kappa; empate: maior acurácia, depois o limiar mais alto (mais rigoroso)."""
    return max(
        sweep,
        key=lambda r: (
            r["kappa"] if r["kappa"] is not None else -2.0,
            r["accuracy"],
            r["threshold"],
        ),
    )


def calibrate(items: list[LabeledItem], scores: list[JudgeScore]) -> dict[str, Any]:
    """Concordância juiz x rótulos por critério, com varredura de limiar e o limiar escolhido."""
    if len(items) != len(scores) or not items:
        raise ValueError("itens e notas precisam ter o mesmo tamanho (> 0)")
    criteria: dict[str, Any] = {}
    for name in CRITERIA:
        truth = [bool(getattr(i.labels, name)) for i in items]
        graded = [int(getattr(s, name)) for s in scores]
        sweep = [_at_threshold(truth, graded, t) for t in THRESHOLDS]
        criteria[name] = {
            "positives": sum(truth),
            "negatives": len(truth) - sum(truth),
            "score_distribution": {str(k): v for k, v in sorted(Counter(graded).items())},
            "sweep": sweep,
            "chosen": best_threshold(sweep),
        }
    return {
        "n": len(items),
        "reviewed_by_human": sum(i.reviewed_by_human for i in items),
        "labelers": sorted({i.labeler for i in items}),
        "criteria": criteria,
    }


# --- juízes: fake, replay e live ----------------------------------------------------------

_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
_COMPLAINT = re.compile(r"absurdo|quebrad|terceira vez|reclam|atras", re.IGNORECASE)


def heuristic_judge(question: str, answer: str) -> tuple[JudgeScore, Usage]:
    """Juiz de brinquedo, determinístico e sem rede: pontua traços de superfície.

    Não é um LLM nem pretende acertar: serve só para o CI exercitar o pipeline de calibração.
    """
    shouting = answer.count("!") >= 3 or bool(re.search(r"[A-ZÁÉÍÓÚÇ]{6,}", answer))
    tone = 5 - 2 * bool(_EMOJI.search(answer) and _COMPLAINT.search(question)) - 2 * shouting
    clarity = 5 - (2 if len(answer) > 900 else 0) - (2 if len(answer) < 25 else 0)
    completeness = 5 - (2 if len(answer) < 40 else 0)
    score = JudgeScore(
        tone=max(1, tone), clarity=max(1, clarity), completeness=max(1, completeness)
    )
    return score, Usage()


def cassette_key(question: str, answer: str) -> str:
    """Chave estável: muda se a pergunta, a resposta ou a rubrica mudarem (cassete velho = erro)."""
    raw = json.dumps([RUBRIC, question, answer], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def replay_judge(path: Path) -> JudgeFn:
    """Repete as notas gravadas; par sem gravação é erro explícito (nunca nota inventada)."""
    if not path.exists():
        raise SystemExit(f"cassete do juiz ausente: {path} (grave com --mode live).")
    rows: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    def run(question: str, answer: str) -> tuple[JudgeScore, Usage]:
        row = rows.get(cassette_key(question, answer))
        if row is None:
            raise LookupError("par sem gravação no cassete (rubrica ou rótulos mudaram?)")
        return JudgeScore.model_validate(row["score"]), Usage.from_dict(row["usage"])

    return run


def live_judge(llm: LLMProvider, record: dict[str, dict[str, Any]]) -> JudgeFn:
    """Chama o juiz real e acumula as notas em ``record`` (para gravar o cassete)."""

    def run(question: str, answer: str) -> tuple[JudgeScore, Usage]:
        score, usage = judge(llm, question, answer)
        record[cassette_key(question, answer)] = {
            "score": score.model_dump(),
            "usage": usage.to_dict(),
        }
        return score, usage

    return run


Mode = Literal["fake", "replay", "live"]


def run_calibration(items: list[LabeledItem], judge_fn: JudgeFn, mode: Mode) -> dict[str, Any]:
    """Roda o juiz sobre todos os rótulos e devolve o relatório de concordância."""
    scores: list[JudgeScore] = []
    usage = Usage()
    for item in items:
        score, used = judge_fn(item.question, item.answer)
        scores.append(score)
        usage += used
    report = calibrate(items, scores)
    report.update(
        {
            "mode": mode,
            "judge_is_llm": mode != "fake",
            "cost_usd": round(usage.cost_usd, 6),
            "warning": (
                "rótulos são rascunho de IA (reviewed_by_human=false): concordância medida "
                "contra o rascunho, não contra a verdade"
                if not all(i.reviewed_by_human for i in items)
                else None
            ),
        }
    )
    return report


def render(report: dict[str, Any]) -> str:
    """Tabela em texto: por critério, o limiar escolhido e a matriz de confusão."""
    lines = [
        f"calibração do juiz: modo={report['mode']} n={report['n']} "
        f"revisados por humano={report['reviewed_by_human']}"
    ]
    if report["mode"] == "fake":
        lines.append("AVISO: juiz heurístico (não é LLM); números só exercitam o pipeline.")
    if report["warning"]:
        lines.append(f"AVISO: {report['warning']}")
    lines.append(f"{'critério':<13}{'limiar':>7}{'acc':>8}{'kappa':>8}   tp  fp  fn  tn")
    for name, data in report["criteria"].items():
        c = data["chosen"]
        k = "n/d" if c["kappa"] is None else f"{c['kappa']:.3f}"
        m = c["confusion"]
        lines.append(
            f"{name:<13}{c['threshold']:>7}{c['accuracy']:>8.3f}{k:>8}"
            f"  {m['tp']:>3} {m['fp']:>3} {m['fn']:>3} {m['tn']:>3}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """``python -m emporio.evals judge-calibrate``: concordância do juiz com os rótulos."""
    parser = argparse.ArgumentParser(prog="emporio.evals judge-calibrate", description=__doc__)
    parser.add_argument("--mode", choices=["fake", "replay", "live"], default="fake")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--cassette", type=Path, default=CASSETTE_PATH)
    parser.add_argument("--out", type=Path, help="grava o relatório JSON neste arquivo")
    args = parser.parse_args(argv)
    items = load_labels(args.labels)
    record: dict[str, dict[str, Any]] = {}
    if args.mode == "fake":
        fn: JudgeFn = heuristic_judge
    elif args.mode == "replay":
        fn = replay_judge(args.cassette)
    else:
        settings = load_settings()
        if not settings.eval_judge_model:
            raise SystemExit("live exige EMPORIO_EVAL_JUDGE_MODEL (família diferente do gerador).")
        llm = factory.build(
            settings.model_copy(update={"llm_model_small": settings.eval_judge_model})
        )
        fn = live_judge(llm, record)
    try:
        report = run_calibration(items, fn, args.mode)
    except LookupError as exc:
        raise SystemExit(f"replay: {exc}") from exc
    print(render(report))
    if args.mode == "live":
        args.cassette.parent.mkdir(parents=True, exist_ok=True)
        args.cassette.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"cassete: {args.cassette}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"relatório: {args.out}")
    return 0
