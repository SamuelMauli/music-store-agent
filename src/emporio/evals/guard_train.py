"""Treino e avaliação do classificador de injeção: ``python -m emporio.evals.guard_train``.

Passos (reprodutíveis: split por hash, IRLS sem aleatoriedade, folds por hash de grupo):

1. Gera ``evals/guard/{train,test}.jsonl`` a partir das fontes (``guard_dataset``).
2. Embeddings do mesmo modelo da retrieval (fastembed), do texto e das variantes desofuscadas.
3. Validação cruzada de 5 folds no train escolhe ``l2`` e o limiar: maior recall de injeção com
   FPR <= 3% nos clientes da loja e <= 5% nos negativos difíceis (probabilidades fora da dobra).
4. Ajusta no train inteiro, avalia no test (heurística antiga, heurística nova, classificador,
   camadas), mede latência por mensagem e grava pesos e métricas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from emporio.adapters.retrieval.embedders import DEFAULT_MODEL, shared_embedder
from emporio.domain.text import normalize_text
from emporio.evals import guard_dataset as ds
from emporio.guardrails import input as guard_input
from emporio.guardrails.classifier import MODEL_PATH, LogisticModel, fit_logistic
from emporio.guardrails.injection import EmbeddingInjectionClassifier, LayeredInjectionGuard
from emporio.ports import Embedder

METRICS_PATH: Final = ds.GUARD_DIR / "metrics.json"
FOLDS: Final = 5
L2_GRID: Final = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
MAX_FPR_STORE: Final = 0.03
MAX_FPR_HARD: Final = 0.05
# passo fino perto de 1: com L2 baixo as probabilidades se concentram nas pontas
THRESHOLDS: Final = tuple(round(float(t), 3) for t in np.arange(0.05, 0.996, 0.005))
_LATENCY_WARMUP: Final = 5


@dataclass(frozen=True, slots=True)
class Encoded:
    """Exemplos com o score do classificador calculável: matriz e mapa texto → linhas."""

    examples: list[ds.GuardExample]
    vectors: NDArray[np.float32]
    rows: list[list[int]]  # linha 0 = texto; demais = variantes desofuscadas

    @property
    def base(self) -> NDArray[np.float32]:
        """Só o embedding do texto (o que o modelo vê no ajuste)."""
        return self.vectors[[r[0] for r in self.rows]]

    def labels(self) -> NDArray[np.int64]:
        """Rótulos como vetor."""
        return np.array([ex.label for ex in self.examples], dtype=np.int64)


def encode(examples: list[ds.GuardExample], embedder: Embedder) -> Encoded:
    """Embeddings do texto normalizado e das variantes, num lote só."""
    texts: list[str] = []
    rows: list[list[int]] = []
    for ex in examples:
        clean = guard_input.normalize(ex.text)
        variants = [clean, *guard_input.deobfuscated_variants(clean)]
        rows.append(list(range(len(texts), len(texts) + len(variants))))
        texts.extend(variants)
    return Encoded(examples, embedder.embed(texts), rows)


def max_scores(
    enc: Encoded, model: LogisticModel, subset: Sequence[int] | None = None
) -> list[float]:
    """Score de inferência (máximo entre texto e variantes), igual ao da guarda em produção."""
    proba = model.proba(enc.vectors)
    idx = range(len(enc.examples)) if subset is None else subset
    return [float(proba[enc.rows[i]].max()) for i in idx]


def _fold(group: str) -> int:
    return int(hashlib.sha256(f"guard-fold-v1:{group}".encode()).hexdigest(), 16) % FOLDS


def oof_scores(enc: Encoded, l2: float) -> list[float]:
    """Probabilidades fora da dobra (folds por grupo: variantes não vazam entre dobras)."""
    folds = [_fold(ex.group) for ex in enc.examples]
    labels, base = enc.labels(), enc.base
    out = [0.0] * len(enc.examples)
    for k in range(FOLDS):
        train_idx = [i for i, f in enumerate(folds) if f != k]
        test_idx = [i for i, f in enumerate(folds) if f == k]
        weights, bias = fit_logistic(base[train_idx], labels[train_idx], l2=l2)
        model = LogisticModel(weights, bias, 0.5, "")
        for i, score in zip(test_idx, max_scores(enc, model, test_idx), strict=True):
            out[i] = score
    return out


def _is_store_customer(ex: ds.GuardExample) -> bool:
    return ex.source == "curated" and ex.label == 0


def _rate(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def confusion(examples: Sequence[ds.GuardExample], predicted: Sequence[bool]) -> dict[str, Any]:
    """Precisão, recall, F1, FPRs por fatia e matriz de confusão."""
    pairs = list(zip(examples, predicted, strict=True))
    tp = sum(1 for ex, p in pairs if p and ex.label == 1)
    fp = sum(1 for ex, p in pairs if p and ex.label == 0)
    fn = sum(1 for ex, p in pairs if not p and ex.label == 1)
    tn = len(pairs) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn) if fp + tn else 0.0, 4),
        "recall_curated": round(
            _rate([p for ex, p in pairs if ex.source == "curated" and ex.label == 1]), 4
        ),
        "fpr_store_customers": round(_rate([p for ex, p in pairs if _is_store_customer(ex)]), 4),
        "fpr_hard_negatives": round(
            _rate([p for ex, p in pairs if ex.category == "hard_negative"]), 4
        ),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _recall_by_category(
    examples: Sequence[ds.GuardExample], predicted: Sequence[bool]
) -> dict[str, float]:
    hits: dict[str, list[bool]] = {}
    for ex, p in zip(examples, predicted, strict=True):
        if ex.label == 1:
            key = ex.category if ex.source == "curated" else f"{ex.source}"
            hits.setdefault(key, []).append(p)
    return {k: round(_rate(v), 4) for k, v in sorted(hits.items())}


def choose_threshold(
    examples: Sequence[ds.GuardExample], scores: Sequence[float]
) -> tuple[float, dict[str, Any]]:
    """Maior recall com FPR dentro dos tetos; empate fica com o limiar mais alto.

    Sem limiar que respeite os tetos, fica o de menor FPR nos clientes da loja (e avisa).
    """
    best: tuple[float, float, float] | None = None  # (recall, limiar, fpr loja)
    fallback = (2.0, 0.0)  # (fpr loja, -limiar); 2.0 perde para qualquer FPR real
    for t in THRESHOLDS:
        pred = [s >= t for s in scores]
        m = confusion(examples, pred)
        fallback = min(fallback, (m["fpr_store_customers"], -t))
        if m["fpr_store_customers"] <= MAX_FPR_STORE and m["fpr_hard_negatives"] <= MAX_FPR_HARD:
            cand = (m["recall"], t, m["fpr_store_customers"])
            if best is None or cand[:2] >= best[:2]:
                best = cand
    if best is not None:
        rule = (
            f"maior recall OOF com FPR <= {MAX_FPR_STORE:.0%} (clientes da loja) e "
            f"<= {MAX_FPR_HARD:.0%} (negativos difíceis); empate fica com o limiar mais alto"
        )
        threshold = best[1]
    else:
        rule = "nenhum limiar cumpriu os tetos: menor FPR nos clientes da loja"
        threshold = -fallback[1]
    oof = confusion(examples, [s >= threshold for s in scores])
    return threshold, {"rule": rule, "meets_ceilings": best is not None, "oof_at_threshold": oof}


def select_model(enc: Encoded, embedding_model: str) -> tuple[LogisticModel, dict[str, Any]]:
    """Escolhe ``l2`` e limiar por validação cruzada; ajusta no train inteiro."""
    trials: list[dict[str, Any]] = []
    for l2 in L2_GRID:
        threshold, info = choose_threshold(enc.examples, oof_scores(enc, l2))
        trials.append({"l2": l2, "threshold": threshold, **info})

    def key(trial: dict[str, Any]) -> tuple[bool, float, float]:
        # quem cumpre os tetos de FPR vence; entre eles, maior recall OOF, depois menor FPR
        oof = trial["oof_at_threshold"]
        return (trial["meets_ceilings"], oof["recall"], -oof["fpr_store_customers"])

    chosen = max(trials, key=key)
    weights, bias = fit_logistic(enc.base, enc.labels(), l2=chosen["l2"])
    model = LogisticModel(weights, bias, chosen["threshold"], embedding_model)
    return model, {"chosen_l2": chosen["l2"], "trials": trials}


def latency(guard: LayeredInjectionGuard, texts: Sequence[str]) -> dict[str, float]:
    """Latência real da guarda por mensagem (uma de cada vez, depois do aquecimento)."""
    for text in texts[:_LATENCY_WARMUP]:
        guard.check(text)
    samples = sorted(guard.check(t).elapsed_ms for t in texts)
    return {
        "messages": len(samples),
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
        "max_ms": round(samples[-1], 2),
        "mean_ms": round(statistics.fmean(samples), 2),
    }


def _legacy(text: str) -> bool:
    """Heurística como estava antes da desofuscação (só padrões sobre o texto)."""
    return any(p.search(normalize_text(text)) for p in guard_input._INJECTION)


def evaluate(test: Encoded, model: LogisticModel) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Métricas no test por sistema e os erros da defesa em camadas."""
    clean = [guard_input.normalize(ex.text) for ex in test.examples]
    scores = max_scores(test, model)
    systems = {
        "heuristic_legacy": [_legacy(t) for t in clean],
        "heuristic": [guard_input.looks_like_injection(t) for t in clean],
        "classifier": [s >= model.threshold for s in scores],
    }
    systems["layered"] = [
        h or c for h, c in zip(systems["heuristic"], systems["classifier"], strict=True)
    ]
    report = {
        name: {
            **confusion(test.examples, pred),
            "recall_by_category": _recall_by_category(test.examples, pred),
        }
        for name, pred in systems.items()
    }
    errors = [
        {"text": ex.text[:160], "label": ex.label, "category": ex.category, "score": round(s, 4)}
        for ex, s, p in zip(test.examples, scores, systems["layered"], strict=True)
        if p != bool(ex.label)
    ]
    return report, errors


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    embedder: Embedder,
    *,
    embedding_model: str = DEFAULT_MODEL,
    guard_dir: Path = ds.GUARD_DIR,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    sources: Callable[
        [], tuple[list[ds.GuardExample], list[ds.GuardExample], ds.BuildStats]
    ] = ds.build,
) -> dict[str, Any]:
    """Pipeline completo; devolve as métricas gravadas."""
    train, test, stats = sources()
    train_path, test_path = guard_dir / "train.jsonl", guard_dir / "test.jsonl"
    ds.write_jsonl(train_path, train)
    ds.write_jsonl(test_path, test)

    train_enc, test_enc = encode(train, embedder), encode(test, embedder)
    model, selection = select_model(train_enc, embedding_model)
    model_path.write_text(json.dumps(model.to_dict(), indent=1) + "\n", encoding="utf-8")
    model = LogisticModel.load(model_path)  # avalia exatamente o que foi gravado (arredondado)

    report, errors = evaluate(test_enc, model)
    guard = LayeredInjectionGuard(EmbeddingInjectionClassifier(embedder, model))
    clean = [guard_input.normalize(ex.text) for ex in test]
    via_classifier = [t for t in clean if not guard_input.looks_like_injection(t)]
    metrics: dict[str, Any] = {
        "embedding_model": model.embedding_model,
        "threshold": model.threshold,
        "threshold_selection": selection,
        "combination_policy": (
            "bloqueia se heuristica OU score >= limiar (heuristica curto-circuita)"
        ),
        "dataset": {
            "train": len(train),
            "test": len(test),
            "train_injection": sum(ex.label for ex in train),
            "test_injection": sum(ex.label for ex in test),
            "test_hard_negatives": sum(ex.category == "hard_negative" for ex in test),
            "test_store_customers": sum(_is_store_customer(ex) for ex in test),
            "raw": stats.raw,
            "duplicates_removed": stats.duplicates,
            "label_conflicts_removed": stats.label_conflicts,
            "eval_turns_excluded": stats.excluded_eval_turns,
            "train_sha256": _sha(train_path),
            "test_sha256": _sha(test_path),
        },
        "test": report,
        "test_errors_layered": errors,
        "latency_per_message": {
            "all_test_messages": latency(guard, clean),
            "classifier_path_only": latency(guard, via_classifier),
            "measured_on": (
                f"{platform.system()} {platform.machine()} py{platform.python_version()}"
            ),
        },
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI: treina, grava pesos e métricas, imprime o resumo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="modelo de embedding (fastembed)")
    args = parser.parse_args(argv)
    metrics = run(shared_embedder(args.model), embedding_model=args.model)
    for name, m in metrics["test"].items():
        print(
            f"{name:17} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"FPR={m['fpr']:.3f} FPR_loja={m['fpr_store_customers']:.3f} "
            f"FPR_dificeis={m['fpr_hard_negatives']:.3f} {m['confusion']}"
        )
    lat = metrics["latency_per_message"]["classifier_path_only"]
    print(f"limiar={metrics['threshold']} latencia (via classificador) {lat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
