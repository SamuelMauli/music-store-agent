"""Defesa em camadas contra injeção de prompt na entrada: heurística + classificador treinado.

Política de combinação (OU, com curto-circuito):

1. A heurística (padrões PT/EN sobre texto e variantes desofuscadas) casou → bloqueia. É
   assinatura explícita de ataque e não custa nada; o classificador nem roda.
2. Senão, o classificador pontua o texto, a troca de homóglifos e o base64 decodificado (leet
   e hífen ficam só com a heurística: reescrevem SKU, rastreio e UUID); a maior probabilidade
   ``>= limiar`` → bloqueia. O limiar vem do treino (``make train-guard``), escolhido por
   validação cruzada para maximizar o recall com FPR baixo em clientes legítimos.
3. Classificador indisponível (modelo de embedding não carregou, pesos inválidos) → fica só a
   heurística, com ``degraded=True`` no veredito e um aviso no log; nova tentativa depois de
   ``retry_after`` segundos, para uma falha transitória não desligar a camada de vez.

Bloquear custa uma recusa educada (o cliente reformula); deixar passar custa um ataque que
segue para as defesas seguintes (ferramentas por intenção, guarda de saída). Por isso o OU.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emporio.guardrails.classifier import MODEL_PATH, LogisticModel
from emporio.guardrails.input import classifier_variants, looks_like_injection
from emporio.ports import Embedder

log = logging.getLogger("emporio.guard")


@dataclass(frozen=True, slots=True)
class InjectionVerdict:
    """Resultado da guarda de injeção para uma mensagem.

    Attributes:
        blocked: A mensagem deve receber a recusa padrão.
        reason: ``heuristic``, ``classifier`` ou ``none``.
        score: Probabilidade do classificador; ``None`` se ele não rodou.
        degraded: O classificador deveria rodar mas está indisponível.
        elapsed_ms: Latência da guarda nesta mensagem.
    """

    blocked: bool
    reason: str
    score: float | None
    degraded: bool
    elapsed_ms: float


class EmbeddingInjectionClassifier:
    """Embedding + regressão logística, com fail-safe e nova tentativa após falha."""

    def __init__(
        self,
        embedder: Embedder,
        model: LogisticModel,
        *,
        retry_after: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Cria o classificador.

        Args:
            embedder: Mesmo modelo de embedding usado no treino.
            model: Pesos e limiar.
            retry_after: Segundos sem tentar de novo depois de uma falha.
            monotonic: Relógio (injetável nos testes).
        """
        self._embedder, self._model = embedder, model
        self._retry_after, self._now = retry_after, monotonic
        self._down_until = 0.0

    @property
    def threshold(self) -> float:
        """Limiar de bloqueio."""
        return self._model.threshold

    @property
    def available(self) -> bool:
        """``False`` enquanto durar a pausa após uma falha."""
        return self._now() >= self._down_until

    def score(self, texts: list[str]) -> float | None:
        """Maior probabilidade de injeção entre ``texts``; ``None`` se indisponível."""
        if not texts or not self.available:
            return None
        try:
            vectors = self._embedder.embed(texts)
            score = float(np.max(self._model.proba(vectors)))
            if not np.isfinite(score):
                raise FloatingPointError("score nao finito")
            return score
        except Exception as exc:  # qualquer falha do modelo cai para a heurística
            self._down_until = self._now() + self._retry_after
            # Só o tipo: a mensagem da exceção pode carregar texto do cliente (CPF, e-mail).
            log.warning(
                "classificador de injecao indisponivel (%s); so heuristica por %.0fs",
                type(exc).__name__,
                self._retry_after,
            )
            return None


class LayeredInjectionGuard:
    """Heurística + classificador opcional, na política descrita no módulo."""

    def __init__(
        self,
        classifier: EmbeddingInjectionClassifier | None = None,
        *,
        degraded: bool = False,
        perf_counter: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Cria a guarda.

        Args:
            classifier: ``None`` = só heurística.
            degraded: Marca que o classificador foi pedido mas não pôde ser montado.
            perf_counter: Relógio de latência.
        """
        self._classifier, self._degraded, self._clock = classifier, degraded, perf_counter

    def check(self, text: str) -> InjectionVerdict:
        """Avalia uma mensagem já normalizada por ``normalize``."""
        start = self._clock()
        if looks_like_injection(text):
            return InjectionVerdict(True, "heuristic", None, False, self._ms(start))
        if self._classifier is None:
            return InjectionVerdict(False, "none", None, self._degraded, self._ms(start))
        score = self._classifier.score([text, *classifier_variants(text)])
        if score is None:
            return InjectionVerdict(False, "none", None, True, self._ms(start))
        blocked = score >= self._classifier.threshold
        reason = "classifier" if blocked else "none"
        return InjectionVerdict(blocked, reason, score, False, self._ms(start))

    def _ms(self, start: float) -> float:
        return (self._clock() - start) * 1000.0


def build_injection_guard(
    embedder_for: Callable[[str], Embedder], path: Path = MODEL_PATH
) -> LayeredInjectionGuard:
    """Monta a guarda com o classificador versionado; sem pesos válidos, só heurística.

    Args:
        embedder_for: Devolve o embedder do modelo nomeado nos pesos (carga preguiçosa).
        path: JSON dos pesos.

    Returns:
        Guarda em camadas, ou só heurística com ``degraded=True`` se os pesos falharem.
    """
    try:
        model = LogisticModel.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning(
            "pesos do classificador de injecao invalidos (%s); so heuristica", type(exc).__name__
        )
        return LayeredInjectionGuard(degraded=True)
    return LayeredInjectionGuard(
        EmbeddingInjectionClassifier(embedder_for(model.embedding_model), model)
    )
