"""Classificador de injeção: embeddings (porta ``Embedder``) + regressão logística em numpy.

Sem sklearn: são 384 pesos e um viés; o ajuste é Newton-Raphson (IRLS) com L2, determinístico
e com convergência em poucas iterações. Os pesos versionados ficam em
``guardrails/injection_model.json`` (gerados por ``make train-guard``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

MODEL_PATH: Final = Path(__file__).with_name("injection_model.json")
_MAX_ITER: Final = 50
_TOL: Final = 1e-8


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


@dataclass(frozen=True, slots=True)
class LogisticModel:
    """Regressão logística binária sobre vetores de embedding.

    Attributes:
        weights: Um peso por dimensão.
        bias: Intercepto.
        threshold: Probabilidade a partir da qual a mensagem conta como injeção.
        embedding_model: Nome do modelo de embedding com que os pesos foram treinados.
    """

    weights: NDArray[np.float64]
    bias: float
    threshold: float
    embedding_model: str

    @property
    def dim(self) -> int:
        """Dimensão esperada do embedding."""
        return int(self.weights.shape[0])

    def proba(self, features: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Probabilidade de injeção por linha de ``features``.

        Raises:
            ValueError: Se a dimensão não bater com a dos pesos.
        """
        if features.ndim != 2 or features.shape[1] != self.dim:
            raise ValueError(f"embedding com dimensão {features.shape}, esperado (n, {self.dim})")
        return _sigmoid(features.astype(np.float64) @ self.weights + self.bias)

    def to_dict(self) -> dict[str, Any]:
        """Forma serializável (pesos arredondados em 7 casas: arquivo estável e pequeno)."""
        return {
            "embedding_model": self.embedding_model,
            "dim": self.dim,
            "threshold": round(self.threshold, 4),
            "bias": round(self.bias, 7),
            "weights": [round(float(w), 7) for w in self.weights],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticModel:
        """Reconstrói o modelo do JSON, validando a dimensão.

        Raises:
            ValueError: JSON incompleto ou inconsistente.
        """
        weights = np.asarray(data["weights"], dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != int(data["dim"]):
            raise ValueError("pesos não batem com 'dim'")
        bias = float(data["bias"])
        if not (np.all(np.isfinite(weights)) and np.isfinite(bias)):
            raise ValueError("pesos ou viés não finitos (NaN/inf)")
        threshold = float(data["threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError("limiar fora de (0, 1)")
        return cls(weights, bias, threshold, str(data["embedding_model"]))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> LogisticModel:
        """Lê os pesos versionados."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_logistic(
    features: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    *,
    l2: float = 1.0,
    balanced: bool = True,
) -> tuple[NDArray[np.float64], float]:
    """Ajusta regressão logística com L2 por IRLS (Newton), sem aleatoriedade.

    Args:
        features: Matriz ``(n, d)``.
        labels: Vetor ``(n,)`` com 0 (benigno) e 1 (injeção).
        l2: Força da regularização (o viés não é regularizado).
        balanced: Pesa as classes pelo inverso da frequência.

    Returns:
        ``(pesos, viés)``.

    Raises:
        ValueError: Rótulos fora de {0, 1} ou só uma classe presente.
    """
    y = labels.astype(np.float64)
    if not set(np.unique(y)) == {0.0, 1.0}:
        raise ValueError("os rótulos precisam ter as duas classes, 0 e 1")
    x = np.hstack([features.astype(np.float64), np.ones((features.shape[0], 1))])
    n, d = x.shape
    sample_w = np.ones(n)
    if balanced:
        pos = y.mean()
        sample_w = np.where(y == 1.0, 0.5 / pos, 0.5 / (1.0 - pos))
    reg = np.full(d, l2)
    reg[-1] = 0.0
    theta = np.zeros(d)
    for _ in range(_MAX_ITER):
        p = _sigmoid(x @ theta)
        grad = x.T @ (sample_w * (p - y)) + reg * theta
        hess = (x * (sample_w * p * (1.0 - p))[:, None]).T @ x + np.diag(reg) + 1e-9 * np.eye(d)
        step = np.linalg.solve(hess, grad)
        theta -= step
        if float(np.max(np.abs(step))) < _TOL:
            break
    return theta[:-1], float(theta[-1])
