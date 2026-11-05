"""Índice vetorial em memória (NumPy): produto interno de vetores normalizados."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class InMemoryVectorIndex:
    """Busca exata por cosseno; suficiente para as dezenas de seções do manual."""

    def __init__(self) -> None:
        """Cria o índice vazio."""
        self._ids: list[str] = []
        self._matrix: NDArray[np.float32] | None = None

    def add(self, ids: list[str], vectors: NDArray[np.float32]) -> None:
        """Acrescenta vetores de norma 1 aos já indexados.

        Raises:
            ValueError: Se a quantidade de ids não bate com a de vetores.
        """
        if len(ids) != len(vectors):
            raise ValueError("ids e vetores com tamanhos diferentes")
        self._ids.extend(ids)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, vector: NDArray[np.float32], k: int) -> list[tuple[str, float]]:
        """Top-``k`` por similaridade; empate resolvido pela ordem de inserção."""
        if self._matrix is None or k <= 0:
            return []
        scores = self._matrix @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[i], float(scores[i])) for i in order]
