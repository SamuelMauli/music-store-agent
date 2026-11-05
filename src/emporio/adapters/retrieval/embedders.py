"""Embedders: um determinístico para testes e o fastembed real."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from emporio.domain.text import normalize_text

if TYPE_CHECKING:
    from fastembed import TextEmbedding

# Menor modelo multilíngue (PT incluso) da lista do fastembed: 0,22 GB, 384 dims.
# `multilingual-e5-small` não consta em `TextEmbedding.list_supported_models()`.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = Path.home() / ".cache" / "fastembed"
_WORD = re.compile(r"\w+")


def _unit(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class FakeEmbedder:
    """Saco de palavras por hash: determinístico, sem rede, sem download.

    Não tem semântica (só sobreposição de vocabulário); serve para testar o pipeline.
    """

    def __init__(self, dim: int = 256) -> None:
        """Define a dimensão do espaço de hash."""
        self._dim = dim

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Vetores normalizados; texto sem palavras vira vetor nulo."""
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(normalize_text(text)):
                digest = hashlib.sha256(word.encode()).digest()
                out[row, int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        return _unit(out)


class FastEmbedEmbedder:
    """Modelo real via fastembed (ONNX, cache em ``~/.cache``); carregado sob demanda."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Guarda o nome; o download só acontece no primeiro ``embed``."""
        self._model_name = model_name
        self._model: TextEmbedding | None = None

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Vetores do modelo, renormalizados em L2."""
        if self._model is None:
            from fastembed import TextEmbedding  # noqa: PLC0415 - import caro, só se usado

            self._model = TextEmbedding(model_name=self._model_name, cache_dir=str(CACHE_DIR))
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return _unit(vectors)
