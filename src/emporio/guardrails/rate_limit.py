"""Limite de mensagens (janela deslizante) com memória limitada: TTL e teto de chaves."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimitedError(Exception):
    """O limite de mensagens foi excedido."""

    def __init__(self, retry_after: int) -> None:
        """Cria o erro.

        Args:
            retry_after: Segundos até a próxima mensagem ser aceita.
        """
        super().__init__(f"limite de mensagens excedido; tente em {retry_after}s")
        self.retry_after = retry_after


class SlidingWindowLimiter:
    """Aceita até ``limit`` eventos por chave em ``window`` segundos.

    A chave é uma sessão, um IP etc. Deques vazios são descartados (varredura a cada
    ``window`` segundos) e o número de chaves é limitado a ``max_keys``: ao estourar,
    a chave menos recentemente usada sai.
    """

    def __init__(
        self,
        limit: int = 20,
        window: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        """Cria o limitador."""
        self._limit, self._window, self._now, self._max_keys = limit, window, monotonic, max_keys
        self._hits: dict[str, deque[float]] = {}
        self._swept_at = monotonic()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Número de chaves em memória."""
        return len(self._hits)

    def _sweep(self, now: float) -> None:
        for key in [
            k for k, hits in self._hits.items() if not hits or now - hits[-1] >= self._window
        ]:
            del self._hits[key]
        self._swept_at = now

    def check(self, key: str) -> None:
        """Registra o evento ou levanta :class:`RateLimitedError`."""
        now = self._now()
        with self._lock:
            if now - self._swept_at >= self._window or len(self._hits) >= self._max_keys:
                self._sweep(now)
            hits = self._hits.pop(key, None) or deque()  # reinsere no fim: ordem = recência
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            try:
                if len(hits) >= self._limit:
                    raise RateLimitedError(max(1, int(self._window - (now - hits[0])) + 1))
                hits.append(now)
            finally:
                if hits:
                    self._hits[key] = hits
                while len(self._hits) > self._max_keys:
                    del self._hits[next(iter(self._hits))]
