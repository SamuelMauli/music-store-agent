"""Limite de mensagens por sessão (janela deslizante)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class RateLimitedError(Exception):
    """A sessão excedeu o limite de mensagens."""

    def __init__(self, retry_after: int) -> None:
        """Cria o erro.

        Args:
            retry_after: Segundos até a próxima mensagem ser aceita.
        """
        super().__init__(f"limite de mensagens excedido; tente em {retry_after}s")
        self.retry_after = retry_after


class SlidingWindowLimiter:
    """Aceita até ``limit`` mensagens por sessão em ``window`` segundos."""

    def __init__(
        self,
        limit: int = 20,
        window: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Cria o limitador."""
        self._limit, self._window, self._now = limit, window, monotonic
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, session_id: str) -> None:
        """Registra a mensagem ou levanta :class:`RateLimitedError`."""
        now = self._now()
        with self._lock:
            hits = self._hits[session_id]
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                raise RateLimitedError(max(1, int(self._window - (now - hits[0])) + 1))
            hits.append(now)
