"""Limite de mensagens (janela deslizante) com memória limitada: TTL e teto de chaves."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis


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


class RedisSlidingWindowLimiter:
    """Janela deslizante compartilhada entre réplicas, sobre sorted set no Redis.

    Cada evento é um membro do ``ZSET`` da chave (score = instante em ms). Remover os
    vencidos, adicionar, contar e renovar o TTL rodam num único ``MULTI``/``EXEC``, então
    nenhuma réplica enxerga o estado pela metade. Se a contagem passa do limite, o membro
    recém-adicionado é removido e o evento é negado. Um negado ainda não removido pode
    negar de forma conservadora uma chamada simultânea, mas nunca deixa passar acima do
    limite. A chave vira ``sha256`` truncado: IP e sessão não ficam em claro no Redis.
    O relógio é o do processo (``time.time``); desvio entre réplicas é irrelevante na
    escala de uma janela de segundos.
    """

    def __init__(
        self,
        client: Redis,
        limit: int = 20,
        window: float = 60.0,
        *,
        prefix: str = "emporio:rl:",
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Cria o limitador.

        Args:
            client: Cliente ``redis.Redis`` (``decode_responses`` indiferente).
            limit: Eventos aceitos por chave na janela.
            window: Largura da janela, em segundos.
            prefix: Prefixo das chaves; use um por limitador (sessão, IP).
            clock: Relógio em segundos (epoch); injetável nos testes.
        """
        self._redis, self._limit, self._window = client, limit, window
        self._prefix, self._clock = prefix, clock

    def check(self, key: str) -> None:
        """Registra o evento ou levanta :class:`RateLimitedError`."""
        now_ms = int(self._clock() * 1000)
        window_ms = int(self._window * 1000)
        name = self._prefix + hashlib.sha256(key.encode()).hexdigest()[:32]
        member = f"{now_ms}-{uuid.uuid4().hex}"
        pipe = self._redis.pipeline(transaction=True)
        pipe.zremrangebyscore(name, "-inf", now_ms - window_ms)
        pipe.zadd(name, {member: now_ms})
        pipe.zcard(name)
        pipe.zrange(name, 0, 0, withscores=True)
        pipe.pexpire(name, window_ms)
        results = pipe.execute()
        if int(results[2]) > self._limit:
            self._redis.zrem(name, member)
            oldest = float(results[3][0][1])
            wait = (window_ms - (now_ms - oldest)) / 1000
            raise RateLimitedError(max(1, int(wait) + 1))
