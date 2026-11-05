"""Tentativas de verificação por chave (sessão, pedido, contato, IP), com TTL.

``AttemptStore`` é o protocolo; a implementação em memória serve a um processo só. Para
várias réplicas, ``RedisAttemptStore`` guarda o contador no Redis (``INCR`` + ``PEXPIRE``).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from redis import Redis


class AttemptStore(Protocol):
    """Contador de falhas com expiração."""

    def count(self, key: str) -> int:
        """Falhas registradas para ``key`` dentro do TTL."""
        ...

    def register(self, key: str) -> int:
        """Soma uma falha e devolve o total."""
        ...


def hashed(kind: str, value: str) -> str:
    """Chave sem PII em memória: ``kind:sha256(valor)[:16]``."""
    return f"{kind}:{hashlib.sha256(value.encode()).hexdigest()[:16]}"


class InMemoryAttemptStore:
    """Falhas por chave; cada falha expira ``ttl`` segundos depois da última da chave."""

    def __init__(
        self,
        ttl: float = 900.0,
        max_keys: int = 10_000,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Cria o store."""
        self._ttl, self._max_keys, self._now = ttl, max_keys, monotonic
        self._data: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Número de chaves em memória."""
        return len(self._data)

    def count(self, key: str) -> int:
        """Falhas vigentes de ``key``."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return 0
            if self._now() >= entry[1]:
                del self._data[key]
                return 0
            return entry[0]

    def register(self, key: str) -> int:
        """Soma uma falha (renova o TTL) e devolve o total."""
        now = self._now()
        with self._lock:
            if len(self._data) >= self._max_keys:
                for stale in [k for k, (_, exp) in self._data.items() if now >= exp]:
                    del self._data[stale]
                while len(self._data) >= self._max_keys:
                    del self._data[next(iter(self._data))]
            entry = self._data.pop(key, None)
            total = (entry[0] if entry and now < entry[1] else 0) + 1
            self._data[key] = (total, now + self._ttl)
            return total


class RedisAttemptStore:
    """Falhas por chave no Redis, valendo entre réplicas.

    Mesmas regras da versão em memória: cada falha renova o TTL da chave, então o
    bloqueio expira ``ttl`` segundos depois da última falha. ``INCR`` e ``PEXPIRE`` vão
    num ``MULTI``/``EXEC``: nunca sobra contador sem expiração.
    """

    def __init__(self, client: Redis, ttl: float = 900.0, *, prefix: str = "emporio:att:") -> None:
        """Cria o store.

        Args:
            client: Cliente ``redis.Redis``.
            ttl: Segundos de vida do contador desde a última falha.
            prefix: Prefixo das chaves.
        """
        self._redis, self._ttl_ms, self._prefix = client, int(ttl * 1000), prefix

    def count(self, key: str) -> int:
        """Falhas vigentes de ``key``."""
        value = self._redis.get(self._prefix + key)
        return int(value) if value is not None else 0

    def register(self, key: str) -> int:
        """Soma uma falha (renova o TTL) e devolve o total."""
        pipe = self._redis.pipeline(transaction=True)
        pipe.incr(self._prefix + key)
        pipe.pexpire(self._prefix + key, self._ttl_ms)
        return int(pipe.execute()[0])
