"""Handoff humano e deduplicação de mensagens no Redis (valem entre réplicas)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from emporio.adapters.persistence.sqlite_handoff import DEDUP_TTL_SECONDS
from emporio.domain.handoff import (
    ClaimResult,
    HandoffConversation,
    HandoffMessage,
    HandoffState,
)

if TYPE_CHECKING:
    from redis import Redis
    from redis.client import Pipeline

_GRACE_SECONDS = 3_600  # o Redis descarta sozinho um pouco depois da expiração lógica
type _Data = dict[str, Any]


class RedisHandoffStore:
    """:class:`~emporio.ports.HandoffStore` e :class:`~emporio.ports.MessageDeduplicator`.

    A conversa é um JSON em ``conv:<id>`` (alterações sob ``WATCH``/``MULTI``), o histórico
    uma lista em ``msgs:<id>`` e ``index`` o conjunto de ids ativos (limpo sob demanda).
    """

    def __init__(self, client: Redis, *, prefix: str = "emporio:wa:") -> None:
        """Cria o adaptador.

        Args:
            client: Cliente ``redis.Redis``.
            prefix: Prefixo das chaves.
        """
        self._r, self._p = client, prefix

    def _conv(self, conv_id: str) -> str:
        return f"{self._p}conv:{conv_id}"

    def _msgs(self, conv_id: str) -> str:
        return f"{self._p}msgs:{conv_id}"

    def _read(self, conv_id: str, now: float, pipe: Pipeline | None = None) -> _Data | None:
        raw = (pipe or self._r).get(self._conv(conv_id))
        if raw is None:
            return None
        data: _Data = json.loads(raw)
        return data if data["expires_at"] > now else None

    def _update[T](
        self,
        conv_id: str,
        now: float,
        change: Callable[[_Data | None], tuple[_Data | None, T]],
        effects: Callable[[Pipeline], None] | None = None,
    ) -> T:
        """Lê-altera-grava sob ``WATCH``: ``change(dados) -> (dados novos ou None, resultado)``."""
        key = self._conv(conv_id)

        def run(pipe: Pipeline) -> T:
            updated, result = change(self._read(conv_id, now, pipe))
            if updated is not None:
                pipe.multi()
                pipe.set(key, json.dumps(updated), exat=int(updated["expires_at"]) + _GRACE_SECONDS)
                if effects is not None:
                    effects(pipe)  # entram na mesma MULTI: tudo ou nada
            return result

        return cast(T, self._r.transaction(run, key, value_from_callable=True))  # type: ignore[arg-type]

    def enter(self, conv_id: str, phone: str, now: float, expires_at: float) -> bool:
        """Põe a conversa em handoff; ``False`` se já estava."""

        def change(data: _Data | None) -> tuple[_Data | None, bool]:
            if data is not None:
                return None, False
            fresh = {
                "phone": phone,
                "state": HandoffState.WAITING.value,
                "operator": None,
                "entered_at": now,
                "expires_at": expires_at,
            }
            return fresh, True

        def effects(pipe: Pipeline) -> None:
            pipe.delete(self._msgs(conv_id))  # descarta histórico de uma conversa expirada
            pipe.sadd(f"{self._p}index", conv_id)

        return self._update(conv_id, now, change, effects)

    def is_human(self, conv_id: str, now: float) -> bool:
        """``True`` se a conversa está em handoff e não expirou."""
        return self._read(conv_id, now) is not None

    def append(self, conv_id: str, direction: str, text: str, now: float) -> None:
        """Anexa mensagem ao histórico (sem efeito se a conversa não existe)."""
        data = self._read(conv_id, now)
        if data is None:
            return
        key = self._msgs(conv_id)
        pipe = self._r.pipeline()
        pipe.rpush(key, json.dumps({"d": direction, "t": text, "a": now}))
        pipe.expireat(key, int(data["expires_at"]) + _GRACE_SECONDS)
        pipe.execute()

    def _build(self, conv_id: str, data: _Data) -> HandoffConversation:
        messages = tuple(
            HandoffMessage(m["d"], m["t"], m["a"])
            for m in (json.loads(raw) for raw in self._r.lrange(self._msgs(conv_id), 0, -1))
        )
        return HandoffConversation(
            conv_id,
            data["phone"],
            HandoffState(data["state"]),
            data["operator"],
            data["entered_at"],
            data["expires_at"],
            messages,
        )

    def get(self, conv_id: str, now: float) -> HandoffConversation | None:
        """Conversa com o histórico, ou ``None``."""
        data = self._read(conv_id, now)
        return self._build(conv_id, data) if data else None

    def list_active(self, now: float) -> list[HandoffConversation]:
        """Conversas ativas, da mais antiga à mais recente."""
        found: list[HandoffConversation] = []
        for raw_id in self._r.smembers(f"{self._p}index"):
            conv_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            data = self._read(conv_id, now)
            if data is None:
                self._r.srem(f"{self._p}index", conv_id)
                continue
            found.append(self._build(conv_id, data))
        return sorted(found, key=lambda c: (c.entered_at, c.id))

    def claim(self, conv_id: str, operator: str, now: float, expires_at: float) -> ClaimResult:
        """Assume a conversa; ``TAKEN`` se outro operador é o dono."""

        def change(data: _Data | None) -> tuple[_Data | None, ClaimResult]:
            if data is None:
                return None, ClaimResult.NOT_FOUND
            if data["operator"] not in (None, operator):
                return None, ClaimResult.TAKEN
            data |= {
                "state": HandoffState.ASSIGNED.value,
                "operator": operator,
                "expires_at": expires_at,
            }
            return data, ClaimResult.OK

        return self._update(conv_id, now, change)

    def extend(self, conv_id: str, operator: str, expires_at: float, now: float) -> bool:
        """Renova a expiração se ``operator`` é o dono."""

        def change(data: _Data | None) -> tuple[_Data | None, bool]:
            if data is None or data["operator"] != operator:
                return None, False
            data["expires_at"] = expires_at
            return data, True

        return self._update(conv_id, now, change)

    def release(
        self, conv_id: str, operator: str
    ) -> tuple[ClaimResult, HandoffConversation | None]:
        """Devolve ao bot se ``operator`` é o dono (ou não há dono); apaga conversa e histórico."""
        key = self._conv(conv_id)
        outcome: list[tuple[ClaimResult, HandoffConversation | None]] = []

        def run(pipe: Pipeline) -> None:
            outcome.clear()
            raw = pipe.get(key)  # WATCH em ``key``: mudança de dono até o EXEC aborta e repete
            if raw is None:
                outcome.append((ClaimResult.NOT_FOUND, None))
                return
            data: _Data = json.loads(raw)
            if data["operator"] not in (None, operator):
                outcome.append((ClaimResult.TAKEN, None))
                return
            released = self._build(conv_id, data)
            pipe.multi()
            pipe.delete(key, self._msgs(conv_id))
            pipe.srem(f"{self._p}index", conv_id)
            outcome.append((ClaimResult.OK, released))

        self._r.transaction(run, key)
        return outcome[-1]

    def first_seen(self, message_id: str, now: float) -> bool:
        """``True`` na 1ª vez que o id aparece (``SET NX`` com TTL de 24h)."""
        key = f"{self._p}seen:{hashlib.sha256(message_id.encode()).hexdigest()[:32]}"
        return bool(self._r.set(key, "1", nx=True, ex=int(DEDUP_TTL_SECONDS)))
