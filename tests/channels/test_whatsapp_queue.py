"""Fila do canal WhatsApp: justiça entre clientes, robustez do worker e encerramento."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

from emporio.channels.whatsapp import InboundMessage, WhatsAppChannel
from emporio.guardrails.rate_limit import RateLimitedError

FLOODER = "5511999990000"
OTHER = "5511888880000"


class Rig:
    """Canal com o LLM travado num portão, para observar a fila sem depender de tempo."""

    def __init__(self, *, workers: int = 4, queue_size: int = 64, per_conv: int = 10) -> None:
        self.gate = threading.Semaphore(0)
        self.service = MagicMock()
        self.service.chat.side_effect = self._chat
        self.service.snapshot.return_value = {}
        store = MagicMock()
        store.is_human.return_value = False
        self.dedup = MagicMock()
        self.dedup.first_seen.return_value = True
        self.limiter = MagicMock()
        self.channel = WhatsAppChannel(
            self.service,
            MagicMock(),
            store,
            self.dedup,
            self.limiter,
            pepper="p",
            max_message_chars=1000,
            handoff_ttl_seconds=60,
            workers=workers,
            queue_size=queue_size,
            conversation_queue=per_conv,
        )

    def _chat(self, *args: Any, **kwargs: Any) -> Any:
        self.gate.acquire(timeout=5)
        return MagicMock(answer="ok", handoff=False)

    def send(self, phone: str, n: int, prefix: str = "m") -> int:
        msgs = (InboundMessage(f"{prefix}{phone}{i}", phone, "text", "oi") for i in range(n))
        return sum(self.channel.submit(m) for m in msgs)

    def finish(self) -> None:
        self.gate.release(1000)
        assert self.channel.drain(10)
        self.channel.close()


def test_one_number_burst_does_not_starve_another_customer() -> None:
    rig = Rig(per_conv=10)
    accepted = rig.send(FLOODER, 68)  # a rajada de 68 do cenário provado pelo verificador
    other = rig.channel.submit(InboundMessage("x", OTHER, "text", "socorro"))
    rig.finish()
    assert accepted == 10  # o excedente de UM número morre antes da fila global
    assert other is True


def test_dedup_and_rate_limit_run_before_taking_a_slot() -> None:
    rig = Rig(workers=1, queue_size=1, per_conv=5)
    rig.limiter.check.side_effect = [None, RateLimitedError(1.0)] + [None] * 10
    assert rig.channel.submit(InboundMessage("a", FLOODER, "text", "oi")) is True
    assert rig.channel.submit(InboundMessage("b", FLOODER, "text", "oi")) is False  # limitada
    rig.dedup.first_seen.return_value = False
    assert rig.channel.submit(InboundMessage("a", FLOODER, "text", "oi")) is True  # reentrega
    rig.dedup.first_seen.return_value = True
    # nada disso ocupou a fila global: outro cliente ainda entra
    assert rig.channel.submit(InboundMessage("c", OTHER, "text", "oi")) is True
    rig.finish()


def test_worker_exception_releases_the_conversation_and_the_slot() -> None:
    rig = Rig(workers=1, queue_size=4, per_conv=5)
    rig.gate.release(100)
    rig.channel.handle = MagicMock(side_effect=[RuntimeError("boom"), None])  # type: ignore[method-assign]
    assert rig.channel.submit(InboundMessage("1", FLOODER, "text", "a"))
    assert rig.channel.submit(InboundMessage("2", FLOODER, "text", "b"))
    assert rig.channel.drain(10)  # inflight voltou a zero apesar da exceção
    assert rig.channel.submit(InboundMessage("3", FLOODER, "text", "c"))  # conversa livre
    assert rig.channel.drain(10)
    rig.channel.close()


def test_shutdown_returns_the_slots_of_discarded_messages() -> None:
    rig = Rig(workers=1, queue_size=8, per_conv=8)
    assert rig.send(FLOODER, 3) == 3  # 1 rodando (preso no portão) + 2 na fila da conversa
    assert rig.send(OTHER, 2) == 2  # 1 tarefa cancelada no pool + 1 na fila dela
    closer = threading.Thread(target=rig.channel.close)
    closer.start()
    rig.gate.release(1000)
    closer.join(10)
    assert not closer.is_alive()
    assert rig.channel.drain(1)  # contabilidade zerada: nada vaza
    assert rig.channel.submit(InboundMessage("z", OTHER, "text", "oi")) is False  # fechado
