"""Repositório de chamados em memória, para testes e execuções efêmeras."""

from __future__ import annotations

import threading
from datetime import datetime

from emporio.domain.ticket import Ticket, TicketKind, idempotency_key


class InMemoryTicketRepository:
    """:class:`~emporio.ports.TicketRepository` em memória, para testes e execuções efêmeras."""

    def __init__(self) -> None:
        self._by_key: dict[str, Ticket] = {}
        self._per_day: dict[str, int] = {}
        self._lock = threading.Lock()

    def open(
        self,
        session_id: str,
        kind: TicketKind,
        summary: str,
        order_id: int | None,
        now: datetime,
    ) -> Ticket:
        """Abre o chamado, ou devolve o já aberto para a mesma chave."""
        key = idempotency_key(session_id, kind, summary)
        with self._lock:
            if key in self._by_key:
                previous = self._by_key[key]
                return Ticket(
                    previous.protocol,
                    previous.kind,
                    previous.summary,
                    previous.order_id,
                    previous.session_id,
                    created=False,
                )
            stamp = now.strftime("%Y%m%d")
            self._per_day[stamp] = self._per_day.get(stamp, 0) + 1
            ticket = Ticket(
                f"EMP-{stamp}-{self._per_day[stamp]:04d}", kind, summary, order_id, session_id
            )
            self._by_key[key] = ticket
            return ticket
