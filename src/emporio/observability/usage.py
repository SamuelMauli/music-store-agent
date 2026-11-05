"""Consumo acumulado por conversa (tokens e custo estimado)."""

from __future__ import annotations

import threading

from emporio.agent.llm.types import Usage


class UsageAccumulator:
    """Soma o consumo de cada turno por sessão; seguro para uso concorrente (API)."""

    def __init__(self) -> None:
        """Cria o acumulador vazio."""
        self._lock = threading.Lock()
        self._totals: dict[str, Usage] = {}

    def add(self, session_id: str, usage: Usage) -> Usage:
        """Soma o turno à conversa e devolve o total da conversa."""
        with self._lock:
            total = self._totals.get(session_id, Usage()) + usage
            self._totals[session_id] = total
            return total

    def total(self, session_id: str) -> Usage:
        """Total da conversa (zero se a sessão é desconhecida)."""
        with self._lock:
            return self._totals.get(session_id, Usage())

    def reset(self, session_id: str) -> None:
        """Esquece a conversa."""
        with self._lock:
            self._totals.pop(session_id, None)
