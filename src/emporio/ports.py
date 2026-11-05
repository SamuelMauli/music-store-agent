"""Portas (Protocols) pelas quais o domínio fala com o mundo externo."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    """Fonte de "agora" injetada, para que prazos sejam determinísticos."""

    def now(self) -> datetime:
        """Retorna o instante atual como datetime *aware*."""
        ...
