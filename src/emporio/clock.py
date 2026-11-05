"""Implementações de :class:`emporio.ports.Clock`."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from emporio.domain.errors import InvalidValueError


class FixedClock:
    """Relógio congelado em ``REFERENCE_DATETIME`` (dados vão de out/2025 a mar/2026)."""

    def __init__(self, instant: datetime) -> None:
        """Cria o relógio.

        Args:
            instant: Instante fixo, obrigatoriamente com fuso.

        Raises:
            InvalidValueError: Se ``instant`` for naive.
        """
        if instant.tzinfo is None:
            raise InvalidValueError("FixedClock exige datetime com fuso")
        self._instant = instant

    def now(self) -> datetime:
        """Retorna sempre o mesmo instante."""
        return self._instant


class SystemClock:
    """Relógio real no fuso da loja."""

    def __init__(self, timezone: str = "America/Campo_Grande") -> None:
        """Cria o relógio.

        Args:
            timezone: Nome IANA do fuso.
        """
        self._tz = ZoneInfo(timezone)

    def now(self) -> datetime:
        """Retorna o instante atual no fuso configurado."""
        return datetime.now(self._tz)
