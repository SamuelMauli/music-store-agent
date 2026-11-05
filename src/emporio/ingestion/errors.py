"""Erros do pipeline de ingestão."""

from __future__ import annotations

from emporio.domain.errors import DomainError


class IngestionError(DomainError):
    """Falha estrutural na ingestão (arquivo ausente, contrato inválido, cabeçalho inesperado)."""
