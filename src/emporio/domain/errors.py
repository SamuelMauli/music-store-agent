"""Exceções de domínio."""

from __future__ import annotations


class DomainError(Exception):
    """Base de todos os erros de regra de negócio."""


class InvalidValueError(DomainError, ValueError):
    """Valor que viola o invariante de um value object ou entidade."""
