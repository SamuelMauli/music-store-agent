"""Exceções de domínio."""

from __future__ import annotations


class DomainError(Exception):
    """Base de todos os erros de regra de negócio."""


class InvalidValueError(DomainError, ValueError):
    """Valor que viola o invariante de um value object ou entidade."""


class MessageDeliveryError(DomainError):
    """Falha ao entregar uma mensagem a um canal externo (WhatsApp)."""


class HandoffNotFoundError(DomainError):
    """A conversa não está em atendimento humano (nunca esteve, voltou ao bot ou expirou)."""


class OperatorMismatchError(DomainError):
    """O nome de operador informado difere do dono da chave de API que autenticou."""


class HandoffConflictError(DomainError):
    """A ação conflita com o dono atual da conversa (outro operador, ou ninguém assumiu)."""
