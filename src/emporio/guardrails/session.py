"""Sessões emitidas pelo servidor: id aleatório de 128 bits assinado com HMAC-SHA256."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_ID_BYTES = 16  # 128 bits
_TAG_BYTES = 16  # tag truncada em 128 bits


class InvalidSessionError(ValueError):
    """O token de sessão não foi emitido por este servidor (ou foi adulterado)."""


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class SessionSigner:
    """Emite e valida tokens ``<id>.<tag>``; o ``id`` é o ``thread_id`` do checkpointer."""

    def __init__(self, secret: bytes) -> None:
        """Cria o assinador.

        Args:
            secret: Segredo HMAC (mínimo 16 bytes).
        """
        if len(secret) < 16:
            raise ValueError("segredo de sessão curto demais (mínimo 16 bytes)")
        self._secret = secret

    def _tag(self, session_id: str) -> str:
        digest = hmac.new(self._secret, session_id.encode(), hashlib.sha256).digest()
        return _b64(digest[:_TAG_BYTES])

    def issue(self) -> str:
        """Gera um token novo."""
        session_id = _b64(secrets.token_bytes(_ID_BYTES))
        return f"{session_id}.{self._tag(session_id)}"

    def verify(self, token: str) -> str:
        """Devolve o id da sessão se a assinatura confere.

        Raises:
            InvalidSessionError: Token malformado ou não assinado por este servidor.
        """
        session_id, _, tag = token.partition(".")
        if not session_id or not tag or not hmac.compare_digest(tag, self._tag(session_id)):
            raise InvalidSessionError("sessão inválida")
        return session_id
