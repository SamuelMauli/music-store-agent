"""Middleware ASGI da API: request_id, autenticação, limite de corpo, cabeçalhos, log e métricas.

É ASGI puro (não ``BaseHTTPMiddleware``) para medir o SSE até o último byte e não bufferizar.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from emporio.observability.logging import request_id_var
from emporio.observability.metrics import Metrics

REQUEST_ID = "X-Request-ID"
PUBLIC_PATHS = frozenset({"/healthz", "/readyz"})
_API_PREFIX = "/v1/"
WEBHOOK_PATH = "/webhooks/whatsapp"  # sem chave de API: a Meta autentica por assinatura HMAC
OPERATOR_PREFIX = "/v1/operator/"  # só aceita as chaves de operador, nunca as do chat
_SAFE_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
log = logging.getLogger("emporio.access")


def error_response(
    status: int, code: str, message: str, request_id: str, **headers: str
) -> JSONResponse:
    """Envelope de erro único da API."""
    body = {"error": {"code": code, "message": message, "request_id": request_id}}
    return JSONResponse(body, status_code=status, headers={REQUEST_ID: request_id, **headers})


def _safe_id(value: str) -> bool:
    return 1 <= len(value) <= 64 and all(c in _SAFE_ID_CHARS for c in value)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


class _BodyTooLargeError(Exception):
    pass


class ApiGuard:
    """Aplica as políticas transversais da API a toda requisição HTTP."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_keys: Sequence[str] = (),
        max_body_bytes: int,
        metrics: Metrics | None = None,
        operator_keys: Sequence[tuple[str, str]] = (),
        body_limits: dict[str, int] | None = None,
    ) -> None:
        """Cria o middleware.

        Args:
            app: Aplicação ASGI interna.
            api_keys: Chaves aceitas; vazio deixa a API aberta (só permitido em dev).
            max_body_bytes: Maior corpo aceito (``413`` acima disso).
            metrics: Métricas HTTP; ``None`` desliga.
            operator_keys: Pares ``(nome, chave)`` dos endpoints ``/v1/operator/``; vazio nega
                tudo ali (diferente de ``api_keys``, nunca fica aberto). O nome da chave que
                autenticou vira ``request.state.operator``.
            body_limits: Limite de corpo por prefixo de caminho (vence ``max_body_bytes``).
        """
        self._app = app
        self._digests = [_digest(k) for k in api_keys if k]
        self._operators = [(n, _digest(k)) for n, k in operator_keys if k]
        self._body_limits = body_limits or {}
        self._max_body = max_body_bytes
        self._metrics = metrics

    def _identify(self, headers: Headers, path: str) -> str | None:
        """Nome do operador ("" fora da área de operador) ou ``None`` se a chave não vale."""
        if path == WEBHOOK_PATH:
            return ""
        names, digests = [""] * len(self._digests), self._digests
        if path.startswith(OPERATOR_PREFIX):
            if not self._operators:
                return None
            names = [n for n, _ in self._operators]
            digests = [d for _, d in self._operators]
        elif not digests:
            return ""
        supplied = headers.get("x-api-key", "")
        auth = headers.get("authorization", "")
        if auth[:7].lower() == "bearer ":
            supplied = auth[7:].strip()
        candidate = _digest(supplied)
        # Percorre todas as chaves, sem curto-circuito: tempo independe de qual casou.
        matches = [hmac.compare_digest(candidate, known) for known in digests]
        if not supplied or not any(matches):
            return None
        return names[matches.index(True)]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Ponto de entrada ASGI."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        incoming = headers.get(REQUEST_ID, "")
        rid = incoming if _safe_id(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = 500
        path = scope["path"]

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                out = MutableHeaders(scope=message)
                out[REQUEST_ID] = rid
                out["X-Content-Type-Options"] = "nosniff"
                out["Referrer-Policy"] = "no-referrer"
                if path.startswith(_API_PREFIX) or path == "/metrics":
                    out["Cache-Control"] = "no-store"
            await send(message)

        try:
            await self._handle(scope, receive, send_wrapper, headers, rid)
        finally:
            elapsed = time.perf_counter() - started
            self._record(scope, status, elapsed)
            request_id_var.reset(token)

    async def _handle(
        self, scope: Scope, receive: Receive, send: Send, headers: Headers, rid: str
    ) -> None:
        path = scope["path"]
        who = "" if path in PUBLIC_PATHS else self._identify(headers, path)
        if who is None:
            denied = error_response(
                401,
                "unauthorized",
                "Chave de API ausente ou inválida.",
                rid,
                **{"WWW-Authenticate": "Bearer"},
            )
            await denied(scope, receive, send)
            return
        if who:
            scope.setdefault("state", {})["operator"] = who
        limit = next(
            (v for p, v in self._body_limits.items() if path.startswith(p)), self._max_body
        )
        declared = headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            await self._too_large(scope, receive, send, rid)
            return

        seen = 0
        exceeded = False

        async def counting_receive() -> Message:
            nonlocal seen, exceeded
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    exceeded = True
                    raise _BodyTooLargeError
            return message

        async def dropping_send(message: Message) -> None:
            # O FastAPI converte a falha de leitura em 400; com o limite estourado a resposta
            # do app é descartada e quem responde é o 413 abaixo.
            if not exceeded:
                await send(message)

        with contextlib.suppress(_BodyTooLargeError):
            await self._app(scope, counting_receive, dropping_send)
        if exceeded:
            await self._too_large(scope, receive, send, rid)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send, rid: str) -> None:
        response = error_response(
            413, "payload_too_large", "Corpo da requisição grande demais.", rid
        )
        await response(scope, receive, send)

    def _record(self, scope: Scope, status: int, elapsed: float) -> None:
        route = getattr(scope.get("route"), "path", None) or "unmatched"
        method = str(scope["method"])
        if self._metrics is not None:
            self._metrics.observe_request(method, route, status, elapsed)
        extra: dict[str, Any] = {
            "event": "http_request",
            "method": method,
            "route": route,
            "status": status,
            "latency_ms": round(elapsed * 1000, 1),
            "session": scope.get("state", {}).get("session_hash"),
        }
        log.info("%s %s %s", method, route, status, extra=extra)
