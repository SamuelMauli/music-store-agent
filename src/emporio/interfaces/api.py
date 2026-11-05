"""API HTTP: ``POST /v1/chat`` (SSE), ``GET /healthz`` e ``GET /readyz``."""

from __future__ import annotations

import json
import logging
import re
import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse, ServerSentEvent

from emporio.agent.service import ChatResult
from emporio.bootstrap import Container, build_container
from emporio.guardrails.rate_limit import RateLimitedError
from emporio.guardrails.session import InvalidSessionError, SessionSigner
from emporio.settings import Settings

log = logging.getLogger("emporio.api")
REQUEST_ID = "X-Request-ID"
_SAFE_ID = re.compile(r"^[\w.-]{1,64}$")
_CHUNK_WORDS = 4


class ChatRequest(BaseModel):
    """Corpo de ``POST /v1/chat``."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        pattern=r"^[\w.-]{1,128}$",
        description="Token emitido pelo servidor (campo session_id do evento done); omita na 1ª.",
    )


def _error(status: int, code: str, message: str, request_id: str, **headers: str) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "request_id": request_id}}
    return JSONResponse(body, status_code=status, headers={REQUEST_ID: request_id, **headers})


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def _chunks(text: str) -> Iterator[str]:
    words = text.split(" ")
    for start in range(0, len(words), _CHUNK_WORDS):
        yield " ".join(words[start : start + _CHUNK_WORDS]) + (
            " " if start + _CHUNK_WORDS < len(words) else ""
        )


def _client_ip(request: Request, trusted_hops: int) -> str:
    """IP do cliente; com proxies confiáveis, o N-ésimo da direita de ``X-Forwarded-For``."""
    if trusted_hops > 0:
        hops = [
            h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()
        ]
        if len(hops) >= trusted_hops:
            return hops[-trusted_hops]
    return request.client.host if request.client else ""


def _done(result: ChatResult, token: str) -> dict[str, Any]:
    data = asdict(result)
    data["session_id"] = token
    data["usage"] = result.usage.to_dict()
    data["conversation_usage"] = result.conversation_usage.to_dict()
    del data["answer"]
    return data


def _events(
    stream: Iterator[tuple[str, Any]], request_id: str, token: str
) -> Iterator[ServerSentEvent]:
    try:
        for kind, payload in stream:
            if kind == "node":
                yield ServerSentEvent(json.dumps({"node": payload}), event="step")
                continue
            for chunk in _chunks(payload.answer):
                yield ServerSentEvent(
                    json.dumps({"text": chunk}, ensure_ascii=False), event="token"
                )
            yield ServerSentEvent(
                json.dumps(_done(payload, token), ensure_ascii=False), event="done"
            )
    except Exception:
        log.exception("falha no turno request_id=%s", request_id)
        error = {"code": "internal_error", "message": "Falha ao processar a mensagem."}
        yield ServerSentEvent(json.dumps(error), event="error")


def _resolve(signer: SessionSigner, supplied: str | None) -> tuple[str, str] | None:
    """(token, id) da sessão: emite uma nova se não veio token; ``None`` se não assinado."""
    token = supplied or signer.issue()
    try:
        return token, signer.verify(token)
    except InvalidSessionError:
        return None


def _signer(config: Settings) -> SessionSigner:
    configured = config.session_secret.get_secret_value() if config.session_secret else ""
    if not configured:
        log.warning(
            "EMPORIO_SESSION_SECRET ausente: usando segredo aleatório deste processo; "
            "as sessões deixam de valer ao reiniciar e não funcionam com várias réplicas"
        )
        return SessionSigner(secrets.token_bytes(32))
    return SessionSigner(configured.encode())


def _register_errors(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) for e in exc.errors())
        return _error(422, "invalid_request", f"Campos inválidos: {fields}", _request_id(request))

    @app.exception_handler(RateLimitedError)
    async def limited(request: Request, exc: RateLimitedError) -> JSONResponse:
        return _error(
            429,
            "rate_limited",
            str(exc),
            _request_id(request),
            **{"Retry-After": str(exc.retry_after)},
        )


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Monta a aplicação.

    Args:
        settings: Configuração; default lê o ambiente.
        container: Container já pronto (testes); default é criado no startup.
    """
    config = settings or Settings()
    signer = _signer(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or build_container(config)
        yield
        if container is None:
            app.state.container.close()

    app = FastAPI(title="Empório da Música · agente Tom", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID, "")
        request.state.request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        response = await call_next(request)
        response.headers[REQUEST_ID] = request.state.request_id
        return response

    _register_errors(app)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request) -> JSONResponse:
        try:
            request.app.state.container.connection.execute("select 1").fetchone()
        except Exception:
            return _error(503, "not_ready", "Banco indisponível.", _request_id(request))
        return JSONResponse({"status": "ready"})

    @app.post("/v1/chat", response_class=EventSourceResponse)
    def chat(body: ChatRequest, request: Request) -> Response:
        rid = _request_id(request)
        current: Container = request.app.state.container
        resolved = _resolve(signer, body.session_id)
        if resolved is None:
            return _error(400, "invalid_session", "Sessão inválida ou expirada.", rid)
        token, session_id = resolved
        ip = _client_ip(request, config.trusted_proxy_hops)
        events = current.service.stream(session_id, body.message, rid, ip)
        return EventSourceResponse(
            _events(events, rid, token), headers={REQUEST_ID: rid, "X-Session-Id": token}
        )

    return app
