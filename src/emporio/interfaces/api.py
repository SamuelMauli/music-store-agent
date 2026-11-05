"""API HTTP: ``POST /v1/chat`` (SSE), ``GET /healthz`` e ``GET /readyz``."""

from __future__ import annotations

import json
import logging
import re
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
from emporio.settings import Settings

log = logging.getLogger("emporio.api")
REQUEST_ID = "X-Request-ID"
_SAFE_ID = re.compile(r"^[\w.-]{1,64}$")
_CHUNK_WORDS = 4


class ChatRequest(BaseModel):
    """Corpo de ``POST /v1/chat``."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[\w-]{1,64}$")


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


def _done(result: ChatResult) -> dict[str, Any]:
    data = asdict(result)
    data["usage"] = result.usage.to_dict()
    del data["answer"]
    return data


def _events(stream: Iterator[tuple[str, Any]], request_id: str) -> Iterator[ServerSentEvent]:
    try:
        for kind, payload in stream:
            if kind == "node":
                yield ServerSentEvent(json.dumps({"node": payload}), event="step")
                continue
            for chunk in _chunks(payload.answer):
                yield ServerSentEvent(
                    json.dumps({"text": chunk}, ensure_ascii=False), event="token"
                )
            yield ServerSentEvent(json.dumps(_done(payload), ensure_ascii=False), event="done")
    except Exception:
        log.exception("falha no turno request_id=%s", request_id)
        error = {"code": "internal_error", "message": "Falha ao processar a mensagem."}
        yield ServerSentEvent(json.dumps(error), event="error")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Monta a aplicação.

    Args:
        settings: Configuração; default lê o ambiente.
        container: Container já pronto (testes); default é criado no startup.
    """
    config = settings or Settings()

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
    def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
        rid = _request_id(request)
        current: Container = request.app.state.container
        events = current.service.stream(body.session_id, body.message, rid)
        return EventSourceResponse(_events(events, rid), headers={REQUEST_ID: rid})

    return app
