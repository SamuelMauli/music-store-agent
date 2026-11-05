"""Rotas do canal WhatsApp: webhook da Meta e endpoints de operador do handoff."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Path, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from emporio.adapters.whatsapp.client import MAX_TEXT_CHARS
from emporio.channels.whatsapp import InboundMessage, WhatsAppChannel, parse_webhook
from emporio.domain.handoff import HandoffConversation
from emporio.interfaces.middleware import OPERATOR_PREFIX, WEBHOOK_PATH, error_response

log = logging.getLogger("emporio.whatsapp")

CONV_ID = r"^[0-9a-f]{32}$"
_CHALLENGE = r"^[\w-]{1,128}$"
SIGNATURE_PREFIX = "sha256="


class OperatorBody(BaseModel):
    """Quem age. A chave de API é compartilhada pela equipe: o nome só rotula o dono."""

    operator: str = Field(pattern=r"^[\w .@-]{1,64}$")


class ReplyBody(OperatorBody):
    """Resposta do operador ao cliente."""

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


def _eq(supplied: str, expected: str) -> bool:
    """Igualdade em tempo constante (compara digests, então o tamanho também não vaza)."""
    return hmac.compare_digest(
        hashlib.sha256(supplied.encode()).digest(), hashlib.sha256(expected.encode()).digest()
    )


def valid_signature(app_secret: str, body: bytes, header: str) -> bool:
    """``X-Hub-Signature-256`` = ``sha256=`` + HMAC-SHA256 do corpo bruto com o app secret."""
    if not header.startswith(SIGNATURE_PREFIX):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(header[len(SIGNATURE_PREFIX) :].encode(), expected.encode())


def _summary(c: HandoffConversation, preview: str) -> dict[str, Any]:
    return {
        "id": c.id,
        "state": c.state.value,
        "operator": c.operator,
        "entered_at": c.entered_at,
        "expires_at": c.expires_at,
        "customer_messages": sum(m.direction == "in" for m in c.messages),
        "preview": preview,
    }


def _detail(c: HandoffConversation, preview: str) -> dict[str, Any]:
    messages = [{"direction": m.direction, "text": m.text, "at": m.at} for m in c.messages]
    return {**_summary(c, preview), "messages": messages}


def channel_of(request: Request) -> WhatsAppChannel:
    """Canal montado no startup (``app.state.whatsapp``)."""
    channel: WhatsAppChannel = request.app.state.whatsapp
    return channel


def _process(channel: WhatsAppChannel, messages: list[InboundMessage]) -> None:
    for message in messages:
        channel.handle(message)


def build_router(*, verify_token: str, app_secret: str, phone_number_id: str) -> APIRouter:
    """Rotas do canal: webhook da Meta e endpoints de operador."""
    router = _webhook_router(verify_token, app_secret, phone_number_id)
    router.include_router(_operator_router())
    return router


def _webhook_router(verify_token: str, app_secret: str, phone_number_id: str) -> APIRouter:
    router = APIRouter()

    @router.get(WEBHOOK_PATH, include_in_schema=False)
    def verify(
        request: Request,
        mode: Annotated[str, Query(alias="hub.mode")] = "",
        token: Annotated[str, Query(alias="hub.verify_token")] = "",
        challenge: Annotated[str, Query(alias="hub.challenge", pattern=_CHALLENGE)] = "",
    ) -> Response:
        rid = str(getattr(request.state, "request_id", ""))
        # Compara sempre o token (sem curto-circuito no modo): tempo independe do que falhou.
        token_ok = _eq(token, verify_token)
        if mode != "subscribe" or not token_ok or not challenge:
            return error_response(403, "forbidden", "Verificação recusada.", rid)
        return PlainTextResponse(challenge)

    @router.post(WEBHOOK_PATH, include_in_schema=False)
    async def receive(request: Request, background: BackgroundTasks) -> Response:
        rid = str(getattr(request.state, "request_id", ""))
        body = await request.body()  # corpo BRUTO: a assinatura cobre os bytes, não o JSON
        signature = request.headers.get("x-hub-signature-256", "")
        if not valid_signature(app_secret, body, signature):
            log.warning("webhook com assinatura ausente ou inválida")
            return error_response(401, "invalid_signature", "Assinatura inválida.", rid)
        try:
            payload = json.loads(body)
        except ValueError:
            return error_response(400, "invalid_request", "Corpo não é JSON.", rid)
        messages = parse_webhook(payload, phone_number_id)
        if messages:  # responde 200 já; o agente (lento) roda depois, fora do ciclo da resposta
            background.add_task(_process, channel_of(request), messages)
        return JSONResponse({"status": "ok"})

    return router


def _operator_router() -> APIRouter:
    router = APIRouter()

    @router.get(f"{OPERATOR_PREFIX}handoffs")
    def queue(request: Request) -> dict[str, Any]:
        channel = channel_of(request)
        return {"items": [_summary(c, channel.preview(c)) for c in channel.queue()]}

    @router.get(OPERATOR_PREFIX + "handoffs/{conv_id}")
    def detail(request: Request, conv_id: Annotated[str, Path(pattern=CONV_ID)]) -> dict[str, Any]:
        channel = channel_of(request)
        conversation = channel.detail(conv_id)
        return _detail(conversation, channel.preview(conversation))

    @router.post(OPERATOR_PREFIX + "handoffs/{conv_id}/claim")
    def claim(
        request: Request, conv_id: Annotated[str, Path(pattern=CONV_ID)], body: OperatorBody
    ) -> dict[str, Any]:
        channel = channel_of(request)
        conversation = channel.claim(conv_id, body.operator)
        return _detail(conversation, channel.preview(conversation))

    @router.post(OPERATOR_PREFIX + "handoffs/{conv_id}/reply")
    def reply(
        request: Request, conv_id: Annotated[str, Path(pattern=CONV_ID)], body: ReplyBody
    ) -> dict[str, str]:
        channel_of(request).reply(conv_id, body.operator, body.text)
        return {"status": "sent"}

    @router.post(OPERATOR_PREFIX + "handoffs/{conv_id}/release")
    def release(
        request: Request, conv_id: Annotated[str, Path(pattern=CONV_ID)], body: OperatorBody
    ) -> dict[str, str]:
        channel_of(request).release(conv_id, body.operator)
        return {"status": "released"}

    return router
