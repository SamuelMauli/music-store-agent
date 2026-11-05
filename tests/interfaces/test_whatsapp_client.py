"""Cliente da Graph API com transporte falso (nenhuma chamada de rede)."""

from __future__ import annotations

import json

import httpx
import pytest

from emporio.adapters.whatsapp.client import MAX_TEXT_CHARS, GraphApiSender
from emporio.domain.errors import MessageDeliveryError


def sender(handler: httpx.MockTransport, sleeps: list[float] | None = None) -> GraphApiSender:
    return GraphApiSender(
        "tok-secreto",
        "1234567890",
        api_version="v99.0",
        transport=handler,
        sleep=(sleeps if sleeps is not None else []).append,
    )


def test_posts_the_graph_payload_with_bearer_token() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"messages": [{"id": "wamid.out"}]})

    sender(httpx.MockTransport(handle)).send_text("5567999990001", "Olá")
    (request,) = seen
    assert str(request.url) == "https://graph.facebook.com/v99.0/1234567890/messages"
    assert request.headers["authorization"] == "Bearer tok-secreto"
    assert json.loads(request.content) == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "5567999990001",
        "type": "text",
        "text": {"preview_url": False, "body": "Olá"},
    }


def test_truncates_to_the_whatsapp_limit() -> None:
    bodies: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content)["text"]["body"])
        return httpx.Response(200)

    sender(httpx.MockTransport(handle)).send_text("5567999990001", "x" * 5000)
    assert len(bodies[0]) == MAX_TEXT_CHARS


def test_retries_transient_status_then_succeeds() -> None:
    codes = iter([503, 429, 200])
    sleeps: list[float] = []
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(next(codes))

    sender(httpx.MockTransport(handle), sleeps).send_text("5567999990001", "oi")
    assert calls == 3
    assert sleeps == [0.5, 1.0]  # backoff exponencial


def test_retries_connect_errors_and_gives_up() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("sem rota", request=request)

    with pytest.raises(MessageDeliveryError, match="3 tentativas"):
        sender(httpx.MockTransport(handle)).send_text("5567999990001", "oi")
    assert calls == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
def test_definitive_failures_are_not_retried(status: int) -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "detalhe sensível do token"}})

    with pytest.raises(MessageDeliveryError) as caught:
        sender(httpx.MockTransport(handle)).send_text("5567999990001", "oi")
    assert calls == 1
    assert "detalhe" not in str(caught.value)  # o corpo da Meta não vaza na exceção


def test_read_timeout_is_not_retried_to_avoid_duplicates() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("lento", request=request)

    with pytest.raises(MessageDeliveryError, match="transporte"):
        sender(httpx.MockTransport(handle)).send_text("5567999990001", "oi")
    assert calls == 1
