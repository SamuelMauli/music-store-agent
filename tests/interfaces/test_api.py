from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from emporio.bootstrap import Container, build_container
from emporio.interfaces.api import create_app
from emporio.settings import Settings
from tests.agent.conftest import fake_settings


@pytest.fixture
def client(ingested: Settings) -> Iterator[TestClient]:
    settings = fake_settings(ingested, rate_limit_per_minute=3)
    container: Container = build_container(settings)
    with TestClient(create_app(settings, container)) as test_client:
        yield test_client
    container.close()


def new_session(client: TestClient) -> str:
    """Abre uma sessão pelo servidor (1ª mensagem sem session_id) e devolve o token."""
    response = client.post("/v1/chat", json={"message": "Oi"})
    assert response.status_code == 200
    return response.headers["x-session-id"]


def parse_sse(raw: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in raw.replace("\r\n", "\n").strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}


def test_chat_streams_steps_tokens_and_done(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"message": "Quais são as promoções ativas?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "step"
    assert "token" in kinds
    assert kinds[-1] == "done"
    text = "".join(str(data["text"]) for kind, data in events if kind == "token")
    assert "Crafter HT-100" in text
    done = events[-1][1]
    assert done["tools"] == ["list_active_promotions"]
    assert done["request_id"] == response.headers["x-request-id"]
    assert done["session_id"] == response.headers["x-session-id"]
    assert "answer" not in done


def test_request_id_is_echoed_when_valid_and_replaced_when_not(client: TestClient) -> None:
    ok = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert ok.headers["x-request-id"] == "abc-123"
    bad = client.get("/healthz", headers={"X-Request-ID": "x y"})
    assert bad.headers["x-request-id"] != "x y"
    assert len(bad.headers["x-request-id"]) == 32


@pytest.mark.parametrize(
    "body",
    [{}, {"message": ""}, {"message": "x" * 4001}, {"message": "oi", "session_id": "../etc"}],
)
def test_invalid_body_uses_the_error_envelope(client: TestClient, body: dict[str, str]) -> None:
    response = client.post("/v1/chat", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["request_id"] == response.headers["x-request-id"]


def test_rate_limit_returns_429_with_retry_after(client: TestClient) -> None:
    first = new_session(client)  # 1ª mensagem
    for _ in range(2):
        body = {"message": "Oi", "session_id": first}
        assert client.post("/v1/chat", json=body).status_code == 200
    blocked = client.post("/v1/chat", json={"message": "Oi", "session_id": first})
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert int(blocked.headers["retry-after"]) >= 1


def test_session_memory_is_shared_across_requests(client: TestClient) -> None:
    token = new_session(client)
    second = client.post("/v1/chat", json={"message": "Oi", "session_id": token})
    assert parse_sse(second.text)[-1][1]["turn"] == 2


def test_server_issues_distinct_unguessable_sessions(client: TestClient) -> None:
    tokens = {new_session(client) for _ in range(2)}
    assert len(tokens) == 2
    for token in tokens:
        session_id, _, tag = token.partition(".")
        assert len(session_id) >= 22  # 128 bits em base64url
        assert tag


@pytest.mark.parametrize("forged", ["a1", "abc.def", "x.", ".y", "A" * 22 + "." + "A" * 22])
def test_unsigned_or_forged_session_is_refused_with_400(client: TestClient, forged: str) -> None:
    response = client.post("/v1/chat", json={"message": "Oi", "session_id": forged})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_session"
    assert error["request_id"] == response.headers["x-request-id"]


def test_tampered_signature_is_refused(client: TestClient) -> None:
    token = new_session(client)
    session_id, _, tag = token.partition(".")
    flipped = ("A" if tag[0] != "A" else "B") + tag[1:]
    response = client.post(
        "/v1/chat", json={"message": "Oi", "session_id": f"{session_id}.{flipped}"}
    )
    assert response.status_code == 400


def test_session_from_another_secret_is_refused(ingested: Settings) -> None:
    def app_with(secret: str) -> tuple[TestClient, Container]:
        settings = fake_settings(ingested, session_secret=SecretStr(secret * 4))
        built = build_container(settings)
        return TestClient(create_app(settings, built)), built

    (one, c1), (two, c2) = app_with("aaaa"), app_with("bbbb")
    try:
        with one, two:
            token = new_session(one)
            forged = {"message": "Oi", "session_id": token}
            assert two.post("/v1/chat", json=forged).status_code == 400
    finally:
        c1.close()
        c2.close()


def test_rate_limit_per_ip_spans_sessions(ingested: Settings) -> None:
    settings = fake_settings(ingested, rate_limit_per_minute=50, rate_limit_per_ip_per_minute=2)
    built = build_container(settings)
    with TestClient(create_app(settings, built)) as client:
        assert client.post("/v1/chat", json={"message": "Oi"}).status_code == 200
        assert client.post("/v1/chat", json={"message": "Oi"}).status_code == 200  # outra sessão
        blocked = client.post("/v1/chat", json={"message": "Oi"})
    built.close()
    assert blocked.status_code == 429


def test_forwarded_for_is_only_trusted_behind_configured_proxies(ingested: Settings) -> None:
    def codes(hops: int) -> list[int]:
        settings = fake_settings(
            ingested,
            rate_limit_per_minute=50,
            rate_limit_per_ip_per_minute=1,
            trusted_proxy_hops=hops,
        )
        built = build_container(settings)
        with TestClient(create_app(settings, built)) as client:
            out = [
                client.post(
                    "/v1/chat", json={"message": "Oi"}, headers={"X-Forwarded-For": f"1.1.1.{n}"}
                ).status_code
                for n in (1, 2)
            ]
        built.close()
        return out

    assert codes(0) == [200, 429]  # cabeçalho ignorado: todos viram o IP do socket
    assert codes(1) == [200, 200]  # atrás de 1 proxy confiável: IPs distintos


def test_readiness_fails_when_database_is_closed(ingested: Settings) -> None:
    settings = fake_settings(ingested)
    container = build_container(settings)
    with TestClient(create_app(settings, container)) as test_client:
        container.connection.close()
        response = test_client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_ready"
    container.close()


def test_empty_session_secret_falls_back_to_random_with_warning(
    ingested: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    settings = fake_settings(ingested, session_secret=SecretStr(""))
    built = build_container(settings)
    with TestClient(create_app(settings, built)) as client:
        assert client.post("/v1/chat", json={"message": "Oi"}).status_code == 200
    built.close()
    assert "EMPORIO_SESSION_SECRET ausente" in caplog.text
