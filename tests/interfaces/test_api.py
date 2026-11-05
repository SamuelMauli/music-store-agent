from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

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
    response = client.post(
        "/v1/chat", json={"message": "Quais são as promoções ativas?", "session_id": "a1"}
    )
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
    for _ in range(3):
        assert (
            client.post("/v1/chat", json={"message": "Oi", "session_id": "r1"}).status_code == 200
        )
    blocked = client.post("/v1/chat", json={"message": "Oi", "session_id": "r1"})
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert int(blocked.headers["retry-after"]) >= 1
    other = client.post("/v1/chat", json={"message": "Oi", "session_id": "r2"})
    assert other.status_code == 200


def test_session_memory_is_shared_across_requests(client: TestClient) -> None:
    client.post("/v1/chat", json={"message": "Oi", "session_id": "mem"})
    second = client.post("/v1/chat", json={"message": "Oi", "session_id": "mem"})
    assert parse_sse(second.text)[-1][1]["turn"] == 2


def test_readiness_fails_when_database_is_closed(ingested: Settings) -> None:
    settings = fake_settings(ingested)
    container = build_container(settings)
    with TestClient(create_app(settings, container)) as test_client:
        container.connection.close()
        response = test_client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_ready"
    container.close()
