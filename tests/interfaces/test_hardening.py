"""Autenticação, métricas, logs JSON, limites e perfil de produção da API."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from emporio.bootstrap import build_container
from emporio.interfaces.api import create_app
from emporio.observability.logging import configure_logging
from emporio.production import ProductionConfigError, production_problems
from emporio.settings import Settings
from tests.agent.conftest import fake_settings

KEYS = [SecretStr("key-old-1234567890"), SecretStr("key-new-1234567890")]


@pytest.fixture
def make(ingested: Settings) -> Callable[..., Any]:
    @contextmanager
    def _make(**extra: Any) -> Iterator[TestClient]:
        settings = fake_settings(ingested, rate_limit_per_minute=50, **extra)
        container = build_container(settings)
        try:
            with TestClient(create_app(settings, container)) as client:
                yield client
        finally:
            container.close()

    return _make


def test_open_api_in_dev_without_keys(make: Callable[..., Any]) -> None:
    with make() as client:
        assert client.post("/v1/chat", json={"message": "Oi"}).status_code == 200


def test_missing_and_wrong_key_get_401_in_the_envelope(make: Callable[..., Any]) -> None:
    with make(api_keys=KEYS) as client:
        missing = client.post("/v1/chat", json={"message": "Oi"})
        wrong = client.post("/v1/chat", json={"message": "Oi"}, headers={"X-API-Key": "nope"})
        empty = client.post("/v1/chat", json={"message": "Oi"}, headers={"X-API-Key": ""})
    for response in (missing, wrong, empty):
        assert response.status_code == 401
        error = response.json()["error"]
        assert error["code"] == "unauthorized"
        assert error["request_id"] == response.headers["x-request-id"]
        assert response.headers["www-authenticate"] == "Bearer"


def test_valid_key_by_bearer_or_header_and_rotation(make: Callable[..., Any]) -> None:
    with make(api_keys=KEYS) as client:
        for headers in (
            {"Authorization": "Bearer key-old-1234567890"},
            {"Authorization": "bearer key-new-1234567890"},
            {"X-API-Key": "key-old-1234567890"},
            {"X-API-Key": "key-new-1234567890"},
        ):
            sent = client.post("/v1/chat", json={"message": "Oi"}, headers=headers)
            assert sent.status_code == 200


def test_health_is_public_and_metrics_needs_a_key(make: Callable[..., Any]) -> None:
    with make(api_keys=KEYS) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/metrics").status_code == 401
        ok = client.get("/metrics", headers={"X-API-Key": "key-new-1234567890"})
        assert ok.status_code == 200


def test_metrics_expose_prometheus_counters_after_a_chat(make: Callable[..., Any]) -> None:
    with make() as client:
        client.post("/v1/chat", json={"message": "Quais são as promoções ativas?"})
        client.post("/v1/chat", json={"message": "quero falar com um atendente humano"})
        body = client.get("/metrics")
    assert body.headers["content-type"].startswith("text/plain; version=")
    text = body.text
    assert 'emporio_http_requests_total{method="POST",route="/v1/chat",status="200"} 2.0' in text
    assert "emporio_http_request_duration_seconds_bucket{" in text
    assert "emporio_turns_total{intent=" in text
    for name in (
        "emporio_handoffs_total",
        "emporio_guardrail_blocks_total",
        "emporio_llm_tokens_total",
        "emporio_llm_cost_usd_total",
    ):
        assert f"# TYPE {name}" in text


def test_metrics_registries_are_isolated_per_app(make: Callable[..., Any]) -> None:
    with make() as first:
        first.post("/v1/chat", json={"message": "Oi"})
    with make() as second:
        assert 'route="/v1/chat"' not in second.get("/metrics").text


def test_metrics_can_be_disabled(make: Callable[..., Any]) -> None:
    with make(metrics_enabled=False) as client:
        assert client.get("/metrics").status_code == 404


def test_json_logs_have_fields_and_never_the_client_message(make: Callable[..., Any]) -> None:
    stream = io.StringIO()
    client_text = "meu cpf 123.456.789-09 e a senha zebra-azul"
    with make(log_format="json", log_level="INFO") as client:
        configure_logging("json", "INFO", stream)  # redireciona o handler para o buffer
        token = client.post("/v1/chat", json={"message": "Oi"}).headers["x-session-id"]
        client.post("/v1/chat", json={"message": client_text, "session_id": token})
    configure_logging("text", "INFO", io.StringIO())
    raw = stream.getvalue()
    lines = [json.loads(line) for line in raw.splitlines()]
    access = [r for r in lines if r.get("event") == "http_request" and r["route"] == "/v1/chat"]
    assert len(access) == 2
    for record in access:
        assert {"ts", "level", "request_id", "route", "status", "latency_ms", "session"} <= set(
            record
        )
        assert record["status"] == 200
        assert len(record["session"]) == 12
    assert "zebra-azul" not in raw
    assert "123.456.789" not in raw
    assert token.partition(".")[0] not in raw
    assert any("turno sessao=" in r["message"] for r in lines)


def test_text_log_format_is_reinstalled_cleanly() -> None:
    configure_logging("text", "WARNING", io.StringIO())
    assert logging.getLogger().level == logging.WARNING
    configure_logging("text", "INFO", io.StringIO())
    marked = [h for h in logging.getLogger().handlers if getattr(h, "_emporio_handler", False)]
    assert len(marked) == 1


def test_oversized_body_gets_413_and_long_message_422(make: Callable[..., Any]) -> None:
    with make(max_message_chars=50) as client:
        huge = client.post(
            "/v1/chat", content=b"x" * 5000, headers={"Content-Type": "application/json"}
        )
        long = client.post("/v1/chat", json={"message": "a" * 51})
        fine = client.post("/v1/chat", json={"message": "a" * 50})
    assert huge.status_code == 413
    assert huge.json()["error"]["code"] == "payload_too_large"
    assert long.status_code == 422
    assert long.json()["error"]["code"] == "invalid_request"
    assert fine.status_code == 200


def test_chunked_body_over_the_limit_gets_413(make: Callable[..., Any]) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(10):
            yield b"x" * 1000

    with make(max_message_chars=50) as client:
        response = client.post("/v1/chat", content=chunks())
    assert response.status_code == 413


def test_security_headers_and_cors(make: Callable[..., Any]) -> None:
    with make(cors_origins=["https://loja.example"], api_keys=KEYS) as client:
        chat = client.post(
            "/v1/chat", json={"message": "Oi"}, headers={"X-API-Key": "key-old-1234567890"}
        )
        preflight = client.options(
            "/v1/chat",
            headers={
                "Origin": "https://loja.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        other = client.options(
            "/v1/chat",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
    assert chat.headers["x-content-type-options"] == "nosniff"
    assert chat.headers["cache-control"] == "no-store"
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://loja.example"
    assert "access-control-allow-origin" not in other.headers


def test_no_cors_headers_by_default(make: Callable[..., Any]) -> None:
    with make() as client:
        response = client.get("/healthz", headers={"Origin": "https://x.example"})
    assert "access-control-allow-origin" not in response.headers


def _prod(**over: Any) -> Settings:
    complete: dict[str, Any] = {
        "env": "prod",
        "session_secret": SecretStr("s" * 32),
        "api_keys": [SecretStr("k" * 20)],
        "llm_provider": "openai_compatible",
        "state_backend": "postgres",
        "database_url": SecretStr("postgresql://u:p@db/emporio"),
        "redis_url": SecretStr("redis://cache:6379/0"),
    }
    return Settings(**{**complete, **over})


def test_prod_profile_lists_every_problem_at_once() -> None:
    settings = _prod(
        session_secret=None,
        api_keys=[],
        llm_provider="fake",
        state_backend="sqlite",
        database_url=None,
        redis_url=None,
    )
    problems = production_problems(settings)
    joined = "\n".join(problems)
    for needle in ("SESSION_SECRET", "API_KEYS", "LLM_PROVIDER", "STATE_BACKEND", "REDIS_URL"):
        assert needle in joined
    assert len(problems) == 5
    with pytest.raises(ProductionConfigError) as caught:
        create_app(settings)
    assert caught.value.problems == problems


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("session_secret", SecretStr("  "), "SESSION_SECRET"),
        ("api_keys", [SecretStr("")], "API_KEYS"),
        ("llm_provider", "fake", "LLM_PROVIDER"),
        ("state_backend", "sqlite", "STATE_BACKEND"),
        ("database_url", None, "DATABASE_URL"),
        ("redis_url", None, "REDIS_URL"),
    ],
)
def test_prod_profile_flags_each_missing_item(field: str, value: Any, needle: str) -> None:
    problems = production_problems(_prod(**{field: value}))
    assert len(problems) == 1
    assert needle in problems[0]


def test_prod_profile_with_complete_config_boots() -> None:
    settings = _prod()
    assert production_problems(settings) == []
    assert create_app(settings) is not None  # monta; o container só nasce no startup


def test_dev_profile_never_fails() -> None:
    assert production_problems(Settings(env="dev")) == []
