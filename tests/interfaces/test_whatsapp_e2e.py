"""Canal WhatsApp ponta a ponta: TestClient + LLM fake do repo + Graph API com transporte falso."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import fakeredis
import httpx
import pytest
import redis
from fastapi.testclient import TestClient
from pydantic import SecretStr

from emporio.bootstrap import Container, build_container
from emporio.channels.whatsapp import InboundMessage, WhatsAppChannel
from emporio.interfaces.api import create_app
from emporio.settings import Settings
from tests.agent.conftest import fake_settings

APP_SECRET = "app-secret-de-teste"  # noqa: S105
VERIFY_TOKEN = "verify-de-teste"  # noqa: S105
PHONE_ID = "1234567890"
OPERATOR_KEY = "operador-chave-de-teste"  # chave da Ana
BIA_KEY = "operadora-bia-chave-de-teste"
CHAT_KEY = "chat-chave-de-teste"
CUSTOMER = "5567999990001"
OTHER_CUSTOMER = "5567999990002"
TTL = 3_600


@dataclass
class World:
    """Tudo o que o teste observa: o cliente HTTP, o que saiu pela Graph API e o relógio."""

    client: TestClient
    sent: list[dict[str, Any]] = field(default_factory=list)
    clock: list[float] = field(default_factory=lambda: [1_800_000_000.0])
    seq: int = 0

    def bodies_to(self, phone: str) -> list[str]:
        return [m["text"]["body"] for m in self.sent if m["to"] == phone]

    def post(self, payload: dict[str, Any], *, key: str | None = None) -> httpx.Response:
        raw = json.dumps(payload).encode()
        digest = hmac.new((key or APP_SECRET).encode(), raw, hashlib.sha256).hexdigest()
        response = self.client.post(
            "/webhooks/whatsapp",
            content=raw,
            headers={"X-Hub-Signature-256": f"sha256={digest}", "Content-Type": "application/json"},
        )
        assert self.channel.drain(), "o canal não terminou de processar"  # o 200 vem antes
        return response

    @property
    def channel(self) -> WhatsAppChannel:
        return self.client.app.state.whatsapp  # type: ignore[attr-defined,no-any-return]

    def say(self, phone: str, text: str, message_id: str | None = None) -> httpx.Response:
        self.seq += 1
        return self.post(inbound(phone, text, message_id or f"wamid.{self.seq}"))


def inbound(
    phone: str, text: str | None, message_id: str, kind: str = "text", phone_id: str = PHONE_ID
) -> dict[str, Any]:
    message: dict[str, Any] = {"from": phone, "id": message_id, "type": kind}
    if text is not None:
        message["text"] = {"body": text}
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "waba",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": phone_id},
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }


def operator(name: str = "ana") -> dict[str, str]:
    key = {"ana": OPERATOR_KEY, "bia": BIA_KEY}.get(name, name)
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def world(ingested: Settings) -> Iterator[World]:
    settings = fake_settings(
        ingested,
        api_keys=[SecretStr(CHAT_KEY)],
        operator_api_keys=[SecretStr(f"ana:{OPERATOR_KEY}"), SecretStr(f"bia:{BIA_KEY}")],
        whatsapp_verify_token=SecretStr(VERIFY_TOKEN),
        whatsapp_app_secret=SecretStr(APP_SECRET),
        whatsapp_access_token=SecretStr("token-graph"),
        whatsapp_phone_number_id=PHONE_ID,
        whatsapp_hash_pepper=SecretStr("pepper-de-teste"),
        whatsapp_handoff_ttl_seconds=TTL,
        whatsapp_rate_limit_per_minute=5,
        handoff_path=":memory:",
        rate_limit_per_minute=100,
    )
    sent: list[dict[str, Any]] = []

    def graph(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"messages": [{"id": "wamid.out"}]})

    container: Container = build_container(settings)
    clock = [1_800_000_000.0]
    app = create_app(
        settings,
        container,
        whatsapp_transport=httpx.MockTransport(graph),
        whatsapp_now=lambda: clock[0],
    )
    with TestClient(app) as test_client:
        yield World(test_client, sent, clock)
    container.close()


def start_handoff(w: World, phone: str = CUSTOMER) -> str:
    w.say(phone, "Quero falar com um atendente")
    queue = w.client.get("/v1/operator/handoffs", headers=operator()).json()["items"]
    return str(queue[-1]["id"]) if phone == CUSTOMER else str(queue[0]["id"])


# --- verificação do webhook --------------------------------------------------------------
def test_verification_echoes_challenge(world: World) -> None:
    ok = world.client.get(
        "/webhooks/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "4242"},
    )
    assert (ok.status_code, ok.text) == (200, "4242")


@pytest.mark.parametrize(
    "params",
    [
        {"hub.mode": "subscribe", "hub.verify_token": "errado", "hub.challenge": "1"},
        {"hub.mode": "unsubscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "1"},
        {"hub.mode": "subscribe", "hub.challenge": "1"},
        {"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN},
    ],
)
def test_verification_rejects_bad_requests(world: World, params: dict[str, str]) -> None:
    assert world.client.get("/webhooks/whatsapp", params=params).status_code == 403


def test_verification_rejects_challenge_with_markup(world: World) -> None:
    response = world.client.get(
        "/webhooks/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "<script>1</script>",
        },
    )
    assert response.status_code == 422


# --- assinatura ----------------------------------------------------------------------------
def test_invalid_signature_is_401_and_nothing_runs(world: World) -> None:
    response = world.post(inbound(CUSTOMER, "Oi", "wamid.x"), key="outro-segredo")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_signature"
    assert world.sent == []


@pytest.mark.parametrize("header", [None, "", "sha256=", "abc", "sha1=deadbeef"])
def test_missing_or_malformed_signature_is_401(world: World, header: str | None) -> None:
    headers = {"X-Hub-Signature-256": header} if header is not None else {}
    body = json.dumps(inbound(CUSTOMER, "Oi", "wamid.y")).encode()
    assert world.client.post("/webhooks/whatsapp", content=body, headers=headers).status_code == 401
    assert world.sent == []


def test_signature_covers_the_raw_bytes_not_the_json(world: World) -> None:
    raw = json.dumps(inbound(CUSTOMER, "Oi", "wamid.z"), indent=2).encode()
    digest = hmac.new(APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    tampered = raw.replace(b"Oi", b"Ol")
    response = world.client.post(
        "/webhooks/whatsapp", content=tampered, headers={"X-Hub-Signature-256": f"sha256={digest}"}
    )
    assert response.status_code == 401


def test_signed_garbage_is_400(world: World) -> None:
    raw = b"{nao-e-json"
    digest = hmac.new(APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    response = world.client.post(
        "/webhooks/whatsapp", content=raw, headers={"X-Hub-Signature-256": f"sha256={digest}"}
    )
    assert response.status_code == 400


def test_webhook_does_not_need_the_chat_api_key(world: World) -> None:
    assert world.say(CUSTOMER, "Quais são as promoções ativas?").status_code == 200
    assert world.client.post("/v1/chat", json={"message": "oi"}).status_code == 401


# --- caminho feliz, idempotência, mídia, status ---------------------------------------------
def test_text_message_is_answered_by_the_bot(world: World) -> None:
    response = world.say(CUSTOMER, "Quais são as promoções ativas?")
    assert response.status_code == 200
    (answer,) = world.bodies_to(CUSTOMER)
    assert "Crafter HT-100" in answer


def test_redelivery_of_the_same_message_id_is_processed_once(world: World) -> None:
    for _ in range(3):
        assert world.say(CUSTOMER, "Quais são as promoções ativas?", "wamid.dup").status_code == 200
    assert len(world.sent) == 1


def test_media_gets_a_polite_reply_without_calling_the_agent(world: World) -> None:
    world.post(inbound(CUSTOMER, None, "wamid.img", kind="image"))
    (reply,) = world.bodies_to(CUSTOMER)
    assert "texto" in reply


def test_status_events_and_reactions_are_ignored(world: World) -> None:
    status = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "metadata": {"phone_number_id": PHONE_ID},
                            "statuses": [{"id": "wamid.s", "status": "delivered"}],
                        }
                    }
                ]
            }
        ]
    }
    assert world.post(status).status_code == 200
    reaction = inbound(CUSTOMER, None, "wamid.r", kind="reaction")
    assert world.post(reaction).status_code == 200
    assert world.sent == []


def test_event_for_another_phone_number_id_is_ignored(world: World) -> None:
    world.post(inbound(CUSTOMER, "Oi", "wamid.o", phone_id="999"))
    assert world.sent == []


def test_too_long_message_is_refused_politely(world: World) -> None:
    world.say(CUSTOMER, "x" * 2001)
    assert "longa" in world.bodies_to(CUSTOMER)[0]


def test_rate_limit_per_number_drops_excess_and_isolates_numbers(world: World) -> None:
    for _ in range(5):
        world.say(CUSTOMER, "Quais são as promoções ativas?")
    assert len(world.bodies_to(CUSTOMER)) == 5
    world.say(CUSTOMER, "Quais são as promoções ativas?")
    assert len(world.bodies_to(CUSTOMER)) == 5  # 6ª descartada
    world.say(OTHER_CUSTOMER, "Quais são as promoções ativas?")
    assert len(world.bodies_to(OTHER_CUSTOMER)) == 1


def test_agent_failure_answers_with_a_generic_message(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("falha interna com detalhe")

    monkeypatch.setattr(world.client.app.state.container.service, "chat", boom)  # type: ignore[attr-defined]
    assert world.say(CUSTOMER, "Oi").status_code == 200
    (reply,) = world.bodies_to(CUSTOMER)
    assert "problema técnico" in reply
    assert "detalhe" not in reply


# --- handoff humano --------------------------------------------------------------------------
def test_handoff_stops_the_bot_and_queues_customer_messages(world: World) -> None:
    conv_id = start_handoff(world)
    first = world.bodies_to(CUSTOMER)
    assert len(first) == 1
    assert "EMP-" in first[0]  # protocolo do chamado do agente
    assert "atendente responde por aqui" in first[0]  # mensagem de transição
    world.say(CUSTOMER, "Alguém aí?")
    world.say(CUSTOMER, "Quais são as promoções ativas?")  # o bot responderia, mas está mudo
    assert world.bodies_to(CUSTOMER) == first
    detail = world.client.get(f"/v1/operator/handoffs/{conv_id}", headers=operator()).json()
    assert [m["text"] for m in detail["messages"]] == [
        "Quero falar com um atendente",
        "Alguém aí?",
        "Quais são as promoções ativas?",
    ]
    assert detail["state"] == "waiting"
    assert CUSTOMER not in json.dumps(detail)  # a API nunca expõe o telefone


def test_media_during_handoff_is_queued_not_answered(world: World) -> None:
    conv_id = start_handoff(world)
    before = len(world.sent)
    world.post(inbound(CUSTOMER, None, "wamid.foto", kind="image"))
    assert len(world.sent) == before
    detail = world.client.get(f"/v1/operator/handoffs/{conv_id}", headers=operator()).json()
    assert detail["messages"][-1]["text"] == "[mensagem não textual: image]"


def test_operator_claims_replies_and_returns_to_the_bot(world: World) -> None:
    conv_id = start_handoff(world)
    base = f"/v1/operator/handoffs/{conv_id}"
    world.say(CUSTOMER, "Meu pedido atrasou")

    claim = world.client.post(f"{base}/claim", headers=operator(), json={})
    assert claim.status_code == 200
    assert claim.json()["state"] == "assigned"
    assert claim.json()["operator"] == "ana"

    reply = world.client.post(
        f"{base}/reply", headers=operator(), json={"operator": "ana", "text": "Oi, sou a Ana."}
    )
    assert reply.status_code == 200
    assert world.bodies_to(CUSTOMER)[-1] == "Oi, sou a Ana."  # saiu pelo WhatsApp
    detail = world.client.get(base, headers=operator()).json()
    assert detail["messages"][-1] == {
        "direction": "out",
        "text": "Oi, sou a Ana.",
        "at": detail["messages"][-1]["at"],
    }

    released = world.client.post(f"{base}/release", headers=operator(), json={})
    assert released.status_code == 200
    assert "Voltei a ajudar" in world.bodies_to(CUSTOMER)[-1]
    assert world.client.get("/v1/operator/handoffs", headers=operator()).json()["items"] == []
    assert world.client.get(base, headers=operator()).status_code == 404

    sent_before = len(world.sent)
    world.say(CUSTOMER, "Quais são as promoções ativas?")  # o bot volta a responder
    assert len(world.sent) == sent_before + 1
    assert "Crafter HT-100" in world.bodies_to(CUSTOMER)[-1]


def test_reply_requires_ownership(world: World) -> None:
    conv_id = start_handoff(world)
    base = f"/v1/operator/handoffs/{conv_id}"
    body = {"text": "oi"}
    assert world.client.post(f"{base}/reply", headers=operator(), json=body).status_code == 409
    world.client.post(f"{base}/claim", headers=operator(), json={})
    intruder = world.client.post(f"{base}/claim", headers=operator("bia"), json={})
    assert intruder.status_code == 409
    assert intruder.json()["error"]["code"] == "handoff_conflict"
    assert (
        world.client.post(f"{base}/reply", headers=operator("bia"), json={"text": "oi"}).status_code
        == 409
    )
    assert world.client.post(f"{base}/release", headers=operator("bia"), json={}).status_code == 409
    assert len(world.bodies_to(CUSTOMER)) == 1  # nenhuma resposta indevida saiu


def test_operator_name_comes_from_the_key_and_a_foreign_name_is_403(world: World) -> None:
    conv_id = start_handoff(world)
    base = f"/v1/operator/handoffs/{conv_id}"
    forged = world.client.post(f"{base}/claim", headers=operator("bia"), json={"operator": "ana"})
    assert forged.status_code == 403
    assert forged.json()["error"]["code"] == "operator_mismatch"
    assert world.client.get(base, headers=operator()).json()["operator"] is None  # não assumiu
    own = world.client.post(f"{base}/claim", headers=operator("bia"), json={"operator": "bia"})
    assert own.status_code == 200
    assert own.json()["operator"] == "bia"


def test_unknown_conversation_is_404(world: World) -> None:
    ghost = "0" * 32
    assert world.client.get(f"/v1/operator/handoffs/{ghost}", headers=operator()).status_code == 404
    claim = world.client.post(f"/v1/operator/handoffs/{ghost}/claim", headers=operator(), json={})
    assert claim.status_code == 404
    assert world.client.get("/v1/operator/handoffs/nao-e-id", headers=operator()).status_code == 422


def test_reply_delivery_failure_is_502_and_not_recorded(ingested: Settings) -> None:
    settings = fake_settings(
        ingested,
        operator_api_keys=[SecretStr(f"ana:{OPERATOR_KEY}"), SecretStr(f"bia:{BIA_KEY}")],
        whatsapp_verify_token=SecretStr(VERIFY_TOKEN),
        whatsapp_app_secret=SecretStr(APP_SECRET),
        whatsapp_access_token=SecretStr("token-graph"),
        whatsapp_phone_number_id=PHONE_ID,
        whatsapp_hash_pepper=SecretStr("pepper-de-teste"),
        handoff_path=":memory:",
    )
    fail = {"on": False}

    def graph(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400 if fail["on"] else 200)

    container = build_container(settings)
    app = create_app(settings, container, whatsapp_transport=httpx.MockTransport(graph))
    with TestClient(app) as client:
        w = World(client)
        conv_id = start_handoff(w)
        base = f"/v1/operator/handoffs/{conv_id}"
        client.post(f"{base}/claim", headers=operator(), json={})
        fail["on"] = True
        response = client.post(f"{base}/reply", headers=operator(), json={"text": "oi"})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "delivery_failed"
        detail = client.get(base, headers=operator()).json()
        assert all(m["direction"] == "in" for m in detail["messages"])
    container.close()


def test_handoff_expires_back_to_the_bot(world: World) -> None:
    start_handoff(world)
    world.say(CUSTOMER, "Quais são as promoções ativas?")
    muted = len(world.sent)
    world.clock[0] += TTL - 1
    world.say(CUSTOMER, "Quais são as promoções ativas?")
    assert len(world.sent) == muted  # ainda mudo
    world.clock[0] += 2
    world.say(CUSTOMER, "Quais são as promoções ativas?")
    assert "Crafter HT-100" in world.bodies_to(CUSTOMER)[-1]
    assert world.client.get("/v1/operator/handoffs", headers=operator()).json()["items"] == []


def test_operator_activity_extends_the_expiry(world: World) -> None:
    conv_id = start_handoff(world)
    base = f"/v1/operator/handoffs/{conv_id}"
    world.clock[0] += TTL - 10
    world.client.post(f"{base}/claim", headers=operator(), json={})
    world.clock[0] += TTL - 10  # passou do prazo original, mas o claim renovou
    assert world.client.get(base, headers=operator()).status_code == 200


def test_queue_lists_numbers_by_opaque_id_with_masked_preview(world: World) -> None:
    world.say(CUSTOMER, "Quero falar com um atendente, meu e-mail é ana@exemplo.com")
    items = world.client.get("/v1/operator/handoffs", headers=operator()).json()["items"]
    assert len(items) == 1
    assert "ana@exemplo.com" not in items[0]["preview"]
    assert CUSTOMER not in json.dumps(items)
    assert len(items[0]["id"]) == 32


def test_handoffs_of_two_customers_are_independent(world: World) -> None:
    start_handoff(world)
    world.say(OTHER_CUSTOMER, "Quais são as promoções ativas?")
    assert "Crafter HT-100" in world.bodies_to(OTHER_CUSTOMER)[-1]  # o outro segue com o bot


# --- autorização -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer errada"}, {"Authorization": f"Bearer {CHAT_KEY}"}],
)
def test_operator_endpoints_reject_missing_wrong_and_chat_keys(
    world: World, headers: dict[str, str]
) -> None:
    for method, path in (
        ("GET", "/v1/operator/handoffs"),
        ("GET", f"/v1/operator/handoffs/{'0' * 32}"),
        ("POST", f"/v1/operator/handoffs/{'0' * 32}/reply"),
    ):
        response = world.client.request(method, path, headers=headers, json=None)
        assert response.status_code == 401


def test_operator_key_does_not_open_the_chat_api(world: World) -> None:
    response = world.client.post("/v1/chat", headers=operator(), json={"message": "oi"})
    assert response.status_code == 401


def test_operator_key_also_works_via_x_api_key(world: World) -> None:
    response = world.client.get("/v1/operator/handoffs", headers={"X-API-Key": OPERATOR_KEY})
    assert response.status_code == 200


# --- logs sem PII -----------------------------------------------------------------------------
def test_logs_never_contain_the_phone_number_or_message_text(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        conv_id = start_handoff(world)
        world.say(CUSTOMER, "mensagem-secreta-do-cliente")
        world.client.post(f"/v1/operator/handoffs/{conv_id}/claim", headers=operator(), json={})
        world.client.post(
            f"/v1/operator/handoffs/{conv_id}/reply",
            headers=operator(),
            json={"operator": "ana", "text": "resposta-secreta-do-operador"},
        )
        world.post(inbound(CUSTOMER, "x", "wamid.bad"), key="errada")
    logs = caplog.text
    assert CUSTOMER not in logs
    assert "mensagem-secreta-do-cliente" not in logs
    assert "resposta-secreta-do-operador" not in logs
    assert "token-graph" not in logs
    assert conv_id[:12] in logs  # o contato aparece só pelo hash


# --- configuração -----------------------------------------------------------------------------
def test_channel_is_off_by_default(ingested: Settings) -> None:
    settings = fake_settings(ingested)
    assert not settings.whatsapp_enabled
    container = build_container(settings)
    with TestClient(create_app(settings, container)) as client:
        assert client.get("/webhooks/whatsapp").status_code == 404
        assert client.get("/v1/operator/handoffs").status_code == 401  # prefixo fechado
    container.close()


def test_partial_configuration_fails_fast_listing_what_is_missing(ingested: Settings) -> None:
    settings = fake_settings(ingested, whatsapp_verify_token=SecretStr("x"))
    with pytest.raises(ValueError, match="EMPORIO_WHATSAPP_APP_SECRET"):
        create_app(settings)


def test_operator_key_must_differ_from_chat_keys(ingested: Settings) -> None:
    settings = fake_settings(
        ingested,
        api_keys=[SecretStr("mesma")],
        operator_api_keys=[SecretStr("mesma")],
        whatsapp_verify_token=SecretStr(VERIFY_TOKEN),
        whatsapp_app_secret=SecretStr(APP_SECRET),
        whatsapp_access_token=SecretStr("t"),
        whatsapp_phone_number_id=PHONE_ID,
        whatsapp_hash_pepper=SecretStr("p"),
    )
    with pytest.raises(ValueError, match="reutilizar"):
        create_app(settings)


def test_redis_backend_is_used_when_redis_url_is_set(
    ingested: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        redis.Redis,
        "from_url",
        classmethod(lambda cls, *_a, **_k: fakeredis.FakeRedis(server=server)),
    )
    settings = fake_settings(
        ingested,
        redis_url=SecretStr("redis://fake:6379/0"),
        operator_api_keys=[SecretStr(f"ana:{OPERATOR_KEY}"), SecretStr(f"bia:{BIA_KEY}")],
        whatsapp_verify_token=SecretStr(VERIFY_TOKEN),
        whatsapp_app_secret=SecretStr(APP_SECRET),
        whatsapp_access_token=SecretStr("token-graph"),
        whatsapp_phone_number_id=PHONE_ID,
        whatsapp_hash_pepper=SecretStr("pepper-de-teste"),
    )
    sent: list[dict[str, Any]] = []

    def graph(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200)

    container = build_container(settings)
    app = create_app(settings, container, whatsapp_transport=httpx.MockTransport(graph))
    with TestClient(app) as client:
        w = World(client, sent)
        start_handoff(w)
        w.say(CUSTOMER, "Quais são as promoções ativas?")
        assert len(sent) == 1  # mudo: handoff e dedup vivem no Redis
        w.say(CUSTOMER, "Quais são as promoções ativas?", "wamid.1")  # reentrega do 1º id
        assert len(sent) == 1
    assert list(fakeredis.FakeRedis(server=server).scan_iter("emporio:wa:conv:*"))
    container.close()


def test_operator_credentials_accept_legacy_key_without_name(ingested: Settings) -> None:
    config = fake_settings(
        ingested,
        operator_api_keys=[SecretStr("ana:k1"), SecretStr("so-a-chave")],
    )
    (ana, legacy) = config.operator_credentials()
    assert ana == ("ana", "k1")
    assert legacy[0].startswith("operador-")
    assert legacy[1] == "so-a-chave"


def test_flood_uses_at_most_the_dedicated_workers_and_chat_stays_responsive(
    ingested: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muitos números com o LLM travado: só ``workers`` threads, 200 imediato, /v1/chat segue."""
    workers, queue = 2, 3
    settings = fake_settings(
        ingested,
        api_keys=[SecretStr(CHAT_KEY)],
        operator_api_keys=[SecretStr(f"ana:{OPERATOR_KEY}")],
        whatsapp_verify_token=SecretStr(VERIFY_TOKEN),
        whatsapp_app_secret=SecretStr(APP_SECRET),
        whatsapp_access_token=SecretStr("token-graph"),
        whatsapp_phone_number_id=PHONE_ID,
        whatsapp_hash_pepper=SecretStr("pepper-de-teste"),
        whatsapp_workers=workers,
        whatsapp_queue_size=queue,
        handoff_path=":memory:",
        rate_limit_per_minute=1000,
    )
    container = build_container(settings)
    real_chat = container.service.chat
    gate = threading.Event()
    running = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def slow_chat(session_id: str, text: str, request_id: str, *args: Any, **kw: Any) -> Any:
        if not session_id.startswith("wa-"):
            return real_chat(session_id, text, request_id, *args, **kw)
        with lock:
            running["now"] += 1
            running["peak"] = max(running["peak"], running["now"])
        gate.wait(10)
        with lock:
            running["now"] -= 1
        return real_chat(session_id, text, request_id, *args, **kw)

    monkeypatch.setattr(container.service, "chat", slow_chat)
    sent: list[dict[str, Any]] = []

    def graph(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200)

    app = create_app(settings, container, whatsapp_transport=httpx.MockTransport(graph))
    with TestClient(app) as client:
        channel: WhatsAppChannel = app.state.whatsapp
        total = workers + queue + 4  # 4 além da capacidade
        for i in range(total):
            payload = inbound(f"55679999{i:05d}", "Qual o horário de vocês?", f"wamid.flood{i}")
            raw = json.dumps(payload).encode()
            sig = hmac.new(APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
            response = client.post(
                "/webhooks/whatsapp",
                content=raw,
                headers={"X-Hub-Signature-256": f"sha256={sig}"},
            )
            assert response.status_code == 200  # nunca espera o LLM
        alive = [t for t in threading.enumerate() if t.name.startswith("wa-worker")]
        assert len(alive) <= workers
        assert client.get("/healthz").status_code == 200  # o servidor segue livre
        chat = client.post(
            "/v1/chat", headers={"X-API-Key": CHAT_KEY}, json={"message": "Quais os horários?"}
        )
        assert chat.status_code == 200
        gate.set()
        assert channel.drain(30)
    assert running["peak"] <= workers
    assert len({m["to"] for m in sent}) == workers + queue  # o excedente foi descartado
    container.close()


def test_same_conversation_is_processed_in_order_one_at_a_time(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    active = {"now": 0, "peak": 0}
    lock = threading.Lock()
    service = world.client.app.state.container.service  # type: ignore[attr-defined]
    real_chat = service.chat

    def tracked(session_id: str, text: str, request_id: str, *a: Any, **k: Any) -> Any:
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        order.append(text)
        try:
            return real_chat(session_id, text, request_id, *a, **k)
        finally:
            with lock:
                active["now"] -= 1

    monkeypatch.setattr(service, "chat", tracked)
    for i in range(4):  # direto no canal, sem esperar entre uma e outra (concorrência real)
        message = InboundMessage(
            f"wamid.ord{i}", CUSTOMER, "text", f"Pergunta {i}: qual o horário?"
        )
        assert world.channel.submit(message)
    assert world.channel.drain(30)
    assert order == [f"Pergunta {i}: qual o horário?" for i in range(4)]
    assert active["peak"] == 1
