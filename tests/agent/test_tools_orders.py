from __future__ import annotations

from emporio.agent.tools import ToolRegistry
from emporio.agent.tools.results import (
    Ambiguous,
    NotFound,
    Ok,
    PolicyBlocked,
    ToolError,
)
from tests.agent.conftest import Caller, caller

EMAIL_14 = "leticia.rocha@jmail.com"


def test_verify_customer_with_email(call: Caller) -> None:
    result = call("verify_customer", order_id=7, email=EMAIL_14.upper())
    assert isinstance(result.result, Ok)
    assert result.result.value.verified
    assert result.state_updates == {"verified_customer_id": 14}


def test_verify_customer_id_never_reaches_llm_payload(call: Caller) -> None:
    payload = call("verify_customer", order_id=7, email=EMAIL_14).payload
    assert "verified_customer_id" not in payload
    assert "leticia.rocha" not in payload


def test_verify_customer_failures_look_identical(call: Caller) -> None:
    wrong_contact = call("verify_customer", order_id=7, email="outra@jmail.com")
    no_such_order = call("verify_customer", order_id=999, email="outra@jmail.com")
    bad_email = call("verify_customer", order_id=7, email="isso-nao-e-email")
    for execution in (wrong_contact, no_such_order, bad_email):
        assert isinstance(execution.result, Ok)
        assert not execution.result.value.verified
        assert execution.state_updates == {}
    assert wrong_contact.payload == no_such_order.payload == bad_email.payload


def test_verify_customer_needs_contact_and_respects_attempt_limit(
    registry: ToolRegistry,
) -> None:
    assert isinstance(caller(registry)("verify_customer", order_id=7).result, Ambiguous)
    blocked = caller(registry, verification_attempts=3)(
        "verify_customer", order_id=7, email=EMAIL_14
    )
    assert isinstance(blocked.result, PolicyBlocked)
    assert blocked.state_updates == {}


def test_order_status_for_owner_delayed(registry: ToolRegistry) -> None:
    result = caller(registry, verified_customer_id=2)("get_order_status", order_id=8)
    assert isinstance(result.result, Ok)
    view = result.result.value
    assert view.estimated_delivery == "28/02/2026"
    assert view.is_delayed
    assert view.delay_note
    assert view.items


def test_order_status_requires_verification(call: Caller) -> None:
    assert isinstance(call("get_order_status", order_id=8).result, PolicyBlocked)


def test_order_status_delivered_is_not_delayed(registry: ToolRegistry) -> None:
    result = caller(registry, verified_customer_id=14)("get_order_status", order_id=7)
    assert isinstance(result.result, Ok)
    assert not result.result.value.is_delayed


def test_return_gherkin_regret_online_day_4_of_7(registry_feb21: ToolRegistry) -> None:
    ask = caller(registry_feb21, verified_customer_id=14)
    channel = ask("check_return_eligibility", order_id=7, reason="regret")
    assert isinstance(channel.result, Ambiguous)
    assert "online" in channel.result.question
    decision = ask("check_return_eligibility", order_id=7, reason="regret", channel="online")
    assert isinstance(decision.result, Ok)
    view = decision.result.value
    assert view.eligible
    assert view.window == "dia 4 de 7 dias corridos"
    assert view.estimate_note


def test_return_in_transit_is_blocked_with_options(registry_feb21: ToolRegistry) -> None:
    result = caller(registry_feb21, verified_customer_id=2)(
        "check_return_eligibility", order_id=8, reason="regret", channel="online"
    )
    assert isinstance(result.result, PolicyBlocked)


def test_return_defect_and_missing_order_and_bad_reason(registry_feb21: ToolRegistry) -> None:
    ask = caller(registry_feb21, verified_customer_id=14)
    assert isinstance(ask("check_return_eligibility", order_id=7, reason="defect").result, Ok)
    assert isinstance(
        ask("check_return_eligibility", order_id=999, reason="defect").result, NotFound
    )
    assert isinstance(ask("check_return_eligibility", order_id=7, reason="troca").result, ToolError)
