"""Classificação de erro transitório x permanente, com as exceções reais de cada SDK."""

from __future__ import annotations

import anthropic
import httpx
import openai
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from google.api_core import exceptions as google_exc
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from emporio.agent.llm.errors import classify, http_status
from emporio.agent.llm.types import LLMError, LLMTransientError

REQUEST = httpx.Request("POST", "https://api.exemplo/v1")


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=REQUEST)


def _openai(cls: type[openai.APIStatusError], status: int) -> Exception:
    return cls("x", response=_resp(status), body=None)


def _anthropic(cls: type[anthropic.APIStatusError], status: int) -> Exception:
    return cls("x", response=_resp(status), body=None)


def _bedrock(code: str, status: int) -> Exception:
    return ClientError(
        {"Error": {"Code": code, "Message": "m"}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "Converse",
    )


def _wrapped_gemini(cause: Exception) -> Exception:
    try:
        raise ChatGoogleGenerativeAIError("Invalid argument provided to Gemini") from cause
    except ChatGoogleGenerativeAIError as exc:
        return exc


TRANSIENT = {
    "openai 429": _openai(openai.RateLimitError, 429),
    "openai 500": _openai(openai.InternalServerError, 500),
    "openai timeout": openai.APITimeoutError(REQUEST),
    "openai conexão": openai.APIConnectionError(request=REQUEST),
    "anthropic 529 overloaded": _anthropic(anthropic.OverloadedError, 529),
    "anthropic 429": _anthropic(anthropic.RateLimitError, 429),
    "anthropic 503": _anthropic(anthropic.ServiceUnavailableError, 503),
    "anthropic timeout": anthropic.APITimeoutError(REQUEST),
    "anthropic 529 genérico": anthropic.APIStatusError("x", response=_resp(529), body=None),
    "gemini 429": google_exc.ResourceExhausted("quota"),
    "gemini 503": google_exc.ServiceUnavailable("indisponível"),
    "gemini 504": google_exc.DeadlineExceeded("prazo"),
    "gemini 500": google_exc.InternalServerError("x"),
    "bedrock throttling": _bedrock("ThrottlingException", 429),
    "bedrock indisponível": _bedrock("ServiceUnavailableException", 503),
    "bedrock código sem status": _bedrock("ModelNotReadyException", 0),
    "bedrock rede": EndpointConnectionError(endpoint_url="https://bedrock"),
    "groq/mistral via httpx": httpx.ConnectTimeout("lento"),
    "timeout nativo": TimeoutError(),
    "status genérico 502": type("BadGatewayish", (Exception,), {"status_code": 502})(),
}

PERMANENT = {
    "openai 401": _openai(openai.AuthenticationError, 401),
    "openai 400": _openai(openai.BadRequestError, 400),
    "anthropic 401": _anthropic(anthropic.AuthenticationError, 401),
    "anthropic 400": _anthropic(anthropic.BadRequestError, 400),
    "gemini 400 embrulhado": _wrapped_gemini(google_exc.InvalidArgument("schema")),
    "gemini 404": google_exc.NotFound("modelo aposentado"),
    "gemini 403": google_exc.PermissionDenied("chave"),
    "bedrock validação": _bedrock("ValidationException", 400),
    "bedrock acesso": _bedrock("AccessDeniedException", 403),
    "valor inválido": ValueError("ruim"),
}


@pytest.mark.parametrize("error", TRANSIENT.values(), ids=TRANSIENT.keys())
def test_transient_errors_by_provider(error: Exception) -> None:
    assert type(classify(error)) is LLMTransientError


@pytest.mark.parametrize("error", PERMANENT.values(), ids=PERMANENT.keys())
def test_permanent_errors_by_provider(error: Exception) -> None:
    assert type(classify(error)) is LLMError


def test_transient_cause_inside_a_wrapper_is_found() -> None:
    wrapper = RuntimeError("adaptador")
    wrapper.__cause__ = google_exc.ServiceUnavailable("x")
    assert type(classify(wrapper)) is LLMTransientError


def test_message_carries_only_class_and_status() -> None:
    error = openai.AuthenticationError(
        "Incorrect API key provided: sk-segredo", response=_resp(401), body=None
    )
    message = str(classify(error))
    assert "401" in message
    assert "AuthenticationError" in message
    assert "sk-segredo" not in message


def test_already_classified_errors_pass_through() -> None:
    original = LLMTransientError("x")
    assert classify(original) is original


def test_http_status_sources() -> None:
    assert http_status(_openai(openai.RateLimitError, 429)) == 429
    assert http_status(google_exc.ResourceExhausted("q")) == 429
    assert http_status(_bedrock("ThrottlingException", 429)) == 429
    assert http_status(type("E", (Exception,), {"response": _resp(503)})()) == 503
    assert http_status(type("E", (Exception,), {"code": True})()) is None
    assert http_status(ValueError()) is None


def test_cyclic_exception_chain_terminates() -> None:
    first, second = ValueError("a"), ValueError("b")
    first.__cause__, second.__cause__ = second, first
    assert type(classify(first)) is LLMError
