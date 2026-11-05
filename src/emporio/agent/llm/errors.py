"""Classificação de erros de qualquer SDK de LLM em transitório ou permanente.

Não importa SDK nenhum: olha o status HTTP exposto pela exceção (``status_code``,
``http_status``, ``code``, ``response``) e o nome das classes na hierarquia, percorrendo a
cadeia ``__cause__``/``__context__`` (adaptadores do LangChain embrulham o erro original).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from emporio.agent.llm.types import LLMError, LLMTransientError

#: 408 timeout, 409 conflito passageiro, 425/429 limite, 5xx e 529 (Anthropic sobrecarregada).
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

_MIN_STATUS, _MAX_STATUS = 100, 600

#: Nomes de classe (em qualquer ponto da hierarquia) que indicam falha passageira.
TRANSIENT_NAMES = frozenset(
    {
        # OpenAI, Anthropic, Groq, Mistral... (SDKs gerados pelo Stainless)
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
        "GatewayTimeoutError",
        "RequestTimeoutError",
        "DeadlineExceededError",
        "RetryableError",
        # Google (api_core)
        "ResourceExhausted",
        "ServiceUnavailable",
        "DeadlineExceeded",
        "TooManyRequests",
        "GatewayTimeout",
        "BadGateway",
        # AWS (botocore / Bedrock)
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "InternalServerException",
        "ModelTimeoutException",
        "EndpointConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        # httpx / rede
        "TimeoutException",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
    }
)

#: Códigos de erro textuais (AWS ``Error.Code``, Google ``status``) que indicam falha passageira.
TRANSIENT_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "InternalServerException",
        "TooManyRequestsException",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "overloaded_error",
        "rate_limit_error",
    }
)


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def http_status(exc: BaseException) -> int | None:
    """Status HTTP exposto pela exceção, se houver.

    Args:
        exc: Exceção de qualquer SDK.

    Returns:
        O status (ex.: 429) ou ``None``.
    """
    for attr in ("status_code", "http_status", "code", "status"):
        value = getattr(exc, attr, None)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and _MIN_STATUS <= value < _MAX_STATUS
        ):
            return value
    response: Any = getattr(exc, "response", None)
    if isinstance(response, dict):  # botocore.ClientError
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status if isinstance(status, int) else None
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _error_code(exc: BaseException) -> str | None:
    response: Any = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        return code if isinstance(code, str) else None
    status = getattr(exc, "status", None)
    return status if isinstance(status, str) else None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & TRANSIENT_NAMES or _error_code(exc) in TRANSIENT_CODES:
        return True
    return http_status(exc) in TRANSIENT_STATUS


def classify(exc: BaseException) -> LLMError:
    """Traduz o erro do provedor para a hierarquia própria.

    A mensagem só leva o nome da classe e o status HTTP: o texto original pode ecoar
    trechos do pedido e nunca vai para log.

    Args:
        exc: Exceção levantada pelo SDK ou pelo adaptador do LangChain.

    Returns:
        ``LLMTransientError`` (vale tentar de novo) ou ``LLMError`` (permanente).
    """
    if isinstance(exc, LLMError):
        return exc
    chain = list(_chain(exc))
    status = next((s for s in map(http_status, chain) if s is not None), None)
    detail = type(exc).__name__ + (f" (HTTP {status})" if status else "")
    if any(_is_transient(e) for e in chain):
        return LLMTransientError(detail)
    return LLMError(f"provedor recusou a chamada: {detail}")
