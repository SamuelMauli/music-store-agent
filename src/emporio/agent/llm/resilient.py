"""Resiliência: retentativa com backoff, circuit breaker e modelo de fallback."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from emporio.agent.llm.types import (
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMTransientError,
    LLMUnavailableError,
    Message,
    Tier,
    Usage,
)

logger = logging.getLogger("emporio.llm")


class CircuitBreaker:
    """Abre após falhas seguidas e deixa passar uma tentativa depois do período de espera."""

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_seconds: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Cria o disjuntor.

        Args:
            failure_threshold: Falhas seguidas que abrem o circuito.
            reset_seconds: Espera até permitir uma tentativa de prova (half-open).
            monotonic: Relógio monotônico, injetável.
        """
        self._threshold = failure_threshold
        self._reset = reset_seconds
        self._now = monotonic
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """Verdadeiro enquanto o circuito bloqueia chamadas."""
        return self._opened_at is not None and self._now() - self._opened_at < self._reset

    def record_success(self) -> None:
        """Fecha o circuito."""
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Conta a falha e abre (ou reabre) o circuito ao atingir o limite."""
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._now()


class ResilientLLM:
    """Decorator de :class:`LLMProvider` com retentativa, circuit breaker e fallback."""

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider | None = None,
        *,
        attempts: int = 5,
        breaker: CircuitBreaker | None = None,
        wait: wait_base | None = None,
    ) -> None:
        """Cria o decorator.

        Args:
            primary: Provedor principal.
            fallback: Provedor alternativo usado quando o principal falha ou o circuito abre.
            attempts: Tentativas por chamada no mesmo provedor (só falhas transitórias).
            breaker: Disjuntor; um novo por padrão.
            wait: Estratégia de espera do tenacity (``wait_none`` nos testes).
        """
        self._primary = primary
        self._fallback = fallback
        self._attempts = attempts
        self._breaker = breaker or CircuitBreaker()
        self._wait = wait or wait_exponential(multiplier=2.0, max=15.0)

    def complete(
        self,
        messages: list[Message],
        *,
        tier: Tier = "medium",
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Chat protegido."""
        return self._guarded(
            lambda p: p.complete(messages, tier=tier, temperature=temperature, tools=tools)
        )

    def structured[T: BaseModel](
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "small",
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Saída estruturada protegida."""
        return self._guarded(
            lambda p: p.structured(messages, schema, tier=tier, temperature=temperature)
        )

    def _guarded[R](self, call: Callable[[LLMProvider], R]) -> R:
        if not self._breaker.is_open:
            try:
                result = self._retrying(call, self._primary)
            except LLMError as exc:
                self._breaker.record_failure()
                logger.warning("provedor principal falhou: %s", type(exc).__name__)
            else:
                self._breaker.record_success()
                return result
        if self._fallback is not None:
            try:
                return self._retrying(call, self._fallback)
            except LLMError as exc:
                logger.warning("fallback falhou: %s", type(exc).__name__)
        raise LLMUnavailableError("LLM indisponível")

    def _retrying[R](self, call: Callable[[LLMProvider], R], provider: LLMProvider) -> R:
        retrying = Retrying(
            stop=stop_after_attempt(self._attempts),
            wait=self._wait,
            retry=retry_if_exception_type(LLMTransientError),
            reraise=True,
        )
        return retrying(call, provider)
