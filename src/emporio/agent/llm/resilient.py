"""Resiliência: retentativa com backoff, circuit breaker por provedor e cadeia de fallback."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


@dataclass(slots=True)
class _Link:
    provider: LLMProvider
    breaker: CircuitBreaker
    name: str


def _name(provider: LLMProvider, index: int) -> str:
    return str(getattr(provider, "name", None) or f"{type(provider).__name__}[{index}]")


class ResilientLLM:
    """Decorator de :class:`LLMProvider` sobre uma cadeia de provedores.

    Percorre a cadeia em ordem; cada elo tem circuit breaker próprio. Falha transitória
    é retentada com backoff no mesmo elo e, esgotada, passa ao próximo; falha permanente
    passa direto ao próximo. Elo com circuito aberto é pulado.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider | None = None,
        *,
        fallbacks: Sequence[LLMProvider] = (),
        attempts: int = 5,
        breaker: CircuitBreaker | None = None,
        breaker_factory: Callable[[], CircuitBreaker] = CircuitBreaker,
        wait: wait_base | None = None,
    ) -> None:
        """Cria o decorator.

        Args:
            primary: Provedor principal.
            fallback: Primeiro provedor reserva (forma legada de ``fallbacks``).
            fallbacks: Demais reservas, em ordem.
            attempts: Tentativas por chamada no mesmo provedor (só falhas transitórias).
            breaker: Disjuntor do principal; um novo por padrão.
            breaker_factory: Cria o disjuntor de cada reserva.
            wait: Estratégia de espera do tenacity (``wait_none`` nos testes).
        """
        chain = [primary, *([fallback] if fallback is not None else []), *fallbacks]
        self._links = [
            _Link(p, breaker if i == 0 and breaker else breaker_factory(), _name(p, i))
            for i, p in enumerate(chain)
        ]
        self._attempts = attempts
        self._wait = wait or wait_exponential(multiplier=2.0, max=15.0)

    @property
    def chain(self) -> list[str]:
        """Nomes dos elos, em ordem (diagnóstico)."""
        return [link.name for link in self._links]

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
        for position, link in enumerate(self._links):
            if link.breaker.is_open:
                logger.info("circuito aberto, pulando %s", link.name)
                continue
            try:
                result = self._retrying(call, link.provider)
            except LLMError as exc:
                link.breaker.record_failure()
                logger.warning(
                    "provedor %s (elo %d) falhou: %s", link.name, position, type(exc).__name__
                )
            else:
                link.breaker.record_success()
                if position:
                    logger.info("atendido pelo fallback %s", link.name)
                return result
        raise LLMUnavailableError("LLM indisponível")

    def _retrying[R](self, call: Callable[[LLMProvider], R], provider: LLMProvider) -> R:
        retrying = Retrying(
            stop=stop_after_attempt(self._attempts),
            wait=self._wait,
            retry=retry_if_exception_type(LLMTransientError),
            reraise=True,
        )
        return retrying(call, provider)
